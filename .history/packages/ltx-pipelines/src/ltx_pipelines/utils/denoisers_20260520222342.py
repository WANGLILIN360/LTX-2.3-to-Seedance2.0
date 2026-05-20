"""Flat denoiser classes — transformer received at call time, not stored.
Three implementations of the :class:`~ltx_pipelines.utils.types.Denoiser` protocol:
* :class:`SimpleDenoiser` — single transformer call, no guidance.
* :class:`GuidedDenoiser` — static guiders, handles CFG + STG + isolated modality.
* :class:`FactoryGuidedDenoiser` — resolves guiders per-step from sigma.
``GuidedDenoiser`` and ``FactoryGuidedDenoiser`` share the core multi-pass
logic via the module-level :func:`_guided_denoise` function, which batches
all guidance passes into a single transformer call.
"""

import torch

from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderFactory, MultiModalGuiderParams
from ltx_core.guidance.perturbations import (
    BatchedPerturbationConfig,
    Perturbation,
    PerturbationConfig,
    PerturbationType,
)
from ltx_core.model.transformer import X0Model
from ltx_core.types import LatentState
from ltx_pipelines.utils.helpers import modality_from_latent_state
from ltx_pipelines.utils.types import DenoisedLatentResult

_POSITIVE_ONLY_GUIDER = MultiModalGuider(
    params=MultiModalGuiderParams(cfg_scale=1.0, stg_scale=0.0, modality_scale=1.0),
)
"""Guider that only runs the conditioned pass and returns cond unchanged."""


def _ensure_guider(guider: MultiModalGuider | None) -> MultiModalGuider:
    """Return the guider as-is, or a positive-only guider for absent modalities."""
    return guider if guider is not None else _POSITIVE_ONLY_GUIDER


def _repeat_state(state: LatentState, n: int) -> LatentState:
    """Repeat a ``LatentState`` *n* times along the batch dimension.
    ``(B, ...) → (n*B, ...)`` by tiling the whole tensor n times, so the
    ordering is ``[item0, item1, ..., item0, item1, ...]`` — matching
    ``torch.cat`` of n per-pass contexts.
    """

    def _repeat(t: torch.Tensor) -> torch.Tensor:
        repeats = [1] * t.dim()
        repeats[0] = n
        return t.repeat(repeats)

    return LatentState(
        latent=_repeat(state.latent),
        denoise_mask=_repeat(state.denoise_mask),
        positions=_repeat(state.positions),
        clean_latent=_repeat(state.clean_latent),
        attention_mask=_repeat(state.attention_mask) if state.attention_mask is not None else None,
    )


def _concat_states(states: list[LatentState]) -> LatentState:
    """Concatenate a list of ``LatentState`` along the batch dimension.
    Used for building per-pass batched states where different passes
    may use different latent states (e.g., noref pass).
    """
    if len(states) == 1:
        return states[0]

    def _cat(tensors: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(tensors, dim=0)

    attention_masks = [s.attention_mask for s in states]
    has_mask = any(m is not None for m in attention_masks)

    return LatentState(
        latent=_cat([s.latent for s in states]),
        denoise_mask=_cat([s.denoise_mask for s in states]),
        positions=_cat([s.positions for s in states]),
        clean_latent=_cat([s.clean_latent for s in states]),
        attention_mask=_cat(attention_masks) if has_mask else None,
    )


def _guided_denoise(  # noqa: PLR0913,PLR0915
    transformer: X0Model,
    video_state: LatentState | None,
    audio_state: LatentState | None,
    sigma: torch.Tensor,
    video_guider: MultiModalGuider,
    audio_guider: MultiModalGuider,
    v_context: torch.Tensor | None,
    a_context: torch.Tensor | None,
    *,
    last_denoised_video: torch.Tensor | None,
    last_denoised_audio: torch.Tensor | None,
    step_index: int,
    force_uncond_pass: bool = False,
    noref_video_state: LatentState | None = None,
    noref_audio_state: LatentState | None = None,
) -> tuple[DenoisedLatentResult | None, DenoisedLatentResult | None]:
    """Core guided denoising — batches all guidance passes into one transformer call.
    Collects per-pass contexts first, then builds a single batched Modality
    per present modality via :func:`modality_from_latent_state`.  When wrapped
    with :class:`~ltx_core.batch_split.BatchSplitAdapter`, the transformer may
    split this batch into sequential chunks internally.
    Guiders must not be ``None``. For absent modalities, callers should pass
    :data:`_POSITIVE_ONLY_GUIDER` (via :func:`_ensure_guider`) so that only
    the conditioned pass runs and ``calculate()`` returns cond unchanged.
    """
    v_skip = video_guider.should_skip_step(step_index)
    a_skip = audio_guider.should_skip_step(step_index)

    if v_skip and a_skip:
        video_result = DenoisedLatentResult.result_or_none(denoised=last_denoised_video)
        audio_result = DenoisedLatentResult.result_or_none(denoised=last_denoised_audio)
        return video_result, audio_result

    if video_state is not None and v_context is None:
        raise ValueError("v_context is required when video_state is provided")
    if audio_state is not None and a_context is None:
        raise ValueError("a_context is required when audio_state is provided")
    # Define passes: (name, video_context, audio_context, perturbation_config).
    # Context is None for absent modalities — filtered out during collection.
    _pass = tuple[str, torch.Tensor | None, torch.Tensor | None, PerturbationConfig]
    passes: list[_pass] = [("cond", v_context, a_context, PerturbationConfig.empty())]

    v_needs_neg = video_guider.do_unconditional_generation() or (force_uncond_pass and video_state is not None)
    a_needs_neg = audio_guider.do_unconditional_generation() or (force_uncond_pass and audio_state is not None)
    if v_needs_neg or a_needs_neg:
        if v_needs_neg and video_guider.negative_context is None:
            raise ValueError("Negative context is required for unconditioned denoising")
        if a_needs_neg and audio_guider.negative_context is None:
            raise ValueError("Negative context is required for unconditioned denoising")
        v_neg = video_guider.negative_context if video_guider.negative_context is not None else v_context
        a_neg = audio_guider.negative_context if audio_guider.negative_context is not None else a_context
        passes.append(("uncond", v_neg, a_neg, PerturbationConfig.empty()))

    stg_perturbations: list[Perturbation] = []
    if video_guider.do_perturbed_generation():
        stg_perturbations.append(
            Perturbation(type=PerturbationType.SKIP_VIDEO_SELF_ATTN, blocks=video_guider.params.stg_blocks)
        )
    if audio_guider.do_perturbed_generation():
        stg_perturbations.append(
            Perturbation(type=PerturbationType.SKIP_AUDIO_SELF_ATTN, blocks=audio_guider.params.stg_blocks)
        )
    if stg_perturbations:
        passes.append(("ptb", v_context, a_context, PerturbationConfig(stg_perturbations)))

    # Identity guidance pass (ID-LoRA style): same text context but without reference tokens.
    # This creates a "no-reference" prediction that we extrapolate away from to amplify
    # identity-specific features from the references.
    v_needs_noref = video_guider.do_identity_guidance() and noref_video_state is not None
    a_needs_noref = audio_guider.do_identity_guidance() and noref_audio_state is not None
    if v_needs_noref or a_needs_noref:
        passes.append(("noref", v_context, a_context, PerturbationConfig.empty()))

    if video_guider.do_isolated_modality_generation() or audio_guider.do_isolated_modality_generation():
        passes.append(
            (
                "mod",
                v_context,
                a_context,
                PerturbationConfig(
                    [
                        Perturbation(type=PerturbationType.SKIP_A2V_CROSS_ATTN, blocks=None),
                        Perturbation(type=PerturbationType.SKIP_V2A_CROSS_ATTN, blocks=None),
                    ]
                ),
            )
        )

    # Collect contexts, repeat states, and build batched modalities.
    pass_names = [name for name, _, _, _ in passes]
    ptb_configs = [ptb for _, _, _, ptb in passes]
    n = len(passes)

    def _batched_sigma(state: LatentState) -> torch.Tensor:
        """Expand scalar sigma to (n * B,) matching the repeated state."""
        return sigma.expand(state.latent.shape[0] * n)

    # Build per-pass video states: noref pass uses noref_video_state, others use video_state.
    batched_video = None
    if video_state is not None:
        v_context = torch.cat([vc for _, vc, _, _ in passes], dim=0)
        per_pass_v_states = []
        for name, _, _, _ in passes:
            if name == "noref" and noref_video_state is not None:
                per_pass_v_states.append(noref_video_state)
            else:
                per_pass_v_states.append(video_state)
        # Concatenate per-pass states along batch dimension
        batched_v_state = _concat_states(per_pass_v_states)
        batched_video = modality_from_latent_state(
            batched_v_state,
            v_context,
            _batched_sigma(video_state),
            enabled=not v_skip,
        )

    # Build per-pass audio states: noref pass uses noref_audio_state, others use audio_state.
    batched_audio = None
    if audio_state is not None:
        a_context = torch.cat([ac for _, _, ac, _ in passes], dim=0)
        per_pass_a_states = []
        for name, _, _, _ in passes:
            if name == "noref" and noref_audio_state is not None:
                per_pass_a_states.append(noref_audio_state)
            else:
                per_pass_a_states.append(audio_state)
        batched_a_state = _concat_states(per_pass_a_states)
        batched_audio = modality_from_latent_state(
            batched_a_state,
            a_context,
            _batched_sigma(audio_state),
            enabled=not a_skip,
        )

    all_v, all_a = transformer(
        video=batched_video, audio=batched_audio, perturbations=BatchedPerturbationConfig(ptb_configs)
    )

    # Split results back and combine via guiders.
    splits_v = list(all_v.chunk(n)) if all_v is not None else [0.0] * n
    splits_a = list(all_a.chunk(n)) if all_a is not None else [0.0] * n
    r = dict(zip(pass_names, zip(splits_v, splits_a, strict=True), strict=True))

    cond_v, cond_a = r["cond"]
    uncond_v, uncond_a = r.get("uncond", (0.0, 0.0))
    ptb_v, ptb_a = r.get("ptb", (0.0, 0.0))
    mod_v, mod_a = r.get("mod", (0.0, 0.0))
    noref_v, noref_a = r.get("noref", (0.0, 0.0))

    denoised_video = last_denoised_video if v_skip else video_guider.calculate(cond_v, uncond_v, ptb_v, mod_v, noref_v)
    denoised_audio = last_denoised_audio if a_skip else audio_guider.calculate(cond_a, uncond_a, ptb_a, mod_a, noref_a)
    return (
        DenoisedLatentResult.result_or_none(
            denoised=denoised_video, uncond=uncond_v, cond=cond_v, ptb=ptb_v, mod=mod_v
        ),
        DenoisedLatentResult.result_or_none(
            denoised=denoised_audio, uncond=uncond_a, cond=cond_a, ptb=ptb_a, mod=mod_a
        ),
    )


class SimpleDenoiser:
    """Single transformer call, no guidance.
    Passes ``None`` Modality for absent modalities.
    """

    def __init__(
        self,
        v_context: torch.Tensor | None,
        a_context: torch.Tensor | None,
    ) -> None:
        self.v_context = v_context
        self.a_context = a_context

    def __call__(
        self,
        transformer: X0Model,
        video_state: LatentState | None,
        audio_state: LatentState | None,
        sigmas: torch.Tensor,
        step_index: int,
    ) -> tuple[DenoisedLatentResult | None, DenoisedLatentResult | None]:
        sigma = sigmas[step_index]
        pos_video = modality_from_latent_state(video_state, self.v_context, sigma) if video_state is not None else None
        pos_audio = modality_from_latent_state(audio_state, self.a_context, sigma) if audio_state is not None else None
        denoised_video, denoised_audio = transformer(video=pos_video, audio=pos_audio, perturbations=None)
        return (
            DenoisedLatentResult.result_or_none(denoised=denoised_video),
            DenoisedLatentResult.result_or_none(denoised=denoised_audio),
        )


class GuidedDenoiser:
    """Static guiders — handles CFG + STG + isolated modality.
    Context/guider can be ``None`` for absent modalities (a positive-only
    guider is substituted at call time).
    """

    def __init__(
        self,
        v_context: torch.Tensor | None,
        a_context: torch.Tensor | None,
        video_guider: MultiModalGuider | None = None,
        audio_guider: MultiModalGuider | None = None,
        force_uncond_pass: bool = False,
        noref_video_state: LatentState | None = None,
        noref_audio_state: LatentState | None = None,
    ) -> None:
        self.v_context = v_context
        self.a_context = a_context
        self.video_guider = video_guider
        self.audio_guider = audio_guider
        self.force_uncond_pass = force_uncond_pass
        self.noref_video_state = noref_video_state
        self.noref_audio_state = noref_audio_state
        self._last_denoised_video: torch.Tensor | None = None
        self._last_denoised_audio: torch.Tensor | None = None

    def __call__(
        self,
        transformer: X0Model,
        video_state: LatentState | None,
        audio_state: LatentState | None,
        sigmas: torch.Tensor,
        step_index: int,
    ) -> tuple[DenoisedLatentResult | None, DenoisedLatentResult | None]:
        guided_denoise_result_v, guided_denoise_result_a = _guided_denoise(
            transformer=transformer,
            video_state=video_state,
            audio_state=audio_state,
            sigma=sigmas[step_index],
            video_guider=_ensure_guider(self.video_guider),
            audio_guider=_ensure_guider(self.audio_guider),
            v_context=self.v_context,
            a_context=self.a_context,
            last_denoised_video=self._last_denoised_video,
            last_denoised_audio=self._last_denoised_audio,
            step_index=step_index,
            force_uncond_pass=self.force_uncond_pass,
            noref_video_state=self.noref_video_state,
            noref_audio_state=self.noref_audio_state,
        )
        self._last_denoised_video = guided_denoise_result_v.denoised
        self._last_denoised_audio = guided_denoise_result_a.denoised
        return guided_denoise_result_v, guided_denoise_result_a


class FactoryGuidedDenoiser:
    """Resolves guiders per-step from sigma, then delegates to shared guided logic."""

    def __init__(
        self,
        v_context: torch.Tensor | None,
        a_context: torch.Tensor | None,
        video_guider_factory: MultiModalGuiderFactory | None = None,
        audio_guider_factory: MultiModalGuiderFactory | None = None,
        force_uncond_pass: bool = False,
        noref_video_state: LatentState | None = None,
        noref_audio_state: LatentState | None = None,
    ) -> None:
        self.v_context = v_context
        self.a_context = a_context
        self.video_guider_factory = video_guider_factory
        self.audio_guider_factory = audio_guider_factory
        self.force_uncond_pass = force_uncond_pass
        self.noref_video_state = noref_video_state
        self.noref_audio_state = noref_audio_state
        self._last_denoised_video: torch.Tensor | None = None
        self._last_denoised_audio: torch.Tensor | None = None
        self._sigma_vals_cached: list[float] | None = None

    def __call__(
        self,
        transformer: X0Model,
        video_state: LatentState | None,
        audio_state: LatentState | None,
        sigmas: torch.Tensor,
        step_index: int,
    ) -> tuple[DenoisedLatentResult | None, DenoisedLatentResult | None]:
        if self._sigma_vals_cached is None:
            self._sigma_vals_cached = sigmas.detach().cpu().tolist()
        sigma_val = self._sigma_vals_cached[step_index]

        video_guider = _ensure_guider(
            self.video_guider_factory.build_from_sigma(sigma_val) if self.video_guider_factory else None
        )
        audio_guider = _ensure_guider(
            (self.audio_guider_factory or self.video_guider_factory).build_from_sigma(sigma_val)
            if self.video_guider_factory or self.audio_guider_factory
            else None
        )

        guided_denoise_result_v, guided_denoise_result_a = _guided_denoise(
            transformer=transformer,
            video_state=video_state,
            audio_state=audio_state,
            sigma=sigmas[step_index],
            video_guider=video_guider,
            audio_guider=audio_guider,
            v_context=self.v_context,
            a_context=self.a_context,
            last_denoised_video=self._last_denoised_video,
            last_denoised_audio=self._last_denoised_audio,
            step_index=step_index,
            force_uncond_pass=self.force_uncond_pass,
            noref_video_state=self.noref_video_state,
            noref_audio_state=self.noref_audio_state,
        )
        self._last_denoised_video = guided_denoise_result_v.denoised
        self._last_denoised_audio = guided_denoise_result_a.denoised
        return guided_denoise_result_v, guided_denoise_result_a
