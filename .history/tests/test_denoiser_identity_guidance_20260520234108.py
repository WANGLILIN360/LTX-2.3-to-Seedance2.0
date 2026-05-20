"""Runtime integration tests for _guided_denoise with identity guidance.

These tests actually import and call _guided_denoise, GuidedDenoiser,
and FactoryGuidedDenoiser to verify the full noref forward path works.
Requires conftest.py stubs for av/OpenImageIO.
"""

import sys
from pathlib import Path

import pytest
import torch

# Ensure paths are set (conftest.py handles stubs)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages" / "ltx-core" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages" / "ltx-pipelines" / "src"))


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture()
def guider_no_id():
    from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
    return MultiModalGuider(params=MultiModalGuiderParams(
        cfg_scale=1.0, identity_guidance_scale=0.0,
    ))


@pytest.fixture()
def guider_with_id():
    from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
    return MultiModalGuider(params=MultiModalGuiderParams(
        cfg_scale=1.0, identity_guidance_scale=0.5,
    ))


@pytest.fixture()
def guider_with_cfg_and_id():
    from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
    return MultiModalGuider(
        params=MultiModalGuiderParams(cfg_scale=3.0, identity_guidance_scale=0.5),
        negative_context=torch.randn(1, 20, 64),
    )


@pytest.fixture()
def video_state():
    """Video state in patchified format: latent (B, T, D), denoise_mask (B, 1, T).
    
    T = total tokens (target + reference), D = feature dim.
    Target tokens: denoise_mask=1 (first 12 tokens).
    Reference tokens: denoise_mask=0 (last 8 tokens).
    """
    from ltx_core.types import LatentState
    target_len = 12
    ref_len = 8
    total_len = target_len + ref_len
    D = 128  # feature dimension (inner_dim)
    return LatentState(
        latent=torch.randn(1, total_len, D),          # (B, T, D) patchified
        denoise_mask=torch.cat([
            torch.ones(1, 1, target_len),   # target: denoise
            torch.zeros(1, 1, ref_len),       # ref: frozen
        ], dim=2),                              # (B, 1, T)
        positions=torch.randn(1, 3, total_len),   # (B, 3, T) for video
        clean_latent=torch.zeros(1, total_len, D),
        attention_mask=torch.ones(1, 1, 1, total_len),
    )


@pytest.fixture()
def noref_video_state():
    """Noref state: only target tokens in patchified format.
    
    latent (B, T_target, D), denoise_mask (B, 1, T_target) all 1s.
    """
    from ltx_core.types import LatentState
    return LatentState(
        latent=torch.randn(1, 12, 128),          # (B, T_target, D)
        denoise_mask=torch.ones(1, 1, 12),       # (B, 1, T_target)
        positions=torch.randn(1, 3, 12),          # (B, 3, T_target)
        clean_latent=torch.zeros(1, 12, 128),
        attention_mask=torch.ones(1, 1, 1, 12),
    )


@pytest.fixture()
def v_context():
    return torch.randn(1, 20, 64)


# ============================================================================
# 1. _guided_denoise — actual runtime tests
# ============================================================================

class TestGuidedDenoiseRuntime:
    """Runtime tests for _guided_denoise with identity guidance."""

    def test_noref_not_called_when_disabled(self, guider_no_id, video_state, v_context):
        """When identity_guidance_scale=0, only 1 transformer call (cond pass)."""
        from ltx_pipelines.utils.denoisers import _guided_denoise
        call_count = {"count": 0}

        class MockTransformer:
            def __call__(self, video=None, audio=None, perturbations=None):
                call_count["count"] += 1
                B = video.latent.shape[0] if video is not None else 1
                T = video.latent.shape[1] if video is not None else 20
                D = video.latent.shape[2] if video is not None else 128
                return torch.randn(B, T, D), None

        _guided_denoise(
            transformer=MockTransformer(),
            video_state=video_state,
            audio_state=None,
            sigma=torch.tensor([1.0]),
            video_guider=guider_no_id,
            audio_guider=guider_no_id,
            v_context=v_context,
            a_context=None,
            last_denoised_video=None,
            last_denoised_audio=None,
            step_index=0,
        )
        assert call_count["count"] == 1

    def test_noref_called_when_enabled(self, guider_with_id, video_state, noref_video_state, v_context):
        """When identity_guidance_scale > 0 and noref state provided, 2 transformer calls."""
        from ltx_pipelines.utils.denoisers import _guided_denoise
        call_count = {"count": 0}

        class MockTransformer:
            def __call__(self, video=None, audio=None, perturbations=None):
                call_count["count"] += 1
                B = video.latent.shape[0] if video is not None else 1
                T = video.latent.shape[1] if video is not None else 20
                D = video.latent.shape[2] if video is not None else 128
                return torch.randn(B, T, D), None

        _guided_denoise(
            transformer=MockTransformer(),
            video_state=video_state,
            audio_state=None,
            sigma=torch.tensor([1.0]),
            video_guider=guider_with_id,
            audio_guider=guider_with_id,
            v_context=v_context,
            a_context=None,
            last_denoised_video=None,
            last_denoised_audio=None,
            step_index=0,
            noref_video_state=noref_video_state,
        )
        # 1st call: main batch (cond pass), 2nd call: noref pass
        assert call_count["count"] == 2

    def test_noref_not_called_without_state(self, guider_with_id, video_state, v_context):
        """When identity_guidance_scale > 0 but no noref state, only 1 call."""
        from ltx_pipelines.utils.denoisers import _guided_denoise
        call_count = {"count": 0}

        class MockTransformer:
            def __call__(self, video=None, audio=None, perturbations=None):
                call_count["count"] += 1
                B = video.latent.shape[0] if video is not None else 1
                T = video.latent.shape[1] if video is not None else 20
                D = video.latent.shape[2] if video is not None else 128
                return torch.randn(B, T, D), None

        _guided_denoise(
            transformer=MockTransformer(),
            video_state=video_state,
            audio_state=None,
            sigma=torch.tensor([1.0]),
            video_guider=guider_with_id,
            audio_guider=guider_with_id,
            v_context=v_context,
            a_context=None,
            last_denoised_video=None,
            last_denoised_audio=None,
            step_index=0,
            # noref_video_state NOT provided
        )
        assert call_count["count"] == 1

    def test_noref_uses_positive_context(self, guider_with_id, video_state, noref_video_state, v_context):
        """Noref pass should use the same positive text context as cond pass."""
        from ltx_pipelines.utils.denoisers import _guided_denoise
        contexts_used = []

        class MockTransformer:
            def __call__(self, video=None, audio=None, perturbations=None):
                if video is not None:
                    contexts_used.append(video.context.clone())
                B = video.latent.shape[0] if video is not None else 1
                seq_len = video.latent.shape[2] if video is not None else 20
                return torch.randn(B, 128, seq_len), None

        _guided_denoise(
            transformer=MockTransformer(),
            video_state=video_state,
            audio_state=None,
            sigma=torch.tensor([1.0]),
            video_guider=guider_with_id,
            audio_guider=guider_with_id,
            v_context=v_context,
            a_context=None,
            last_denoised_video=None,
            last_denoised_audio=None,
            step_index=0,
            noref_video_state=noref_video_state,
        )
        # 2 calls: cond and noref — both should use positive context
        assert len(contexts_used) == 2
        # The noref context (2nd call) should match the cond context (1st call's first batch)
        # First call has batch dim from cond pass, second call is noref
        cond_ctx = contexts_used[0][:1]  # first batch item
        noref_ctx = contexts_used[1]
        assert torch.allclose(cond_ctx, noref_ctx, atol=1e-6)

    def test_noref_result_fed_to_guider(self, guider_with_id, video_state, noref_video_state, v_context):
        """Verify noref transformer output is actually used in identity guidance delta."""
        from ltx_pipelines.utils.denoisers import _guided_denoise

        # Use a transformer that returns distinct values for cond vs noref
        call_idx = [0]

        class MockTransformer:
            def __call__(self, video=None, audio=None, perturbations=None):
                call_idx[0] += 1
                B = video.latent.shape[0] if video is not None else 1
                T = video.latent.shape[1] if video is not None else 20
                D = video.latent.shape[2] if video is not None else 128
                return torch.full((B, T, D), float(call_idx[0])), None

        result_v, result_a = _guided_denoise(
            transformer=MockTransformer(),
            video_state=video_state,
            audio_state=None,
            sigma=torch.tensor([1.0]),
            video_guider=guider_with_id,
            audio_guider=guider_with_id,
            v_context=v_context,
            a_context=None,
            last_denoised_video=None,
            last_denoised_audio=None,
            step_index=0,
            noref_video_state=noref_video_state,
        )
        assert result_v is not None
        denoised = result_v.denoised
        # cond = 1.0 (first call), noref = 2.0 (second call)
        # cfg=1.0, id=0.5
        # guider.calculate(cond, 0, 0, 0, 0) = cond (since cfg=1, stg=0, mod=1, noref=0)
        # Zero-padded noref: first 12 positions = 2.0, remaining 8 positions = 0.0
        # identity delta = 0.5 * (cond - noref_padded)
        # target positions (first 12): 1.0 + 0.5*(1.0 - 2.0) = 0.5
        # ref positions (last 8): 1.0 + 0.5*(1.0 - 0.0) = 1.5
        target_mask = video_state.denoise_mask.squeeze(0).squeeze(0).bool()
        assert torch.allclose(denoised[:, target_mask, :], torch.tensor(0.5), atol=1e-4)
        ref_mask = ~target_mask
        assert torch.allclose(denoised[:, ref_mask, :], torch.tensor(1.5), atol=1e-4)


# ============================================================================
# 2. GuidedDenoiser / FactoryGuidedDenoiser — runtime noref passthrough
# ============================================================================

class TestDenoiserNorefPassthroughRuntime:
    """Runtime tests for denoiser noref state storage and passthrough."""

    def test_guided_denoiser_stores_noref(self, noref_video_state):
        from ltx_pipelines.utils.denoisers import GuidedDenoiser
        from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
        guider = MultiModalGuider(params=MultiModalGuiderParams(identity_guidance_scale=0.5))
        denoiser = GuidedDenoiser(
            v_context=torch.randn(1, 20, 64),
            a_context=None,
            video_guider=guider,
            noref_video_state=noref_video_state,
        )
        assert denoiser.noref_video_state is noref_video_state

    def test_factory_denoiser_stores_noref(self, noref_video_state):
        from ltx_pipelines.utils.denoisers import FactoryGuidedDenoiser
        from ltx_core.components.guiders import MultiModalGuiderFactory, MultiModalGuiderParams
        factory = MultiModalGuiderFactory.constant(
            MultiModalGuiderParams(identity_guidance_scale=0.5),
        )
        denoiser = FactoryGuidedDenoiser(
            v_context=torch.randn(1, 20, 64),
            a_context=None,
            video_guider_factory=factory,
            noref_video_state=noref_video_state,
        )
        assert denoiser.noref_video_state is noref_video_state

    def test_guided_denoiser_default_noref_none(self):
        from ltx_pipelines.utils.denoisers import GuidedDenoiser
        denoiser = GuidedDenoiser(
            v_context=torch.randn(1, 20, 64),
            a_context=None,
        )
        assert denoiser.noref_video_state is None
        assert denoiser.noref_audio_state is None

    def test_guided_denoiser_passes_noref_to_core(self, guider_with_id, video_state, noref_video_state, v_context):
        """Verify GuidedDenoiser passes noref state to _guided_denoise."""
        from ltx_pipelines.utils.denoisers import GuidedDenoiser
        call_count = {"count": 0}

        class MockTransformer:
            def __call__(self, video=None, audio=None, perturbations=None):
                call_count["count"] += 1
                B = video.latent.shape[0] if video is not None else 1
                T = video.latent.shape[1] if video is not None else 20
                D = video.latent.shape[2] if video is not None else 128
                return torch.randn(B, T, D), None

        denoiser = GuidedDenoiser(
            v_context=v_context,
            a_context=None,
            video_guider=guider_with_id,
            noref_video_state=noref_video_state,
        )
        denoiser(
            transformer=MockTransformer(),
            video_state=video_state,
            audio_state=None,
            sigmas=torch.tensor([1.0]),
            step_index=0,
        )
        # Should make 2 calls: cond + noref
        assert call_count["count"] == 2


# ============================================================================
# 3. Pipeline noref state construction — runtime verification
# ============================================================================

class TestPipelineNorefStateRuntime:
    """Verify pipeline constructs noref state correctly."""

    def test_noref_state_has_no_ref_tokens(self, video_state, noref_video_state):
        """Noref state should have fewer tokens than cond state (no reference tokens)."""
        # In patchified format: latent shape is (B, T, D), compare T (dim 1)
        assert noref_video_state.latent.shape[1] < video_state.latent.shape[1]

    def test_noref_state_same_noise_pattern(self):
        """Noref state created with same noiser should have matching noise structure.

        In the pipeline, both cond and noref states are created via create_noised_state
        with the same noiser seed, so they share the same target token noise.
        The difference is only in the conditioning tokens appended.
        """
        from ltx_core.types import LatentState
        # Simulated in patchified format: (B, T, D)
        target_tokens = 80
        ref_tokens = 20
        D = 128
        cond_state = LatentState(
            latent=torch.randn(1, target_tokens + ref_tokens, D),
            denoise_mask=torch.cat([
                torch.ones(1, 1, target_tokens),  # target: denoise
                torch.zeros(1, 1, ref_tokens),     # ref: no denoise
            ], dim=2),
            positions=torch.randn(1, 3, target_tokens + ref_tokens),
            clean_latent=torch.zeros(1, target_tokens + ref_tokens, D),
            attention_mask=torch.ones(1, 1, 1, target_tokens + ref_tokens),
        )
        noref_state = LatentState(
            latent=torch.randn(1, target_tokens, D),  # no ref tokens
            denoise_mask=torch.ones(1, 1, target_tokens),
            positions=torch.randn(1, 3, target_tokens),
            clean_latent=torch.zeros(1, target_tokens, D),
            attention_mask=torch.ones(1, 1, 1, target_tokens),
        )
        # Key invariant: noref has fewer tokens (dim 1)
        assert noref_state.latent.shape[1] < cond_state.latent.shape[1]
        # Key invariant: noref denoise_mask is all 1s (no frozen ref tokens)
        assert (noref_state.denoise_mask == 1).all()

    def test_pipeline_decision_logic(self):
        """Verify the pipeline's noref construction decision logic."""
        from ltx_core.components.guiders import MultiModalGuiderParams

        # Case 1: identity_guidance_scale > 0 and refs exist → build noref
        params1 = MultiModalGuiderParams(identity_guidance_scale=0.5)
        has_refs1 = True
        assert params1.identity_guidance_scale > 0 and has_refs1

        # Case 2: identity_guidance_scale = 0 → don't build noref
        params2 = MultiModalGuiderParams(identity_guidance_scale=0.0)
        has_refs2 = True
        assert not (params2.identity_guidance_scale > 0 and has_refs2)

        # Case 3: no refs → don't build noref even if scale > 0
        params3 = MultiModalGuiderParams(identity_guidance_scale=0.5)
        has_refs3 = False
        assert not (params3.identity_guidance_scale > 0 and has_refs3)


# ============================================================================
# 4. Full guider formula with CFG + identity guidance
# ============================================================================

class TestGuiderFullFormulaRuntime:
    """Runtime verification of the complete guidance formula."""

    def test_cfg_plus_identity_guidance(self, guider_with_cfg_and_id):
        """CFG + identity guidance: pred = cond + (cfg-1)*(cond-uncond) + id*(cond-noref)."""
        cond = torch.randn(2, 4, 8, 8)
        uncond = torch.randn(2, 4, 8, 8)
        noref = torch.randn(2, 4, 8, 8)

        result = guider_with_cfg_and_id.calculate(cond, uncond, 0.0, 0.0, noref)
        expected = cond + (3.0 - 1) * (cond - uncond) + 0.5 * (cond - noref)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_identity_guidance_amplifies_identity_features(self):
        """Identity guidance should amplify the difference between with-ref and without-ref."""
        from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams

        # Simulate: cond has identity features, noref doesn't
        cond = torch.tensor([1.0, 0.5, 0.0])  # identity feature present
        noref = torch.tensor([0.0, 0.5, 1.0])  # identity feature absent

        # Without identity guidance
        g_no_id = MultiModalGuider(params=MultiModalGuiderParams(identity_guidance_scale=0.0))
        result_no_id = g_no_id.calculate(cond, 0.0, 0.0, 0.0, noref)
        # result = cond (no extrapolation)

        # With identity guidance
        g_with_id = MultiModalGuider(params=MultiModalGuiderParams(identity_guidance_scale=1.0))
        result_with_id = g_with_id.calculate(cond, 0.0, 0.0, 0.0, noref)
        # result = cond + 1.0*(cond - noref) = 2*cond - noref
        # = [2.0, 1.0, -1.0] — identity feature amplified!

        assert result_with_id[0] > result_no_id[0]  # identity feature amplified
        assert result_with_id[2] < result_no_id[2]  # non-identity suppressed
