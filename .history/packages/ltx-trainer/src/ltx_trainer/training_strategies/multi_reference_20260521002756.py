"""Multi-reference text-to-video training strategy.

Implements training with multiple simultaneous reference inputs (images, videos,
audio), each with attribute-aware attention routing. Inspired by Seedance 2.0's
unified multimodal conditioning approach.

This strategy extends the video_to_video (IC-LoRA) pattern to support:
- Multiple image references (identity, appearance, style)
- Multiple video references (motion, camera, scene)
- Multiple audio references (rhythm, mood)
- Attribute-aware attention masks that control per-reference influence
- Joint audio-video training

Training data structure:
    preprocessed_data_root/
    ├── latents/              # Target video latents
    ├── conditions/           # Text embeddings
    ├── audio_latents/        # Target audio latents (optional)
    ├── ref_image_latents/    # Image reference latents (multiple per sample)
    ├── ref_video_latents/    # Video reference latents (multiple per sample)
    └── ref_audio_latents/    # Audio reference latents (multiple per sample)

Each ref_* directory contains subdirectories per reference index:
    ref_image_latents/
    ├── 0/  # First image reference
    ├── 1/  # Second image reference
    └── ...
"""

from __future__ import annotations

from typing import Any, Literal

import torch
from pydantic import Field
from torch import Tensor

from ltx_core.conditioning.types.multi_reference_cond import (
    DEFAULT_ATTRIBUTE_WEIGHTS,
    ReferenceAttribute,
    ReferenceItem,
    ReferenceModality,
    ReferenceSlotEmbedding,
)
from ltx_core.conditioning.mask_utils import build_attention_mask
from ltx_core.model.transformer.modality import Modality
from ltx_trainer import logger
from ltx_trainer.timestep_samplers import TimestepSampler
from ltx_trainer.training_strategies.base_strategy import (
    DEFAULT_FPS,
    ModelInputs,
    TrainingStrategy,
    TrainingStrategyConfigBase,
)


class MultiReferenceConfig(TrainingStrategyConfigBase):
    """Configuration for multi-reference training strategy."""

    name: Literal["multi_reference"] = "multi_reference"

    first_frame_conditioning_p: float = Field(
        default=0.1,
        description="Probability of conditioning on the first frame during training",
        ge=0.0,
        le=1.0,
    )

    with_audio: bool = Field(
        default=False,
        description="Whether to include audio in training (joint audio-video generation)",
    )

    audio_latents_dir: str = Field(
        default="audio_latents",
        description="Directory name for target audio latents",
    )

    # Reference directory configuration
    ref_image_latents_dir: str = Field(
        default="ref_image_latents",
        description="Directory name for image reference latents",
    )

    ref_video_latents_dir: str = Field(
        default="ref_video_latents",
        description="Directory name for video reference latents",
    )

    ref_audio_latents_dir: str = Field(
        default="ref_audio_latents",
        description="Directory name for audio reference latents",
    )

    # Reference count limits
    max_image_references: int = Field(
        default=9,
        description="Maximum number of image references per sample",
        ge=0,
        le=9,
    )

    max_video_references: int = Field(
        default=3,
        description="Maximum number of video references per sample",
        ge=0,
        le=3,
    )

    max_audio_references: int = Field(
        default=3,
        description="Maximum number of audio references per sample",
        ge=0,
        le=3,
    )

    # Reference dropout (for robustness)
    reference_dropout_p: float = Field(
        default=0.1,
        description="Probability of dropping each reference during training (classifier-free guidance style)",
        ge=0.0,
        le=1.0,
    )

    # Default attribute assignment for references without explicit tags
    default_image_attributes: list[str] = Field(
        default=["identity", "appearance"],
        description="Default attribute tags for image references",
    )

    default_video_attributes: list[str] = Field(
        default=["motion", "camera"],
        description="Default attribute tags for video references",
    )

    default_audio_attributes: list[str] = Field(
        default=["audio_rhythm", "audio_mood"],
        description="Default attribute tags for audio references",
    )

    # Reference downscale
    reference_downscale_factor: int = Field(
        default=1,
        description="Spatial downscale factor for video/image references",
        ge=1,
    )

    # Audio reference conditioning
    with_audio_references: bool = Field(
        default=False,
        description="Whether to include audio references in training",
    )

    # Identity guidance (noref loss)
    identity_guidance_scale: float = Field(
        default=0.0,
        description="Identity guidance scale for training. Extrapolates between "
        "with-reference and without-reference predictions. "
        "0.0 = disabled. Requires a separate transformer forward pass.",
        ge=0.0,
    )

    # Slot embedding for @mention binding
    with_slot_embeddings: bool = Field(
        default=False,
        description="Whether to use learnable ReferenceSlotEmbedding as auxiliary "
        "training signal. Adds a unique embedding per reference slot to both "
        "reference tokens and text @mention positions, helping the model "
        "learn which reference corresponds to which @mention.",
    )

    slot_embed_dim: int = Field(
        default=128,
        description="Dimension of slot embeddings. Should match the latent channel dim.",
    )


class MultiReferenceStrategy(TrainingStrategy):
    """Multi-reference training strategy.

    Concatenates multiple reference latents (images, videos, audio) with the
    target latent, using attribute-aware attention masks to control per-reference
    influence. Loss is computed only on the target portion.

    The sequence layout is:
        [ref_img_1 | ref_img_2 | ... | ref_vid_1 | ... | ref_aud_1 | ... | target]

    Each reference group has its own attention mask weight based on its
    attribute tags, enabling the model to learn attribute routing.
    """

    config: MultiReferenceConfig

    def __init__(self, config: MultiReferenceConfig):
        super().__init__(config)
        self._reference_downscale_factor: int | None = None
        self._slot_embedding: ReferenceSlotEmbedding | None = None
        if config.with_slot_embeddings:
            self._slot_embedding = ReferenceSlotEmbedding(
                embed_dim=config.slot_embed_dim,
            )

    @property
    def requires_audio(self) -> bool:
        return self.config.with_audio or self.config.with_audio_references

    def get_extra_trainable_params(self) -> list[torch.nn.Parameter]:
        """Return extra trainable parameters from this strategy (e.g., slot embeddings)."""
        if self._slot_embedding is not None:
            return list(self._slot_embedding.parameters())
        return []

    def get_data_sources(self) -> dict[str, str]:
        """Multi-reference training requires latents, conditions, and reference latents."""
        sources: dict[str, str] = {
            "latents": "latents",
            "conditions": "conditions",
            self.config.ref_image_latents_dir: "ref_image_latents",
            self.config.ref_video_latents_dir: "ref_video_latents",
        }

        if self.config.with_audio:
            sources[self.config.audio_latents_dir] = "audio_latents"

        if self.config.with_audio_references:
            sources[self.config.ref_audio_latents_dir] = "ref_audio_latents"

        return sources

    def prepare_training_inputs(  # noqa: PLR0915
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,
    ) -> ModelInputs:
        """Prepare inputs for multi-reference training."""
        # Get target latents
        latents = batch["latents"]
        target_latents = latents["latents"]

        num_frames = latents["num_frames"][0].item()
        height = latents["height"][0].item()
        width = latents["width"][0].item()

        # Patchify target
        target_latents = self._video_patchifier.patchify(target_latents)

        # Handle FPS
        fps = latents.get("fps", None)
        if fps is not None and not torch.all(fps == fps[0]):
            logger.warning(
                f"Different FPS values found in the batch. Found: {fps.tolist()}, using the first one: {fps[0].item()}"
            )
        fps = fps[0].item() if fps is not None else DEFAULT_FPS

        # Get text embeddings
        conditions = batch["conditions"]
        video_prompt_embeds = conditions["video_prompt_embeds"]
        audio_prompt_embeds = conditions["audio_prompt_embeds"]
        prompt_attention_mask = conditions["prompt_attention_mask"]

        batch_size = target_latents.shape[0]
        target_seq_len = target_latents.shape[1]
        device = target_latents.device
        dtype = target_latents.dtype

        # ---- Collect and patchify reference latents ----
        ref_groups: list[_ReferenceGroup] = []

        # Image references
        ref_image_data = batch.get("ref_image_latents", {})
        if isinstance(ref_image_data, dict) and "latents" in ref_image_data:
            ref_groups.extend(self._process_image_references(
                ref_image_data, batch_size, device, dtype, fps,
            ))

        # Video references
        ref_video_data = batch.get("ref_video_latents", {})
        if isinstance(ref_video_data, dict) and "latents" in ref_video_data:
            ref_groups.extend(self._process_video_references(
                ref_video_data, batch_size, device, dtype, fps,
            ))

        # Audio references
        if self.config.with_audio_references:
            ref_audio_data = batch.get("ref_audio_latents", {})
            if isinstance(ref_audio_data, dict) and "latents" in ref_audio_data:
                ref_groups.extend(self._process_audio_references(
                    ref_audio_data, batch_size, device, dtype,
                ))

        # Apply reference dropout
        if self.config.reference_dropout_p > 0:
            ref_groups = self._apply_reference_dropout(ref_groups, batch_size, device)

        # ---- Apply slot embeddings to reference tokens ----
        if self._slot_embedding is not None:
            self._slot_embedding = self._slot_embedding.to(device=device, dtype=dtype)
            for group in ref_groups:
                group.tokens = self._slot_embedding.add_to_tokens(
                    group.tokens, group.slot_id,
                )

        # ---- Build combined sequence ----
        # Concatenate all reference tokens + target tokens
        all_ref_tokens = []
        all_ref_positions = []
        all_ref_conditioning_masks = []
        total_ref_seq_len = 0

        for group in ref_groups:
            all_ref_tokens.append(group.tokens)
            all_ref_positions.append(group.positions)
            all_ref_conditioning_masks.append(
                torch.ones(batch_size, group.seq_len, dtype=torch.bool, device=device)
            )
            total_ref_seq_len += group.seq_len

        # Target conditioning mask (first frame conditioning)
        target_conditioning_mask = self._create_first_frame_conditioning_mask(
            batch_size=batch_size,
            sequence_length=target_seq_len,
            height=height,
            width=width,
            device=device,
            first_frame_conditioning_p=self.config.first_frame_conditioning_p,
        )

        # Combined conditioning mask
        if all_ref_conditioning_masks:
            ref_conditioning_mask = torch.cat(all_ref_conditioning_masks, dim=1)
            conditioning_mask = torch.cat([ref_conditioning_mask, target_conditioning_mask], dim=1)
        else:
            conditioning_mask = target_conditioning_mask

        # Sample noise and sigmas
        sigmas = timestep_sampler.sample_for(target_latents)
        noise = torch.randn_like(target_latents)
        sigmas_expanded = sigmas.view(-1, 1, 1)

        # Apply noise to target
        noisy_target = (1 - sigmas_expanded) * target_latents + sigmas_expanded * noise
        target_conditioning_mask_expanded = target_conditioning_mask.unsqueeze(-1)
        noisy_target = torch.where(target_conditioning_mask_expanded, target_latents, noisy_target)

        # Targets for loss computation
        targets = noise - target_latents

        # Concatenate reference (clean) and target (noisy)
        if all_ref_tokens:
            ref_combined = torch.cat(all_ref_tokens, dim=1)
            combined_latents = torch.cat([ref_combined, noisy_target], dim=1)
        else:
            combined_latents = noisy_target

        # Create per-token timesteps
        timesteps = self._create_per_token_timesteps(conditioning_mask, sigmas.squeeze())

        # Generate positions
        target_positions = self._get_video_positions(
            num_frames=num_frames,
            height=height,
            width=width,
            batch_size=batch_size,
            fps=fps,
            device=device,
            dtype=dtype,
        )

        if all_ref_positions:
            ref_positions_combined = torch.cat(all_ref_positions, dim=2)
            positions = torch.cat([ref_positions_combined, target_positions], dim=2)
        else:
            positions = target_positions

        # Build attribute-aware attention mask for the combined sequence
        attention_mask = self._build_attribute_aware_attention_mask(
            ref_groups=ref_groups,
            batch_size=batch_size,
            target_seq_len=target_seq_len,
            total_ref_seq_len=total_ref_seq_len,
            device=device,
            dtype=dtype,
        )

        # Create video Modality
        video_modality = Modality(
            enabled=True,
            latent=combined_latents,
            sigma=sigmas,
            timesteps=timesteps,
            positions=positions,
            context=video_prompt_embeds,
            context_mask=prompt_attention_mask,
            attention_mask=attention_mask,
        )

        # Loss mask: only compute loss on non-conditioning target tokens
        ref_loss_mask = torch.zeros(batch_size, total_ref_seq_len, dtype=torch.bool, device=device)
        target_loss_mask = ~target_conditioning_mask
        video_loss_mask = torch.cat([ref_loss_mask, target_loss_mask], dim=1)

        # Handle audio if enabled
        audio_modality = None
        audio_targets = None
        audio_loss_mask = None

        if self.config.with_audio:
            audio_modality, audio_targets, audio_loss_mask = self._prepare_audio_inputs(
                batch=batch,
                sigmas=sigmas,
                audio_prompt_embeds=audio_prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )

        # Build no-reference state for identity guidance
        noref_video_modality = None
        if self.config.identity_guidance_scale > 0 and total_ref_seq_len > 0:
            # No-ref state: same noisy target but WITHOUT reference tokens
            # This requires a separate forward pass due to different sequence length
            noref_conditioning_mask = target_conditioning_mask
            noref_timesteps = self._create_per_token_timesteps(noref_conditioning_mask, sigmas.squeeze())

            noref_video_modality = Modality(
                enabled=True,
                latent=noisy_target,
                sigma=sigmas,
                timesteps=noref_timesteps,
                positions=target_positions,
                context=video_prompt_embeds,
                context_mask=prompt_attention_mask,
                attention_mask=None,  # No special mask needed (no refs)
            )

        return ModelInputs(
            video=video_modality,
            audio=audio_modality,
            video_targets=targets,
            audio_targets=audio_targets,
            video_loss_mask=video_loss_mask,
            audio_loss_mask=audio_loss_mask,
            ref_seq_len=total_ref_seq_len,
            noref_video=noref_video_modality,
        )

    def _process_image_references(
        self,
        ref_data: dict[str, Any],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        fps: float,
    ) -> list[_ReferenceGroup]:
        """Process image reference latents into reference groups.

        Supports both a single image tensor [B, C, 1, H, W] and a list of
        image tensors (multiple image references).
        """
        groups: list[_ReferenceGroup] = []
        ref_latents = ref_data["latents"]

        # Handle list of image references
        if isinstance(ref_latents, list):
            for i, img_latent in enumerate(ref_latents):
                if img_latent.dim() == 5:
                    # Use per-slot metadata if available
                    slot_data = self._get_slot_metadata(ref_data, i)
                    group = self._make_image_ref_group(
                        img_latent, slot_data, batch_size, device, dtype, fps,
                        ref_index=i,
                    )
                    if group is not None:
                        groups.append(group)
        elif ref_latents.dim() == 5:
            group = self._make_image_ref_group(
                ref_latents, ref_data, batch_size, device, dtype, fps, ref_index=0,
            )
            if group is not None:
                groups.append(group)

        return groups

    @staticmethod
    def _get_slot_metadata(ref_data: dict[str, Any], slot_index: int) -> dict[str, Any]:
        """Extract metadata for a specific slot from merged multi-slot data.

        When _merge_multi_slot_data stores per-slot metadata as lists,
        this method extracts the metadata for a single slot so that
        _make_image_ref_group / _make_video_ref_group can use the correct
        dimensions for each reference independently.
        """
        slot_data: dict[str, Any] = {}
        for key, value in ref_data.items():
            if key == "latents":
                continue  # Skip the latents list; caller already has the right tensor
            if isinstance(value, list) and len(value) > slot_index:
                slot_data[key] = value[slot_index]
            else:
                # Not a per-slot list — use as-is (e.g., shared metadata)
                slot_data[key] = value
        return slot_data

    def _make_image_ref_group(
        self,
        ref_latent: Tensor,
        ref_data: dict[str, Any],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        fps: float,
        ref_index: int = 0,
    ) -> _ReferenceGroup | None:
        """Create a single image reference group from a latent tensor."""
        tokens = self._video_patchifier.patchify(ref_latent)
        ref_height = ref_data["height"][0].item()
        ref_width = ref_data["width"][0].item()

        positions = self._get_video_positions(
            num_frames=1,
            height=ref_height,
            width=ref_width,
            batch_size=batch_size,
            fps=fps,
            device=device,
            dtype=dtype,
        )

        if self.config.reference_downscale_factor != 1:
            positions = positions.clone()
            positions[:, 1, ...] *= self.config.reference_downscale_factor
            positions[:, 2, ...] *= self.config.reference_downscale_factor

        attributes = [ReferenceAttribute(a) for a in self.config.default_image_attributes]
        attn_weight = self._compute_group_attention_weight(attributes)

        return _ReferenceGroup(
            tokens=tokens,
            positions=positions,
            seq_len=tokens.shape[1],
            modality=ReferenceModality.IMAGE,
            attributes=attributes,
            attention_weight=attn_weight,
            slot_id=f"Image{ref_index + 1}",
        )

    def _process_video_references(
        self,
        ref_data: dict[str, Any],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        fps: float,
    ) -> list[_ReferenceGroup]:
        """Process video reference latents into reference groups.

        Supports both a single video tensor [B, C, F, H, W] and a list of
        video tensors (multiple video references).
        """
        groups: list[_ReferenceGroup] = []
        ref_latents = ref_data["latents"]

        if isinstance(ref_latents, list):
            for i, vid_latent in enumerate(ref_latents):
                if vid_latent.dim() == 5:
                    # Use per-slot metadata if available
                    slot_data = self._get_slot_metadata(ref_data, i)
                    group = self._make_video_ref_group(
                        vid_latent, slot_data, batch_size, device, dtype, fps,
                        ref_index=i,
                    )
                    if group is not None:
                        groups.append(group)
        elif ref_latents.dim() == 5:
            group = self._make_video_ref_group(
                ref_latents, ref_data, batch_size, device, dtype, fps, ref_index=0,
            )
            if group is not None:
                groups.append(group)

        return groups

    def _make_video_ref_group(
        self,
        ref_latent: Tensor,
        ref_data: dict[str, Any],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        fps: float,
        ref_index: int = 0,
    ) -> _ReferenceGroup | None:
        """Create a single video reference group from a latent tensor."""
        tokens = self._video_patchifier.patchify(ref_latent)
        ref_frames = ref_data["num_frames"][0].item()
        ref_height = ref_data["height"][0].item()
        ref_width = ref_data["width"][0].item()

        target_height = ref_data.get("target_height", None)
        if target_height is not None:
            downscale = self._infer_reference_downscale_factor(
                target_height=target_height.item() if hasattr(target_height, 'item') else target_height,
                target_width=ref_data.get("target_width", ref_width),
                ref_height=ref_height,
                ref_width=ref_width,
            )
        else:
            downscale = self.config.reference_downscale_factor

        positions = self._get_video_positions(
            num_frames=ref_frames,
            height=ref_height,
            width=ref_width,
            batch_size=batch_size,
            fps=fps,
            device=device,
            dtype=dtype,
        )

        if downscale != 1:
            positions = positions.clone()
            positions[:, 1, ...] *= downscale
            positions[:, 2, ...] *= downscale

        attributes = [ReferenceAttribute(a) for a in self.config.default_video_attributes]
        attn_weight = self._compute_group_attention_weight(attributes)

        return _ReferenceGroup(
            tokens=tokens,
            positions=positions,
            seq_len=tokens.shape[1],
            modality=ReferenceModality.VIDEO,
            attributes=attributes,
            attention_weight=attn_weight,
            slot_id=f"Video{ref_index + 1}",
        )

    def _process_audio_references(
        self,
        ref_data: dict[str, Any],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> list[_ReferenceGroup]:
        """Process audio reference latents into reference groups."""
        groups: list[_ReferenceGroup] = []

        ref_latents = ref_data["latents"]
        if ref_latents.dim() == 4:
            # Audio: [B, C, T, mel_bins]
            tokens = self._audio_patchifier.patchify(ref_latents)
            audio_seq_len = tokens.shape[1]

            # Audio positions are 1D temporal, padded to match video position dims
            audio_positions = self._get_audio_positions(
                num_time_steps=audio_seq_len,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )
            # Pad to [B, 3, T, 2] to match video positions
            padded_positions = torch.zeros(
                batch_size, 3, audio_seq_len, 2,
                device=device, dtype=dtype,
            )
            padded_positions[:, 0, :, :] = audio_positions[:, 0, :, :]

            attributes = [ReferenceAttribute(a) for a in self.config.default_audio_attributes]
            attn_weight = self._compute_group_attention_weight(attributes)

            groups.append(_ReferenceGroup(
                tokens=tokens,
                positions=padded_positions,
                seq_len=audio_seq_len,
                modality=ReferenceModality.AUDIO,
                attributes=attributes,
                attention_weight=attn_weight,
                slot_id="Audio1",
            ))

        return groups

    @staticmethod
    def _compute_group_attention_weight(attributes: list[ReferenceAttribute]) -> float:
        """Compute the attention weight for a reference group from its attribute tags.

        Uses weighted average of attribute weights instead of max, so that
        weaker attributes still contribute to the overall weight.
        """
        if not attributes:
            return 1.0
        weights = [DEFAULT_ATTRIBUTE_WEIGHTS.get(attr, 0.8) for attr in attributes]
        return sum(weights) / len(weights)

    def _build_attribute_aware_attention_mask(
        self,
        ref_groups: list[_ReferenceGroup],
        batch_size: int,
        target_seq_len: int,
        total_ref_seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        """Build a 2-D self-attention mask with isolated reference groups.

        Uses full cross-attention (weight=1.0) between target tokens and each
        reference group. The model learns semantic routing from the text
        encoder's context (e.g., "for identity" in the prompt), NOT from
        hand-crafted mask weights. This is the Seedance 2.0 approach.

        The mask has the block structure:
                     target      ref_grp_0   ref_grp_1   ...
                 ┌───────────┬───────────┬───────────┬─────┐
        target    │     1     │     1     │     1     │ ... │
                 ├───────────┼───────────┼───────────┼─────┤
        ref_grp_0 │     1     │     1     │     0     │ ... │
                 ├───────────┼───────────┼───────────┼─────┤
        ref_grp_1 │     1     │     0     │     1     │ ... │
                 ├───────────┼───────────┼───────────┼─────┤
        ...       │  ...      │  ...      │  ...      │ ... │
                 └───────────┴───────────┴───────────┴─────┘

        Different reference groups do NOT attend to each other (0) to prevent
        attribute confusion. Full cross-attention (1) between target and each
        reference group lets the model learn which reference influences which
        aspect from the text context.
        """
        if not ref_groups:
            return None

        total_tokens = target_seq_len + total_ref_seq_len
        mask = torch.zeros((batch_size, total_tokens, total_tokens), device=device, dtype=dtype)

        # Target tokens attend to each other fully
        mask[:, :target_seq_len, :target_seq_len] = 1.0

        # Build mask incrementally for each reference group
        offset = target_seq_len
        for group in ref_groups:
            grp_start = offset
            grp_end = offset + group.seq_len

            # Full cross-attention between target and this reference group.
            # The model learns from text context (via cross-attention with
            # the text encoder output) which reference to attend to for what.
            mask[:, :target_seq_len, grp_start:grp_end] = 1.0
            mask[:, grp_start:grp_end, :target_seq_len] = 1.0
            # This reference group attends to itself fully
            mask[:, grp_start:grp_end, grp_start:grp_end] = 1.0
            # Cross-reference attention remains 0 (already initialized)

            offset = grp_end

        return mask

    def _apply_reference_dropout(
        self,
        groups: list[_ReferenceGroup],
        batch_size: int,
        device: torch.device,
    ) -> list[_ReferenceGroup]:
        """Apply random dropout to individual references for robustness.

        Each reference is independently dropped with probability reference_dropout_p.
        At least one reference is always kept if any are available.
        """
        if not groups:
            return groups

        kept: list[_ReferenceGroup] = []
        for group in groups:
            if torch.rand(1).item() >= self.config.reference_dropout_p:
                kept.append(group)
            else:
                logger.debug(f"Dropping {group.modality.value} reference (dropout)")

        # Always keep at least one reference if available
        if not kept and groups:
            kept = [groups[0]]

        return kept

    def _prepare_audio_inputs(
        self,
        batch: dict[str, Any],
        sigmas: Tensor,
        audio_prompt_embeds: Tensor,
        prompt_attention_mask: Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Modality, Tensor, Tensor]:
        """Prepare target audio inputs for joint audio-video training."""
        audio_data = batch["audio_latents"]
        audio_latents = audio_data["latents"]
        audio_latents = self._audio_patchifier.patchify(audio_latents)

        audio_seq_len = audio_latents.shape[1]
        audio_noise = torch.randn_like(audio_latents)

        sigmas_expanded = sigmas.view(-1, 1, 1)
        noisy_audio = (1 - sigmas_expanded) * audio_latents + sigmas_expanded * audio_noise
        audio_targets = audio_noise - audio_latents

        audio_timesteps = sigmas.view(-1, 1).expand(-1, audio_seq_len)
        audio_positions = self._get_audio_positions(
            num_time_steps=audio_seq_len,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

        audio_modality = Modality(
            enabled=True,
            latent=noisy_audio,
            sigma=sigmas,
            timesteps=audio_timesteps,
            positions=audio_positions,
            context=audio_prompt_embeds,
            context_mask=prompt_attention_mask,
        )

        audio_loss_mask = torch.ones(batch_size, audio_seq_len, dtype=torch.bool, device=device)

        return audio_modality, audio_targets, audio_loss_mask

    def compute_loss(
        self,
        video_pred: Tensor,
        audio_pred: Tensor | None,
        inputs: ModelInputs,
        noref_video_pred: Tensor | None = None,
    ) -> Tensor:
        """Compute masked loss on target portion with optional identity guidance.

        When identity_guidance_scale > 0 and noref_video_pred is provided, the
        loss target is modified to include the identity guidance extrapolation:
            guided_target = target + identity_guidance_scale * (target - noref_target)
        This amplifies identity-specific features by extrapolating away from
        the no-reference prediction.
        Returns [B,].
        """
        ref_seq_len = inputs.ref_seq_len or 0
        target_pred = video_pred[:, ref_seq_len:, :]
        target_loss_mask = inputs.video_loss_mask[:, ref_seq_len:]

        # Base loss target
        loss_target = inputs.video_targets

        # Apply identity guidance if noref prediction is available
        if (
            self.config.identity_guidance_scale > 0
            and noref_video_pred is not None
            and ref_seq_len > 0
        ):
            # noref prediction is on target-only sequence (no ref tokens)
            # Identity guidance: extrapolate away from no-reference prediction
            # guided = pred + scale * (pred - noref_pred)
            # Equivalently, adjust the target to match this guided prediction
            identity_scale = self.config.identity_guidance_scale
            # The noref_pred corresponds to the target portion only
            loss_target = inputs.video_targets + identity_scale * (
                inputs.video_targets - noref_video_pred.detach()
            )

        loss = (target_pred - loss_target).pow(2)
        loss_mask = target_loss_mask.unsqueeze(-1).float()
        masked = loss.mul(loss_mask)
        video_loss = masked.mean(dim=[-2, -1]) / loss_mask.mean(dim=[-2, -1]).clamp(min=1e-8)

        if not self.config.with_audio or audio_pred is None or inputs.audio_targets is None:
            return video_loss

        audio_loss = (audio_pred - inputs.audio_targets).pow(2).mean(dim=[-2, -1])
        return video_loss + audio_loss

    def get_checkpoint_metadata(self) -> dict[str, Any]:
        """Get metadata for checkpoint files."""
        metadata: dict[str, Any] = {
            "training_strategy": "multi_reference",
            "max_image_references": self.config.max_image_references,
            "max_video_references": self.config.max_video_references,
            "max_audio_references": self.config.max_audio_references,
            "reference_downscale_factor": self.config.reference_downscale_factor,
        }
        if self._reference_downscale_factor is not None:
            metadata["reference_downscale_factor"] = self._reference_downscale_factor
        return metadata

    @staticmethod
    def _infer_reference_downscale_factor(
        target_height: int,
        target_width: int,
        ref_height: int,
        ref_width: int,
    ) -> int:
        """Infer the reference downscale factor from target and reference dimensions."""
        if target_height == ref_height and target_width == ref_width:
            return 1
        if target_height % ref_height != 0 or target_width % ref_width != 0:
            raise ValueError(
                f"Target dimensions ({target_height}x{target_width}) must be exact multiples "
                f"of reference dimensions ({ref_height}x{ref_width})"
            )
        scale_h = target_height // ref_height
        scale_w = target_width // ref_width
        if scale_h != scale_w:
            raise ValueError(
                f"Reference scale must be uniform. Got height scale {scale_h} and width scale {scale_w}."
            )
        if scale_h < 1:
            raise ValueError(
                f"Reference dimensions ({ref_height}x{ref_width}) cannot be larger than "
                f"target dimensions ({target_height}x{target_width})"
            )
        return scale_h


class _ReferenceGroup:
    """Internal container for a group of reference tokens."""

    __slots__ = ("tokens", "positions", "seq_len", "modality", "attributes", "attention_weight", "slot_id")

    def __init__(
        self,
        tokens: Tensor,
        positions: Tensor,
        seq_len: int,
        modality: ReferenceModality,
        attributes: list[ReferenceAttribute],
        attention_weight: float = 1.0,
        slot_id: str = "",
    ):
        self.tokens = tokens
        self.positions = positions
        self.seq_len = seq_len
        self.modality = modality
        self.attributes = attributes
        self.attention_weight = attention_weight
        self.slot_id = slot_id
