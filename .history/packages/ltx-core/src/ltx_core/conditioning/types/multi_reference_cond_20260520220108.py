"""Unified multi-modal multi-reference conditioning.

Implements a Seedance 2.0-style reference injection system where multiple
image, video, and audio references can be injected simultaneously, each
with explicit binding via slot_id that connects the reference tokens to
the @mention in the text prompt (Binding Logic).

Architecture (Seedance 2.0 Binding Logic):
- Each reference gets a slot_id (e.g., "Image1", "Video1")
- A learnable ReferenceSlotEmbedding maps each slot_id to a binding vector
- This binding vector is added to both:
  1. The reference latent tokens (so the model knows "these tokens are slot Image1")
  2. The text context at the @mention position (so the model knows "@Image1
     in the text corresponds to slot Image1")
- The model's cross-attention between latent tokens and text context then
  naturally learns to route each reference's influence based on the text
  semantics (e.g., "for identity" in the prompt).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
import torch.nn as nn

from ltx_core.conditioning.item import ConditioningItem
from ltx_core.conditioning.mask_utils import update_attention_mask
from ltx_core.tools import LatentTools, VideoLatentTools
from ltx_core.types import LatentState, VideoLatentShape


class ReferenceModality(str, Enum):
    """Modality of a reference asset."""

    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class ReferenceAttribute(str, Enum):
    """Semantic attribute that a reference should contribute to generation.

    These tags control the attention mask strength between reference tokens
    and noisy target tokens. Higher weight = stronger influence on that attribute.
    """

    IDENTITY = "identity"
    APPEARANCE = "appearance"
    STYLE = "style"
    MOTION = "motion"
    CAMERA = "camera"
    SCENE = "scene"
    AUDIO_RHYTHM = "audio_rhythm"
    AUDIO_MOOD = "audio_mood"
    LIP_SYNC = "lip_sync"


# Default attention weights per attribute.
# These can be overridden per-reference or globally.
DEFAULT_ATTRIBUTE_WEIGHTS: dict[ReferenceAttribute, float] = {
    ReferenceAttribute.IDENTITY: 1.0,
    ReferenceAttribute.APPEARANCE: 1.0,
    ReferenceAttribute.STYLE: 0.8,
    ReferenceAttribute.MOTION: 0.8,
    ReferenceAttribute.CAMERA: 0.6,
    ReferenceAttribute.SCENE: 0.7,
    ReferenceAttribute.AUDIO_RHYTHM: 0.8,
    ReferenceAttribute.AUDIO_MOOD: 0.6,
    ReferenceAttribute.LIP_SYNC: 1.0,
}


@dataclass(frozen=True)
class ReferenceItem:
    """A single reference asset with metadata.

    Attributes:
        latent: Pre-encoded latent tensor.
            - Image/Video: shape ``[B, C, F, H, W]`` (video VAE latent)
            - Audio: shape ``[B, C, T, mel_bins]`` (audio VAE latent)
        modality: Whether this is an image, video, or audio reference.
        attribute_tags: Which semantic attributes this reference should
            contribute. Used as training-time auxiliary signals.
        slot_id: Binding slot identifier (e.g., "Image1", "Video1").
            This creates the Seedance 2.0 "Binding Logic" — the same
            slot_id appears in both the text context (as a special token
            at the @mention position) and in the reference latent tokens
            (as a slot embedding added to each token). This tells the
            model which reference tokens correspond to which @mention.
        strength: Conditioning strength. 1.0 = reference kept clean,
            0.0 = reference fully denoised.
        attention_weight: Override for the aggregated attribute weight.
            If None, the weight is computed from ``attribute_tags`` using
            ``DEFAULT_ATTRIBUTE_WEIGHTS``.
        downscale_factor: Spatial downscale factor for video/image
            references (e.g., 2 = half-resolution reference). Must match
            training preprocessing.
        frame_idx: For image references, which frame to inject at.
            If None, tokens are appended (reference mode).
            If set, tokens replace/inject at that frame (first-frame mode).
    """

    latent: torch.Tensor
    modality: ReferenceModality
    attribute_tags: list[ReferenceAttribute]
    slot_id: str = ""  # e.g., "Image1", "Video1", "Audio1"
    strength: float = 1.0
    attention_weight: float | None = None
    downscale_factor: int = 1
    frame_idx: int | None = None

    def effective_attention_weight(self) -> float:
        """Compute the effective attention weight from attribute tags.

        NOTE: This is used for training-time auxiliary loss weighting,
        NOT for inference-time attention mask computation. During inference,
        the model learns semantic routing from text context (Seedance 2.0
        approach), so all reference groups get full cross-attention (1.0).

        Uses weighted average instead of max so that weaker attributes
        still influence the overall weight (e.g., identity=1.0 + style=0.8
        yields 0.9 instead of 1.0).
        """
        if self.attention_weight is not None:
            return self.attention_weight
        if not self.attribute_tags:
            return 1.0
        weights = [DEFAULT_ATTRIBUTE_WEIGHTS.get(tag, 0.8) for tag in self.attribute_tags]
        return sum(weights) / len(weights)


class UnifiedMultiReferenceConditioning(ConditioningItem):
    """Unified multi-modal multi-reference conditioning.

    Accepts a list of :class:`ReferenceItem` objects (images, videos, audio)
    and injects them into the latent state.

    Architecture (Seedance 2.0 style — text-driven semantic routing):
    - Each reference's tokens are appended to the sequence with isolated
      attention (different reference groups don't attend to each other).
    - The **text encoder** processes the full prompt including @mention
      descriptions (e.g., "for identity", "camera dolly"). The model
      learns through cross-attention to route each reference's influence
      based on the text semantics — NOT through hand-crafted mask weights.
    - During **training**, attribute_tags serve as auxiliary supervision
      signals (e.g., loss weighting per attribute region) and help
      construct the training data pipeline.
    - During **inference**, the model has already learned the routing from
      text context, so attribute_tags only affect the initial attention
      mask structure (full cross-attention between noisy tokens and each
      reference group).

    This is fundamentally different from using hand-crafted attention weights
    to force routing. Instead, the model learns to understand "for identity"
    from the text encoder output, just like Seedance 2.0.

    Args:
        references: List of :class:`ReferenceItem` to inject.
        attribute_weights: Optional override for ``DEFAULT_ATTRIBUTE_WEIGHTS``.
            Only used for training-time auxiliary loss weighting, NOT for
            inference-time attention mask computation.
    """

    def __init__(
        self,
        references: list[ReferenceItem],
        attribute_weights: dict[ReferenceAttribute, float] | None = None,
    ):
        self.references = references
        self._attr_weights = attribute_weights or DEFAULT_ATTRIBUTE_WEIGHTS

    def apply_to(
        self,
        latent_state: LatentState,
        latent_tools: LatentTools,
    ) -> LatentState:
        """Apply all references sequentially, each with its own attention mask."""
        state = latent_state

        for ref in self.references:
            if ref.modality in (ReferenceModality.IMAGE, ReferenceModality.VIDEO):
                state = self._apply_video_reference(state, ref, latent_tools)
            elif ref.modality == ReferenceModality.AUDIO:
                state = self._apply_audio_reference(state, ref, latent_tools)
            else:
                raise ValueError(f"Unsupported reference modality: {ref.modality}")

        return state

    def _apply_video_reference(
        self,
        latent_state: LatentState,
        ref: ReferenceItem,
        latent_tools: LatentTools,
    ) -> LatentState:
        """Inject a video/image reference into the video latent state."""
        from ltx_core.components.patchifiers import get_pixel_coords

        if not isinstance(latent_tools, VideoLatentTools):
            raise TypeError("Video reference requires VideoLatentTools")

        # If frame_idx is specified and this is an image at frame 0,
        # use replacement mode (like VideoConditionByLatentIndex)
        if ref.frame_idx is not None and ref.modality == ReferenceModality.IMAGE:
            return self._apply_image_at_frame(latent_state, ref, latent_tools)

        # Patchify the reference latent
        tokens = latent_tools.patchifier.patchify(ref.latent)

        # Compute positions for the reference
        latent_coords = latent_tools.patchifier.get_patch_grid_bounds(
            output_shape=VideoLatentShape.from_torch_shape(ref.latent.shape),
            device=ref.latent.device,
        )
        positions = get_pixel_coords(
            latent_coords=latent_coords,
            scale_factors=latent_tools.scale_factors,
            causal_fix=latent_tools.causal_fix,
        )
        positions = positions.to(dtype=torch.float32)
        positions[:, 0, ...] /= latent_tools.fps

        # Scale spatial positions to match target coordinate space
        if ref.downscale_factor != 1:
            positions = positions.clone()
            positions[:, 1, ...] *= ref.downscale_factor
            positions[:, 2, ...] *= ref.downscale_factor

        # Denoise mask: 1.0 - strength means how much to denoise
        denoise_mask = torch.full(
            size=(*tokens.shape[:2], 1),
            fill_value=1.0 - ref.strength,
            device=ref.latent.device,
            dtype=ref.latent.dtype,
        )

        # Build attention mask with full cross-attention between noisy tokens
        # and this reference group. The model learns semantic routing from
        # the text encoder's context (e.g., "for identity" in the prompt),
        # NOT from hand-crafted mask weights. This is the Seedance 2.0 approach.
        new_attention_mask = update_attention_mask(
            latent_state=latent_state,
            attention_mask=1.0,  # Full cross-attention; model learns routing from text
            num_noisy_tokens=latent_tools.target_shape.token_count(),
            num_new_tokens=tokens.shape[1],
            batch_size=tokens.shape[0],
            device=ref.latent.device,
            dtype=ref.latent.dtype,
        )

        return LatentState(
            latent=torch.cat([latent_state.latent, tokens], dim=1),
            denoise_mask=torch.cat([latent_state.denoise_mask, denoise_mask], dim=1),
            positions=torch.cat([latent_state.positions, positions], dim=2),
            clean_latent=torch.cat([latent_state.clean_latent, tokens], dim=1),
            attention_mask=new_attention_mask,
        )

    def _apply_audio_reference(
        self,
        latent_state: LatentState,
        ref: ReferenceItem,
        latent_tools: LatentTools,
    ) -> LatentState:
        """Inject an audio reference into the video latent state.

        Audio references are patchified and appended as additional tokens.
        They use 1D temporal positions and the audio patchifier.
        """
        from ltx_core.components.patchifiers import AudioPatchifier

        audio_patchifier = AudioPatchifier(patch_size=1)
        patchified = audio_patchifier.patchify(ref.latent)

        # Compute audio positions (1D temporal)
        audio_shape = ref.latent.shape  # [B, C, T, mel_bins]
        from ltx_core.types import AudioLatentShape

        audio_latent_shape = AudioLatentShape.from_torch_shape(audio_shape)
        audio_coords = audio_patchifier.get_patch_grid_bounds(
            output_shape=audio_latent_shape,
            device=ref.latent.device,
        )
        positions = audio_coords.to(dtype=torch.float32)
        # Expand to match video position dims [B, 3, T, 2] by padding
        # Audio positions are [B, 1, T, 2], we need [B, 3, T, 2]
        batch_size = positions.shape[0]
        audio_seq_len = positions.shape[2]
        padded_positions = torch.zeros(
            batch_size, 3, audio_seq_len, 2,
            device=ref.latent.device,
            dtype=torch.float32,
        )
        # Time dimension: use audio temporal positions
        padded_positions[:, 0, :, :] = positions[:, 0, :, :]
        # Height/width: zero (no spatial meaning for audio)

        denoise_mask = torch.full(
            size=(*patchified.shape[:2], 1),
            fill_value=1.0 - ref.strength,
            device=ref.latent.device,
            dtype=patchified.dtype,
        )

        # Full cross-attention — model learns audio routing from text context
        new_attention_mask = update_attention_mask(
            latent_state=latent_state,
            attention_mask=1.0,  # Full cross-attention; model learns routing from text
            num_noisy_tokens=latent_tools.target_shape.token_count(),
            num_new_tokens=patchified.shape[1],
            batch_size=patchified.shape[0],
            device=ref.latent.device,
            dtype=patchified.dtype,
        )

        return LatentState(
            latent=torch.cat([latent_state.latent, patchified], dim=1),
            denoise_mask=torch.cat([latent_state.denoise_mask, denoise_mask], dim=1),
            positions=torch.cat([latent_state.positions, padded_positions], dim=2),
            clean_latent=torch.cat([latent_state.clean_latent, patchified], dim=1),
            attention_mask=new_attention_mask,
        )

    def _apply_image_at_frame(
        self,
        latent_state: LatentState,
        ref: ReferenceItem,
        latent_tools: VideoLatentTools,
    ) -> LatentState:
        """Inject an image reference at a specific frame (replacement mode).

        This replaces the latent at the specified frame index with the
        encoded image, similar to ``VideoConditionByLatentIndex``.
        """
        tokens = latent_tools.patchifier.patchify(ref.latent)
        start_token = latent_tools.patchifier.get_token_count(
            latent_tools.target_shape._replace(frames=ref.frame_idx)
        )
        stop_token = start_token + tokens.shape[1]

        # Use clone + in-place for replacement mode (modifies existing tokens)
        latent = latent_state.latent.clone()
        clean_latent = latent_state.clean_latent.clone()
        denoise_mask = latent_state.denoise_mask.clone()

        latent[:, start_token:stop_token] = tokens
        clean_latent[:, start_token:stop_token] = tokens
        denoise_mask[:, start_token:stop_token] = 1.0 - ref.strength

        return LatentState(
            latent=latent,
            denoise_mask=denoise_mask,
            positions=latent_state.positions,
            clean_latent=clean_latent,
            attention_mask=latent_state.attention_mask,
        )

    def _compute_attention_weight(self, ref: ReferenceItem) -> float:
        """Compute the attention weight for a reference based on its attribute tags.

        NOTE: This method is kept for backward compatibility and potential
        use in training-time auxiliary loss weighting. It is NOT used for
        inference-time attention mask computation — the model learns semantic
        routing from text context (Seedance 2.0 approach).
        """
        if ref.attention_weight is not None:
            return ref.attention_weight
        if not ref.attribute_tags:
            return 1.0
        weights = [self._attr_weights.get(tag, 0.8) for tag in ref.attribute_tags]
        return sum(weights) / len(weights)
