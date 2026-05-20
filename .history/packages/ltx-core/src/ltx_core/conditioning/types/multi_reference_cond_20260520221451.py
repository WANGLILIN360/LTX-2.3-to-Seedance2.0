"""Unified multi-modal multi-reference conditioning.

Implements a Seedance 2.0-style reference injection system where multiple
image, video, and audio references can be injected simultaneously, each
with explicit binding via slot_id and positional encoding.

Architecture (based on ID-LoRA/IC-LoRA + Seedance 2.0 Binding Logic):
- Each reference's tokens are concatenated along the sequence dimension
  with the noisy target tokens (same as IC-LoRA).
- The model learns reference-target correspondence through self-attention,
  guided by positional encoding and text cross-attention.
- Different reference groups are isolated (don't attend to each other)
  to prevent attribute confusion.
- The text encoder processes the full prompt including @mention descriptions
  (e.g., "for identity", "camera dolly"). The model's cross-attention with
  text context naturally routes each reference's influence based on the
  text semantics — NOT through hand-crafted mask weights.
- During training, attribute_tags serve as auxiliary supervision signals.

Reference binding mechanism:
  The binding between @mentions and reference tokens works through two
  complementary signals:
  1. Positional encoding: Each reference group has distinct positions that
     separate them from target tokens and from each other (following IC-LoRA).
  2. Text cross-attention: The text encoder processes "@Image1 for identity"
     as a single sequence. When the model cross-attends to this text, it
     learns that the reference at position group "Image1" should influence
     the identity aspect. This is the Seedance 2.0 "Binding Logic".
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

# Maximum number of reference slots per modality (matches Seedance 2.0 limits)
MAX_IMAGE_SLOTS = 9
MAX_VIDEO_SLOTS = 3
MAX_AUDIO_SLOTS = 3
MAX_TOTAL_SLOTS = MAX_IMAGE_SLOTS + MAX_VIDEO_SLOTS + MAX_AUDIO_SLOTS


class ReferenceSlotEmbedding(nn.Module):
    """Learnable slot embeddings for Seedance 2.0 Binding Logic.

    Each reference slot (e.g., @Image1, @Video1) gets a unique learnable
    embedding vector. This embedding serves as the **binding signal** that
    connects the @mention in the text prompt to the reference latent tokens.

    How it works:
    1. When the text encoder processes "@Image1 for identity", it produces
       context tokens. We inject the slot embedding at the @mention position
       in the context, so the model knows "this text position = slot Image1".
    2. When reference latent tokens are appended to the sequence, we add the
       same slot embedding to each token, so the model knows "these tokens
       belong to slot Image1".
    3. The model's cross-attention between latent tokens and text context
       then naturally learns the binding: when it sees the slot embedding
       in both the query and key/value, it strengthens the association.

    This is the core of Seedance 2.0's "Binding Logic" — the model can
    distinguish which reference tokens correspond to which @mention without
    any external hard-coded routing.

    Args:
        embed_dim: Dimension of the embedding vector. Should match the
            transformer's hidden dimension.
        max_slots: Maximum number of reference slots (default 15 = 9+3+3).
    """

    def __init__(self, embed_dim: int, max_slots: int = MAX_TOTAL_SLOTS):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_slots = max_slots
        self.embedding = nn.Parameter(
            torch.randn(max_slots, embed_dim) * 0.02
        )

    def get_slot_index(self, slot_id: str) -> int:
        """Map a slot_id string (e.g., "Image1") to an index in the embedding table.

        Layout: [Image1..Image9, Video1..Video3, Audio1..Audio3]
        """
        slot_id = slot_id.strip()
        for prefix, offset, max_count in [
            ("Image", 0, MAX_IMAGE_SLOTS),
            ("Video", MAX_IMAGE_SLOTS, MAX_VIDEO_SLOTS),
            ("Audio", MAX_IMAGE_SLOTS + MAX_VIDEO_SLOTS, MAX_AUDIO_SLOTS),
        ]:
            if slot_id.lower().startswith(prefix.lower()):
                try:
                    num = int(slot_id[len(prefix):])
                    if 1 <= num <= max_count:
                        return offset + num - 1
                except ValueError:
                    pass
        # Fallback: hash to a valid index
        return hash(slot_id) % self.max_slots

    def get_embedding(self, slot_id: str) -> torch.Tensor:
        """Get the embedding vector for a given slot_id.

        Returns:
            Tensor of shape (embed_dim,)
        """
        idx = self.get_slot_index(slot_id)
        return self.embedding[idx]

    def inject_into_context(
        self,
        context: torch.Tensor,
        slot_ids: list[str],
        mention_positions: list[int],
    ) -> torch.Tensor:
        """Inject slot embeddings into the text context at @mention positions.

        This creates the text-side binding signal. After this operation,
        the text context at the @mention positions contains the slot
        embedding, so cross-attention can find the matching reference tokens.

        Args:
            context: Text context tensor of shape (B, S, D) from the text encoder.
            slot_ids: List of slot IDs corresponding to each @mention.
            mention_positions: Token positions in the context where each
                @mention appears (e.g., the position of the "@Image1" token).

        Returns:
            Modified context tensor with slot embeddings injected.
        """
        context = context.clone()
        for slot_id, pos in zip(slot_ids, mention_positions):
            if 0 <= pos < context.shape[1]:
                emb = self.get_embedding(slot_id)
                # Additive injection: add slot embedding to the existing token
                context[:, pos, :] = context[:, pos, :] + emb.to(context.dtype)
        return context

    def add_to_tokens(
        self,
        tokens: torch.Tensor,
        slot_id: str,
    ) -> torch.Tensor:
        """Add slot embedding to reference latent tokens.

        This creates the latent-side binding signal. After this operation,
        each reference token carries the slot embedding, so the model knows
        "these tokens belong to slot X".

        Args:
            tokens: Reference latent tokens of shape (B, T, D).
            slot_id: The slot ID for these tokens (e.g., "Image1").

        Returns:
            Modified tokens with slot embedding added.
        """
        emb = self.get_embedding(slot_id)
        return tokens + emb.to(tokens.dtype).unsqueeze(0).unsqueeze(0)


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
            Used for tracking which @mention corresponds to which reference.
            The actual binding mechanism works through positional encoding
            (each reference group has distinct positions, following IC-LoRA)
            and text cross-attention (the text encoder processes "@Image1
            for identity" and the model learns the routing via cross-attention).
            This field is primarily for bookkeeping and training-time
            auxiliary supervision.
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
    """Unified multi-modal multi-reference conditioning with Binding Logic.

    Accepts a list of :class:`ReferenceItem` objects (images, videos, audio)
    and injects them into the latent state.

    Architecture (based on ID-LoRA/IC-LoRA + Seedance 2.0 Binding Logic):
    - Each reference's tokens are concatenated along the sequence dimension
      with the noisy target tokens, following the IC-LoRA paradigm.
    - The model learns reference-target correspondence through self-attention,
      guided by positional encoding (each reference group has distinct
      positions that separate them from target tokens).
    - Different reference groups are isolated (don't attend to each other)
      to prevent attribute confusion.
    - The text encoder processes the full prompt including @mention
      descriptions (e.g., "for identity", "camera dolly"). The model's
      cross-attention with text context naturally routes each reference's
      influence based on the text semantics — NOT through hand-crafted
      mask weights. This is the Seedance 2.0 "Binding Logic".
    - During training, attribute_tags serve as auxiliary supervision signals.

    The binding chain:
        Prompt: "Man @Image1 for identity walks in @Image2 park. @Video1 camera."
        Text encoder → context tokens with "@Image1 for identity" semantics
                                    ↕ cross-attention
        Latent seq:  [target] [ref_Image1_tokens] [ref_Image2_tokens] [ref_Video1_tokens]
                     (each group with distinct positions → model distinguishes them)

    Args:
        references: List of :class:`ReferenceItem` to inject.
        attribute_weights: Optional override for ``DEFAULT_ATTRIBUTE_WEIGHTS``.
            Only used for training-time auxiliary loss weighting.
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

        # Add slot binding embedding to reference tokens (Binding Logic)
        if self.slot_embedding is not None and ref.slot_id:
            tokens = self.slot_embedding.add_to_tokens(tokens, ref.slot_id)

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

        # Add slot binding embedding to audio reference tokens (Binding Logic)
        if self.slot_embedding is not None and ref.slot_id:
            patchified = self.slot_embedding.add_to_tokens(patchified, ref.slot_id)

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
