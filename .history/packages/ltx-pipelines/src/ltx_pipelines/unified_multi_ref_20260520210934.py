"""Unified multi-reference video generation pipeline.

Supports simultaneous conditioning on multiple image, video, and audio
references with attribute-aware attention routing, inspired by
Seedance 2.0's @mention reference system.

This pipeline can:
- Accept text + multiple images + multiple videos + multiple audio as input
- Use @mention syntax in prompts to specify what each reference contributes
- Route each reference's influence via attribute-aware attention masks
- Work with the 22b-dev or 22b-distilled checkpoints
- Load existing IC-LoRA weights (pose, camera, motion-track, etc.)

Usage:
    python -m ltx_pipelines.unified_multi_ref \\
        --checkpoint-path ltx-2.3-22b-dev.safetensors \\
        --distilled-lora ltx-2.3-22b-distilled-lora-384-1.1.safetensors 0.8 \\
        --spatial-upsampler-path ltx-2.3-spatial-upscaler-x2-1.1.safetensors \\
        --gemma-root gemma-3-12b-it-qat-q4_0-unquantized/ \\
        --prompt "Man @Image1 for appearance walks in @Image2's park. \\
                   Replicate @Video1's camera. @Audio1 for rhythm." \\
        --image-references image1.jpg image2.jpg \\
        --video-references video1.mp4 \\
        --audio-references audio1.wav \\
        --output-path output.mp4
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import replace

import torch

from ltx_core.components.guiders import (
    MultiModalGuiderFactory,
    MultiModalGuiderParams,
    create_multimodal_guider_factory,
)
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.conditioning import (
    ConditioningItem,
    ReferenceAttribute,
    ReferenceItem,
    ReferenceModality,
    UnifiedMultiReferenceConditioning,
)
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.loader.registry import Registry
from ltx_core.model.video_vae import TilingConfig
from ltx_core.quantization import QuantizationPolicy
from ltx_core.types import Audio, VideoPixelShape
from ltx_pipelines.utils.args import (
    ImageConditioningInput,
    default_2_stage_arg_parser,
    detect_checkpoint_path,
)
from ltx_pipelines.utils.blocks import (
    AudioDecoder,
    DiffusionStage,
    ImageConditioner,
    PromptEncoder,
    VideoDecoder,
    VideoUpsampler,
)
from ltx_pipelines.utils.constants import (
    STAGE_2_DISTILLED_SIGMAS,
    detect_params,
)
from ltx_pipelines.utils.denoisers import FactoryGuidedDenoiser, SimpleDenoiser
from ltx_pipelines.utils.helpers import (
    assert_resolution,
    audio_latent_from_file,
    combined_image_conditionings,
    get_device,
    video_latent_from_file,
)
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.prompt_parser import ParsedPrompt, parse_prompt
from ltx_pipelines.utils.types import ModalitySpec, OffloadMode

logger = logging.getLogger(__name__)


class UnifiedMultiRefPipeline:
    """Two-stage pipeline with unified multi-modal multi-reference conditioning.

    Supports simultaneous conditioning on text + multiple images/videos/audio
    with attribute-aware attention routing via @mention prompt syntax.

    Stage 1: Generate at half resolution with multi-reference conditioning + CFG
    Stage 2: Upsample 2x with distilled LoRA refinement
    """

    def __init__(
        self,
        checkpoint_path: str,
        distilled_lora: list[LoraPathStrengthAndSDOps],
        spatial_upsampler_path: str,
        gemma_root: str,
        loras: list[LoraPathStrengthAndSDOps],
        device: torch.device | None = None,
        quantization: QuantizationPolicy | None = None,
        registry: Registry | None = None,
        torch_compile: bool = False,
        offload_mode: OffloadMode = OffloadMode.NONE,
    ):
        self.device = device or get_device()
        self.dtype = torch.bfloat16
        self._scheduler = LTX2Scheduler()

        self.prompt_encoder = PromptEncoder(
            checkpoint_path, gemma_root, self.dtype, self.device,
            registry=registry, offload_mode=offload_mode,
        )
        self.image_conditioner = ImageConditioner(
            checkpoint_path, self.dtype, self.device, registry=registry,
        )
        self.upsampler = VideoUpsampler(
            checkpoint_path, spatial_upsampler_path, self.dtype, self.device, registry=registry,
        )
        self.video_decoder = VideoDecoder(
            checkpoint_path, self.dtype, self.device, registry=registry,
        )
        self.audio_decoder = AudioDecoder(
            checkpoint_path, self.dtype, self.device, registry=registry,
        )

        self.stage_1 = DiffusionStage(
            checkpoint_path, self.dtype, self.device,
            loras=tuple(loras),
            quantization=quantization,
            registry=registry,
            torch_compile=torch_compile,
            offload_mode=offload_mode,
        )
        self.stage_2 = DiffusionStage(
            checkpoint_path, self.dtype, self.device,
            loras=(*tuple(loras), *distilled_lora),
            quantization=quantization,
            registry=registry,
            torch_compile=torch_compile,
            offload_mode=offload_mode,
        )

    def __call__(  # noqa: PLR0913
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        num_inference_steps: int,
        video_guider_params,
        audio_guider_params,
        # Multi-reference inputs
        image_references: list[str] | None = None,
        video_references: list[str] | None = None,
        audio_references: list[str] | None = None,
        # Parsed prompt (optional, auto-parsed if not provided)
        parsed_prompt: ParsedPrompt | None = None,
        # Standard image conditioning (backward compatible)
        images: list[ImageConditioningInput] | None = None,
        # Reference settings
        reference_strength: float = 1.0,
        reference_downscale_factor: int = 1,
        attribute_weights: dict[ReferenceAttribute, float] | None = None,
        # Pipeline settings
        tiling_config: TilingConfig | None = None,
        enhance_prompt: bool = False,
        max_batch_size: int = 1,
        stage_1_sigmas: torch.Tensor | None = None,
        stage_2_sigmas: torch.Tensor = STAGE_2_DISTILLED_SIGMAS,
    ) -> tuple[Iterator[torch.Tensor], Audio]:
        """Generate video with unified multi-reference conditioning."""
        assert_resolution(height, width, is_two_stage=True)

        # Parse @mention references from prompt
        if parsed_prompt is None:
            parsed_prompt = parse_prompt(prompt)

        # Build conditioning items
        conditionings = self._build_conditionings(
            parsed_prompt=parsed_prompt,
            image_references=image_references or [],
            video_references=video_references or [],
            audio_references=audio_references or [],
            images=images or [],
            height=height,
            width=width,
            strength=reference_strength,
            downscale_factor=reference_downscale_factor,
            attribute_weights=attribute_weights,
        )

        # Encode prompt
        video_prompt_embeds, audio_prompt_embeds, prompt_attention_mask = (
            self.prompt_encoder(
                prompts=[prompt],
                negative_prompts=[negative_prompt],
                enhance_prompt=enhance_prompt,
                seed=seed,
            )
        )

        # Split into positive/negative contexts for CFG
        v_context_p, v_context_n = video_prompt_embeds.chunk(2)
        a_context_p, a_context_n = audio_prompt_embeds.chunk(2)

        # Build video ModalitySpec with multi-reference conditionings
        video_spec = ModalitySpec(
            context=v_context_p,
            conditionings=conditionings,
        )

        # Build audio ModalitySpec (no special conditionings for audio target)
        audio_spec = ModalitySpec(
            context=a_context_p,
        )

        # Stage 1: Generate at half resolution
        stage_1_height, stage_1_width = height // 2, width // 2
        stage_1_sigmas = stage_1_sigmas or self._scheduler.get_sigmas(num_inference_steps)

        noiser = GaussianNoiser()
        denoiser = FactoryGuidedDenoiser(
            v_context=v_context_p,
            a_context=a_context_p,
            video_guider_factory=create_multimodal_guider_factory(
                params=video_guider_params,
                negative_context=v_context_n,
            ),
            audio_guider_factory=create_multimodal_guider_factory(
                params=audio_guider_params,
                negative_context=a_context_n,
            ),
        )

        video_state, audio_state = self.stage_1(
            denoiser=denoiser,
            sigmas=stage_1_sigmas,
            noiser=noiser,
            width=stage_1_width,
            height=stage_1_height,
            frames=num_frames,
            fps=frame_rate,
            video=video_spec,
            audio=audio_spec,
            max_batch_size=max_batch_size,
        )

        # Stage 2: Upsample with distilled LoRA
        if video_state is not None:
            upsampled_latent = self.upsampler(video_state.latent)
            # Create a new ModalitySpec for stage 2 with the upsampled latent
            stage_2_video_spec = ModalitySpec(
                context=video_prompt_embeds,
                initial_latent=upsampled_latent,
            )
        else:
            stage_2_video_spec = ModalitySpec(
                context=video_prompt_embeds,
            )

        stage_2_audio_spec = ModalitySpec(
            context=audio_prompt_embeds,
        )

        stage_2_denoiser = SimpleDenoiser()

        video_state_2, audio_state_2 = self.stage_2(
            denoiser=stage_2_denoiser,
            sigmas=stage_2_sigmas,
            noiser=noiser,
            width=width,
            height=height,
            frames=num_frames,
            fps=frame_rate,
            video=stage_2_video_spec,
            audio=stage_2_audio_spec,
            max_batch_size=max_batch_size,
        )

        # Decode video and audio
        if video_state_2 is not None:
            video_frames = self.video_decoder(
                video_state_2.latent, height, width, num_frames,
                tiling_config=tiling_config,
            )
        else:
            video_frames = None

        if audio_state_2 is not None:
            audio = self.audio_decoder(audio_state_2.latent)
        else:
            audio = Audio(waveform=torch.zeros(1, 1, 16000), sampling_rate=16000)

        if video_frames is not None:
            return encode_video(video_frames, audio, frame_rate), audio

        raise RuntimeError("Pipeline failed to produce video output")

    def _build_conditionings(
        self,
        parsed_prompt: ParsedPrompt,
        image_references: list[str],
        video_references: list[str],
        audio_references: list[str],
        images: list[ImageConditioningInput],
        height: int,
        width: int,
        strength: float,
        downscale_factor: int,
        attribute_weights: dict[ReferenceAttribute, float] | None,
    ) -> list[ConditioningItem]:
        """Build all conditioning items for the pipeline."""
        conditionings: list[ConditioningItem] = []

        # Backward-compatible image conditioning
        if images:
            conditionings.extend(
                combined_image_conditionings(
                    images=images,
                    height=height,
                    width=width,
                    video_encoder=self.image_conditioner.encoder,
                    dtype=self.dtype,
                    device=self.device,
                )
            )

        # Build multi-reference conditioning items
        ref_items = self._build_reference_items(
            parsed_prompt=parsed_prompt,
            image_references=image_references,
            video_references=video_references,
            audio_references=audio_references,
            height=height,
            width=width,
            strength=strength,
            downscale_factor=downscale_factor,
        )

        if ref_items:
            conditionings.append(
                UnifiedMultiReferenceConditioning(
                    references=ref_items,
                    attribute_weights=attribute_weights,
                )
            )

        return conditionings

    def _build_reference_items(
        self,
        parsed_prompt: ParsedPrompt,
        image_references: list[str],
        video_references: list[str],
        audio_references: list[str],
        height: int,
        width: int,
        strength: float,
        downscale_factor: int,
    ) -> list[ReferenceItem]:
        """Build ReferenceItem list from parsed prompt and file paths."""
        items: list[ReferenceItem] = []

        for binding in parsed_prompt.bindings:
            ref_path = self._resolve_ref_path(
                binding, image_references, video_references, audio_references,
            )
            if ref_path is None:
                logger.warning(f"No file path found for @{binding.ref_id}, skipping")
                continue

            if binding.ref_type == ReferenceModality.IMAGE:
                latent = self._encode_image_ref(ref_path, height, width)
                items.append(ReferenceItem(
                    latent=latent,
                    modality=ReferenceModality.IMAGE,
                    attribute_tags=binding.attributes,
                    strength=strength,
                    downscale_factor=downscale_factor,
                    frame_idx=0 if binding.index == 1 else None,
                ))
            elif binding.ref_type == ReferenceModality.VIDEO:
                latent = self._encode_video_ref(ref_path, height, width)
                items.append(ReferenceItem(
                    latent=latent,
                    modality=ReferenceModality.VIDEO,
                    attribute_tags=binding.attributes,
                    strength=strength,
                    downscale_factor=downscale_factor,
                ))
            elif binding.ref_type == ReferenceModality.AUDIO:
                latent = self._encode_audio_ref(ref_path, height, width)
                if latent is not None:
                    items.append(ReferenceItem(
                        latent=latent,
                        modality=ReferenceModality.AUDIO,
                        attribute_tags=binding.attributes,
                        strength=strength,
                    ))

        # Add unbound references (files not mentioned in prompt)
        items.extend(self._add_unbound_references(
            parsed_prompt, image_references, video_references,
            audio_references, height, width, strength, downscale_factor,
        ))

        return items

    def _resolve_ref_path(self, binding, image_references, video_references, audio_references):
        """Resolve a binding to a file path."""
        from ltx_pipelines.utils.prompt_parser import ReferenceBinding
        idx = binding.index - 1
        if binding.ref_type == ReferenceModality.IMAGE and idx < len(image_references):
            return image_references[idx]
        elif binding.ref_type == ReferenceModality.VIDEO and idx < len(video_references):
            return video_references[idx]
        elif binding.ref_type == ReferenceModality.AUDIO and idx < len(audio_references):
            return audio_references[idx]
        return None

    def _add_unbound_references(
        self,
        parsed_prompt: ParsedPrompt,
        image_references: list[str],
        video_references: list[str],
        audio_references: list[str],
        height: int,
        width: int,
        strength: float,
        downscale_factor: int,
    ) -> list[ReferenceItem]:
        """Add references that have file paths but no @mention in the prompt."""
        items: list[ReferenceItem] = []
        bound_image_indices = {
            b.index for b in parsed_prompt.bindings if b.ref_type == ReferenceModality.IMAGE
        }
        bound_video_indices = {
            b.index for b in parsed_prompt.bindings if b.ref_type == ReferenceModality.VIDEO
        }
        bound_audio_indices = {
            b.index for b in parsed_prompt.bindings if b.ref_type == ReferenceModality.AUDIO
        }

        for i, path in enumerate(image_references):
            if (i + 1) not in bound_image_indices:
                latent = self._encode_image_ref(path, height, width)
                items.append(ReferenceItem(
                    latent=latent,
                    modality=ReferenceModality.IMAGE,
                    attribute_tags=[ReferenceAttribute.IDENTITY, ReferenceAttribute.APPEARANCE],
                    strength=strength,
                    downscale_factor=downscale_factor,
                    frame_idx=0 if i == 0 else None,
                ))

        for i, path in enumerate(video_references):
            if (i + 1) not in bound_video_indices:
                latent = self._encode_video_ref(path, height, width)
                items.append(ReferenceItem(
                    latent=latent,
                    modality=ReferenceModality.VIDEO,
                    attribute_tags=[ReferenceAttribute.MOTION, ReferenceAttribute.CAMERA],
                    strength=strength,
                    downscale_factor=downscale_factor,
                ))

        for i, path in enumerate(audio_references):
            if (i + 1) not in bound_audio_indices:
                latent = self._encode_audio_ref(path, height, width)
                if latent is not None:
                    items.append(ReferenceItem(
                        latent=latent,
                        modality=ReferenceModality.AUDIO,
                        attribute_tags=[ReferenceAttribute.AUDIO_RHYTHM, ReferenceAttribute.AUDIO_MOOD],
                        strength=strength,
                    ))

        return items

    def _encode_image_ref(self, path: str, height: int, width: int) -> torch.Tensor:
        """Encode an image reference using the video VAE."""
        from ltx_pipelines.utils.helpers import load_image_and_preprocess
        image = load_image_and_preprocess(
            image_path=path, height=height, width=width,
            dtype=self.dtype, device=self.device, crf=33,
        )
        return self.image_conditioner.encoder(image)

    def _encode_video_ref(self, path: str, height: int, width: int) -> torch.Tensor:
        """Encode a video reference using the video VAE."""
        output_shape = VideoPixelShape(
            batch=1, frames=89, height=height, width=width, fps=25.0,
        )
        latent = video_latent_from_file(
            video_encoder=self.image_conditioner.encoder,
            file_path=path,
            output_shape=output_shape,
            device=self.device,
            dtype=self.dtype,
        )
        if latent is None:
            raise ValueError(f"Failed to encode video reference: {path}")
        return latent

    def _encode_audio_ref(self, path: str, height: int, width: int) -> torch.Tensor | None:
        """Encode an audio reference using the audio VAE."""
        output_shape = VideoPixelShape(
            batch=1, frames=89, height=height, width=width, fps=25.0,
        )
        return audio_latent_from_file(
            audio_encoder=self.audio_decoder.encoder,
            file_path=path,
            output_shape=output_shape,
            device=self.device,
            dtype=self.dtype,
        )
