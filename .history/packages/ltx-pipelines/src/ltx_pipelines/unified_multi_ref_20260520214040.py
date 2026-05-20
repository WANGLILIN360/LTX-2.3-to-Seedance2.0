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

import torch

from ltx_core.components.guiders import (
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
from ltx_core.model.video_vae import TilingConfig, VideoEncoder, get_video_chunks_number
from ltx_core.quantization import QuantizationPolicy
from ltx_core.types import Audio, VideoPixelShape
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.blocks import (
    AudioConditioner,
    AudioDecoder,
    DiffusionStage,
    ImageConditioner,
    PromptEncoder,
    VideoDecoder,
    VideoUpsampler,
)
from ltx_pipelines.utils.constants import STAGE_2_DISTILLED_SIGMAS
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
from ltx_pipelines.utils.semantic_prompt_parser import SemanticPromptParser
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
        use_semantic_parser: bool = False,
    ):
        self.device = device or get_device()
        self.dtype = torch.bfloat16
        self._scheduler = LTX2Scheduler()
        self.use_semantic_parser = use_semantic_parser
        self._semantic_parser = None

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
        self.audio_conditioner = AudioConditioner(
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
        video_guider_params: MultiModalGuiderParams,
        audio_guider_params: MultiModalGuiderParams,
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

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)

        # Parse @mention references from prompt
        if parsed_prompt is None:
            if self.use_semantic_parser:
                if self._semantic_parser is None:
                    self._semantic_parser = SemanticPromptParser(
                        gemma_root=self.prompt_encoder._gemma_root if hasattr(self.prompt_encoder, '_gemma_root') else "",
                        device=self.device,
                        dtype=self.dtype,
                    )
                parsed_prompt = self._semantic_parser.parse(prompt)
            else:
                parsed_prompt = parse_prompt(prompt)

        # Encode prompt — returns (positive_context, negative_context)
        ctx_p, ctx_n = self.prompt_encoder(
            [prompt, negative_prompt],
            enhance_first_prompt=enhance_prompt,
            enhance_prompt_image=images[0][0] if images and len(images) > 0 else None,
            enhance_prompt_seed=seed,
        )
        v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
        v_context_n, a_context_n = ctx_n.video_encoding, ctx_n.audio_encoding

        # Stage 1: Generate at half resolution
        stage_1_height, stage_1_width = height // 2, width // 2

        # Build conditionings using the ImageConditioner callback pattern
        # First, standard image conditionings (backward compatible)
        stage_1_image_conditionings: list[ConditioningItem] = []
        if images:
            stage_1_image_conditionings = self.image_conditioner(
                lambda enc: combined_image_conditionings(
                    images=images,
                    height=stage_1_height,
                    width=stage_1_width,
                    video_encoder=enc,
                    dtype=self.dtype,
                    device=self.device,
                )
            )

        # Build multi-reference conditionings using the video encoder
        # (image/video refs) and audio encoder (audio refs)
        stage_1_ref_conditionings = self.image_conditioner(
            lambda enc: self._build_multi_ref_conditionings(
                encoder=enc,
                parsed_prompt=parsed_prompt,
                image_references=image_references or [],
                video_references=video_references or [],
                audio_references=audio_references or [],
                height=stage_1_height,
                width=stage_1_width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                strength=reference_strength,
                downscale_factor=reference_downscale_factor,
                attribute_weights=attribute_weights,
            )
        )

        all_conditionings = stage_1_image_conditionings + stage_1_ref_conditionings

        sigmas = (
            stage_1_sigmas if stage_1_sigmas is not None
            else self._scheduler.execute(steps=num_inference_steps)
        ).to(dtype=torch.float32, device=self.device)

        video_state, audio_state = self.stage_1(
            denoiser=FactoryGuidedDenoiser(
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
            ),
            sigmas=sigmas,
            noiser=noiser,
            width=stage_1_width,
            height=stage_1_height,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(context=v_context_p, conditionings=all_conditionings),
            audio=ModalitySpec(context=a_context_p),
            max_batch_size=max_batch_size,
        )

        # Stage 2: Upsample and refine at full resolution with distilled LoRA
        upscaled_video_latent = self.upsampler(video_state.latent[:1])

        # Build stage 2 conditionings at full resolution
        stage_2_image_conditionings: list[ConditioningItem] = []
        if images:
            stage_2_image_conditionings = self.image_conditioner(
                lambda enc: combined_image_conditionings(
                    images=images,
                    height=height,
                    width=width,
                    video_encoder=enc,
                    dtype=self.dtype,
                    device=self.device,
                )
            )

        # Build stage 2 multi-reference conditionings at full resolution
        stage_2_ref_conditionings = self.image_conditioner(
            lambda enc: self._build_multi_ref_conditionings(
                encoder=enc,
                parsed_prompt=parsed_prompt,
                image_references=image_references or [],
                video_references=video_references or [],
                audio_references=audio_references or [],
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                strength=reference_strength,
                downscale_factor=reference_downscale_factor,
                attribute_weights=attribute_weights,
            )
        )

        stage_2_all_conditionings = stage_2_image_conditionings + stage_2_ref_conditionings

        stage_2_sigmas = stage_2_sigmas.to(dtype=torch.float32, device=self.device)

        video_state, audio_state = self.stage_2(
            denoiser=SimpleDenoiser(v_context=v_context_p, a_context=a_context_p),
            sigmas=stage_2_sigmas,
            noiser=noiser,
            width=width,
            height=height,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(
                context=v_context_p,
                conditionings=stage_2_all_conditionings,
                noise_scale=stage_2_sigmas[0].item(),
                initial_latent=upscaled_video_latent,
            ),
            audio=ModalitySpec(
                context=a_context_p,
                noise_scale=stage_2_sigmas[0].item(),
                initial_latent=audio_state.latent,
            ),
            max_batch_size=max_batch_size,
        )

        # Decode outputs
        decoded_video = self.video_decoder(video_state.latent, tiling_config, generator)
        decoded_audio = self.audio_decoder(audio_state.latent)
        return decoded_video, decoded_audio

    def _build_multi_ref_conditionings(
        self,
        encoder: VideoEncoder,
        parsed_prompt: ParsedPrompt,
        image_references: list[str],
        video_references: list[str],
        audio_references: list[str],
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        strength: float,
        downscale_factor: int,
        attribute_weights: dict[ReferenceAttribute, float] | None,
    ) -> list[ConditioningItem]:
        """Build multi-reference conditioning items using the video encoder.

        Audio references are encoded separately via AudioConditioner since
        the video encoder callback does not have access to the audio encoder.
        They are appended after the video/image references are built.
        """
        from ltx_pipelines.utils.helpers import load_image_and_preprocess

        ref_items: list[ReferenceItem] = []
        audio_ref_paths: list[tuple[str, list[ReferenceAttribute]]] = []

        for binding in parsed_prompt.bindings:
            ref_path = self._resolve_ref_path(
                binding, image_references, video_references, audio_references,
            )
            if ref_path is None:
                logger.warning(f"No file path found for @{binding.ref_id}, skipping")
                continue

            if binding.ref_type == ReferenceModality.IMAGE:
                image = load_image_and_preprocess(
                    image_path=ref_path, height=height, width=width,
                    dtype=self.dtype, device=self.device, crf=33,
                )
                latent = encoder(image)
                ref_items.append(ReferenceItem(
                    latent=latent,
                    modality=ReferenceModality.IMAGE,
                    attribute_tags=binding.attributes,
                    strength=strength,
                    downscale_factor=downscale_factor,
                    frame_idx=0 if binding.index == 1 else None,
                ))

            elif binding.ref_type == ReferenceModality.VIDEO:
                output_shape = VideoPixelShape(
                    batch=1, frames=num_frames, height=height, width=width, fps=frame_rate,
                )
                latent = video_latent_from_file(
                    video_encoder=encoder,
                    file_path=ref_path,
                    output_shape=output_shape,
                    device=self.device,
                    dtype=self.dtype,
                )
                if latent is not None:
                    ref_items.append(ReferenceItem(
                        latent=latent,
                        modality=ReferenceModality.VIDEO,
                        attribute_tags=binding.attributes,
                        strength=strength,
                        downscale_factor=downscale_factor,
                    ))

            elif binding.ref_type == ReferenceModality.AUDIO:
                audio_ref_paths.append((ref_path, binding.attributes))

        # Add unbound references (files not mentioned in prompt)
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
                image = load_image_and_preprocess(
                    image_path=path, height=height, width=width,
                    dtype=self.dtype, device=self.device, crf=33,
                )
                latent = encoder(image)
                ref_items.append(ReferenceItem(
                    latent=latent,
                    modality=ReferenceModality.IMAGE,
                    attribute_tags=[ReferenceAttribute.IDENTITY, ReferenceAttribute.APPEARANCE],
                    strength=strength,
                    downscale_factor=downscale_factor,
                    frame_idx=0 if i == 0 else None,
                ))

        for i, path in enumerate(video_references):
            if (i + 1) not in bound_video_indices:
                output_shape = VideoPixelShape(
                    batch=1, frames=num_frames, height=height, width=width, fps=frame_rate,
                )
                latent = video_latent_from_file(
                    video_encoder=encoder,
                    file_path=path,
                    output_shape=output_shape,
                    device=self.device,
                    dtype=self.dtype,
                )
                if latent is not None:
                    ref_items.append(ReferenceItem(
                        latent=latent,
                        modality=ReferenceModality.VIDEO,
                        attribute_tags=[ReferenceAttribute.MOTION, ReferenceAttribute.CAMERA],
                        strength=strength,
                        downscale_factor=downscale_factor,
                    ))

        # Collect unbound audio reference paths
        for i, path in enumerate(audio_references):
            if (i + 1) not in bound_audio_indices:
                audio_ref_paths.append((path, [ReferenceAttribute.AUDIO_RHYTHM, ReferenceAttribute.AUDIO_MOOD]))

        # Encode audio references using AudioConditioner
        if audio_ref_paths:
            output_shape = VideoPixelShape(
                batch=1, frames=num_frames, height=height, width=width, fps=frame_rate,
            )
            audio_ref_items = self.audio_conditioner(
                lambda audio_enc: self._encode_audio_references(
                    audio_encoder=audio_enc,
                    audio_ref_paths=audio_ref_paths,
                    output_shape=output_shape,
                    strength=strength,
                )
            )
            ref_items.extend(audio_ref_items)

        if not ref_items:
            return []

        return [
            UnifiedMultiReferenceConditioning(
                references=ref_items,
                attribute_weights=attribute_weights,
            )
        ]

    def _encode_audio_references(
        self,
        audio_encoder: torch.nn.Module,
        audio_ref_paths: list[tuple[str, list[ReferenceAttribute]]],
        output_shape: VideoPixelShape,
        strength: float,
    ) -> list[ReferenceItem]:
        """Encode audio reference files into ReferenceItem objects."""
        ref_items: list[ReferenceItem] = []
        for ref_path, attributes in audio_ref_paths:
            latent = audio_latent_from_file(
                audio_encoder=audio_encoder,
                file_path=ref_path,
                output_shape=output_shape,
                device=self.device,
                dtype=self.dtype,
            )
            if latent is not None:
                ref_items.append(ReferenceItem(
                    latent=latent,
                    modality=ReferenceModality.AUDIO,
                    attribute_tags=attributes,
                    strength=strength,
                ))
            else:
                logger.warning(f"Failed to encode audio reference: {ref_path}")
        return ref_items

    def _resolve_ref_path(self, binding, image_references, video_references, audio_references):
        """Resolve a binding to a file path."""
        idx = binding.index - 1
        if binding.ref_type == ReferenceModality.IMAGE and idx < len(image_references):
            return image_references[idx]
        elif binding.ref_type == ReferenceModality.VIDEO and idx < len(video_references):
            return video_references[idx]
        elif binding.ref_type == ReferenceModality.AUDIO and idx < len(audio_references):
            return audio_references[idx]
        return None
