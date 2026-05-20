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
from pathlib import Path

import torch

from ltx_core.components.guiders import (
    MultiModalGuiderFactory,
    MultiModalGuiderParams,
    create_multimodal_guider_factory,
)
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.conditioning import (
    ReferenceAttribute,
    ReferenceItem,
    ReferenceModality,
    UnifiedMultiReferenceConditioning,
)
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.loader.registry import Registry
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
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
    state_with_conditionings,
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
        video_guider_params: MultiModalGuiderParams | MultiModalGuiderFactory,
        audio_guider_params: MultiModalGuiderParams | MultiModalGuiderFactory,
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
        """Generate video with unified multi-reference conditioning.

        Args:
            prompt: Text prompt, may contain @mention references.
            image_references: Paths to image reference files.
            video_references: Paths to video reference files.
            audio_references: Paths to audio reference files.
            parsed_prompt: Pre-parsed prompt. If None, auto-parsed from prompt.
            reference_strength: Default strength for all references.
            reference_downscale_factor: Downscale factor for video/image refs.
            attribute_weights: Override default attribute attention weights.
        """
        assert_resolution(height, width, is_two_stage=True)

        # Parse @mention references from prompt
        if parsed_prompt is None:
            parsed_prompt = parse_prompt(prompt)

        # Build reference items from parsed prompt + file paths
        reference_items = self._build_reference_items(
            parsed_prompt=parsed_prompt,
            image_references=image_references or [],
            video_references=video_references or [],
            audio_references=audio_references or [],
            height=height,
            width=width,
            strength=reference_strength,
            downscale_factor=reference_downscale_factor,
        )

        # Also support backward-compatible image conditioning
        image_conditionings = []
        if images:
            image_conditionings = combined_image_conditionings(
                images=images,
                height=height,
                width=width,
                video_encoder=self.image_conditioner.encoder,
                dtype=self.dtype,
                device=self.device,
            )

        # Create the unified multi-reference conditioning
        multi_ref_conditioning = UnifiedMultiReferenceConditioning(
            references=reference_items,
            attribute_weights=attribute_weights,
        )

        # Combine all conditionings
        all_conditionings = list(image_conditionings) + [multi_ref_conditioning]

        # Encode prompt
        video_prompt_embeds, audio_prompt_embeds, prompt_attention_mask = (
            self.prompt_encoder.encode(prompt, negative_prompt, enhance_prompt)
        )

        # Stage 1: Generate at half resolution
        stage_1_height, stage_1_width = height // 2, width // 2
        stage_1_shape = VideoPixelShape(
            batch=1, frames=num_frames, height=stage_1_height,
            width=stage_1_width, fps=frame_rate,
        )

        stage_1_sigmas = stage_1_sigmas or self._scheduler.get_sigmas(num_inference_steps)

        # Run stage 1 with multi-reference conditioning
        stage_1_result = self._run_stage(
            stage=self.stage_1,
            shape=stage_1_shape,
            sigmas=stage_1_sigmas,
            video_prompt_embeds=video_prompt_embeds,
            audio_prompt_embeds=audio_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            conditionings=all_conditionings,
            video_guider_params=video_guider_params,
            audio_guider_params=audio_guider_params,
            seed=seed,
            modality_spec=ModalitySpec.AUDIO_VIDEO,
            tiling_config=tiling_config,
            max_batch_size=max_batch_size,
        )

        # Stage 2: Upsample with distilled LoRA
        stage_2_result = self._run_stage_2_upsample(
            stage_1_latent=stage_1_result,
            sigmas=stage_2_sigmas,
            video_prompt_embeds=video_prompt_embeds,
            audio_prompt_embeds=audio_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            seed=seed,
            tiling_config=tiling_config,
        )

        # Decode video and audio
        video_frames = self.video_decoder.decode(
            stage_2_result, height, width, num_frames, tiling_config=tiling_config,
        )
        audio = self.audio_decoder.decode(stage_2_result)

        return encode_video(video_frames, audio, frame_rate), audio

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
            # Find the corresponding file path
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

        # Also add any file paths not mentioned in the prompt
        items.extend(self._add_unbound_references(
            parsed_prompt, image_references, video_references,
            audio_references, height, width, strength, downscale_factor,
        ))

        return items

    def _resolve_ref_path(
        self,
        binding: "ReferenceBinding",
        image_references: list[str],
        video_references: list[str],
        audio_references: list[str],
    ) -> str | None:
        """Resolve a binding to a file path."""
        idx = binding.index - 1  # 1-indexed to 0-indexed
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

    def _run_stage(
        self,
        stage: DiffusionStage,
        shape: VideoPixelShape,
        sigmas: torch.Tensor,
        video_prompt_embeds: torch.Tensor,
        audio_prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        conditionings: list,
        video_guider_params: MultiModalGuiderParams | MultiModalGuiderFactory,
        audio_guider_params: MultiModalGuiderParams | MultiModalGuiderFactory,
        seed: int,
        modality_spec: ModalitySpec,
        tiling_config: TilingConfig | None = None,
        max_batch_size: int = 1,
    ) -> torch.Tensor:
        """Run a diffusion stage with the given conditionings."""
        from ltx_core.tools import VideoLatentTools
        from ltx_core.components.patchifiers import VideoLatentPatchifier
        from ltx_core.types import VideoLatentShape, SpatioTemporalScaleFactors

        scale_factors = SpatioTemporalScaleFactors.default()
        patchifier = VideoLatentPatchifier(patch_size=1)
        latent_shape = VideoLatentShape.from_pixel_shape(shape)

        tools = VideoLatentTools(
            patchifier=patchifier,
            target_shape=latent_shape,
            fps=shape.fps,
            scale_factors=scale_factors,
        )

        # Create initial state and apply conditionings
        from ltx_pipelines.utils.helpers import create_noised_state
        noiser = GaussianNoiser()
        state = create_noised_state(
            tools=tools,
            conditionings=conditionings,
            noiser=noiser,
            dtype=self.dtype,
            device=self.device,
        )

        # Run denoising
        guider_factory = create_multimodal_guider_factory(
            video_guider_params=video_guider_params,
            audio_guider_params=audio_guider_params,
        )
        denoiser = FactoryGuidedDenoiser(
            model=stage.model,
            guider_factory=guider_factory,
            scheduler=self._scheduler,
        )

        result = denoiser(
            state=state,
            sigmas=sigmas,
            video_context=video_prompt_embeds,
            audio_context=audio_prompt_embeds,
            video_context_mask=prompt_attention_mask,
            audio_context_mask=prompt_attention_mask,
            seed=seed,
            modality_spec=modality_spec,
        )

        # Clear conditioning tokens
        return tools.clear_conditioning(result)

    def _run_stage_2_upsample(
        self,
        stage_1_latent: torch.Tensor,
        sigmas: torch.Tensor,
        video_prompt_embeds: torch.Tensor,
        audio_prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        seed: int,
        tiling_config: TilingConfig | None = None,
    ) -> torch.Tensor:
        """Run stage 2: spatial upsampling with distilled LoRA."""
        # Upsample latent spatially
        upsampled = self.upsampler.upsample(
            stage_1_latent, tiling_config=tiling_config,
        )

        # Denoise with distilled LoRA
        denoiser = SimpleDenoiser(
            model=self.stage_2.model,
            scheduler=self._scheduler,
        )

        return denoiser(
            latent=upsampled,
            sigmas=sigmas,
            video_context=video_prompt_embeds,
            audio_context=audio_prompt_embeds,
            video_context_mask=prompt_attention_mask,
            audio_context_mask=prompt_attention_mask,
            seed=seed,
        )


# ---- CLI entry point ----

def _build_arg_parser():
    """Build argument parser for the unified multi-ref pipeline."""
    parser = default_2_stage_arg_parser(
        description="Unified multi-reference video generation pipeline",
    )

    # Multi-reference arguments
    ref_group = parser.add_argument_group("Multi-Reference Options")
    ref_group.add_argument(
        "--image-references", nargs="*", default=[],
        help="Image reference file paths (referenced as @Image1, @Image2, ...)",
    )
    ref_group.add_argument(
        "--video-references", nargs="*", default=[],
        help="Video reference file paths (referenced as @Video1, @Video2, ...)",
    )
    ref_group.add_argument(
        "--audio-references", nargs="*", default=[],
        help="Audio reference file paths (referenced as @Audio1, @Audio2, ...)",
    )
    ref_group.add_argument(
        "--reference-strength", type=float, default=1.0,
        help="Default conditioning strength for all references (0.0-1.0)",
    )
    ref_group.add_argument(
        "--reference-downscale-factor", type=int, default=1,
        help="Spatial downscale factor for video/image references",
    )

    return parser


if __name__ == "__main__":
    import argparse
    from ltx_pipelines.utils.args import ImageConditioningInput

    parser = _build_arg_parser()
    args = parser.parse_args()
    args = detect_params(args)
    args = detect_checkpoint_path(args)

    # Build pipeline
    distilled_lora = [
        LoraPathStrengthAndSDOps(
            args.distilled_lora_path,
            args.distilled_lora_strength,
            LTXV_LORA_COMFY_RENAMING_MAP,
        ),
    ] if args.distilled_lora_path else []

    loras = []
    for lora_path, lora_strength in args.loras:
        loras.append(LoraPathStrengthAndSDOps(
            lora_path, lora_strength, LTXV_LORA_COMFY_RENAMING_MAP,
        ))

    pipeline = UnifiedMultiRefPipeline(
        checkpoint_path=args.checkpoint_path,
        distilled_lora=distilled_lora,
        spatial_upsampler_path=args.spatial_upsampler_path,
        gemma_root=args.gemma_root,
        loras=loras,
        quantization=QuantizationPolicy.fp8_cast() if args.quantization == "fp8-cast" else None,
    )

    # Parse prompt for @mentions
    parsed = parse_prompt(args.prompt)
    logger.info(f"Parsed {len(parsed.bindings)} reference bindings from prompt")

    # Run pipeline
    video_iter, audio = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        num_inference_steps=args.num_inference_steps,
        video_guider_params=args.video_guider_params,
        audio_guider_params=args.audio_guider_params,
        image_references=args.image_references,
        video_references=args.video_references,
        audio_references=args.audio_references,
        parsed_prompt=parsed,
        reference_strength=args.reference_strength,
        reference_downscale_factor=args.reference_downscale_factor,
        tiling_config=TilingConfig.default() if args.spatial_tile else None,
        enhance_prompt=args.enhance_prompt,
    )

    # Save output
    for frame_chunk in video_iter:
        pass  # Video is written incrementally by encode_video
