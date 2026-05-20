#!/usr/bin/env python3
"""Preprocess multi-reference training data for LTX-2.3 LoRA training.

Extends the standard process_videos.py to also encode image, video, and
audio reference files as separate latent directories that the
multi_reference training strategy can load.

Dataset input format (CSV/JSON/JSONL):
    video_path, caption, ref_image_1, ref_image_2, ref_video_1, ref_audio_1

Output directory structure:
    preprocessed_data_root/
    ├── latents/              # Target video latents
    ├── conditions/           # Text embeddings
    ├── audio_latents/        # Target audio latents (optional)
    ├── ref_image_latents/    # Image reference latents
    ├── ref_video_latents/    # Video reference latents
    └── ref_audio_latents/    # Audio reference latents (optional)

Usage:
    python scripts/process_multi_ref.py dataset.csv \
        --resolution-buckets 768x768x25 \
        --output-dir /path/to/output \
        --model-source /path/to/ltx-2.3-22b-dev.safetensors \
        --ref-image-columns ref_image_1 ref_image_2 \
        --ref-video-columns ref_video_1 \
        --ref-audio-columns ref_audio_1
"""

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import typer
from accelerate import PartialState
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from transformers.utils.logging import disable_progress_bar

from ltx_core.model.audio_vae import AudioProcessor
from ltx_core.types import Audio
from ltx_trainer import logger
from ltx_trainer.model_loader import load_audio_vae_encoder, load_video_vae_encoder
from ltx_trainer.utils import open_image_as_srgb
from ltx_trainer.video_utils import get_video_frame_count, read_video

disable_progress_bar()

# Constants
VAE_SPATIAL_FACTOR = 32
VAE_TEMPORAL_FACTOR = 8
AUDIO_LATENT_CHANNELS = 8
AUDIO_FREQUENCY_BINS = 16

app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    help="Process multi-reference training data for LTX-2.3 LoRA training.",
)


class MultiRefDataset(torch.utils.data.Dataset):
    """Dataset for processing multi-reference training data."""

    def __init__(
        self,
        dataset_file: str | Path,
        video_column: str,
        caption_column: str,
        ref_image_columns: list[str],
        ref_video_columns: list[str],
        ref_audio_columns: list[str],
        resolution_buckets: list[tuple[int, int, int]],
        reshape_mode: str = "center",
        with_audio: bool = False,
    ):
        super().__init__()
        self.dataset_file = Path(dataset_file)
        self.data_root = self.dataset_file.parent
        self.video_column = video_column
        self.caption_column = caption_column
        self.ref_image_columns = ref_image_columns
        self.ref_video_columns = ref_video_columns
        self.ref_audio_columns = ref_audio_columns
        self.resolution_buckets = resolution_buckets
        self.reshape_mode = reshape_mode
        self.with_audio = with_audio

        # Load data
        self._load_dataset()

    def _load_dataset(self):
        """Load dataset from CSV/JSON/JSONL."""
        path = self.dataset_file
        if path.suffix == ".csv":
            self.df = pd.read_csv(path)
        elif path.suffix == ".json":
            self.df = pd.read_json(path)
        elif path.suffix == ".jsonl":
            self.df = pd.read_json(path, lines=True)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        logger.info(f"Loaded {len(self.df)} samples from {path}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]

        result: dict[str, Any] = {
            "video_path": str(self.data_root / row[self.video_column]),
            "caption": str(row[self.caption_column]),
            "ref_images": [],
            "ref_videos": [],
            "ref_audios": [],
        }

        # Collect reference paths
        for col in self.ref_image_columns:
            if col in row and pd.notna(row[col]):
                result["ref_images"].append(str(self.data_root / row[col]))

        for col in self.ref_video_columns:
            if col in row and pd.notna(row[col]):
                result["ref_videos"].append(str(self.data_root / row[col]))

        for col in self.ref_audio_columns:
            if col in row and pd.notna(row[col]):
                result["ref_audios"].append(str(self.data_root / row[col]))

        return result


@app.command()
def process(
    dataset_file: str = typer.Argument(help="Path to CSV/JSON/JSONL dataset file"),
    output_dir: str = typer.Option(..., "--output-dir", help="Output directory"),
    model_source: str = typer.Option(..., "--model-source", help="Path to model safetensors"),
    text_encoder_source: str = typer.Option(None, "--text-encoder-source", help="Path to text encoder"),
    resolution_buckets: list[str] = typer.Option(
        ["768x768x25"], "--resolution-buckets",
        help="Resolution buckets as WxHxF (e.g., 768x768x25)",
    ),
    video_column: str = typer.Option("video_path", "--video-column", help="Video path column name"),
    caption_column: str = typer.Option("caption", "--caption-column", help="Caption column name"),
    ref_image_columns: list[str] = typer.Option(
        [], "--ref-image-columns", help="Column names for image references",
    ),
    ref_video_columns: list[str] = typer.Option(
        [], "--ref-video-columns", help="Column names for video references",
    ),
    ref_audio_columns: list[str] = typer.Option(
        [], "--ref-audio-columns", help="Column names for audio references",
    ),
    with_audio: bool = typer.Option(False, "--with-audio", help="Extract audio from videos"),
    batch_size: int = typer.Option(1, "--batch-size", help="Batch size for encoding"),
    device: str = typer.Option("cuda", "--device", help="Device to use"),
    dtype: str = typer.Option("bfloat16", "--dtype", help="Data type"),
    max_frames: int = typer.Option(89, "--max-frames", help="Maximum number of frames"),
    fps: float = typer.Option(25.0, "--fps", help="Target FPS"),
) -> None:
    """Process multi-reference training data."""
    console = Console()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Parse resolution buckets
    buckets = []
    for b in resolution_buckets:
        w, h, f = b.split("x")
        buckets.append((int(f), int(h), int(w)))

    # Load models
    console.print("[bold blue]Loading models...[/]")
    torch_dtype = getattr(torch, dtype)
    device_obj = torch.device(device)

    video_vae_encoder = load_video_vae_encoder(model_source, device=device_obj, dtype=torch_dtype)
    audio_vae_encoder = None
    if with_audio or ref_audio_columns:
        audio_vae_encoder = load_audio_vae_encoder(model_source, device=device_obj, dtype=torch_dtype)

    # Create dataset
    dataset = MultiRefDataset(
        dataset_file=dataset_file,
        video_column=video_column,
        caption_column=caption_column,
        ref_image_columns=ref_image_columns,
        ref_video_columns=ref_video_columns,
        ref_audio_columns=ref_audio_columns,
        resolution_buckets=buckets,
        with_audio=with_audio,
    )

    # Create output directories
    dirs = {
        "latents": output_path / "latents",
        "conditions": output_path / "conditions",
        "ref_image_latents": output_path / "ref_image_latents",
        "ref_video_latents": output_path / "ref_video_latents",
    }
    if with_audio:
        dirs["audio_latents"] = output_path / "audio_latents"
    if ref_audio_columns:
        dirs["ref_audio_latents"] = output_path / "ref_audio_latents"

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # Process each sample
    console.print(f"[bold green]Processing {len(dataset)} samples...[/]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Encoding...", total=len(dataset))

        for i in range(len(dataset)):
            sample = dataset[i]
            sample_name = f"sample_{i:06d}"

            # Encode target video
            video_path = sample["video_path"]
            try:
                video_tensor, video_fps = _load_and_preprocess_video(
                    video_path, buckets, max_frames, fps, torch_dtype, device_obj,
                )
                video_latent = video_vae_encoder(video_tensor)

                # Save target video latent
                _save_latent(video_latent, dirs["latents"] / f"{sample_name}.safetensors",
                             num_frames=video_tensor.shape[1],
                             height=video_tensor.shape[2],
                             width=video_tensor.shape[3],
                             fps=video_fps)

                # Save caption
                _save_caption(sample["caption"], dirs["conditions"] / f"{sample_name}.json")

                # Encode reference images
                for j, ref_path in enumerate(sample["ref_images"]):
                    ref_img = _load_and_preprocess_image(
                        ref_path, buckets, torch_dtype, device_obj,
                    )
                    ref_latent = video_vae_encoder(ref_img)
                    _save_latent(ref_latent, dirs["ref_image_latents"] / f"{sample_name}_img{j}.safetensors",
                                 num_frames=1,
                                 height=ref_img.shape[2],
                                 width=ref_img.shape[3],
                                 fps=fps)

                # Encode reference videos
                for j, ref_path in enumerate(sample["ref_videos"]):
                    ref_vid, ref_fps = _load_and_preprocess_video(
                        ref_path, buckets, max_frames, fps, torch_dtype, device_obj,
                    )
                    ref_latent = video_vae_encoder(ref_vid)
                    _save_latent(ref_latent, dirs["ref_video_latents"] / f"{sample_name}_vid{j}.safetensors",
                                 num_frames=ref_vid.shape[1],
                                 height=ref_vid.shape[2],
                                 width=ref_vid.shape[3],
                                 fps=ref_fps)

                # Encode reference audio
                if audio_vae_encoder is not None:
                    for j, ref_path in enumerate(sample["ref_audios"]):
                        ref_audio = _load_and_preprocess_audio(ref_path, torch_dtype, device_obj)
                        if ref_audio is not None:
                            ref_latent = audio_vae_encoder(ref_audio)
                            _save_latent(ref_latent, dirs["ref_audio_latents"] / f"{sample_name}_aud{j}.safetensors")

                # Encode target audio
                if with_audio and audio_vae_encoder is not None:
                    target_duration = video_tensor.shape[1] / fps
                    target_audio = _extract_audio_from_video(video_path, target_duration, torch_dtype, device_obj)
                    if target_audio is not None:
                        audio_latent = audio_vae_encoder(target_audio)
                        _save_latent(audio_latent, dirs["audio_latents"] / f"{sample_name}.safetensors")

            except Exception as e:
                logger.warning(f"Failed to process sample {i}: {e}")
                continue

            progress.advance(task)

    console.print("[bold green]Done![/]")
    console.print(f"Output saved to: {output_path.resolve()}")


def _load_and_preprocess_video(
    path: str,
    buckets: list[tuple[int, int, int]],
    max_frames: int,
    target_fps: float,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    """Load and preprocess a video file."""
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    video, fps, _ = read_video(path, output_format="TCHW")
    if video is None:
        raise ValueError(f"Failed to read video: {path}")

    # Select closest bucket
    num_frames = min(video.shape[0], max_frames)
    best_bucket = min(buckets, key=lambda b: abs(b[0] - num_frames))
    target_frames, target_h, target_w = best_bucket

    # Sample frames
    if video.shape[0] > target_frames:
        indices = torch.linspace(0, video.shape[0] - 1, target_frames).long()
        video = video[indices]

    # Convert to float and normalize
    video = video.float() / 255.0
    video = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(video)

    # Resize
    video = transforms.Resize(
        size=(target_h, target_w),
        interpolation=InterpolationMode.BILINEAR,
    )(video)

    # Reformat to [C, F, H, W] for VAE
    video = video.permute(1, 0, 2, 3)  # [T, C, H, W] -> [C, T, H, W]

    return video.to(dtype=dtype, device=device), fps


def _load_and_preprocess_image(
    path: str,
    buckets: list[tuple[int, int, int]],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Load and preprocess an image file as a single-frame video."""
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    image = open_image_as_srgb(path)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")

    # Use the first bucket's spatial dimensions
    _, target_h, target_w = buckets[0]

    image = image.convert("RGB")
    image_tensor = transforms.functional.pil_to_tensor(image).float() / 255.0
    image_tensor = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(image_tensor)
    image_tensor = transforms.Resize(
        size=(target_h, target_w),
        interpolation=InterpolationMode.BILINEAR,
    )(image_tensor)

    # [C, H, W] -> [C, 1, H, W] (single frame video)
    image_tensor = image_tensor.unsqueeze(1)

    return image_tensor.to(dtype=dtype, device=device)


def _load_and_preprocess_audio(
    path: str,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor | None:
    """Load and preprocess an audio file."""
    try:
        import torchaudio
        waveform, sr = torchaudio.load(path)
        processor = AudioProcessor()
        audio = processor(waveform, sr)
        return audio.to(dtype=dtype, device=device)
    except Exception as e:
        logger.warning(f"Failed to process audio {path}: {e}")
        return None


def _extract_audio_from_video(
    video_path: str,
    target_duration: float,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor | None:
    """Extract audio from a video file."""
    try:
        import torchaudio
        waveform, sr = torchaudio.load(video_path)
        # Trim to target duration
        max_samples = int(target_duration * sr)
        if waveform.shape[-1] > max_samples:
            waveform = waveform[:, :max_samples]
        processor = AudioProcessor()
        audio = processor(waveform, sr)
        return audio.to(dtype=dtype, device=device)
    except Exception as e:
        logger.warning(f"Failed to extract audio from {video_path}: {e}")
        return None


def _save_latent(latent: torch.Tensor, path: Path, **metadata):
    """Save a latent tensor to a safetensors file."""
    from safetensors.torch import save_file
    latent = latent.cpu().to(torch.bfloat16)
    save_file({"latent": latent}, str(path), metadata={k: str(v) for k, v in metadata.items()})


def _save_caption(caption: str, path: Path):
    """Save a caption to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"caption": caption}, f, ensure_ascii=False)


if __name__ == "__main__":
    app()
