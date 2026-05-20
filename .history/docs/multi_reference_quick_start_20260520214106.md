# Multi-Reference LoRA Training — Quick Start Guide

This guide covers the unified multi-modal multi-reference conditioning system
for LTX-2.3, inspired by Seedance 2.0's @mention reference binding approach.

## Overview

The multi-reference system allows simultaneous conditioning on multiple image,
video, and audio references, each with explicit attribute tags controlling what
the model should extract (identity, motion, style, camera, audio rhythm, etc.).

### Architecture

```
Prompt: "Man @Image1 for identity walks in @Image2's park.
         Replicate @Video1's camera. @Audio1 for rhythm."

┌─────────────────────────────────────────────────────────┐
│                   @mention Prompt Parser                 │
│  Extracts: @Image1→identity, @Image2→scene,             │
│            @Video1→camera, @Audio1→audio_rhythm          │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│         UnifiedMultiReferenceConditioning                │
│  - Patchifies each reference latent                     │
│  - Computes attribute-aware attention masks              │
│  - Appends reference tokens to LatentState              │
│  - Isolated cross-attention between reference groups     │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│              LTX-2.3 Dual-Stream Transformer            │
│  Video Stream (14B) ←→ Cross-Modal Attn ←→ Audio (5B)  │
│  LoRA on to_q/k/v/out for attribute routing             │
└─────────────────────────────────────────────────────────┘
```

## New Files

| File | Description |
|------|-------------|
| `ltx-core/conditioning/types/multi_reference_cond.py` | Core conditioning module: `UnifiedMultiReferenceConditioning`, `ReferenceItem`, `ReferenceModality`, `ReferenceAttribute` |
| `ltx-pipelines/utils/prompt_parser.py` | @mention prompt parser: `parse_prompt()`, `ParsedPrompt`, `ReferenceBinding` |
| `ltx-pipelines/unified_multi_ref.py` | Two-stage inference pipeline with multi-reference support |
| `ltx-trainer/training_strategies/multi_reference.py` | Training strategy: `MultiReferenceStrategy`, `MultiReferenceConfig` |
| `ltx-trainer/configs/ltx2_multi_ref_lora.yaml` | Training config YAML |
| `ltx-trainer/scripts/process_multi_ref.py` | Data preprocessing script |

## Modified Files

| File | Change |
|------|--------|
| `ltx-core/conditioning/__init__.py` | Export new types |
| `ltx-core/conditioning/types/__init__.py` | Export new types |
| `ltx-trainer/training_strategies/__init__.py` | Register `multi_reference` strategy |
| `ltx-trainer/training_strategies/base_strategy.py` | Add `"multi_reference"` to Literal type |
| `ltx-trainer/config.py` | Import + register `MultiReferenceConfig` in discriminated union |
| `ltx-trainer/.gitignore` | Allow new config YAML |

## Step 1: Prepare Your Dataset

Create a CSV/JSON/JSONL file with columns for the target video, caption, and
reference file paths:

```csv
video_path,caption,ref_image_1,ref_image_2,ref_video_1,ref_audio_1
videos/clip1.mp4,"A woman walks through a garden",refs/face.jpg,refs/park.jpg,refs/walk.mp4,refs/birds.wav
videos/clip2.mp4,"A man plays guitar in a studio",refs/face2.jpg,,refs/guitar.mp4,
```

## Step 2: Preprocess Data

```bash
python packages/ltx-trainer/scripts/process_multi_ref.py dataset.csv \
    --output-dir /path/to/preprocessed \
    --model-source ./models/LTX-2.3/ltx-2.3-22b-dev.safetensors \
    --text-encoder-source ./models/text_encoder \
    --resolution-buckets 768x768x25 \
    --video-column video_path \
    --caption-column caption \
    --ref-image-columns ref_image_1 ref_image_2 \
    --ref-video-columns ref_video_1 \
    --ref-audio-columns ref_audio_1 \
    --with-audio
```

Output structure:
```
/path/to/preprocessed/
├── latents/              # Target video latents
├── conditions/           # Text embeddings
├── audio_latents/        # Target audio latents
├── ref_image_latents/    # Image reference latents
├── ref_video_latents/    # Video reference latents
└── ref_audio_latents/    # Audio reference latents
```

## Step 3: Configure Training

Edit `packages/ltx-trainer/configs/ltx2_multi_ref_lora.yaml`:

```yaml
model:
  model_path: "./models/LTX-2.3/ltx-2.3-22b-dev.safetensors"
  text_encoder_path: "./models/text_encoder"
  training_mode: "lora"

lora:
  rank: 64
  alpha: 64
  target_modules:
    - "to_k"
    - "to_q"
    - "to_v"
    - "to_out.0"

training_strategy:
  name: "multi_reference"
  with_audio: true
  with_audio_references: true
  reference_dropout_p: 0.1
  default_image_attributes: ["identity", "appearance"]
  default_video_attributes: ["motion", "camera"]
  default_audio_attributes: ["audio_rhythm", "audio_mood"]

data:
  preprocessed_data_root: "/path/to/preprocessed"
```

## Step 4: Train

```bash
cd packages/ltx-trainer
python -m ltx_trainer.train --config configs/ltx2_multi_ref_lora.yaml
```

## Step 5: Inference

```bash
python -m ltx_pipelines.unified_multi_ref \
    --checkpoint-path ./models/LTX-2.3/ltx-2.3-22b-dev.safetensors \
    --distilled-lora ./models/LTX-2.3/ltx-2.3-22b-distilled-lora-384-1.1.safetensors 0.8 \
    --spatial-upsampler-path ./models/LTX-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
    --gemma-root ./models/text_encoder \
    --prompt "A woman @Image1 for identity walks through @Image2's garden. Replicate @Video1's camera dolly. @Audio1 for background rhythm." \
    --image-references refs/face.jpg refs/park.jpg \
    --video-references refs/walk.mp4 \
    --audio-references refs/birds.wav \
    --output-path output.mp4
```

## @mention Syntax Reference

### Two Parsing Modes

| Mode | How it works | When to use |
|------|-------------|-------------|
| **Rule-based** (default) | Regex + keyword matching | English prompts, fast inference |
| **Semantic** (`--use-semantic-parser`) | Gemma 3 LLM understands context | Chinese/mixed-language prompts, nuanced semantics |

The semantic parser uses Gemma 3 to truly understand what each reference should contribute.
It supports Chinese prompts like:

```
"一个人 @Image1 这个人的长相和身份 走在 @Image2 的公园场景里。
 参考 @Video1 的运镜方式。 @Audio1 的背景节奏。"
```

Gemma 3 will output structured JSON:
```json
{
  "bindings": [
    {"ref_id": "Image1", "ref_type": "image", "attributes": ["identity", "appearance"]},
    {"ref_id": "Image2", "ref_type": "image", "attributes": ["scene"]},
    {"ref_id": "Video1", "ref_type": "video", "attributes": ["motion", "camera"]},
    {"ref_id": "Audio1", "ref_type": "audio", "attributes": ["audio_rhythm", "audio_mood"]}
  ]
}
```

If the semantic parser fails (e.g., JSON decode error), it automatically falls back
to the rule-based parser.

| Mention | Modality | Default Attributes |
|---------|----------|-------------------|
| `@Image1` | Image | identity, appearance |
| `@Video1` | Video | motion, camera |
| `@Audio1` | Audio | audio_rhythm, audio_mood |

### Attribute Keywords

The parser recognizes these keywords near @mentions:

- **Identity**: identity, face, character
- **Appearance**: appearance, look, facial
- **Style**: style, aesthetic
- **Motion**: motion, movement, choreography, dance, action
- **Camera**: camera, dolly, tracking, crane, pan, orbit, zoom
- **Scene**: scene, background, environment, setting
- **Audio Rhythm**: rhythm, beat, tempo, pacing
- **Audio Mood**: mood, atmosphere, music
- **Lip Sync**: lip sync, lip-sync, lipsync

Example with explicit attributes:
```
"A dancer @Image1 for identity performs @Video1 for motion in a @Image2 style setting."
```

## Compatible Models

| Model | Usage |
|-------|-------|
| `ltx-2.3-22b-dev.safetensors` | Base model for training (recommended) |
| `ltx-2.3-22b-distilled-1.1.safetensors` | Faster inference |
| `ltx-2.3-22b-distilled-lora-384-1.1.safetensors` | Stage 2 refinement |
| `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | 2x spatial upsampling |
| `LTX-2.3-22b-IC-LoRA-*` | Can be loaded alongside multi-ref LoRA |
| `LTX-2-19b-LoRA-Camera-Control-*` | Compatible camera LoRAs |

## Key Design Decisions

1. **Attribute-aware attention masks**: Each reference group gets its own
   attention weight based on its attribute tags, enabling the model to learn
   which reference influences which aspect of generation.

2. **Isolated reference groups**: Different reference groups do NOT attend to
   each other (0 in the attention mask), preventing attribute confusion.

3. **Reference dropout**: During training, references are randomly dropped
   (default 10%) to enable classifier-free guidance style robustness.

4. **LoRA-only training**: The multi-reference strategy requires LoRA mode
   to avoid catastrophic forgetting of the base model's capabilities.

5. **Sequential conditioning application**: References are applied one at a
   time to the `LatentState`, each building on the previous attention mask,
   maintaining the existing LTX-2 conditioning protocol.
