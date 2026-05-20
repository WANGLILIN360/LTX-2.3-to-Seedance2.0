# LTX-2.3 → Seedance 2.0

[![Base Model](https://img.shields.io/badge/Base%20Model-LTX--2.3-orange?logo=huggingface)](https://huggingface.co/Lightricks/LTX-2.3)
[![Inspired By](https://img.shields.io/badge/Inspired%20By-Seedance%202.0-blue)](https://seedance.ai)
[![Paper](https://img.shields.io/badge/Paper-ID--LoRA-EC1C24?logo=adobeacrobatreader&logoColor=white)](https://arxiv.org/abs/2603.10256)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)

**LTX-2.3 → Seedance 2.0** 是基于 [Lightricks/LTX-2](https://github.com/Lightricks/LTX-2) 开源项目及 LTX-2.3 开源模型权重，进行微调训练，整合成一个**统一 LLM 输入、多模态参考、音画同出**的视频生成模型。

核心思路借鉴 [Seedance 2.0](https://seedance.ai) 的 `@mention` 引用绑定机制与 [ID-LoRA](https://arxiv.org/abs/2603.10256) 的参考条件注入方式，在 LTX-2.3 的 Dual-Stream DiT 架构上实现多参考条件融合，使模型能够同时接受图像、视频、音频等多种参考输入，并生成音画同步的高质量视频。

## ✨ 核心特性

- **多模态参考输入** — 同时支持图像（身份/外观/风格）、视频（动作/运镜）、音频（节奏/氛围）参考
- **@mention 语义绑定** — 类似 Seedance 2.0，通过 `@Image1 for identity` 语法将参考与语义属性绑定
- **音画同出** — 基于 LTX-2.3 的 Dual-Stream 架构，视频与音频联合生成
- **中文语义解析** — 支持 Gemma 3 语义解析器处理中文/混合语言 prompt
- **LoRA 微调** — 仅训练 LoRA 权重，保留基座模型全部能力
- **Identity Guidance** — CFG 变体，在"有参考"与"无参考"预测之间外推，增强身份保真度

## 🏗️ 架构概览

```
Prompt: "A woman @Image1 for identity walks in @Image2's park.
         Replicate @Video1's camera. @Audio1 for rhythm."

┌─────────────────────────────────────────────────────────┐
│                   @mention Prompt Parser                 │
│  Rule-based / Semantic (Gemma 3)                         │
│  Extracts: @Image1→identity, @Image2→scene,             │
│            @Video1→camera, @Audio1→audio_rhythm          │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│         UnifiedMultiReferenceConditioning                │
│  - Patchify each reference latent                       │
│  - Concatenate reference tokens to sequence (IC-LoRA)   │
│  - Negative temporal positions for reference groups      │
│  - Isolated attention between reference groups (0 mask)  │
│  - Full cross-attention target↔each reference (1.0)     │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│              LTX-2.3 Dual-Stream Transformer            │
│  Video Stream (14B) ←→ Cross-Modal Attn ←→ Audio (5B)  │
│  LoRA on to_q/k/v/out for attribute routing             │
│  Text cross-attention handles semantic routing           │
└─────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/WANGLILIN360/LTX-2.3-to-Seedance2.0.git
cd LTX-2.3-to-Seedance2.0

# Set up the environment
uv sync --frozen
source .venv/bin/activate   # Linux/Mac
# or .venv\Scripts\activate  # Windows
```

### Required Models

Download the following models from the [LTX-2.3 HuggingFace repository](https://huggingface.co/Lightricks/LTX-2.3):

**LTX-2.3 Model Checkpoint** (choose one)
  * [`ltx-2.3-22b-dev.safetensors`](https://huggingface.co/Lightricks/LTX-2.3/blob/main/ltx-2.3-22b-dev.safetensors) — [Download](https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-22b-dev.safetensors)
  * [`ltx-2.3-22b-distilled-1.1.safetensors`](https://huggingface.co/Lightricks/LTX-2.3/blob/main/ltx-2.3-22b-distilled-1.1.safetensors) — [Download](https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-22b-distilled-1.1.safetensors)

**Spatial Upscaler**
  * [`ltx-2.3-spatial-upscaler-x2-1.1.safetensors`](https://huggingface.co/Lightricks/LTX-2.3/blob/main/ltx-2.3-spatial-upscaler-x2-1.1.safetensors) — [Download](https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x2-1.1.safetensors)

**Distilled LoRA** (required for two-stage pipelines)
  * [`ltx-2.3-22b-distilled-lora-384-1.1.safetensors`](https://huggingface.co/Lightricks/LTX-2.3/blob/main/ltx-2.3-22b-distilled-lora-384-1.1.safetensors) — [Download](https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-22b-distilled-lora-384-1.1.safetensors)

**Gemma Text Encoder**
  * [`Gemma 3`](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized/tree/main)

**Official IC-LoRAs** (compatible, can be loaded alongside multi-ref LoRA)
  * [`LTX-2.3-22b-IC-LoRA-Union-Control`](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control)
  * [`LTX-2.3-22b-IC-LoRA-Motion-Track-Control`](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control)
  * [`LTX-2.3-22b-IC-LoRA-HDR`](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-HDR)
  * [`LTX-2.3-22b-IC-LoRA-LipDub`](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-LipDub)

### Multi-Reference Inference

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

### Original Pipelines

All original LTX-2 pipelines remain available:

* **TI2VidTwoStagesPipeline** — Production-quality text/image-to-video with 2x upsampling (recommended)
* **TI2VidTwoStagesHQPipeline** — Second-order sampler, fewer steps, better quality
* **TI2VidOneStagePipeline** — Single-stage generation for quick prototyping
* **DistilledPipeline** — Fastest inference with 8 predefined sigmas
* **ICLoraPipeline** — Video-to-video and image-to-video transformations
* **KeyframeInterpolationPipeline** — Interpolate between keyframe images
* **A2VidPipelineTwoStage** — Audio-to-video generation
* **RetakePipeline** — Regenerate a specific time region of an existing video
* **HDRICLoraPipeline** — Video-to-video with HDR output
* **LipDubPipeline** — Lip dubbing with speaker identity matching

## 🎯 Multi-Reference Training

### Step 1: Prepare Dataset

Create a CSV with target video, caption, and reference file paths:

```csv
video_path,caption,ref_image_1,ref_image_2,ref_video_1,ref_audio_1
videos/clip1.mp4,"A woman walks through a garden",refs/face.jpg,refs/park.jpg,refs/walk.mp4,refs/birds.wav
```

### Step 2: Preprocess

```bash
python packages/ltx-trainer/scripts/process_multi_ref.py dataset.csv \
    --output-dir /path/to/preprocessed \
    --model-source ./models/LTX-2.3/ltx-2.3-22b-dev.safetensors \
    --text-encoder-source ./models/text_encoder \
    --resolution-buckets 768x768x25 \
    --with-audio
```

### Step 3: Configure & Train

Edit `packages/ltx-trainer/configs/ltx2_multi_ref_lora.yaml`, then:

```bash
cd packages/ltx-trainer
python -m ltx_trainer.train --config configs/ltx2_multi_ref_lora.yaml
```

> 📖 Full training guide: [docs/multi_reference_quick_start.md](docs/multi_reference_quick_start.md) | [中文版](docs/multi_reference_quick_start_zh.md)

## 📝 @mention Syntax

| Mention | Modality | Default Attributes |
|---------|----------|-------------------|
| `@Image1` | Image | identity, appearance |
| `@Video1` | Video | motion, camera |
| `@Audio1` | Audio | audio_rhythm, audio_mood |

**Attribute keywords**: identity, face, appearance, style, motion, camera, scene, rhythm, mood, lip sync

Example:
```
"A dancer @Image1 for identity performs @Video1 for motion in a @Image2 style setting."
```

**Chinese prompts** are supported via the semantic parser (`--use-semantic-parser`):
```
"一个人 @Image1 这个人的长相和身份 走在 @Image2 的公园场景里。参考 @Video1 的运镜方式。"
```

## 🔑 Key Design Decisions

1. **Text-driven semantic routing (Seedance 2.0 approach)** — The model learns to understand "@Image1 for identity" from text encoder output via cross-attention, NOT from hand-crafted attention mask weights
2. **IC-LoRA style token concatenation** — Reference tokens concatenated along sequence dimension with negative temporal positions
3. **Isolated reference groups** — Different reference groups do NOT attend to each other (0 mask), preventing attribute confusion
4. **Full target↔reference cross-attention** — Target tokens attend to each reference group with weight 1.0
5. **LoRA-only training** — Avoids catastrophic forgetting of base model capabilities
6. **Reference dropout** — Random 10% dropout for classifier-free guidance style robustness

## 📦 Packages

This repository is organized as a monorepo with three main packages:

* **[ltx-core](packages/ltx-core/)** — Core model implementation, inference stack, and utilities
* **[ltx-pipelines](packages/ltx-pipelines/)** — High-level pipeline implementations including multi-reference pipeline
* **[ltx-trainer](packages/ltx-trainer/)** — Training and fine-tuning tools with multi-reference strategy

## 📚 Documentation

* **[Multi-Reference Quick Start](docs/multi_reference_quick_start.md)** — Training & inference guide for multi-reference system
* **[Multi-Reference Quick Start (中文)](docs/multi_reference_quick_start_zh.md)** — 中文版快速入门
* **[LTX-Core README](packages/ltx-core/README.md)** — Core model implementation
* **[LTX-Pipelines README](packages/ltx-pipelines/README.md)** — Pipeline implementations
* **[LTX-Trainer README](packages/ltx-trainer/README.md)** — Training documentation

## 🙏 Acknowledgements

- **[Lightricks/LTX-2](https://github.com/Lightricks/LTX-2)** — Base model and original codebase
- **[Seedance 2.0](https://seedance.ai)** — Inspiration for @mention reference binding and text-driven semantic routing
- **[ID-LoRA (arxiv 2603.10256)](https://arxiv.org/abs/2603.10256)** — Reference conditioning via token concatenation and identity guidance
- **[IC-LoRA](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control)** — Image-conditioned LoRA for reference-based video generation

## 📄 License

This project is licensed under the [Apache 2.0 License](LICENSE), same as the original LTX-2 project.
