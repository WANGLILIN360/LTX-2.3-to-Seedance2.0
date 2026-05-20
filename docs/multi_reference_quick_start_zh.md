# 多参考源 LoRA 训练 — 快速入门指南

本指南涵盖 LTX-2.3 的统一多模态多参考源条件系统，灵感来源于 Seedance 2.0 的 @mention 参考绑定方法。

## 概述

多参考源系统允许同时对多个图像、视频和音频参考进行条件注入，每个参考带有显式属性标签，控制模型应提取的内容（身份、运动、风格、镜头、音频节奏等）。

### 架构

```
提示词: "男人 @Image1 提取身份 在 @Image2 的公园中行走。
         复刻 @Video1 的镜头运动。@Audio1 提供节奏。"

┌─────────────────────────────────────────────────────────┐
│                   @mention 提示词解析器                    │
│  提取: @Image1→identity, @Image2→scene,                 │
│        @Video1→camera, @Audio1→audio_rhythm              │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│         UnifiedMultiReferenceConditioning                │
│  - 将每个参考 latent patchify                            │
│  - 计算属性感知的注意力掩码                                │
│  - 将参考 token 追加到 LatentState                       │
│  - 参考组之间隔离的交叉注意力                              │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│              LTX-2.3 双流 Transformer                     │
│  视频流 (14B) ←→ 跨模态注意力 ←→ 音频流 (5B)             │
│  LoRA 作用于 to_q/k/v/out 实现属性路由                    │
└─────────────────────────────────────────────────────────┘
```

## 新增文件

| 文件 | 说明 |
|------|------|
| `ltx-core/conditioning/types/multi_reference_cond.py` | 核心条件模块：`UnifiedMultiReferenceConditioning`、`ReferenceItem`、`ReferenceModality`、`ReferenceAttribute` |
| `ltx-pipelines/utils/prompt_parser.py` | @mention 提示词解析器：`parse_prompt()`、`ParsedPrompt`、`ReferenceBinding` |
| `ltx-pipelines/unified_multi_ref.py` | 支持多参考源的两阶段推理管线 |
| `ltx-trainer/training_strategies/multi_reference.py` | 训练策略：`MultiReferenceStrategy`、`MultiReferenceConfig` |
| `ltx-trainer/configs/ltx2_multi_ref_lora.yaml` | 训练配置 YAML |
| `ltx-trainer/scripts/process_multi_ref.py` | 数据预处理脚本 |

## 修改的文件

| 文件 | 变更 |
|------|------|
| `ltx-core/conditioning/__init__.py` | 导出新类型 |
| `ltx-core/conditioning/types/__init__.py` | 导出新类型 |
| `ltx-trainer/training_strategies/__init__.py` | 注册 `multi_reference` 策略 |
| `ltx-trainer/training_strategies/base_strategy.py` | 在 Literal 类型中添加 `"multi_reference"` |
| `ltx-trainer/config.py` | 导入 + 注册 `MultiReferenceConfig` 到判别联合类型 |
| `ltx-trainer/.gitignore` | 允许新配置 YAML |

## 第 1 步：准备数据集

创建包含目标视频、描述和参考文件路径列的 CSV/JSON/JSONL 文件：

```csv
video_path,caption,ref_image_1,ref_image_2,ref_video_1,ref_audio_1
videos/clip1.mp4,"一位女性走过花园",refs/face.jpg,refs/park.jpg,refs/walk.mp4,refs/birds.wav
videos/clip2.mp4,"一位男性在录音室弹吉他",refs/face2.jpg,,refs/guitar.mp4,
```

## 第 2 步：预处理数据

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

输出目录结构：
```
/path/to/preprocessed/
├── latents/              # 目标视频 latent
├── conditions/           # 文本嵌入
├── audio_latents/        # 目标音频 latent
├── ref_image_latents/    # 图像参考 latent
├── ref_video_latents/    # 视频参考 latent
└── ref_audio_latents/    # 音频参考 latent
```

## 第 3 步：配置训练

编辑 `packages/ltx-trainer/configs/ltx2_multi_ref_lora.yaml`：

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

## 第 4 步：训练

```bash
cd packages/ltx-trainer
python -m ltx_trainer.train --config configs/ltx2_multi_ref_lora.yaml
```

## 第 5 步：推理

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

## @mention 语法参考

| 提及 | 模态 | 默认属性 |
|------|------|----------|
| `@Image1` | 图像 | identity, appearance |
| `@Video1` | 视频 | motion, camera |
| `@Audio1` | 音频 | audio_rhythm, audio_mood |

### 属性关键词

解析器识别 @mention 附近的以下关键词：

- **身份 (Identity)**: identity, face, character
- **外观 (Appearance)**: appearance, look, facial
- **风格 (Style)**: style, aesthetic
- **运动 (Motion)**: motion, movement, choreography, dance, action
- **镜头 (Camera)**: camera, dolly, tracking, crane, pan, orbit, zoom
- **场景 (Scene)**: scene, background, environment, setting
- **音频节奏 (Audio Rhythm)**: rhythm, beat, tempo, pacing
- **音频氛围 (Audio Mood)**: mood, atmosphere, music
- **唇形同步 (Lip Sync)**: lip sync, lip-sync, lipsync

显式属性示例：
```
"A dancer @Image1 for identity performs @Video1 for motion in a @Image2 style setting."
```

## 兼容模型

| 模型 | 用途 |
|------|------|
| `ltx-2.3-22b-dev.safetensors` | 训练用基础模型（推荐） |
| `ltx-2.3-22b-distilled-1.1.safetensors` | 更快的推理 |
| `ltx-2.3-22b-distilled-lora-384-1.1.safetensors` | 第 2 阶段精炼 |
| `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | 2x 空间上采样 |
| `LTX-2.3-22b-IC-LoRA-*` | 可与多参考源 LoRA 同时加载 |
| `LTX-2-19b-LoRA-Camera-Control-*` | 兼容的镜头控制 LoRA |

## 关键设计决策

1. **属性感知注意力掩码**：每个参考组根据其属性标签获得独立的注意力权重，使模型能够学习哪个参考影响生成的哪个方面。

2. **隔离参考组**：不同参考组之间**不**互相注意（注意力掩码中为 0），防止属性混淆。

3. **参考 dropout**：训练期间随机丢弃参考（默认 10%），实现分类器自由引导风格的鲁棒性。

4. **仅 LoRA 训练**：多参考源策略要求 LoRA 模式，避免基础模型能力的灾难性遗忘。

5. **顺序条件注入**：参考逐个应用到 `LatentState`，每个在前一个注意力掩码基础上构建，维持现有 LTX-2 条件协议。

---

# 项目实现分析

## 一、模块概览

整个多参考源系统由以下核心模块组成：

### 1. 核心条件模块 (`multi_reference_cond.py`)

- **`ReferenceModality`**: 枚举类型，定义 IMAGE / VIDEO / AUDIO 三种模态
- **`ReferenceAttribute`**: 枚举类型，定义 9 种语义属性（identity, appearance, style, motion, camera, scene, audio_rhythm, audio_mood, lip_sync）
- **`DEFAULT_ATTRIBUTE_WEIGHTS`**: 每种属性的默认注意力权重字典
- **`ReferenceItem`**: 单个参考源的数据容器，包含 latent、模态、属性标签、强度、注意力权重、下采样因子、帧索引
- **`UnifiedMultiReferenceConditioning`**: 核心条件注入类，继承自 `ConditioningItem`，实现 `apply_to()` 方法

### 2. 提示词解析器 (`prompt_parser.py`)

- 使用正则 `@(Image|Video|Audio)(\d+)` 匹配 @mention
- 从上下文中提取属性关键词并映射到 `ReferenceAttribute`
- 输出 `ParsedPrompt`，包含 `ReferenceBinding` 列表

### 3. 推理管线 (`unified_multi_ref.py`)

- `UnifiedMultiRefPipeline`: 两阶段管线（半分辨率生成 → 2x 上采样精炼）
- `_build_multi_ref_conditionings()`: 根据解析结果构建 `ReferenceItem` 列表
- `_resolve_ref_path()`: 将绑定映射到文件路径

### 4. 训练策略 (`multi_reference.py`)

- **`MultiReferenceConfig`**: Pydantic 配置模型，包含参考数量上限、dropout 概率、默认属性等
- **`MultiReferenceStrategy`**: 继承 `TrainingStrategy`，实现 `prepare_training_inputs()` 和 `compute_loss()`
- **`_ReferenceGroup`**: 内部容器，存储一组参考 token、位置、属性

### 5. 数据预处理 (`process_multi_ref.py`)

- `MultiRefDataset`: 加载 CSV/JSON/JSONL 数据集
- 编码目标视频、参考图像/视频/音频为 latent
- 保存到 safetensors 文件

### 6. 注意力掩码工具 (`mask_utils.py`)

- `build_attention_mask()`: 构建块状注意力掩码，实现参考组隔离
- `update_attention_mask()`: 递增式扩展掩码

## 二、存在的问题

### 🔴 严重问题

#### 1. 推理管线中音频参考完全未实现

`@/packages/ltx-pipelines/src/ltx_pipelines/unified_multi_ref.py:362-366`

```python
elif binding.ref_type == ReferenceModality.AUDIO:
    # Audio references are handled via audio_latent_from_file
    # which needs the audio encoder (not available in this callback)
    # They will be added separately if audio encoder is available
    pass
```

`_build_multi_ref_conditionings()` 中的音频参考分支是一个 **空的 `pass`**。注释说"音频编码器在此回调中不可用"，但没有提供任何替代方案。这意味着：
- 推理时 `@Audio1` 语法虽然能被解析，但**永远不会产生实际效果**
- 未绑定音频参考同样被忽略（只处理了未绑定的 image 和 video）
- 这是一个**功能缺失**，与文档和配置中声称的音频参考支持严重不符

#### 2. 训练策略中未使用属性感知注意力权重

`@/packages/ltx-trainer/src/ltx_trainer/training_strategies/multi_reference.py:330-339`

训练策略中构建 `Modality` 时，**没有传递任何注意力掩码信息**：

```python
video_modality = Modality(
    enabled=True,
    latent=combined_latents,
    sigma=sigmas,
    timesteps=timesteps,
    positions=positions,
    context=video_prompt_embeds,
    context_mask=prompt_attention_mask,
)
```

虽然 `_ReferenceGroup` 存储了 `attributes`，但这些属性**从未被用于计算注意力权重或掩码**。所有参考 token 在训练中被同等对待，属性路由机制在训练端完全缺失。这与核心条件模块 `UnifiedMultiReferenceConditioning` 中精心设计的 `effective_attention_weight()` 和 `update_attention_mask()` 逻辑脱节。

#### 3. 训练策略中图像参考只处理单个

`@/packages/ltx-trainer/src/ltx_trainer/training_strategies/multi_reference.py:385-417`

`_process_image_references()` 只处理 `ref_latents.dim() == 5`（即 `[B, C, 1, H, W]`）的情况，且只生成一个 `_ReferenceGroup`。配置中 `max_image_references: 9` 允许最多 9 个图像参考，但代码中**没有任何逻辑处理多个图像参考**。同样的问题也存在于视频参考处理中。

### 🟡 中等问题

#### 4. 提示词解析器的属性提取过于粗糙

`@/packages/ltx-pipelines/src/ltx_pipelines/utils/prompt_parser.py:129-171`

`_extract_attributes_for_mention()` 使用简单的 `keyword in context` 子串匹配：
- **误匹配风险**：`"facial"` 会匹配 `"superficial"`，`"pan"` 会匹配 `"panorama"` 或 `"company"`
- **无词边界检测**：没有使用 `\b` 或分词，导致误报
- **窗口过大**：向前回看 80 个字符，向后看到下一个 @mention，可能导致属性被错误分配给相邻的参考

#### 5. 数据预处理脚本缺少分布式支持

`@/packages/ltx-trainer/scripts/process_multi_ref.py:240-306`

虽然导入了 `PartialState`（来自 accelerate），但处理循环是单进程串行的：
```python
for i in range(len(dataset)):
    sample = dataset[i]
    ...
```
对于大规模数据集，缺少多 GPU / 多进程并行处理能力，预处理效率极低。

#### 6. Stage 2 推理缺少多参考源条件

`@/packages/ltx-pipelines/src/ltx_pipelines/unified_multi_ref.py:258-294`

Stage 2（上采样精炼阶段）只传递了 `stage_2_image_conditionings`（标准图像条件），**没有传递多参考源条件**：

```python
video=ModalitySpec(
    context=v_context_p,
    conditionings=stage_2_image_conditionings,  # 只有标准图像条件
    ...
),
```

这意味着 Stage 2 的精炼过程中，模型**失去了所有参考源信息**（图像参考、视频参考、音频参考），可能导致上采样结果与 Stage 1 的参考约束不一致。

#### 7. 视频参考的帧数硬编码

`@/packages/ltx-pipelines/src/ltx_pipelines/unified_multi_ref.py:343-345`

```python
output_shape = VideoPixelShape(
    batch=1, frames=89, height=height, width=width, fps=25.0,
)
```

帧数 `89` 和 `fps=25.0` 被硬编码，无法根据实际参考视频调整。如果参考视频较短或帧率不同，会导致浪费计算或截断内容。

#### 8. `ReferenceItem.effective_attention_weight()` 使用 `max()` 而非加权聚合

`@/packages/ltx-core/src/ltx_core/conditioning/types/multi_reference_cond.py:100-107`

```python
def effective_attention_weight(self) -> float:
    weights = [DEFAULT_ATTRIBUTE_WEIGHTS.get(tag, 0.8) for tag in self.attribute_tags]
    return max(weights)
```

使用 `max()` 意味着多属性参考的权重完全由最强属性决定，其他属性权重被忽略。例如，一个同时标注 `identity(1.0)` 和 `style(0.8)` 的参考，最终权重为 1.0，`style` 属性的降权意图完全丧失。更合理的做法可能是加权平均或可学习的聚合。

### 🟢 轻微问题

#### 9. 预处理脚本的分辨率桶解析顺序不一致

`@/packages/ltx-trainer/scripts/process_multi_ref.py:184-186`

```python
w, h, f = b.split("x")
buckets.append((int(f), int(h), int(w)))
```

输入格式为 `WxHxF`，但内部存储为 `(frames, height, width)`。这种不一致容易造成混淆和潜在 bug。

#### 10. 训练策略中参考 dropout 的粒度过粗

`@/packages/ltx-trainer/src/ltx_trainer/training_strategies/multi_reference.py:518-539`

`_apply_reference_dropout()` 以**组**为单位进行 dropout，即要么保留整组参考，要么丢弃整组。更精细的做法应该是以**单个参考**为单位进行 dropout，这样模型能学习到部分参考缺失的情况。

#### 11. 训练配置中缺少跨模态注意力 LoRA 目标

`@/packages/ltx-trainer/configs/ltx2_multi_ref_lora.yaml:73-83`

配置注释中提到跨模态注意力模块（`audio_to_video_attn`, `video_to_audio_attn）对属性路由至关重要，但 `target_modules` 列表中**没有包含这些模块**，且被注释掉的 feed-forward 层也未启用。这意味着跨模态属性路由能力可能不足。

#### 12. `_apply_image_at_frame()` 使用 `clone()` 但 `LatentState` 可能不支持

`@/packages/ltx-core/src/ltx_core/conditioning/types/multi_reference_cond.py:310`

```python
state = latent_state.clone()
```

其他方法都使用 `LatentState(...)` 构造新实例（不可变模式），但 `_apply_image_at_frame()` 使用了 `clone()` + 原地修改。如果 `LatentState` 没有实现 `clone()` 方法，此处会运行时出错；即使有，这种不一致的风格也容易引入 bug。

#### 13. 缺少验证和单元测试

整个多参考源系统中没有发现任何单元测试文件。关键逻辑（如提示词解析、注意力掩码构建、属性权重计算）缺少测试覆盖，增加了回归风险。

## 三、问题汇总

| 优先级 | 问题 | 影响 |
|--------|------|------|
| 🔴 严重 | 音频参考推理未实现 | 音频参考功能完全不可用 |
| 🔴 严重 | 训练策略未使用属性感知注意力权重 | 属性路由机制在训练端失效 |
| 🔴 严重 | 图像/视频参考只处理单个 | 多参考源核心功能受限 |
| 🟡 中等 | 提示词解析器属性提取粗糙 | 属性分配可能不准确 |
| 🟡 中等 | 预处理缺少分布式支持 | 大规模数据集效率极低 |
| 🟡 中等 | Stage 2 缺少多参考源条件 | 上采样结果可能与参考约束不一致 |
| 🟡 中等 | 视频参考帧数硬编码 | 灵活性差，可能浪费计算 |
| 🟡 中等 | 注意力权重用 max() 聚合 | 多属性参考的弱属性被忽略 |
| 🟢 轻微 | 分辨率桶解析顺序不一致 | 代码可读性差，易混淆 |
| 🟢 轻微 | 参考 dropout 粒度过粗 | 训练鲁棒性不够精细 |
| 🟢 轻微 | 缺少跨模态 LoRA 目标 | 跨模态属性路由能力不足 |
| 🟢 轻微 | `clone()` 使用不一致 | 潜在运行时错误 |
| 🟢 轻微 | 缺少单元测试 | 回归风险高 |

## 四、建议修复优先级

1. **最高优先级**：实现推理管线中的音频参考编码，或在文档中明确标注音频参考为实验性功能
2. **高优先级**：在训练策略中集成属性感知注意力掩码，使训练与推理的行为一致
3. **高优先级**：支持多个图像/视频参考的处理逻辑
4. **中优先级**：修复提示词解析器的词边界匹配、Stage 2 参考条件传递、帧数参数化
5. **低优先级**：添加分布式预处理、改进 dropout 粒度、添加单元测试
