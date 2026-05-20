#!/usr/bin/env python3
"""Download all official LTX-2.3 models from HuggingFace."""

import os
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

# ========== 镜像配置 ==========
# 国内用户可设置环境变量使用镜像：
#   $env:HF_ENDPOINT="https://hf-mirror.com"  (PowerShell)
#   export HF_ENDPOINT=https://hf-mirror.com    (Linux/Mac)
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")

# ========== 配置 ==========
# 模型下载根目录
DOWNLOAD_DIR = Path("./models")

# 主仓库
MAIN_REPO = "Lightricks/LTX-2.3"

# 2.3 版本专用 LoRA 仓库（独立仓库）
LORA_REPOS_23 = {
    "LTX-2.3-22b-IC-LoRA-Union-Control": "Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control",
    "LTX-2.3-22b-IC-LoRA-Motion-Track-Control": "Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control",
    "LTX-2.3-22b-IC-LoRA-HDR": "Lightricks/LTX-2.3-22b-IC-LoRA-HDR",
    "LTX-2.3-22b-IC-LoRA-LipDub": "Lightricks/LTX-2.3-22b-IC-LoRA-LipDub",
}

# 文本编码器
TEXT_ENCODER_REPO = "google/gemma-3-12b-it-qat-q4_0-unquantized"

# ========== 主仓库文件列表 ==========
MAIN_FILES = [
    # 主模型（二选一即可，这里都下载让用户自己选）
    "ltx-2.3-22b-dev.safetensors",
    "ltx-2.3-22b-distilled-1.1.safetensors",
    # 空间上采样器
    "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
    "ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors",
    # 时间上采样器
    "ltx-2.3-temporal-upscaler-x2-1.0.safetensors",
    # Distilled LoRA
    "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
]


def download_file(repo_id: str, filename: str, local_dir: Path, subfolder: str = ""):
    """下载单个文件。"""
    print(f"  -> {filename}")
    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            subfolder=subfolder,
            local_dir=str(local_dir),
            endpoint=HF_ENDPOINT,
        )
        print(f"     [OK] {filename}")
    except Exception as e:
        print(f"     [FAIL] {filename}: {e}")


def download_repo_snapshot(repo_id: str, local_dir: Path, ignore_patterns=None):
    """下载整个仓库快照。"""
    print(f"  -> {repo_id}")
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            endpoint=HF_ENDPOINT,
            ignore_patterns=ignore_patterns or [],
        )
        print(f"     [OK] {repo_id}")
    except Exception as e:
        print(f"     [FAIL] {repo_id}: {e}")


def main():
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LTX-2.3 模型下载脚本")
    print("=" * 60)
    print(f"下载目录: {DOWNLOAD_DIR.resolve()}")
    print(f"端点: {HF_ENDPOINT}")
    if HF_ENDPOINT != "https://huggingface.co":
        print("  (正在使用镜像站点)")
    print()

    # 1. 下载主仓库文件
    print("[1/4] 下载主模型、上采样器、LoRA...")
    main_dir = DOWNLOAD_DIR / "LTX-2.3"
    main_dir.mkdir(exist_ok=True)
    for filename in MAIN_FILES:
        download_file(MAIN_REPO, filename, main_dir)
    print()

    # 2. 下载 2.3 专用 LoRA
    print("[2/4] 下载 LTX-2.3 专用 IC-LoRA...")
    lora_dir = DOWNLOAD_DIR / "loras"
    lora_dir.mkdir(exist_ok=True)
    for name, repo_id in LORA_REPOS_23.items():
        print(f"  -> {name} ({repo_id})")
        target_dir = lora_dir / name
        target_dir.mkdir(exist_ok=True)
        download_repo_snapshot(repo_id, target_dir, ignore_patterns=["*.md", "*.txt"])
    print()

    # 3. 下载文本编码器
    print("[3/4] 下载 Gemma-3 文本编码器...")
    encoder_dir = DOWNLOAD_DIR / "text_encoder"
    encoder_dir.mkdir(exist_ok=True)
    download_repo_snapshot(TEXT_ENCODER_REPO, encoder_dir, ignore_patterns=["*.md", "*.txt"])
    print()

    print("=" * 60)
    print("下载完成！文件保存在:")
    print(f"  {DOWNLOAD_DIR.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
