#!/usr/bin/env python3
"""Download LTX-2.3 models via ModelScope (China mirror)."""

import os
from pathlib import Path

# ========== 配置 ==========
DOWNLOAD_DIR = Path("./models")
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ModelScope 上的仓库映射
REPOS = {
    # 主模型仓库
    "LTX-2.3": "Lightricks/LTX-2.3",
    # 2.3 专用 LoRA
    "LTX-2.3-22b-IC-LoRA-Union-Control": "Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control",
    "LTX-2.3-22b-IC-LoRA-Motion-Track-Control": "Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control",
    "LTX-2.3-22b-IC-LoRA-HDR": "Lightricks/LTX-2.3-22b-IC-LoRA-HDR",
    "LTX-2.3-22b-IC-LoRA-LipDub": "Lightricks/LTX-2.3-22b-IC-LoRA-LipDub",
    # 文本编码器 (魔搭上可能叫 gemma-3-it，如果没有需手动处理)
    # "text_encoder": "google/gemma-3-12b-it",
}

# 主仓库中需要下载的特定文件（如果只想下部分）
MAIN_FILES = [
    "ltx-2.3-22b-dev.safetensors",
    "ltx-2.3-22b-distilled-1.1.safetensors",
    "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
    "ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors",
    "ltx-2.3-temporal-upscaler-x2-1.0.safetensors",
    "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
]


def ensure_modelscope():
    """确保 modelscope 已安装。"""
    try:
        import modelscope
        print(f"modelscope 已安装: {modelscope.__version__}")
    except ImportError:
        print("modelscope 未安装，正在安装...")
        os.system("pip install modelscope -U")
        print("安装完成，请重新运行本脚本。")
        exit(0)


def download_repo(repo_id: str, local_dir: Path, allow_patterns=None):
    """用 modelscope 下载仓库。"""
    from modelscope import snapshot_download
    print(f"  -> 下载 {repo_id} ...")
    try:
        snapshot_download(
            repo_id,
            local_dir=str(local_dir),
            allow_patterns=allow_patterns,
        )
        print(f"     [OK] {repo_id}")
    except Exception as e:
        print(f"     [FAIL] {repo_id}: {e}")


def download_single_file(repo_id: str, filename: str, local_dir: Path):
    """用 modelscope 下载单个文件。"""
    from modelscope.hub.file_download import model_file_download
    print(f"  -> {filename}")
    try:
        model_file_download(
            model_id=repo_id,
            file_path=filename,
            local_dir=str(local_dir),
        )
        print(f"     [OK] {filename}")
    except Exception as e:
        print(f"     [FAIL] {filename}: {e}")


def main():
    ensure_modelscope()

    print("=" * 60)
    print("LTX-2.3 模型下载脚本 (ModelScope 镜像)")
    print("=" * 60)
    print(f"下载目录: {DOWNLOAD_DIR.resolve()}")
    print()

    # 1. 主仓库文件
    print("[1/3] 下载主模型文件...")
    main_dir = DOWNLOAD_DIR / "LTX-2.3"
    main_dir.mkdir(exist_ok=True)
    for filename in MAIN_FILES:
        download_single_file(REPOS["LTX-2.3"], filename, main_dir)
    print()

    # 2. LoRA 仓库
    print("[2/3] 下载 LTX-2.3 专用 IC-LoRA...")
    lora_dir = DOWNLOAD_DIR / "loras"
    lora_dir.mkdir(exist_ok=True)
    for name, repo_id in REPOS.items():
        if "LoRA" in name:
            target = lora_dir / name
            target.mkdir(exist_ok=True)
            download_repo(repo_id, target)
    print()

    # 3. 文本编码器（可选，魔搭上可能没有 exact 匹配）
    print("[3/3] 文本编码器提示...")
    print("     Gemma-3 文本编码器在国内镜像上可能没有 exact 版本。")
    print("     请尝试用 git 直接克隆或手动下载：")
    print("     https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized")
    print()

    print("=" * 60)
    print("下载完成！文件保存在:")
    print(f"  {DOWNLOAD_DIR.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
