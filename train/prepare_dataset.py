"""
SongPanda 训练数据准备脚本
==========================
将 vRain 生成的古籍合成图像 + 标签文本，转换为训练框架所需的 JSONL 格式。

输入结构（示例）：
    synthetic_images/
        0001.png
        0002.png
        ...
    synthetic_labels/
        0001.txt         # 内容：正文 + 【夹注】标记（vRain 源文本格式）
        0002.txt
        ...

输出：
    PaddleFormers 格式（默认）：
        {"image": "xxx.png", "conversations": [{"role": "user", ...}, {"role": "assistant", ...}]}
    LLaMA-Factory 格式（--format llama_factory）：
        {"messages": [...], "images": ["xxx.png"]}

用法：
    python prepare_dataset.py \\
        --images-dir ./synthetic_images \\
        --labels-dir ./synthetic_labels \\
        --output ./data/train.jsonl

作者：zhengningch
许可证：Apache-2.0
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

# ---------- 统一任务 Prompt（训练 / 推理保持一致）----------
DEFAULT_PROMPT = (
    "请对这张古籍图像进行 OCR：\n"
    "- 自动删去版心无关正文的字段；\n"
    "- 识别正文；\n"
    "- 将双行小字夹注以 <footnote></footnote> 标签标出（Mix 集中 <head> 表示眉批、\\n 表示列末换列、\\f 表示换半页）。"
)


def convert_label(text: str) -> str:
    """将 vRain 源文本标记（【】）转换为训练标签（<footnote>）。"""
    text = text.strip()
    # vRain 的双行小字夹注用【】包裹，统一替换为 <footnote>...</footnote>
    text = re.sub(r"【([^【】]*)】", r"<footnote>\1</footnote>", text)
    return text


def iter_samples(images_dir: Path, labels_dir: Path) -> Iterable[tuple[Path, str]]:
    """遍历图像，并匹配同名标签文件。"""
    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            print(f"[WARN] missing label for {img_path.name}, skipped.")
            continue
        with label_path.open("r", encoding="utf-8") as f:
            raw = f.read()
        yield img_path, convert_label(raw)


def to_paddleformers(image: str, label: str, prompt: str) -> dict:
    """PaddleFormers 对话样本格式。"""
    return {
        "image": image,
        "conversations": [
            {"role": "user", "content": f"<image>\n{prompt}"},
            {"role": "assistant", "content": label},
        ],
    }


def to_llama_factory(image: str, label: str, prompt: str) -> dict:
    """LLaMA-Factory 对话样本格式（路线 B 对照实验用）。"""
    return {
        "messages": [
            {"role": "user", "content": f"<image>{prompt}"},
            {"role": "assistant", "content": label},
        ],
        "images": [image],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SongPanda 训练数据准备")
    parser.add_argument("--images-dir", required=True, type=Path, help="合成图像目录")
    parser.add_argument("--labels-dir", required=True, type=Path, help="标签文本目录")
    parser.add_argument("--output", required=True, type=Path, help="输出 JSONL 路径")
    parser.add_argument(
        "--format",
        choices=["paddleformers", "llama_factory"],
        default="paddleformers",
        help="输出格式，默认 paddleformers",
    )
    parser.add_argument(
        "--image-path-mode",
        choices=["basename", "absolute", "relative"],
        default="absolute",
        help="写入 JSONL 的图像路径形式",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="任务 Prompt")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    converter = to_paddleformers if args.format == "paddleformers" else to_llama_factory

    n_ok, n_total = 0, 0
    with args.output.open("w", encoding="utf-8") as fout:
        for img_path, label in iter_samples(args.images_dir, args.labels_dir):
            n_total += 1
            if not label:
                continue
            if args.image_path_mode == "basename":
                image_str = img_path.name
            elif args.image_path_mode == "relative":
                image_str = str(img_path.relative_to(args.images_dir.parent))
            else:
                image_str = str(img_path.resolve())
            sample = converter(image_str, label, args.prompt)
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
            n_ok += 1

    print(f"[DONE] wrote {n_ok}/{n_total} samples → {args.output}")


if __name__ == "__main__":
    main()
