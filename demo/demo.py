"""
SongPanda 快速推理示例
======================
对一张古籍图像进行 OCR，输出带 <footnote> 标签的纯文本。

使用方式：

    python demo/demo.py --image ./example.jpg

    # 或指定自定义模型 / 输出文件：
    python demo/demo.py \\
        --image ./my_ancient_book.jpg \\
        --model-path ningzhuo/SongPanda2.0 \\
        --output ./result.txt

环境：
    pip install paddlepaddle-gpu paddleformers pillow

作者：zhengningch
许可证：Apache-2.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


DEFAULT_MODEL = "zhengningch/SongPanda2.0"
DEFAULT_PROMPT = (
    "OCR:<footnote><head>"
)


def run_inference(image_path: Path, model_path: str, prompt: str, max_new_tokens: int) -> str:
    """基于 PaddleFormers 加载 SongPanda 并对单张图像推理。"""
    try:
        import paddle
        from PIL import Image
        from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
        from paddleformers.generation import GenerationConfig
    except ImportError as e:
        sys.exit(
            "[ERROR] 缺少依赖，请先安装：\n"
            "    pip install paddlepaddle-gpu paddleformers pillow\n"
            f"详细错误：{e}"
        )

    print(f"[INFO] Loading model from: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForConditionalGeneration.from_pretrained(
        model_path,
        convert_from_hf=True,
    ).eval()
    model.config._attn_implementation = "flashmask"
    model.visual.config._attn_implementation = "flashmask"

    image = Image.open(image_path).convert("RGB")
    print(f"[INFO] Image loaded: {image_path} ({image.size[0]}×{image.size[1]})")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pd"
    )

    print(f"[INFO] Generating (max_new_tokens={max_new_tokens}) ...")
    gen_config = GenerationConfig(
        do_sample=False,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        use_cache=True,
    )

    with paddle.no_grad():
        output_ids = model.generate(
            **inputs,
            generation_config=gen_config,
            max_new_tokens=max_new_tokens,
        )
    
    text = processor.decode(output_ids[0].tolist()[0], skip_special_tokens=True)
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="SongPanda 古籍 OCR 快速推理示例")
    parser.add_argument("--image", required=True, type=Path, help="输入古籍图像路径")
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL,
        help=f"模型路径或 HuggingFace ID（默认：{DEFAULT_MODEL}）",
    )
    parser.add_argument("--output", type=Path, default=None, help="输出文件（默认打印到 stdout）")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="最大生成 token 数")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="任务 Prompt")
    args = parser.parse_args()

    if not args.image.exists():
        sys.exit(f"[ERROR] 图像不存在：{args.image}")

    result = run_inference(args.image, args.model_path, args.prompt, args.max_new_tokens)

    print("\n========== OCR 结果 ==========")
    print(result)
    print("================================")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result, encoding="utf-8")
        print(f"[INFO] 已保存到：{args.output}")


if __name__ == "__main__":
    main()
