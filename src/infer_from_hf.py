#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
infer_from_hf.py — 从 Hugging Face Hub 拉取 SongPanda2.3 权重并推理单张图片

依赖:
    pip install torch transformers>=4.57,<4.58 huggingface_hub Pillow accelerate einops sentencepiece protobuf

用法:
    # 最简 (从 HF 拉 ningzhuo/SongPanda2.3, prompt="OCR:【】<>")
    python infer_from_hf.py --image sample.jpg

    # 指定设备
    python infer_from_hf.py --image sample.jpg --device cpu
    python infer_from_hf.py --image sample.jpg --device cuda

    # 拉私有仓库 (HF 私有 / gated 都需要 token)
    HF_TOKEN=hf_xxx python infer_from_hf.py --image sample.jpg

    # 直接用本地权重 (跳过下载)
    python infer_from_hf.py --image sample.jpg --model /path/to/local_dir
"""
from __future__ import annotations

import argparse
import functools
import os
import re
import sys
import time

# ---------- transformers 4.57 兼容 shim ----------
# 模型代码调用 create_causal_mask(inputs_embeds=...)，而 transformers 4.57.x 是 input_embeds
try:
    import transformers.masking_utils as _mu
    _orig_ccm = _mu.create_causal_mask

    @functools.wraps(_orig_ccm)
    def _patched_ccm(*args, **kwargs):
        if "inputs_embeds" in kwargs and "input_embeds" not in kwargs:
            kwargs["input_embeds"] = kwargs.pop("inputs_embeds")
        return _orig_ccm(*args, **kwargs)

    _mu.create_causal_mask = _patched_ccm
except Exception:
    pass

import torch
from PIL import Image

DEFAULT_MODEL = "ningzhuo/SongPanda2.3"     # Hugging Face Hub 模型 ID
DEFAULT_PROMPT = "OCR:【】<>。、"              


def parse_args():
    p = argparse.ArgumentParser(description="SongPanda2.3 (HF) 单图推理 demo")
    p.add_argument("--image", required=True, help="图片路径")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="HF Hub 模型 ID 或本地目录 (默认 %(default)s)")
    p.add_argument("--prompt", default=DEFAULT_PROMPT,
                   help="推理 prompt (默认: %(default)s)")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                   help="设备 (默认 auto, 自动检测)")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--cache-dir", default=None,
                   help="HF 缓存目录 (默认 ~/.cache/huggingface)")
    p.add_argument("--no-post-process", action="store_true",
                   help="跳过标签→可读符号的替换")
    return p.parse_args()


def resolve_device(device: str) -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        if not torch.cuda.is_available():
            print("[!] 指定了 cuda 但不可用, 回退到 CPU")
            return torch.device("cpu")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def download_model(model_id: str, cache_dir: str | None) -> str:
    """HF Hub ID → 本地路径; 本地目录直接返回."""
    if os.path.isdir(model_id):
        return os.path.abspath(model_id)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    print(f"[*] 从 Hugging Face 拉取模型: {model_id} ...")
    from huggingface_hub import snapshot_download
    kwargs = dict(
        repo_id=model_id,
        repo_type="model",
        cache_dir=cache_dir,
        allow_patterns=[
            "*.json", "*.py", "*.jinja", "*.model", "*.md",
            "*.safetensors", "*.txt", "tokenizer*", "*.tiktoken",
        ],
    )
    # 私有 / gated 仓库需要 token
    if token:
        kwargs["token"] = token
    local_dir = snapshot_download(**kwargs)
    print(f"[+] 下载完成: {local_dir}")
    return local_dir


def post_process(text: str) -> str:
    """
    将模型标签转成可读符号:
      <footnote>...</footnote>  →  【...】
      <head>...</head>          →  <...>
    单独未配对的标签也会被替换。
    """
    text = text.replace("\\n", "\n").replace("\\f", "\n")
    text = re.sub(r"<footnote>(.*?)</footnote>", r"【\1】", text, flags=re.DOTALL)
    text = re.sub(r"<head>(.*?)</head>", r"<\1>", text, flags=re.DOTALL)
    text = re.sub(r"<footnote>", "【", text)
    text = re.sub(r"</footnote>", "】", text)
    text = re.sub(r"<head>", "<", text)
    text = re.sub(r"</head>", ">", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.rstrip("\n")


def main():
    args = parse_args()
    device = resolve_device(args.device)
    use_gpu = device.type == "cuda"
    dtype = torch.bfloat16 if use_gpu else torch.float32

    print(f"[*] 设备: {device}  dtype: {dtype}")
    print(f"[*] 模型: {args.model}")
    print(f"[*] prompt: {args.prompt!r}")

    # 1. 下载 / 定位模型
    model_path = download_model(args.model, args.cache_dir)

    # 2. 加载模型与 processor
    from transformers import AutoModelForCausalLM, AutoProcessor
    print("[*] 加载模型 ...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=dtype,
        attn_implementation="eager",   # CPU 必须 eager; GPU 也兼容
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    print(f"[+] 模型加载完成, 耗时 {time.time()-t0:.1f}s")

    # 3. 推理
    image = Image.open(args.image).convert("RGB")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": args.prompt},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 对齐 vLLM 行为: 同时把 eos (2) 和 <|end_of_sentence|> (100272) 作为停止 token
    eos_ids = list({processor.tokenizer.eos_token_id, 100272})

    t0 = time.time()
    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            num_beams=1,
            use_cache=True,
            eos_token_id=eos_ids,
            pad_token_id=processor.tokenizer.pad_token_id or eos_ids[0],
        )
    gen_time = time.time() - t0
    new_tokens = out_ids.shape[1] - inputs["input_ids"].shape[1]
    raw_text = processor.decode(out_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # 4. 输出
    print("=" * 60)
    print(f"[图片] {args.image}")
    print("-" * 60)
    print("[识别结果 (原始)]")
    print(raw_text)
    if not args.no_post_process:
        print("-" * 60)
        print("[识别结果 (后处理: <footnote>→【】, <head>→<>)]")
        print(post_process(raw_text))
    print("=" * 60)
    print(f"tokens={new_tokens}  time={gen_time:.1f}s  "
          f"speed={new_tokens/gen_time:.2f} tok/s  device={device}")


if __name__ == "__main__":
    main()
