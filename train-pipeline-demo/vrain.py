#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vRain Python版 — 中文古籍刻本风格直排电子书制作工具

用法:
    python vrain.py --text 00 --layout 24_black
    python vrain.py --text 00 --layout 24_black --test 3
    python vrain.py --text 00 --layout 24_black --verbose
    python vrain.py --text 00 --layout 24_black --no-pdf   # 只生成图片，不保留PDF

目录约定:
    books/<name>.txt          原始文本
    layouts/<id>.yaml         版式配置（可复用）
    output/<name>/pdf/        输出 PDF
    output/<name>/images/     输出压缩图片（<name>_001.jpg ...）
    output/<name>/paging/     输出 JSONL 分页日志
"""

import argparse
import sys
from pathlib import Path

from core.config  import load_book_layout_config, load_num2zh
from core.typeset import Typesetter
from core.topcmt  import TopcommentInserter
from core.imgconv import pdf_to_images, pdf_to_images_direct

BASE_DIR = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser(description="vRain 古籍直排电子书制作工具")
    parser.add_argument("--text",    required=True, help="文本文件名（不含.txt，对应 books/<name>.txt）")
    parser.add_argument("--layout",  required=True, help="版式ID（对应 layouts/<id>.yaml）")
    parser.add_argument("--test",    type=int, default=0, metavar="N",
                        help="测试模式：仅生成前 N 页")
    parser.add_argument("--verbose", action="store_true", help="详细输出每字坐标/字体")
    parser.add_argument("--no-topcmt", action="store_true", help="跳过眉批插入步骤")
    parser.add_argument("--no-pdf",    action="store_true",
                        help="生成图片后删除中间 PDF（最终只保留图片）")
    args = parser.parse_args()

    text_file   = BASE_DIR / "books" / f"{args.text}.txt"
    layout_file = BASE_DIR / "layouts" / f"{args.layout}.yaml"

    if not text_file.exists():
        sys.exit(f"错误：文本文件不存在 '{text_file}'")
    if not layout_file.exists():
        sys.exit(f"错误：版式配置不存在 '{layout_file}'")

    print("=" * 60)
    print("  vRain Python版 — 兀雨古籍刻本电子书制作工具")
    print("=" * 60)

    # 加载配置（版式 + 从layout读书目参数）
    cfg = load_book_layout_config(layout_file)
    print(f"  文本：{args.text}.txt  版式：{args.layout}")
    print(f"  画布：{cfg['canvas_width']}×{cfg['canvas_height']}，{cfg['col_num']}列×{cfg['row_num']}字")
    if args.test:
        print(f"  ⚡ 测试模式：仅生成前 {args.test} 页")

    # 输出目录
    out_base   = BASE_DIR / "output" / args.text
    pdf_dir    = out_base / "pdf"
    images_dir = out_base / "images"
    paging_dir = out_base / "paging"
    topcmt_dir = out_base / "topcomments"
    for d in [pdf_dir, images_dir, paging_dir, topcmt_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 清空上次产物，避免旧文件混入
    for f in images_dir.glob("*.jpg"): f.unlink()
    for f in pdf_dir.glob("*.pdf"):    f.unlink()
    for f in paging_dir.glob("*.jsonl"): f.unlink()

    # ── 步骤1：排版 → 主PDF ──────────────────────────────────
    ts = Typesetter(
        text_file  = text_file,
        text_name  = args.text,
        cfg        = cfg,
        base_dir   = BASE_DIR,
        out_pdf_dir    = pdf_dir,
        out_paging_dir = paging_dir,
        out_topcmt_dir = topcmt_dir,
        test_pages = args.test,
        verbose    = args.verbose,
    )
    pdf_path = ts.run()
    print(f"\n✓ 主PDF：{pdf_path.name}")

    # ── 步骤2：插入眉批 ──────────────────────────────────────
    final_pdf = pdf_path
    # 版式层面禁用眉批：yaml 里设 no_topcmt: 1 即可跳过
    cfg_no_topcmt = bool(int(cfg.get("no_topcmt", 0)))
    if cfg_no_topcmt:
        print("  （版式配置 no_topcmt=1，跳过眉批插入）")
    if not args.no_topcmt and not cfg_no_topcmt:
        topcmt_log = topcmt_dir / f"{pdf_path.stem}.txt"
        has_topcmt = topcmt_log.exists() and topcmt_log.stat().st_size > 0
        if has_topcmt:
            ti = TopcommentInserter(
                pdf_path    = pdf_path,
                cfg         = cfg,
                base_dir    = BASE_DIR,
                text_name   = args.text,
                paging_dir  = paging_dir,
                topcmt_dir  = topcmt_dir,
                verbose     = args.verbose,
            )
            final_pdf = ti.run()
            print(f"✓ 眉批PDF：{final_pdf.name}")
        else:
            print("  （无眉批，跳过眉批插入）")

    # ── 步骤3：PDF → 压缩图片 ───────────────────────────────
    print(f"\n  转换图片...")
    img_paths = pdf_to_images_direct(
        pdf_path   = final_pdf,
        out_dir    = images_dir,
        prefix     = args.text,
        dpi        = 150,        # 控制图片大小，约等于压缩PDF效果
        quality    = 75,
    )
    print(f"✓ 生成图片：{len(img_paths)} 张 → output/{args.text}/images/")

    # ── 步骤4：paging → JSONL（绑定图片路径）───────────────
    _update_paging_jsonl(paging_dir, images_dir, args.text)
    print(f"✓ JSONL分页日志 → output/{args.text}/paging/")

    # ── 步骤5：按需删除PDF ───────────────────────────────────
    if args.no_pdf:
        for p in pdf_dir.glob("*.pdf"):
            p.unlink()
        pdf_dir.rmdir()
        print("  （已删除中间PDF，仅保留图片）")

    print("\n完成！")
    print(f"  output/{args.text}/")
    print(f"    ├── images/  {len(img_paths)} 张图")
    print(f"    └── paging/  JSONL分页日志")


def _update_paging_jsonl(paging_dir: Path, images_dir: Path, text_name: str):
    """将 paging/*.jsonl 中的 images 路径更新为实际图片路径（已在typeset阶段写入）"""
    # 实际图片路径列表
    imgs = sorted(images_dir.glob(f"{text_name}_*.jpg"))
    img_map = {i: str(p) for i, p in enumerate(imgs)}

    for jl in paging_dir.glob("*.jsonl"):
        import json
        lines = jl.read_text(encoding="utf-8").splitlines()
        updated = []
        for line in lines:
            if not line.strip():
                continue
            obj = json.loads(line)
            # 更新 images 字段为实际路径
            page_idx = obj.get("_page_idx", 0)
            if page_idx in img_map:
                obj["images"] = [img_map[page_idx]]
            obj.pop("_page_idx", None)
            updated.append(json.dumps(obj, ensure_ascii=False))
        jl.write_text("\n".join(updated) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
