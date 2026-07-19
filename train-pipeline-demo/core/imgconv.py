# -*- coding: utf-8 -*-
"""
PDF → 压缩图片（JPEG）
使用 pypdfium2（速度快，无需 ghostscript/poppler）
回退方案：PIL + reportlab rasterize
"""

from pathlib import Path


def pdf_to_images_direct(pdf_path: Path, out_dir: Path, prefix: str,
                          dpi: int = 150, quality: int = 75) -> list[Path]:
    """
    将 PDF 每页转为压缩 JPEG，返回生成的图片路径列表。
    DPI=150, quality=75 约等于 ghostscript /screen 压缩效果。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    img_paths = []

    try:
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(str(pdf_path))
        for i, page in enumerate(pdf):
            scale = dpi / 72
            bitmap = page.render(scale=scale, rotation=0)
            pil_img = bitmap.to_pil()
            # 转 RGB（去掉 alpha）
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            out_path = out_dir / f"{prefix}_{i+1:03d}.jpg"
            pil_img.save(str(out_path), "JPEG", quality=quality, optimize=True)
            img_paths.append(out_path)
        return img_paths

    except ImportError:
        pass

    # 回退：用 pdf2image（需要 poppler）
    try:
        from pdf2image import convert_from_path
        from PIL import Image
        pages = convert_from_path(str(pdf_path), dpi=dpi)
        for i, img in enumerate(pages):
            if img.mode != "RGB":
                img = img.convert("RGB")
            out_path = out_dir / f"{prefix}_{i+1:03d}.jpg"
            img.save(str(out_path), "JPEG", quality=quality, optimize=True)
            img_paths.append(out_path)
        return img_paths

    except (ImportError, Exception) as e:
        print(f"  警告：PDF转图片失败（{e}），跳过图片生成")
        return []


# 保留旧接口名兼容
pdf_to_images = pdf_to_images_direct
