# -*- coding: utf-8 -*-
"""
图片放置模块：从 image/ 目录随机选取一张图，裁剪/放大适配到目标区域，
返回临时 JPEG 路径，供 typeset.py 用 reportlab 的 drawImage 贴到 PDF。

用法（typeset.py 内部调用）：
    from .imgplace import pick_image_for_zone
    tmp_path = pick_image_for_zone(image_dir, zone_w_px, zone_h_px)
    c.drawImage(tmp_path, x, y, w, h)
    tmp_path.unlink()   # 用完即删
"""

import random
import tempfile
from pathlib import Path
from typing import Optional


# ── 全局图片池（每次进程内缓存，不重复扫描目录）──────────────────
_img_pool: Optional[list[Path]] = None
_img_pool_dir: Optional[Path] = None


def _get_pool(image_dir: Path) -> list[Path]:
    global _img_pool, _img_pool_dir
    if _img_pool is None or _img_pool_dir != image_dir:
        exts = {".jpg", ".jpeg", ".png", ".webp"}
        _img_pool = [p for p in sorted(image_dir.iterdir())
                     if p.suffix.lower() in exts]
        _img_pool_dir = image_dir
    return _img_pool


def pick_image_for_zone(
    image_dir: Path,
    zone_w: int,
    zone_h: int,
    out_dir: Optional[Path] = None,
) -> Optional[Path]:
    """
    从 image_dir 随机抽一张图，裁剪（或放大）后保存为临时 JPEG，
    返回路径（调用方负责用完后 unlink）。
    若无图片则返回 None。

    缩放策略：
      1. 若原图比目标区域小（任一维度），先等比放大到刚好能覆盖区域。
      2. 从放大后的图中随机裁剪 zone_w × zone_h 的子区域。
    """
    try:
        from PIL import Image
    except ImportError:
        return None

    pool = _get_pool(image_dir)
    if not pool:
        return None

    src_path = random.choice(pool)
    try:
        img = Image.open(src_path).convert("RGB")
    except Exception:
        return None

    iw, ih = img.size
    tw, th = int(zone_w), int(zone_h)

    # ── 放大到能覆盖目标区域 ─────────────────────────────────────
    scale = max(tw / iw, th / ih, 1.0)     # 至少 1.0（不缩小）
    if scale > 1.0:
        nw = max(int(iw * scale + 0.5), tw)
        nh = max(int(ih * scale + 0.5), th)
        img = img.resize((nw, nh), Image.LANCZOS)
        iw, ih = nw, nh

    # ── 随机裁剪 ─────────────────────────────────────────────────
    max_x = max(0, iw - tw)
    max_y = max(0, ih - th)
    left  = random.randint(0, max_x)
    top   = random.randint(0, max_y)
    crop  = img.crop((left, top, left + tw, top + th))

    # ── 保存到临时文件 ────────────────────────────────────────────
    tmp_dir = out_dir or Path(tempfile.gettempdir())
    tmp_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_str = tempfile.mkstemp(suffix=".jpg", dir=str(tmp_dir))
    import os; os.close(fd)
    tmp_path = Path(tmp_str)
    crop.save(str(tmp_path), "JPEG", quality=85, optimize=True)
    return tmp_path
