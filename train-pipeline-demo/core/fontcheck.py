# -*- coding: utf-8 -*-
"""
字体检查模块：检测某字符在指定字体中是否有字形
使用 fonttools 替代 Font::FreeType，无需系统依赖
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional
from fontTools.ttLib import TTFont


# 全局字体缓存，避免重复打开
_font_cache: dict[str, TTFont] = {}


def _get_ttfont(font_path: str) -> TTFont:
    if font_path not in _font_cache:
        _font_cache[font_path] = TTFont(font_path, lazy=True)
    return _font_cache[font_path]


def font_has_char(font_path: str, char: str) -> bool:
    """检查字符是否在字体中有字形"""
    try:
        tt = _get_ttfont(font_path)
        cmap = tt.getBestCmap()
        if cmap is None:
            return False
        return ord(char) in cmap
    except Exception:
        return False


def get_font_for_char(char: str, font_paths: list) -> Optional[str]:
    """按 fallback 顺序找第一个支持该字符的字体路径，找不到返回 None"""
    for fp in font_paths:
        if font_has_char(fp, char):
            return fp
    return None
