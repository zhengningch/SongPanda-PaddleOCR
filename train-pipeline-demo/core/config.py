# -*- coding: utf-8 -*-
"""配置加载：只需一个 layout yaml，书目参数也放在 layout 里"""

import yaml
from pathlib import Path


def load_book_layout_config(layout_path: Path) -> dict:
    with open(layout_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    W   = int(cfg["canvas_width"])
    H   = int(cfg["canvas_height"])
    ml  = int(cfg.get("margins_left", 50))
    mr  = int(cfg.get("margins_right", 50))
    mt  = int(cfg.get("margins_top", 200))
    mb  = int(cfg.get("margins_bottom", 50))

    # ── 解析列框标注（col_positions + col_widths）────────────────
    # 格式：col_positions: "x1,x2,x3,..."  col_widths: "w1,w2,w3,..."
    # x 是每列左边界（原图坐标），从右到左排（即 x 从大到小）
    raw_pos = cfg.get("col_positions", "")
    raw_wid = cfg.get("col_widths", "")
    if raw_pos:
        xs = [float(v) for v in str(raw_pos).split(",") if v.strip()]
        ws = [float(v) for v in str(raw_wid).split(",") if v.strip()] if raw_wid else []
        # 按 x 从大到小排（右→左，与排版顺序一致）
        if ws and len(ws) == len(xs):
            cols = sorted(zip(xs, ws), key=lambda t: -t[0])
        else:
            # 没有 col_widths 时，用相邻列左边界差作为列宽
            xs_sorted = sorted(xs, reverse=True)
            cols = []
            for k, x in enumerate(xs_sorted):
                if k + 1 < len(xs_sorted):
                    w = x - xs_sorted[k+1]
                else:
                    w = xs_sorted[0] - xs_sorted[-1]  # fallback
                cols.append((x, w))
        # 多栏模式下合并相近列框（上下栏分别标注同一列时，x 坐标仅差数像素）
        rps_check = int(cfg.get("rows_per_section", 0))
        if rps_check > 0 and len(cols) > 1:
            merged = [cols[0]]
            for x, w in cols[1:]:
                prev_x, prev_w = merged[-1]
                if abs(prev_x - x) < 20:  # 同列的上下栏标注，合并取平均
                    merged[-1] = ((prev_x + x) / 2, (prev_w + w) / 2)
                else:
                    merged.append((x, w))
            cols = merged
        cfg["_col_boxes"] = cols          # [(x_left, width), ...]  右→左
        cfg["_cw"]        = cols[0][1]    # 第一列宽（供兼容代码用）
        cn = len(cols)
        cfg["col_num"]    = cn
        # 版心区域：标注里 tool=center 的框，存为 leaf_x/leaf_w
        # 此处不强制要求，排版时有就用
    else:
        # 无标注 → 均分（兼容旧逻辑）
        lc = int(cfg.get("leaf_center_width", 0))
        cn = int(cfg["col_num"])
        cfg["_cw"]       = (W - ml - mr - lc) / cn
        cfg["_col_boxes"] = None

    # ── 解析行线标注（row_positions）────────────────────────────
    # 格式：row_positions: "y1,y2,y3,..."  y 从小到大（上→下，原图坐标）
    # 多栏模式：行线作为栏分界线，每栏内部由 rows_per_section 指定行数
    raw_row = cfg.get("row_positions", "")
    if raw_row:
        ys = sorted([float(v) for v in str(raw_row).split(",") if v.strip()])
        cfg["_row_ys"] = ys               # 栏分界线 y 坐标列表（原图坐标，上→下）
        rps = int(cfg.get("rows_per_section", 0))
        if rps > 0:
            # 多栏模式：N 条栏线 → N+1 栏，每栏 rows_per_section 行
            n_sections = len(ys) + 1
            cfg["row_num"] = n_sections * rps
            cfg["_rows_per_section"] = rps
            cfg["_n_sections"] = n_sections
        else:
            # 兼容旧逻辑：N 条行线 → N+1 行（行线即行间分界）
            cfg["row_num"] = len(ys) + 1
            cfg["_rows_per_section"] = 0
            cfg["_n_sections"] = 1
    else:
        cfg["_row_ys"] = None
        cfg["_rows_per_section"] = 0
        cfg["_n_sections"] = 1

    return cfg


def load_num2zh(db_path: Path) -> dict:
    result = {}
    with open(db_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "|" in line:
                k, v = line.split("|", 1)
                result[k] = v
    return result
