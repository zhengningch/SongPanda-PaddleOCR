# -*- coding: utf-8 -*-
"""
排版核心：无封面版本
- 输入：books/<name>.txt
- 输出：output/<name>/pdf/<name>.pdf
         output/<name>/paging/<name>.jsonl   （JSONL分页日志，\\h->\\n）
         output/<name>/topcomments/<name>.txt（内部眉批位置日志）
"""

import re
import json
from pathlib import Path
from datetime import datetime

from reportlab.pdfgen import canvas as rl_canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont as RLTTFont

from .fontcheck import get_font_for_char
from .config import load_num2zh
from .imgplace import pick_image_for_zone


# ── 特殊标记常量 ───────────────────────────────────────────
TAG_SPACE     = "@"
TAG_NEWLEAF   = "%"
TAG_HALFPAGE  = "$"
TAG_LASTCOL   = "&"
TAG_BOOKLEFT  = "《"
TAG_BOOKRIGHT = "》"
TAG_CMT_L     = "【"
TAG_CMT_R     = "】"
TAG_TC_L      = "＜"   # 全角小于
TAG_TC_R      = "＞"   # 全角大于


class Typesetter:
    def __init__(self, text_file, text_name, cfg, base_dir,
                 out_pdf_dir, out_paging_dir, out_topcmt_dir,
                 test_pages=0, verbose=False):
        self.text_file     = Path(text_file)
        self.text_name     = text_name
        self.cfg           = cfg
        self.base_dir      = Path(base_dir)
        self.out_pdf_dir   = Path(out_pdf_dir)
        self.out_paging    = Path(out_paging_dir)
        self.out_topcmt    = Path(out_topcmt_dir)
        self.test_pages    = test_pages
        self.verbose       = verbose

        self.fonts_dir  = self.base_dir / "fonts"
        self.canvas_dir = self.base_dir / "canvas"
        self.db_dir     = self.base_dir / "db"
        self.zhnums     = load_num2zh(self.db_dir / "num2zh_jid.txt")

        c = cfg
        self.W         = int(c["canvas_width"])
        self.H         = int(c["canvas_height"])
        self.mt        = int(c["margins_top"])
        self.mb        = int(c["margins_bottom"])
        self.ml        = int(c["margins_left"])
        self.mr        = int(c["margins_right"])
        self.col_num   = int(c["col_num"])
        self.lc_width  = int(c["leaf_center_width"])
        self.row_num   = int(c["row_num"])
        self.row_delta_y = float(c.get("row_delta_y", 0))
        self.cw        = c["_cw"]
        self.rh        = (self.H - self.mt - self.mb) / self.row_num
        self.page_chars = self.col_num * self.row_num

        # 多栏模式：每列实际行数 = rows_per_section，用于文本换行对齐
        _rps = int(c.get("_rows_per_section", 0))
        self.col_rows  = _rps if _rps > 0 else self.row_num

        # 字体
        self.font_names = [c.get(f"font{i}") for i in range(1,6) if c.get(f"font{i}")]
        self.font_paths = [str(self.fonts_dir / fn) for fn in self.font_names]
        self.font_sizes_text = [float(c.get(f"text_font{i}_size", 40)) for i in range(1,6)]
        self.font_sizes_cmt  = [float(c.get(f"comment_font{i}_size", 30)) for i in range(1,6)]
        self.font_rotate     = [float(c.get(f"font{i}_rotate", 0)) for i in range(1,6)]

        tfa = str(c.get("text_fonts_array", "1"))
        cfa = str(c.get("comment_fonts_array", "12"))
        self.text_fps = [self.font_paths[int(i)-1] for i in tfa
                         if i.isdigit() and int(i) <= len(self.font_paths)]
        self.cmt_fps  = [self.font_paths[int(i)-1] for i in cfa
                         if i.isdigit() and int(i) <= len(self.font_paths)]

        self.text_color       = c.get("text_font_color", "black")
        self.cmt_color        = c.get("comment_font_color", "black")
        self.onlyperiod_color = c.get("onlyperiod_color", "")

        self.if_tpcenter      = int(c.get("if_tpcenter", 1))
        self.title_font_size  = float(c.get("title_font_size", 80))
        self.title_font_color = c.get("title_font_color", "black")
        self.title_y          = float(c.get("title_y", 1200))
        self.title_ydis       = float(c.get("title_ydis", 1.2))
        self.title_directory  = int(c.get("title_directory", 0))
        self.pager_font_size  = float(c.get("pager_font_size", 35))
        self.pager_font_color = c.get("pager_font_color", "black")
        self.pager_y          = float(c.get("pager_y", 500))

        # 版心标题从配置取（若无则用文本名）
        self.spine_title   = c.get("spine_title", text_name)
        self.title_postfix = c.get("title_postfix", "")

        # 标点规则
        self.exp_replace_comma  = c.get("exp_replace_comma", "")
        self.exp_replace_number = c.get("exp_replace_number", "")
        self.exp_delete_comma   = c.get("exp_delete_comma", "")
        self.if_nocomma         = int(c.get("if_nocomma", 0))
        self.exp_nocomma        = c.get("exp_nocomma", "")
        self.if_onlyperiod      = int(c.get("if_onlyperiod", 0))
        self.exp_onlyperiod     = c.get("exp_onlyperiod", "")

        self.text_comma_nop      = c.get("text_comma_nop", "")
        self.text_comma_nop_size = float(c.get("text_comma_nop_size", 1.2))
        self.text_comma_nop_x    = float(c.get("text_comma_nop_x", 0.5))
        self.text_comma_nop_y    = float(c.get("text_comma_nop_y", 0.2))
        self.text_comma_90       = c.get("text_comma_90", "")
        self.text_comma_90_size  = float(c.get("text_comma_90_size", 0.8))
        self.text_comma_90_x     = float(c.get("text_comma_90_x", 0.35))
        self.text_comma_90_y     = float(c.get("text_comma_90_y", 0.6))

        self.cmt_comma_nop       = c.get("comment_comma_nop", "")
        self.cmt_comma_nop_size  = float(c.get("comment_comma_nop_size", 0.7))
        self.cmt_comma_nop_x     = float(c.get("comment_comma_nop_x", 0.8))
        self.cmt_comma_nop_y     = float(c.get("comment_comma_nop_y", 0.1))
        self.cmt_comma_90        = c.get("comment_comma_90", "")
        self.cmt_comma_90_size   = float(c.get("comment_comma_90_size", 0.8))
        self.cmt_comma_90_x      = float(c.get("comment_comma_90_x", 0.15))
        self.cmt_comma_90_y      = float(c.get("comment_comma_90_y", 0.5))

        self.if_book_vline = int(c.get("if_book_vline", 0))
        self.bline_w       = float(c.get("book_line_width", 1))
        self.bline_c       = c.get("book_line_color", "black")

        # ── 预计算字符位置数组（1-based）─────────────────────────
        col_boxes = c.get("_col_boxes")   # [(x_left, width), ...] 右→左，或 None
        row_ys    = c.get("_row_ys")      # [y1, y2, ...] 原图坐标上→下，或 None
        rps       = int(c.get("_rows_per_section", 0))  # 多栏：每栏行数
        n_sec     = int(c.get("_n_sections", 1))         # 栏数

        if col_boxes:
            # ── 按标注列框计算 ──────────────────────────────────
            # col_boxes 已按 x 从大到小排（右→左）
            if row_ys and len(row_ys) >= 1 and rps > 0:
                # ── 多栏模式：行线作为栏分界线 ──────────────────
                # 构建每栏的上下边界（原图坐标）
                section_bounds = []  # [(y_top, y_bottom), ...] 原图坐标
                boundaries = [self.mt] + list(row_ys) + [self.H - self.mb]
                for k in range(n_sec):
                    section_bounds.append((boundaries[k], boundaries[k+1]))
                # 为每栏内均分行高，生成每行的 y 坐标（reportlab 坐标系）
                self.row_ys_rl = None  # 多栏模式不使用旧的 row_ys_rl
                section_row_ys = []  # 所有行的 reportlab y 坐标和行高
                for sec_top, sec_bot in section_bounds:
                    sec_h = sec_bot - sec_top
                    sec_rh = sec_h / rps
                    for j in range(1, rps + 1):
                        # 原图坐标 y = sec_top + sec_rh * j
                        # reportlab y = H - (原图 y)，字符底边
                        img_y = sec_top + sec_rh * j
                        rl_y = self.H - img_y + self.row_delta_y
                        section_row_ys.append((rl_y, sec_rh))
                # 用第一栏的行高作为全局 rh（供其他地方兼容使用）
                self.rh = section_row_ys[0][1] if section_row_ys else (self.H - self.mt - self.mb) / self.row_num
            elif row_ys and len(row_ys) >= 1:
                # ── 旧逻辑：行线作为行间分界（不变）──────────────
                row_tops = [self.H - y for y in row_ys]
                row_tops_sorted = sorted(row_tops, reverse=True)
                self.row_ys_rl = row_tops_sorted
                self.rh = (row_tops_sorted[0] - row_tops_sorted[-1]) / max(len(row_tops_sorted)-1, 1)
                section_row_ys = None
            else:
                # 无行线：行高 = 正文区 / row_num（不变）
                body_h = self.H - self.mt - self.mb
                self.rh = body_h / self.row_num
                self.row_ys_rl = None
                section_row_ys = None

            self.pos_l  = [None]
            self.pos_r  = [None]
            self.pos_cw = [None]   # 每个位置对应的列宽
            if section_row_ys and rps > 0:
                # ── 多栏模式：按半页+栏遍历 ────────────────────────
                # 排版顺序（古籍右→左阅读）：
                #   右半页上栏 → 右半页下栏 → 左半页上栏 → 左半页下栏
                # col_boxes 按 x 从大到小排（右→左），版心在中间
                half = len(col_boxes) // 2
                right_cols = col_boxes[:half]   # 右半页列（x 较大）
                left_cols  = col_boxes[half:]   # 左半页列（x 较小）
                for half_cols in [right_cols, left_cols]:
                    for sec_idx in range(n_sec):
                        row_offset = sec_idx * rps
                        for col_x, col_w in half_cols:
                            for j in range(rps):
                                py, _ = section_row_ys[row_offset + j]
                                self.pos_l.append((col_x, py))
                                self.pos_r.append((col_x + col_w / 2, py))
                                self.pos_cw.append(col_w)
            else:
                # ── 非多栏：原有逻辑（逐列逐行，不变）──────────────
                for col_x, col_w in col_boxes:
                    px = col_x
                    cw_this = col_w
                    for j in range(1, self.row_num + 1):
                        if self.row_ys_rl and j - 1 < len(self.row_ys_rl):
                            py = self.row_ys_rl[j-1] - self.rh + self.row_delta_y
                        else:
                            py = self.H - self.mt - self.rh * j + self.row_delta_y
                        self.pos_l.append((px, py))
                        self.pos_r.append((px + cw_this / 2, py))
                        self.pos_cw.append(cw_this)
            # 保存列框供批注宽度计算用
            self._col_boxes  = col_boxes
            self.cw          = col_boxes[0][1]  # 兼容用第一列宽
            self.page_chars  = self.col_num * self.row_num  # 用最终确定的 row_num 重算
        else:
            # ── 无标注：均分（原有逻辑）───────────────────────────
            self.row_ys_rl  = None
            self._col_boxes = None
            self.pos_l  = [None]
            self.pos_r  = [None]
            self.pos_cw = [None]
            for i in range(1, self.col_num + 1):
                if i <= self.col_num // 2:
                    px = self.W - self.mr - self.cw * i
                else:
                    px = self.W - self.mr - self.cw * i - self.lc_width
                for j in range(1, self.row_num + 1):
                    py = self.H - self.mt - self.rh * j + self.row_delta_y
                    self.pos_l.append((px, py))
                    self.pos_r.append((px + self.cw / 2, py))
                    self.pos_cw.append(self.cw)

        # 向 reportlab 注册字体
        self._reg: dict[str, str] = {}
        for fp, fn in zip(self.font_paths, self.font_names):
            rln = re.sub(r'[^A-Za-z0-9]', '_', fn)
            try:
                pdfmetrics.registerFont(RLTTFont(rln, fp))
                self._reg[fp] = rln
            except Exception as e:
                print(f"  警告：注册字体失败 {fn}: {e}")

        # 预计算标点集合
        self._nop_text = set(ch for p in self.text_comma_nop.split("|") for ch in p if ch)
        self._90_text  = set(ch for p in self.text_comma_90.split("|")  for ch in p if ch)
        self._nop_cmt  = set(ch for p in self.cmt_comma_nop.split("|")  for ch in p if ch)
        self._90_cmt   = set(ch for p in self.cmt_comma_90.split("|")   for ch in p if ch)
        # ● U+25CF 实心圆、◎ U+25CE 双圆圈：不占位叠加注记，缩小为 1/5 字号渲染
        # 注：○ U+25CB、△ U+25B3 是正文占位字符，不加入 nop
        # 注：、（U+3001）是普通顿号，占位正常渲染，不加入 nop
        _FIXED_NOP = {"●", "◎"}
        self._nop_text |= _FIXED_NOP
        self._nop_cmt  |= _FIXED_NOP
        # - (U+002D)：不占位，渲染为字格右侧竖旁线（不用字符，用 _line 绘制）
        self._dash_text = {"-"}
        self._dash_cmt  = {"-"}

        # ── 图文排版：解析 image_zones（原图像素坐标，reportlab 坐标系） ──
        # yaml 字段格式：image_zones: "x,y,w,h;x,y,w,h;..."  （原图像素坐标）
        # reportlab 坐标原点在左下角；原图坐标原点在左上角
        # 转换：rl_y = canvas_H - img_y - img_h
        raw_iz = str(c.get("image_zones", "")).strip()
        self._image_zones = []   # [(rl_x, rl_y, w, h), ...]  reportlab 坐标
        if raw_iz:
            for seg in raw_iz.split(";"):
                seg = seg.strip()
                if not seg:
                    continue
                parts = [v.strip() for v in seg.split(",")]
                if len(parts) == 4:
                    try:
                        ix, iy, iw, ih = [float(v) for v in parts]
                        # 原图坐标 → reportlab 坐标（原点在左下）
                        rl_x = ix
                        rl_y = self.H - iy - ih
                        self._image_zones.append((rl_x, rl_y, iw, ih))
                    except ValueError:
                        pass
        self._image_dir = self.base_dir / "image"

        # 版式级禁用眉批：no_topcmt=1 时 JSONL 中也去除 ＜眉批＞ 内容
        self._no_topcmt = bool(int(c.get("no_topcmt", 0)))

    # ── 工具方法 ──────────────────────────────────────────────

    def _rl(self, fp): return self._reg.get(fp, "Helvetica")

    def _font(self, char, fps): return get_font_for_char(char, fps)

    def _fsize(self, fp, mode):
        idx = self.font_paths.index(fp) if fp in self.font_paths else 0
        arr = self.font_sizes_text if mode == "text" else self.font_sizes_cmt
        return arr[idx] if idx < len(arr) else arr[0]

    def _color(self, s):
        if not s or not isinstance(s, str):
            return (0, 0, 0)
        s = s.strip().lower()
        if not s:
            return (0, 0, 0)
        named = {"black":(0,0,0),"white":(1,1,1),"red":(1,0,0),
                 "blue":(0,0,1),"gray":(0.5,0.5,0.5)}
        if s in named: return named[s]
        m = re.match(r'rgb\((\d+),(\d+),(\d+)\)', s)
        if m: return tuple(int(x)/255 for x in m.groups())
        m = re.match(r'#([0-9a-f]{6})', s)
        if m: h=m.group(1); return (int(h[0:2],16)/255,int(h[2:4],16)/255,int(h[4:6],16)/255)
        return (0,0,0)

    def _draw(self, c, x, y, fp, size, char, color="black", rotate=0):
        fn = self._rl(fp)
        r,g,b = self._color(color)
        c.saveState()
        c.setFillColorRGB(r,g,b)
        c.setFont(fn, size)
        if rotate:
            c.translate(x + size/2, y + size/2)
            c.rotate(rotate)
            c.drawString(-size/2, -size/2, char)
        else:
            c.drawString(x, y, char)
        c.restoreState()

    def _line(self, c, x1,y1,x2,y2, w, color):
        r,g,b = self._color(color)
        c.saveState()
        c.setStrokeColorRGB(r,g,b)
        c.setLineWidth(w)
        c.line(x1,y1,x2,y2)
        c.restoreState()

    # ── 文本预处理 ────────────────────────────────────────────

    def _preprocess(self, line: str) -> str:
        if self.exp_replace_comma:
            for kv in self.exp_replace_comma.split("|"):
                if len(kv) >= 2:
                    line = line.replace(kv[0], kv[1])
        if self.exp_replace_number:
            for kv in self.exp_replace_number.split("|"):
                if len(kv) >= 2:
                    line = line.replace(kv[0], kv[1])
        if self.exp_delete_comma:
            for ch in self.exp_delete_comma.split("|"):
                if ch: line = line.replace(ch, "")
        if self.if_nocomma:
            for ch in self.exp_nocomma.split("|"):
                if ch: line = line.replace(ch, "")
        if self.if_onlyperiod:
            for ch in self.exp_onlyperiod.split("|"):
                if ch: line = line.replace(ch, "。")
            line = re.sub("。+", "。", line)
            line = re.sub("^。", "", line)
        line = line.replace(TAG_SPACE, " ")
        return line

    def _cmt_width(self, txt: str) -> int:
        tmp = txt
        for ch in self._nop_cmt: tmp = tmp.replace(ch, "")
        if self.if_book_vline: tmp = tmp.replace("《","").replace("》","")
        n = len(tmp)
        return n // 2 + (1 if n % 2 else 0)

    def _measure_line(self, s: str) -> int:
        """
        深度感知计算一行占用的主文列数，用于行对齐补空格。
        规则与排版渲染保持一致：
          - 顶层 ＜...＞ 眉批：移出正文流，不计宽度
          - 【...】 注释：深度感知截取（含嵌套），按小字双列计宽
          - 游离 】：忽略
          - 不占位标点（_nop）与（开启竖线时的）《》：不计宽度
        """
        main = 0
        i, n = 0, len(s)
        while i < n:
            ch = s[i]
            # 顶层眉批 ＜...＞
            if ch == TAG_TC_L:
                j = s.find(TAG_TC_R, i + 1)
                i = (j + 1) if j >= 0 else n
                continue
            # 注释 【...】，深度感知
            if ch == TAG_CMT_L:
                depth = 1
                i += 1
                buf = []
                while i < n and depth > 0:
                    c2 = s[i]
                    if c2 == TAG_CMT_L:
                        depth += 1
                    elif c2 == TAG_CMT_R:
                        depth -= 1
                        if depth == 0:
                            i += 1
                            break
                    buf.append(c2)
                    i += 1
                main += self._cmt_width("".join(buf))
                continue
            # 游离右括号
            if ch == TAG_CMT_R:
                i += 1
                continue
            # 不占位标点
            if ch in self._nop_text or ch in self._nop_cmt:
                i += 1
                continue
            # 竖线模式下的书名号不占位
            if self.if_book_vline and ch in ("《", "》"):
                i += 1
                continue
            main += 1
            i += 1
        return main

    def _flatten_nested(self, text: str) -> str:
        """
        规整括号层次（在排版与 JSONL 生成前统一处理）：
          - 保留最外层 【】 与 ＜＞，并保证闭合
          - 内层套娃的括号符号一律删除，内容保留
            例：同【一口】【又】采木【一【戀心十】】  →  同一口又采木一戀心十
                說文從乙＜又＞聲                    →  說文從乙又聲
          - 丢弃无法匹配的游离右括号（如 道門定制 的多余 】）
        注：换行符原样保留，供 _load_text 逐行补空格；跨行注释状态连续。
        """
        out = []
        container = None   # None / '【' / '＜'：当前所在的最外层容器
        depth = 0          # 同型括号的嵌套深度
        for ch in text:
            if container is None:
                if ch == TAG_CMT_L:
                    container, depth = TAG_CMT_L, 1
                    out.append(ch)
                elif ch == TAG_TC_L:
                    container, depth = TAG_TC_L, 1
                    out.append(ch)
                elif ch in (TAG_CMT_R, TAG_TC_R):
                    pass   # 游离右括号，丢弃
                else:
                    out.append(ch)
            elif container == TAG_CMT_L:
                if ch == TAG_CMT_L:
                    depth += 1          # 内层【，删符号
                elif ch == TAG_CMT_R:
                    depth -= 1
                    if depth == 0:
                        out.append(ch)
                        container = None
                    # 内层】删符号
                elif ch in (TAG_TC_L, TAG_TC_R):
                    pass                # 【】内的＜＞符号删除，内容保留
                else:
                    out.append(ch)
            else:  # container == TAG_TC_L
                if ch == TAG_TC_R:
                    out.append(ch)
                    container = None
                elif ch in (TAG_TC_L, TAG_CMT_L, TAG_CMT_R):
                    pass                # ＜＞内的套娃符号删除
                else:
                    out.append(ch)
        if container is not None:       # 未闭合，补右括号
            out.append(TAG_CMT_R if container == TAG_CMT_L else TAG_TC_R)
        return "".join(out)

    def _load_text(self) -> str:
        full = self.text_file.read_text(encoding="utf-8")
        full = self._flatten_nested(full)   # 先展平嵌套/游离括号（跨行有状态）
        lines = []
        for raw in full.split("\n"):
            raw = re.sub(r'\s', '', raw)
            if not raw: continue
            processed = self._preprocess(raw)
            tmpstr = processed

            # 计算补空格（行对齐）——深度感知
            body_len = self._measure_line(processed)
            spaces = (self.col_rows - body_len % self.col_rows) % self.col_rows
            lines.append(tmpstr + " " * spaces)
        return "".join(lines)

    # ── 版心标题/页码 ─────────────────────────────────────────

    def _draw_spine(self, c, tpchars):
        for i, ch in enumerate(tpchars):
            fp = self._font(ch, self.text_fps) or self.text_fps[0]
            fs = self.title_font_size
            fx = self.W/2 - fs/2 if self.if_tpcenter else -fs/2
            fy = self.title_y - fs * i * self.title_ydis
            self._draw(c, fx, fy, fp, fs, ch, self.title_font_color)

    def _draw_pager(self, c, pid):
        zh = self.zhnums.get(str(pid), str(pid))
        for i, ch in enumerate(zh):
            fs = self.pager_font_size
            px = self.W/2 - fs/2 if self.if_tpcenter else -fs/2
            py = self.pager_y - fs * i * self.title_ydis
            self._draw(c, px, py, self.text_fps[0], fs, ch, self.pager_font_color)

    # ── paging JSONL ─────────────────────────────────────────

    def _pagetxt_to_content(self, pagetxt: str) -> str:
        """
        将 paging 原始文本转为 JSONL content，与 PDF 内容完全对应：
          - 【批注】＜眉批＞ 原样保留（标签+内容）
          - no_topcmt=1 时 ＜眉批＞ 连同标签一起删除
          - \\h  → \\n（换列）
          - $   → \\f（换半页）
          - %   → 去掉（换叶控制符，不可见）
          - &   → 去掉（跳末列控制符，不可见）
        """
        s = pagetxt
        # no_topcmt 模式：删除 ＜...＞ 眉批（含标签和内容）
        if self._no_topcmt:
            s = re.sub(r'＜[^＞]*＞', '', s)
        # 只转换排版控制符，不动任何可见内容标签
        s = s.replace(r'\h', '\n')   # 换列 → 换行
        s = s.replace('$', '\f')      # 换半页
        s = s.replace('%', '')        # 换叶符（不可见）
        s = s.replace('&', '')        # 跳末列符（不可见）
        # 清理行尾多余空白
        lines = [l.rstrip() for l in s.split('\n')]
        s = '\n'.join(lines)
        s = re.sub(r'\n{3,}', '\n\n', s)
        s = s.strip()
        return s

    # ── 主排版 ────────────────────────────────────────────────

    def run(self) -> Path:
        dat = self._load_text()
        chars = list(dat)

        pdf_name  = self.text_name
        if self.test_pages:
            pdf_name += "_test"
        pdf_path  = self.out_pdf_dir / f"{pdf_name}.pdf"
        jsonl_path = self.out_paging / f"{pdf_name}.jsonl"
        topcmt_path = self.out_topcmt / f"{pdf_name}.txt"

        bg_path = self.canvas_dir / f"{self.cfg['canvas_id']}.jpg"
        if not bg_path.exists():
            raise FileNotFoundError(f"背景图不存在: {bg_path}")

        c = rl_canvas.Canvas(str(pdf_path), pagesize=(self.W, self.H))

        # 版心标题字符
        tpchars = list(self.spine_title)
        if self.title_postfix:
            tpchars = list(self.spine_title + self.title_postfix.replace("X", ""))

        topcmt_fh = open(topcmt_path, "w", encoding="utf-8")
        jsonl_lines = []   # 每页一条 JSONL

        pid   = 0
        pcnt  = 0
        rchars = []
        flag_tbook = False
        flag_rbook = False
        pagetxt = ""       # 当前页 paging 原始文本

        def _draw_image_zones():
            """在当前页面上贴图片（每个 image_zone 随机选一张图）"""
            if not self._image_zones:
                return
            if not self._image_dir.exists():
                return
            for (rl_x, rl_y, iw, ih) in self._image_zones:
                tmp = pick_image_for_zone(self._image_dir, int(iw), int(ih))
                if tmp is None:
                    continue
                try:
                    c.drawImage(str(tmp), rl_x, rl_y, iw, ih,
                                preserveAspectRatio=False, anchor="sw")
                except Exception as e:
                    print(f"  警告：贴图失败 zone=({rl_x:.0f},{rl_y:.0f},{iw:.0f},{ih:.0f}): {e}")
                finally:
                    try:
                        tmp.unlink()
                    except Exception:
                        pass

        def new_page():
            nonlocal pid, pcnt, pagetxt
            # 保存当前页 paging
            _save_paging()
            pid  += 1
            pcnt  = 0
            pagetxt = ""
            c.drawImage(str(bg_path), 0, 0, self.W, self.H)
            self._draw_spine(c, tpchars)
            _draw_image_zones()

        def _save_paging():
            nonlocal pagetxt
            content = self._pagetxt_to_content(pagetxt)
            obj = {
                "messages": [
                    {"role": "user",      "content": "<image>\nOCR:【】＜＞。、"},
                    {"role": "assistant", "content": content},
                ],
                "images": [],        # 后续由 main 填充实际路径
                "_page_idx": pid,    # 临时字段，供 main 匹配图片
            }
            jsonl_lines.append(json.dumps(obj, ensure_ascii=False))

        # 第一页
        print(f"    创建页[{pid}]...")
        c.drawImage(str(bg_path), 0, 0, self.W, self.H)
        self._draw_spine(c, tpchars)
        _draw_image_zones()

        while True:
            # ── 换页检查 ─────────────────────────────────────
            if pcnt == self.page_chars or (not chars and not rchars):
                self._draw_pager(c, pid)
                if not chars and not rchars:
                    _save_paging()
                    break
                c.showPage()
                if self.test_pages and pid + 1 >= self.test_pages:
                    _save_paging()
                    break
                print(f"    创建页[{pid+1}]...")
                new_page()

            # ── 批注处理 ─────────────────────────────────────
            if rchars:
                rctmp = "".join(rchars)
                for ch in self._nop_cmt: rctmp = rctmp.replace(ch, "")
                if self.if_book_vline: rctmp = rctmp.replace("《","").replace("》","")
                rn = len(rctmp)
                cnt = rn // 2 + (1 if rn % 2 else 0)

                pcol = int(pcnt / self.col_rows) + 1
                end  = min(int(pcnt) + cnt, pcol * self.col_rows)
                _max_idx = len(self.pos_r) - 1
                r_pos = (
                    [self.pos_r[min(i, _max_idx)] for i in range(int(pcnt)+1, end+1)] +
                    [self.pos_l[min(i, _max_idx)] for i in range(int(pcnt)+1, end+1)]
                )

                rlast = None
                while rchars:
                    rc = rchars.pop(0)
                    pagetxt += rc
                    if not rchars: pagetxt += TAG_CMT_R

                    if rc == TAG_BOOKLEFT:
                        flag_rbook = True
                        if self.if_book_vline: continue
                    elif rc == TAG_BOOKRIGHT:
                        flag_rbook = False
                        if self.if_book_vline: continue

                    fp = self._font(rc, self.cmt_fps)
                    if fp is None:
                        # 注释中所有字体都无此字形：跳过该字，JSONL 同步去除
                        # （注意 rc 为最后一字时其后可能已追加 】，需保留 】）
                        if pagetxt.endswith(TAG_CMT_R) and \
                           pagetxt[:-len(TAG_CMT_R)].endswith(rc):
                            pagetxt = pagetxt[:-len(TAG_CMT_R)-len(rc)] + TAG_CMT_R
                        elif pagetxt.endswith(rc):
                            pagetxt = pagetxt[:-len(rc)]
                        continue
                    fsize = self._fsize(fp, "comment")
                    fdeg  = self.font_rotate[self.font_paths.index(fp)] if fp in self.font_paths else 0
                    fcolor = self.cmt_color

                    # - 在批注中也渲染为竖旁线（字格右侧），不占位
                    if rc in self._dash_cmt:
                        if rlast:
                            rfx, rfy = rlast
                            _cw_r = self.cw / 2  # 批注字格宽约为列宽一半
                            _rlx = rfx + _cw_r - 1
                            self._line(c, _rlx, rfy, _rlx, rfy + self.rh * 0.9, 1.5, fcolor)
                        continue  # 不走 _draw

                    if rc in self._nop_cmt:
                        if rlast:
                            fx, fy = rlast
                            fsize *= self.cmt_comma_nop_size
                            fx += self.cw/2 * self.cmt_comma_nop_x
                            fy -= self.rh  * self.cmt_comma_nop_y
                            fy = max(fy, self.mb + 10)
                        else:
                            continue
                    else:
                        if not r_pos:
                            rchars.insert(0, rc)
                            # 撤销本次预加的 】（rc 为最后一字时已追加）与 rc，
                            # 留到下一页重新处理，避免出现 “也】也】” 重复
                            if pagetxt.endswith(TAG_CMT_R):
                                pagetxt = pagetxt[:-len(TAG_CMT_R)]
                            if pagetxt.endswith(rc):
                                pagetxt = pagetxt[:-len(rc)]
                            break
                        pos = r_pos.pop(0)
                        fx, fy = pos
                        rlast = (fx, fy)
                        fx += (self.cw - fsize*2) / 4
                        fy += (self.rh - fsize) / 4
                        if rc in self._90_cmt:
                            fdeg   = -90
                            fsize *= self.cmt_comma_90_size
                            fx    += self.cw/2 * self.cmt_comma_90_x
                            fy    += self.rh   * self.cmt_comma_90_y
                        pcnt += 0.5

                    if self.if_onlyperiod and rc == "。" and self.onlyperiod_color:
                        fcolor = self.onlyperiod_color
                    if self.verbose:
                        print(f"      [{pid}/{pcnt:.1f}] {rc} -> {Path(fp).name}")
                    self._draw(c, fx, fy, fp, fsize, rc, fcolor, fdeg)

                    # ── 批注一字两标点：当前批注普通字渲染后，若下一个也是 nop 则叠加 ──
                    if rchars and rchars[0] in self._nop_cmt and rc not in self._nop_cmt and rc not in self._90_cmt:
                        rnxt2 = rchars[0]
                        rchars.pop(0)
                        ov_fp = self._font(rnxt2, self.cmt_fps)   # 按标点自身选字体
                        if ov_fp is not None and rlast:
                            pagetxt += rnxt2
                            rfx2, rfy2 = rlast
                            rfx2 += self.cw/2 * self.cmt_comma_nop_x
                            rfy2 -= self.rh   * self.cmt_comma_nop_y
                            rfy2  = max(rfy2, self.mb + 10)
                            fs_ov2 = self._fsize(ov_fp, "comment") * self.cmt_comma_nop_size
                            self._draw(c, rfx2, rfy2, ov_fp, fs_ov2, rnxt2, fcolor, 0)
                            if self.verbose:
                                print(f"      [{pid}/{pcnt:.1f}+叠] {rnxt2} (批注叠加标点)")
                        # ov_fp 为空（所有字体都无字形）：丢弃该叠加标点，JSONL 同步跳过

                    if self.if_book_vline and flag_rbook:
                        ply = min(fy + self.rh*0.7, self.H - self.mt - 5)
                        self._line(c, fx-1, fy-self.rh*0.3, fx-1, ply, self.bline_w, self.bline_c)

                    if rc not in self._nop_cmt:
                        ipc = int(pcnt)
                        if ipc == self.page_chars // 2: pagetxt += "$"
                        elif ipc > 0 and ipc % self.col_rows == 0: pagetxt += r"\h"

                if rchars: continue
                pcnt = int(pcnt + 0.5)
                if pcnt == self.page_chars: continue
                continue

            # ── 正文处理 ─────────────────────────────────────
            if not chars: continue

            char = chars.pop(0)

            # 眉批
            if char == TAG_TC_L:
                tcdat = []
                while chars and chars[0] != TAG_TC_R:
                    tcdat.append(chars.pop(0))
                if chars: chars.pop(0)
                tc_text = "".join(tcdat)
                topcmt_fh.write(f"{pid}|{int(pcnt)}|{tc_text}\n")
                pagetxt += TAG_TC_L + tc_text + TAG_TC_R
                continue

            pagetxt += char

            if char == TAG_NEWLEAF:
                for _ in range(self.col_rows - 1):
                    if chars: chars.pop(0)
                pcnt = self.page_chars
                continue

            if char == TAG_HALFPAGE:
                for _ in range(self.col_rows - 1):
                    if chars: chars.pop(0)
                if pcnt == 0 or pcnt == self.page_chars // 2: continue
                pcnt = self.page_chars // 2 if pcnt < self.page_chars // 2 else self.page_chars
                continue

            if char == TAG_LASTCOL:
                for _ in range(self.col_rows - 1):
                    if chars: chars.pop(0)
                if pcnt <= self.page_chars - self.col_rows + 1:
                    pcnt = self.page_chars - self.col_rows
                continue

            if char == TAG_BOOKLEFT:
                flag_tbook = True
                if self.if_book_vline: continue
            if char == TAG_BOOKRIGHT:
                flag_tbook = False
                if self.if_book_vline: continue

            if char == TAG_CMT_R:
                # 游离右括号（数据错误，无匹配左括号）：跳过不渲染
                if pagetxt.endswith(char):
                    pagetxt = pagetxt[:-len(char)]
                continue

            if char == TAG_CMT_L:
                # 深度感知截取注释内容，正确处理嵌套【】
                # （如 一切經音義 的 同【一口】【又】...采木【一【戀心十】】）
                rdat = []
                depth = 1
                while chars and depth > 0:
                    c2 = chars[0]
                    if c2 == TAG_CMT_L:
                        depth += 1
                    elif c2 == TAG_CMT_R:
                        depth -= 1
                        if depth == 0:
                            chars.pop(0)
                            break
                    rdat.append(chars.pop(0))
                rchars = rdat
                continue

            # 普通字符
            if pcnt < self.page_chars:
                pcnt += 1
            if pcnt <= self.page_chars:
                fp = self._font(char, self.text_fps)
                if fp is None:
                    # 所有字体都无此字形：跳过该字，JSONL 同步去除，且不占字格
                    if pagetxt.endswith(char):
                        pagetxt = pagetxt[:-len(char)]
                    pcnt -= 1   # 撤销前面的 pcnt += 1
                    continue
                fsize  = self._fsize(fp, "text")
                fdeg   = self.font_rotate[self.font_paths.index(fp)] if fp in self.font_paths else 0
                fcolor = self.text_color
                _safe_idx = min(pcnt, len(self.pos_l) - 1)
                fx, fy = self.pos_l[_safe_idx]
                _cw = self.pos_cw[_safe_idx] or self.cw   # 当前列宽

                if char in self._dash_text:
                    # - 渲染为上一字格右侧竖旁线（着重线），不占位
                    # pcnt 已经 +1，上一字的格号 = pcnt-2（+1之后），即 pos_l[pcnt-1] 之前那格
                    _prev_idx = min(max(pcnt - 2, 1), len(self.pos_l) - 1)
                    if _prev_idx >= 1:
                        lx, ly = self.pos_l[_prev_idx]
                        _cw_prev = self.pos_cw[_prev_idx] or self.cw
                        # 竖线画在字格右边界内侧 2px，线宽 2px，覆盖整字高
                        _lx = lx + _cw_prev - 2
                        self._line(c, _lx, ly, _lx, ly + self.rh * 0.95, 2, fcolor)
                    pcnt -= 1
                    if char not in self._nop_text:
                        if pcnt == self.page_chars // 2: pagetxt += "$"
                        elif pcnt > 0 and pcnt % self.col_rows == 0: pagetxt += r"\h"
                    continue  # 不走下面的 _draw

                if char in self._nop_text:
                    # ●◎ 缩为 1/5 字号，其余 nop 用 nop_size 缩放
                    if char in {"●", "◎"}:
                        fsize = self._fsize(fp, "text") / 5
                    else:
                        fsize *= self.text_comma_nop_size
                    if pcnt > 1:
                        _prev_idx = min(pcnt-1, len(self.pos_l) - 1)
                        lx, ly = self.pos_l[_prev_idx]
                        _cw_prev = self.pos_cw[_prev_idx] or self.cw
                        if char in {"●", "◎"}:
                            # 定位到上一字格右上角
                            fx = lx + _cw_prev - fsize - 1
                            fy = ly + self.rh - fsize - 1
                        else:
                            fx = lx + _cw_prev * self.text_comma_nop_x
                            fy = ly - self.rh * self.text_comma_nop_y
                            fy = max(fy, self.mb + 10)
                    pcnt -= 1
                elif char in self._90_text:
                    fsize *= self.text_comma_90_size
                    fx    += _cw * self.text_comma_90_x
                    fy    += self.rh * self.text_comma_90_y
                    fdeg   = -90
                else:
                    fx += (_cw - fsize) / 2

                if self.if_onlyperiod and char == "。" and self.onlyperiod_color:
                    fcolor = self.onlyperiod_color
                if self.verbose:
                    print(f"    [{pid}/{pcnt}] {char} -> {Path(fp).name}")

                self._draw(c, fx, fy, fp, fsize, char, fcolor, fdeg)

                # ── 一字两标点：渲染完当前字后，若下一字也是 nop 标点则叠加渲染 ──
                if chars and chars[0] in self._nop_text:
                    nxt2 = chars[0]
                    # 仅在当前字是普通字符（非 nop/90/dash）时叠加，避免多个 nop 连续叠加
                    if char not in self._nop_text and char not in self._90_text and char not in self._dash_text:
                        chars.pop(0)
                        ov_fp = self._font(nxt2, self.text_fps)   # 按标点自身选字体
                        if ov_fp is not None:
                            pagetxt += nxt2
                            _ov_idx = min(pcnt, len(self.pos_l) - 1)
                            ox, oy = self.pos_l[_ov_idx]
                            _cw_ov = self.pos_cw[_ov_idx] or self.cw
                            if nxt2 in {"●", "◎"}:
                                fs_ov = self._fsize(ov_fp, "text") / 5
                                ox_ov = ox + _cw_ov - fs_ov - 1
                                oy_ov = oy + self.rh - fs_ov - 1
                            else:
                                fs_ov = self._fsize(ov_fp, "text") * self.text_comma_nop_size
                                ox_ov = ox + _cw_ov * self.text_comma_nop_x
                                oy_ov = oy - self.rh * self.text_comma_nop_y
                                oy_ov = max(oy_ov, self.mb + 10)
                            fc_ov = self.onlyperiod_color if (self.if_onlyperiod and nxt2 == "。" and self.onlyperiod_color) else fcolor
                            self._draw(c, ox_ov, oy_ov, ov_fp, fs_ov, nxt2, fc_ov, 0)
                            if self.verbose:
                                print(f"    [{pid}/{pcnt}+叠] {nxt2} (叠加标点)")
                        # ov_fp 为空：丢弃该叠加标点，JSONL 同步跳过

                if self.if_book_vline and flag_tbook:
                    ply = min(fy + self.rh*0.7, self.H - self.mt - 5)
                    self._line(c, fx-2, fy-self.rh*0.3, fx-2, ply, self.bline_w, self.bline_c)

                if char not in self._nop_text:
                    if pcnt == self.page_chars // 2: pagetxt += "$"
                    elif pcnt > 0 and pcnt % self.col_rows == 0: pagetxt += r"\h"

                # 叶尾超前处理不占位标点
                if pcnt == self.page_chars and chars:
                    nxt = chars[0]
                    if nxt in self._nop_text:
                        chars.pop(0)
                        tail_fp = self._font(nxt, self.text_fps)   # 按标点自身选字体
                        if tail_fp is not None:
                            _tail_idx = min(pcnt, len(self.pos_l) - 1)
                            fx2, fy2 = self.pos_l[_tail_idx]
                            _cw_tail = self.pos_cw[_tail_idx] or self.cw
                            fs2 = self._fsize(tail_fp, "text") * self.text_comma_nop_size
                            fx2 += _cw_tail * self.text_comma_nop_x
                            fy2 -= self.rh * self.text_comma_nop_y
                            fy2 = max(fy2, self.mb + 10)
                            self._draw(c, fx2, fy2, tail_fp, fs2, nxt, fcolor, fdeg)

        topcmt_fh.close()
        c.save()

        # 写 JSONL
        jsonl_path.write_text("\n".join(jsonl_lines) + "\n", encoding="utf-8")
        print(f"  → PDF: {pdf_path.name}  ({pid} 页)")
        print(f"  → JSONL: {jsonl_path.name}  ({len(jsonl_lines)} 条)")

        return pdf_path
