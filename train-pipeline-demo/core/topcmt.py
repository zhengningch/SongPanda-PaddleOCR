# -*- coding: utf-8 -*-
"""眉批插入模块（适配新目录结构）"""

import re
import io
import json
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont as RLTTFont

from .fontcheck import get_font_for_char


class TopcommentInserter:
    FS_MIN_ABS = 15
    LINE_GAP   = 1.3
    TC_MT      = 30
    TC_MB      = 30
    GR         = 0.618

    def __init__(self, pdf_path, cfg, base_dir, text_name,
                 paging_dir, topcmt_dir, verbose=False):
        self.pdf_path   = Path(pdf_path)
        self.cfg        = cfg
        self.base_dir   = Path(base_dir)
        self.text_name  = text_name
        self.paging_dir = Path(paging_dir)
        self.topcmt_dir = Path(topcmt_dir)
        self.verbose    = verbose

        self.fonts_dir  = self.base_dir / "fonts"
        c = cfg
        self.row_num  = int(c["row_num"])
        self.W        = int(c["canvas_width"])
        self.H        = int(c["canvas_height"])
        self.mt       = int(c["margins_top"])
        self.ml       = int(c["margins_left"])
        self.mr       = int(c["margins_right"])
        self.lc       = int(c["leaf_center_width"])
        self.col_num  = int(c["col_num"])
        self.olw      = float(c.get("outline_width", 10))
        self.olvm     = float(c.get("outline_vmargin", 5))

        # ── 从 cfg 读眉批字号和颜色（可在 yaml 中配置）──────────
        self.FS_BASE  = float(c.get("topcmt_font_size", c.get("comment_font1_size", 20)))
        self.TC_COLOR = str(c.get("topcmt_color", "red"))

        self.tccn    = self.col_num * 2
        self.tccw    = (self.W - self.ml - self.mr - self.lc) / self.tccn
        self.avail_h = self.mt - self.TC_MT - self.olw/2 - self.olvm - self.TC_MB

        self.font_names  = [c.get(f"font{i}") for i in range(1,6) if c.get(f"font{i}")]
        self.font_paths  = [str(self.fonts_dir / fn) for fn in self.font_names]
        tfa = str(c.get("text_fonts_array", "123"))
        self.tc_fps = [self.font_paths[int(i)-1] for i in tfa
                       if i.isdigit() and int(i) <= len(self.font_paths)]

        self.nop_set = set(ch for p in c.get("comment_comma_nop","").split("|") for ch in p if ch)

        self._reg: dict[str, str] = {}
        for fp, fn in zip(self.font_paths, self.font_names):
            rln = re.sub(r'[^A-Za-z0-9]', '_', fn)
            try:
                pdfmetrics.registerFont(RLTTFont(rln, fp))
                self._reg[fp] = rln
            except Exception:
                pass

    def _rl(self, fp): return self._reg.get(fp, "Helvetica")

    def _color(self, s):
        if not s or not isinstance(s, str):
            return (0, 0, 0)
        s = s.strip().lower()
        if not s:
            return (0, 0, 0)
        named = {"black":(0,0,0),"white":(1,1,1),"red":(1,0,0),"blue":(0,0,1),"gray":(0.5,0.5,0.5)}
        if s in named: return named[s]
        m = re.match(r'rgb\((\d+),(\d+),(\d+)\)', s)
        if m: return tuple(int(x)/255 for x in m.groups())
        m = re.match(r'#([0-9a-f]{6})', s)
        if m: h=m.group(1); return (int(h[0:2],16)/255,int(h[2:4],16)/255,int(h[4:6],16)/255)
        return (0,0,0)

    def _col_x(self, ci):
        if ci < self.tccn / 2:
            return self.W - self.mr - self.tccw * (ci + 1)
        return self.W - self.mr - self.tccw * (ci + 1) - self.lc

    def _build_pos(self, start_col, fs):
        row_h = fs * self.LINE_GAP
        rpc   = max(1, int(self.avail_h / row_h))
        pos   = []
        for ci in range(start_col, self.tccn):
            px = self._col_x(ci)
            for ri in range(1, rpc + 1):
                py = self.H - self.TC_MT - row_h * (ri - 0.5) - fs / 2
                pos.append((px, py))
        return pos, rpc

    def _fit_fs(self, tc_len, avail_cols):
        fs = self.FS_BASE
        while fs >= self.FS_MIN_ABS:
            rpc  = max(1, int(self.avail_h / (fs * self.LINE_GAP)))
            cols = (tc_len + rpc - 1) // rpc
            if cols <= avail_cols: return fs
            fs -= 1
        return -1

    def run(self) -> Path:
        tc_log = self.topcmt_dir / f"{self.pdf_path.stem}.txt"
        tcs: dict[int, list] = {}
        with open(tc_log, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split("|", 2)
                if len(parts) == 3:
                    pid, pcnt, txt = int(parts[0]), int(parts[1]), parts[2]
                    tcs.setdefault(pid, []).append((pcnt, txt))

        reader = PdfReader(str(self.pdf_path))
        writer = PdfWriter()
        for page in reader.pages:
            writer.add_page(page)

        filtered = []

        for pid in sorted(tcs.keys()):
            pager    = pid + 1          # 无封面，pid=0对应第1页
            page_idx = pager - 1
            if page_idx >= len(writer.pages): continue

            print(f"  叶[{pid}] → PDF第{pager}页，写入 {len(tcs[pid])} 条眉批...")

            buf  = io.BytesIO()
            cv   = rl_canvas.Canvas(buf, pagesize=(self.W, self.H))
            wrote = False
            next_free_col = 0
            last_pos = None

            for pcnt, tcomment in sorted(tcs[pid], key=lambda x: x[0]):
                tccn_begin = int(pcnt / self.row_num) * 2
                tccn_begin = max(tccn_begin, next_free_col)

                tctmp = tcomment
                for ch in self.nop_set: tctmp = tctmp.replace(ch, "")
                tc_len = len(tctmp)

                avail_cols = self.tccn - tccn_begin
                fs = self._fit_fs(tc_len, avail_cols)
                if fs < 0:
                    print(f"    过长被过滤：＜{tcomment[:20]}＞")
                    filtered.append(tcomment)
                    continue
                if fs < self.FS_BASE:
                    print(f"    字号从{self.FS_BASE}→{fs}pt：＜{tcomment[:12]}...＞")

                positions, rpc = self._build_pos(tccn_begin, fs)
                next_free_col  = tccn_begin + (tc_len + rpc - 1) // rpc
                pos_iter = iter(positions)

                for ch in tcomment:
                    fp = get_font_for_char(ch, self.tc_fps) or (self.tc_fps[0] if self.tc_fps else None)
                    if not fp: continue
                    fn = self._rl(fp)
                    r,g,b = self._color(self.TC_COLOR)
                    cv.setFillColorRGB(r,g,b)
                    if ch in self.nop_set:
                        if last_pos:
                            fx = last_pos[0] + self.tccw*(1-1/8) - fs*self.GR/2
                            fy = last_pos[1] + fs*self.LINE_GAP/12 - fs*self.GR/2
                            cv.setFont(fn, fs*self.GR); cv.drawString(fx, fy, ch); wrote=True
                    else:
                        try: fx, fy = next(pos_iter)
                        except StopIteration: break
                        last_pos = (fx, fy)
                        fx += (self.tccw - fs) / 2
                        if self.verbose: print(f"      -> {ch} ({int(fx)},{int(fy)}) {fs}pt")
                        cv.setFont(fn, fs); cv.drawString(fx, fy, ch); wrote=True

            if not wrote: continue
            cv.save(); buf.seek(0)
            overlay = PdfReader(buf).pages[0]
            writer.pages[page_idx].merge_page(overlay)

        # 更新 JSONL paging（删除被过滤眉批的标记）
        if filtered:
            self._update_paging(filtered)

        out_path = self.pdf_path.parent / f"{self.pdf_path.stem}_眉批.pdf"
        with open(out_path, "wb") as f:
            writer.write(f)
        return out_path

    def _update_paging(self, filtered):
        """被过滤的眉批未打入PDF，JSONL里也整体删除（＜标签+内容＞全删）"""
        stem = self.pdf_path.stem
        jl   = self.paging_dir / f"{stem}.jsonl"
        if not jl.exists(): return
        lines = jl.read_text(encoding="utf-8").splitlines()
        updated = []
        for line in lines:
            if not line.strip(): continue
            obj = json.loads(line)
            for txt in filtered:
                for msg in obj.get("messages", []):
                    # 删除 ＜被过滤眉批内容＞ 整体（标签+内容）
                    msg["content"] = msg["content"].replace(
                        "＜" + txt + "＞", "")
            updated.append(json.dumps(obj, ensure_ascii=False))
        jl.write_text("\n".join(updated) + "\n", encoding="utf-8")
        print(f"  paging JSONL 已更新（去除被过滤眉批）")


# 常量（供内部使用）
TAG_TC_L = "＜"
TAG_TC_R = "＞"
