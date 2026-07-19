#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vRain Studio 独立服务器
- 托管 studio/ 前端静态文件
- API: 读取 / 保存 layouts/*.yaml
- API: 调用真实排版引擎生成预览图（/api/preview）
- 完全在 studio/ 目录内运行，不依赖 vRain-py 其他目录

用法: cd studio && python3 server.py
"""
import json
import sys
import subprocess
import tempfile
import shutil
import re as _re
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse

BASE_DIR    = Path(__file__).parent       # studio/ 自身
LAYOUTS_DIR = BASE_DIR / "layouts"
BOOKS_DIR   = BASE_DIR / "books"
STUDIO_DIR  = BASE_DIR
IMAGE_DIR   = BASE_DIR / "image"
FONTS_DIR   = BASE_DIR / "fonts"
CANVAS_DIR  = BASE_DIR / "canvas"

# 内置默认预览文本（当 books/00.txt 不存在时使用）
DEFAULT_PREVIEW_TEXT = """史記集解序
裴駰字龍駒河東聞喜人宋中郎外兵曹參軍父松之字世期太中大夫注三國志班固有言曰司馬遷據左氏國語采世本戰國策述楚漢春秋接其後事訖于天漢其言秦漢詳矣至於采經摭傳分散數家之事甚多疏略或有抵捂亦其所涉獵者廣博貫穿經傳馳騁古今上下數千載閒斯已勤矣又其是非頗謬於聖人論大道則先黃老而後六經序游俠則退處士而進姦雄述貨殖則崇勢利而羞賤貧此其所蔽也然自劉向揚雄博極羣書皆稱遷有良史之才服其善序事理辯而不華質而不俚其文直其事核不虛美不隱惡故謂之實錄駰以為固之所言世稱其當雖時有紕繆實勒成一家緫其大較信命世之宏才也考較此書文句不同有多有少莫辯其實而世之惑者定後從此是非相貿真偽舛雜故中散大夫東莞徐廣研核衆本為音義具列異同兼述訓解粗有所發明而殊恨省略聊以愚管增演徐氏采經傳百家并先儒之說豫是有益悉皆抄內刪其游辭取其要實或義在可疑則數家兼列漢書音義稱臣瓚者莫知氏姓今直云瓚曰又都無姓名者但云漢書音義時見微意有所裨補譬嘒星之繼朝陽飛塵之集華嶽以徐為本號曰集解未詳則闕弗敢臆說人心不同聞見異辭班氏所謂疏略抵捂者依違不悉辯也愧非胥臣之多聞子產之博物妄言末學蕪穢舊史豈足以關諸畜德庶賢無所用心而已
%
史記索隱序
司馬貞撰史記者漢太史司馬遷父子之所述也遷自以承五百之運繼春秋而纂是史其襃貶覈實頗亞於丘明之書於是上始軒轅下訖天漢作十二本紀十表八書三十系家七十列傳凡一百三十篇始變左氏之體而年載悠邈簡冊闕遺勒成一家其勤至矣又其屬稁先據左氏國語系本戰國策楚漢春秋及諸子百家之書而後貫穿經傳馳騁古今錯綜隱括各使成一國一家之事故其意難究詳矣比於班書微為古質故漢晉名賢未知見重所以魏文侯聽古樂則唯恐卧良有以也逮至晉末有中散大夫東莞徐廣始考異同作音義十三卷宋外兵參軍裴駰又取經傳訓釋作集解合為八十卷雖粗見微意而未窮討論南齊輕車錄事鄒誕生亦作音義三卷音則微殊義乃更略爾後其學中廢貞觀中諫議大夫崇賢館學士劉伯莊達學宏才鉤深探賾又作音義二十卷比於徐鄒音則具矣殘文錯節異音微義雖知獨善不見傍通欲使後人從何准的貞謏聞陋識頗事鑽研而家傳是書不敢失墜初欲改更舛錯裨補踈遺義有未通兼重註述然此書殘缺雖多實為古史忽加穿鑿難允物情今止探求異聞採摭典故解其所未解申其所未申者釋文演註又為述贊凡三十卷號曰史記索隱雖未敢藏之書府亦欲以貽厥孫謀云
"""


class StudioHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STUDIO_DIR), **kwargs)

    def log_message(self, fmt, *args):
        print(f"  [{self.address_string()}] {fmt % args}")

    # ── GET ─────────────────────────────────────────────────────
    def do_GET(self):
        p = urlparse(self.path).path

        if p == "/api/layouts":
            self._json(self._list_layouts())

        elif p == "/api/fonts":
            fonts = sorted([f.name for f in FONTS_DIR.glob("*")
                            if f.suffix.lower() in (".ttf", ".otf", ".woff")]) if FONTS_DIR.exists() else []
            self._json(fonts)

        elif p == "/api/images":
            self._json(self._list_images())

        elif p.startswith("/api/layout/") and not p.startswith("/api/layout/save"):
            lid = p.split("/api/layout/")[-1].strip("/")
            fpath = LAYOUTS_DIR / f"{lid}.yaml"
            if fpath.exists():
                text = fpath.read_text(encoding="utf-8")
                self._json({"id": lid, "yaml": text, "cfg": _parse_yaml(text)})
            else:
                self._error(404, f"layout '{lid}' not found")

        # ── 静态文件：预览图片 ────────────────────────────────
        elif p.startswith("/preview_output/"):
            fname = p.split("/preview_output/")[-1].strip("/")
            fpath = STUDIO_DIR / "preview_output" / fname
            if fpath.exists() and fpath.suffix.lower() in (".jpg", ".jpeg", ".png"):
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(fpath.stat().st_size))
                self.send_header("Cache-Control", "no-cache")
                self._cors()
                self.end_headers()
                with open(fpath, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self._error(404, "image not found")

        # ── 静态文件：图片素材缩略图 ─────────────────────────
        elif p.startswith("/image/"):
            fname = p.split("/image/")[-1].strip("/")
            fpath = IMAGE_DIR / fname
            if fpath.exists() and fpath.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(fpath.stat().st_size))
                self.send_header("Cache-Control", "no-cache")
                self._cors()
                self.end_headers()
                with open(fpath, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self._error(404, "image not found")

        # ── 静态文件：canvas 背景图 ───────────────────────────
        elif p.startswith("/canvas/"):
            fname = p.split("/canvas/")[-1].strip("/")
            fpath = CANVAS_DIR / fname
            if fpath.exists() and fpath.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(fpath.stat().st_size))
                self.send_header("Cache-Control", "no-cache")
                self._cors()
                self.end_headers()
                with open(fpath, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self._error(404, "image not found")

        else:
            super().do_GET()

    # ── POST ────────────────────────────────────────────────────
    def do_POST(self):
        p = urlparse(self.path).path
        cl = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(cl) if cl else b""

        if p == "/api/layout/save":
            self._handle_save(body)
        elif p == "/api/layout/delete":
            self._handle_delete(body)
        elif p == "/api/preview":
            self._handle_preview(body)
        elif p == "/api/upload_canvas":
            self._handle_upload_canvas(body)
        else:
            self._error(404, "unknown api")

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors(); self.end_headers()

    # ── 业务处理 ────────────────────────────────────────────────
    def _list_layouts(self):
        result = []
        for f in sorted(LAYOUTS_DIR.glob("*.yaml")):
            text = f.read_text(encoding="utf-8")
            cfg = _parse_yaml(text)
            result.append({"id": f.stem, "name": cfg.get("canvas_id", f.stem),
                           "yaml": text, "cfg": cfg})
        return result

    def _list_images(self):
        """列出图片素材（用于图文排版选择）"""
        if not IMAGE_DIR.exists():
            return []
        exts = {".jpg", ".jpeg", ".png", ".webp"}
        return [{
            "name": f.name,
            "url": f"/image/{f.name}",
            "size": f.stat().st_size,
        } for f in sorted(IMAGE_DIR.iterdir()) if f.suffix.lower() in exts]

    def _handle_save(self, body):
        try:
            data = json.loads(body)
            lid = data.get("id", "").strip()
            yaml = data.get("yaml", "")
            if not lid or not all(c.isalnum() or c in "_-" for c in lid):
                self._error(400, "invalid id"); return
            fpath = LAYOUTS_DIR / f"{lid}.yaml"
            fpath.write_text(yaml, encoding="utf-8")
            self._json({"ok": True, "path": str(fpath)})
        except Exception as e:
            self._error(500, str(e))

    def _handle_delete(self, body):
        try:
            data = json.loads(body)
            lid = data.get("id", "").strip()
            fpath = LAYOUTS_DIR / f"{lid}.yaml"
            if fpath.exists():
                fpath.unlink()
                self._json({"ok": True})
            else:
                self._error(404, "not found")
        except Exception as e:
            self._error(500, str(e))

    def _handle_upload_canvas(self, body):
        """
        接收 multipart/form-data 上传的图片，保存到 canvas/ 目录。
        表单字段：file=<图片>, filename=<期望文件名（不含扩展名）>
        返回：{ ok, canvas_id, path }
        """
        try:
            ctype = self.headers.get("Content-Type", "")
            print(f"  [upload] Content-Type: {ctype}")
            print(f"  [upload] Content-Length: {len(body)}")

            # 从 Content-Type 取 boundary（兼容带引号的情况）
            m = _re.search(r'boundary=[""]?([^\s;"]+)', ctype)
            if not m:
                self._error(400, "missing boundary"); return
            boundary = m.group(1).strip('"').encode()

            # 简单 multipart 解析
            parts = body.split(b"--" + boundary)
            file_data = None
            fname_raw = ""
            orig_fname = ""
            for part in parts:
                if b"Content-Disposition" not in part:
                    continue
                header_end = part.find(b"\r\n\r\n")
                if header_end < 0:
                    continue
                headers_raw = part[:header_end].decode("utf-8", errors="replace")
                data = part[header_end + 4:]
                if data.endswith(b"\r\n"):
                    data = data[:-2]

                name_m = _re.search(r'name="([^"]+)"', headers_raw)
                field_name = name_m.group(1) if name_m else ""
                fname_m = _re.search(r'filename="([^"]*)"', headers_raw)
                if fname_m:
                    orig_fname = fname_m.group(1)

                if field_name == "file":
                    file_data = data
                elif field_name == "filename":
                    fname_raw = data.decode("utf-8", errors="replace").strip()

            if file_data is None:
                self._error(400, "missing file field"); return

            safe = _re.sub(r"[^A-Za-z0-9_\-]", "_", fname_raw or orig_fname.rsplit(".", 1)[0]) or "canvas_img"
            ext = ("." + orig_fname.rsplit(".", 1)[-1].lower()) if "." in orig_fname else ".jpg"
            if ext not in (".jpg", ".jpeg", ".png", ".webp"):
                ext = ".jpg"

            CANVAS_DIR.mkdir(exist_ok=True)
            dest = CANVAS_DIR / f"{safe}{ext}"
            dest.write_bytes(file_data)

            canvas_id = safe
            print(f"  [upload] 已保存背景图: {dest} ({len(file_data)} bytes)")
            self._json({"ok": True, "canvas_id": canvas_id, "path": str(dest), "filename": dest.name})
        except Exception as e:
            import traceback
            self._error(500, traceback.format_exc())

    PREVIEW_OUT = STUDIO_DIR / "preview_output"  # 持久化预览图目录

    def _handle_preview(self, body):
        """
        调用真实 vrain.py 排版引擎生成预览。
        输出图片存到 studio/preview_output/，返回图片 URL 列表。
        可选参数：selected_images = ["a.jpg", "b.jpg", ...] 指定图文排版使用的图片
        """
        try:
            data = json.loads(body)
            pages = min(int(data.get("pages", 1)), 6)
            layout_id = data.get("layout_id", "").strip()
            yaml_txt = data.get("yaml_content", "")
            selected_images = data.get("selected_images", []) or []

            # 清理旧预览图
            if self.PREVIEW_OUT.exists():
                for f in self.PREVIEW_OUT.glob("*.jpg"):
                    f.unlink()

            # 使用持久化目录（不为临时，删除会保留）
            workdir = Path(tempfile.mkdtemp(prefix="vrain_preview_"))
            try:
                result = self._run_preview(workdir, layout_id, yaml_txt, pages, selected_images)
                self._json(result)
            finally:
                shutil.rmtree(workdir, ignore_errors=True)

        except Exception as e:
            import traceback
            self._error(500, traceback.format_exc())

    def _run_preview(self, tmpdir: Path, layout_id: str, yaml_txt: str, pages: int, selected_images: list) -> dict:
        # ── 1. 准备 layout yaml ───────────────────────────────────
        layouts_tmp = tmpdir / "layouts"
        layouts_tmp.mkdir()

        if yaml_txt:
            lid = "preview_tmp"
            cfg = _parse_yaml(yaml_txt)

            # ── 检查 canvas_id 对应背景图是否存在，不存在则 fallback ──
            canvas_id = str(cfg.get("canvas_id", "24_black"))
            canvas_exists = any(
                (CANVAS_DIR / f"{canvas_id}{ext}").exists()
                for ext in (".jpg", ".jpeg", ".png")
            )
            if not canvas_exists:
                print(f"  [preview] canvas_id '{canvas_id}' 背景图不存在，fallback 到 24_black")
                yaml_txt = _re.sub(
                    r'^(canvas_id\s*:)\s*\S+',
                    r'\g<1>          24_black',
                    yaml_txt, flags=_re.MULTILINE
                )
                cfg["canvas_id"] = "24_black"

            (layouts_tmp / f"{lid}.yaml").write_text(yaml_txt, encoding="utf-8")
        else:
            lid = layout_id
            fpath = LAYOUTS_DIR / f"{lid}.yaml"
            if not fpath.exists():
                raise FileNotFoundError(f"layout '{lid}' not found")
            yaml_txt = fpath.read_text(encoding="utf-8")
            cfg = _parse_yaml(yaml_txt)

            canvas_id = str(cfg.get("canvas_id", "24_black"))
            canvas_exists = any(
                (CANVAS_DIR / f"{canvas_id}{ext}").exists()
                for ext in (".jpg", ".jpeg", ".png")
            )
            if not canvas_exists:
                print(f"  [preview] canvas_id '{canvas_id}' 背景图不存在，fallback 到 24_black")
                yaml_txt = _re.sub(
                    r'^(canvas_id\s*:)\s*\S+',
                    r'\g<1>          24_black',
                    yaml_txt, flags=_re.MULTILINE
                )
                cfg["canvas_id"] = "24_black"

            (layouts_tmp / f"{lid}.yaml").write_text(yaml_txt, encoding="utf-8")

        # ── 2. 准备示例文本（优先用 books/00.txt，否则用内置文本） ─
        books_tmp = tmpdir / "books"
        books_tmp.mkdir()
        text_name = "preview"
        real_book = BOOKS_DIR / "00.txt"
        if real_book.exists():
            src_text = real_book.read_text(encoding="utf-8")
        else:
            src_text = DEFAULT_PREVIEW_TEXT
        (books_tmp / f"{text_name}.txt").write_text(src_text, encoding="utf-8")

        # ── 3. 其他资源目录 ──────────────────────────────────────
        # 字体直接引用本地
        for d in ["fonts", "db"]:
            src = BASE_DIR / d
            dst = tmpdir / d
            if src.exists():
                try:
                    dst.symlink_to(src.resolve())
                except Exception:
                    if src.is_dir():
                        shutil.copytree(src, dst)
                    else:
                        shutil.copy2(src, dst)

        # 图片目录：仅在版式包含 image_zones 时复制图片
        image_tmp = tmpdir / "image"
        image_tmp.mkdir()
        has_image_zones = bool(str(cfg.get("image_zones", "")).strip())
        if has_image_zones:
            if selected_images:
                copied = 0
                for img_name in selected_images:
                    safe = img_name.replace("..", "").replace("/", "").replace("\\", "")
                    src = IMAGE_DIR / safe
                    if src.exists() and src.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                        shutil.copy2(src, image_tmp / src.name)
                        copied += 1
                print(f"  [preview] 图文排版：已复制 {copied} 张选中图片到临时 image/ 目录")
            else:
                if IMAGE_DIR.exists():
                    for src in IMAGE_DIR.iterdir():
                        if src.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                            shutil.copy2(src, image_tmp / src.name)
                print(f"  [preview] 图文排版：未指定图片，使用全部 {len(list(image_tmp.iterdir()))} 张图片")
        else:
            print("  [preview] 版式无 image_zones，跳过图片复制")

        # canvas 背景图引用本地
        canvas_tmp = tmpdir / "canvas"
        canvas_tmp.mkdir()
        if CANVAS_DIR.exists():
            for src in CANVAS_DIR.iterdir():
                if src.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".cfg", ".pl"):
                    shutil.copy2(src, canvas_tmp / src.name)

        # ── 4. core 模块 ─────────────────────────────────────────
        core_src = BASE_DIR / "core"
        core_dst = tmpdir / "core"
        try:
            core_dst.symlink_to(core_src.resolve())
        except Exception:
            shutil.copytree(core_src, core_dst)

        # ── 5. 输出目录 ──────────────────────────────────────────
        out_dir = tmpdir / "output"
        out_dir.mkdir()

        # ── 6. 复制 vrain.py ─────────────────────────────────────
        vrain_src = BASE_DIR / "vrain.py"
        vrain_dst = tmpdir / "vrain.py"
        shutil.copy(vrain_src, vrain_dst)

        # ── 7. 运行排版引擎（完整流程，包含眉批插入）────────────
        cmd = [
            sys.executable, str(vrain_dst),
            "--text", text_name,
            "--layout", lid,
            "--test", str(pages),
            "--no-pdf",     # 最终只保留图片
        ]
        print(f"  [preview] 运行: {' '.join(cmd)}")
        proc = subprocess.run(
            cmd, cwd=str(tmpdir),
            capture_output=True, text=True, timeout=120
        )
        if proc.returncode != 0:
            raise RuntimeError(f"排版失败:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")

        # ── 8. 收集图片，拷贝到 studio/preview_output/ ──────────
        img_dir = out_dir / text_name / "images"
        self.PREVIEW_OUT.mkdir(parents=True, exist_ok=True)
        image_urls = []
        if img_dir.exists():
            for img_path in sorted(img_dir.glob("*.jpg")):
                dst = self.PREVIEW_OUT / img_path.name
                shutil.copy(img_path, dst)
                # 返回相对 URL，前端可直接加载
                image_urls.append(f"/preview_output/{img_path.name}")

        cw_val = cfg.get("canvas_width", 2480)
        ch_val = cfg.get("canvas_height", 1860)
        col_num = cfg.get("col_num", 24)
        row_num = cfg.get("row_num", 30)
        lcw = cfg.get("leaf_center_width", 120)
        ml, mr = cfg.get("margins_left", 50), cfg.get("margins_right", 50)
        mt, mb = cfg.get("margins_top", 200), cfg.get("margins_bottom", 50)
        body_w = cw_val - ml - mr - lcw
        body_h = ch_val - mt - mb
        cw_px = body_w / col_num if col_num else 0
        rh_px = body_h / row_num if row_num else 0

        return {
            "images": image_urls,
            "pages_total": len(image_urls),
            "log": proc.stdout[-2000:] if proc.stdout else "",
            "info": {
                "canvas": f"{cw_val}×{ch_val}",
                "cols": col_num,
                "rows": row_num,
                "col_w_px": round(cw_px, 2),
                "row_h_px": round(rh_px, 2),
                "chars_per_page": col_num * row_num,
            }
        }

    # ── HTTP 工具 ────────────────────────────────────────────────
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json(self, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors(); self.end_headers()
        self.wfile.write(body)

    def _error(self, code, msg):
        body = json.dumps({"error": msg}, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors(); self.end_headers()
        self.wfile.write(body)


# ── YAML 解析（极简） ────────────────────────────────────────────
def _parse_yaml(text: str) -> dict:
    result = {}
    for line in text.splitlines():
        line = line.split("#")[0].strip()
        if not line or ":" not in line:
            continue
        k, _, v = line.partition(":")
        k = k.strip(); v = v.strip().strip('"').strip("'")
        if not k: continue
        try:    result[k] = int(v)
        except ValueError:
            try:    result[k] = float(v)
            except ValueError: result[k] = v
    return result


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8766
    print("=" * 50)
    print(f"  vRain Studio  →  http://localhost:{port}/")
    print(f"  layouts: {LAYOUTS_DIR}")
    print(f"  image:   {IMAGE_DIR}")
    print(f"  fonts:   {FONTS_DIR}")
    print("  Ctrl+C 停止")
    print("=" * 50)
    # 多线程以免预览时阻塞
    server = HTTPServer(("", port), StudioHandler)
    server.timeout = 180
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
