#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量为所有 layout 生成预览图，存入 preview_output/
用法: cd train-pipeline-demo && python3 gen_previews.py
"""
import sys, shutil, subprocess, tempfile, re
from pathlib import Path

BASE_DIR    = Path(__file__).parent
LAYOUTS_DIR = BASE_DIR / "layouts"
BOOKS_DIR   = BASE_DIR / "books"
CANVAS_DIR  = BASE_DIR / "canvas"
IMAGE_DIR   = BASE_DIR / "image"
PREVIEW_OUT = BASE_DIR / "preview_output"

DEFAULT_TEXT = """史記集解序
裴駰字龍駒河東聞喜人宋中郎外兵曹參軍父松之字世期太中大夫注三國志班固有言曰司馬遷據左氏國語采世本戰國策述楚漢春秋接其後事訖于天漢其言秦漢詳矣至於采經摭傳分散數家之事甚多疏略或有抵捂亦其所涉獵者廣博貫穿經傳馳騁古今上下數千載閒斯已勤矣
%
史記索隱序
司馬貞撰史記者漢太史司馬遷父子之所述也遷自以承五百之運繼春秋而纂是史其襃貶覈實頗亞於丘明之書於是上始軒轅下訖天漢作十二本紀十表八書三十系家七十列傳凡一百三十篇始變左氏之體而年載悠邈簡冊闕遺勒成一家其勤至矣
"""

PAGES = 2  # 每个 layout 生成几页

def run_layout(layout_id: str, yaml_path: Path):
    print(f"\n{'='*50}")
    print(f"  layout: {layout_id}")
    tmpdir = Path(tempfile.mkdtemp(prefix=f"vrain_{layout_id}_"))
    try:
        # layouts
        ld = tmpdir / "layouts"; ld.mkdir()
        shutil.copy(yaml_path, ld / yaml_path.name)

        # books
        bd = tmpdir / "books"; bd.mkdir()
        real = BOOKS_DIR / "00.txt"
        txt = real.read_text("utf-8") if real.exists() else DEFAULT_TEXT
        (bd / "preview.txt").write_text(txt, "utf-8")

        # symlink 资源目录
        for d in ["fonts", "db", "canvas", "image"]:
            src = BASE_DIR / d
            dst = tmpdir / d
            if src.exists():
                try: dst.symlink_to(src.resolve())
                except: shutil.copytree(src, dst) if src.is_dir() else shutil.copy2(src, dst)
            else:
                dst.mkdir(exist_ok=True)

        # core
        try: (tmpdir/"core").symlink_to((BASE_DIR/"core").resolve())
        except: shutil.copytree(BASE_DIR/"core", tmpdir/"core")

        # vrain.py
        shutil.copy(BASE_DIR/"vrain.py", tmpdir/"vrain.py")
        (tmpdir/"output").mkdir()

        cmd = [sys.executable, str(tmpdir/"vrain.py"),
               "--text", "preview", "--layout", layout_id,
               "--test", str(PAGES), "--no-pdf"]
        print(f"  运行: {' '.join(cmd[-6:])}")
        proc = subprocess.run(cmd, cwd=str(tmpdir),
                              capture_output=True, text=True, timeout=180)
        if proc.returncode != 0:
            print(f"  ✗ 失败:\n{proc.stderr[-800:]}")
            return False

        # 收集输出图片
        imgs = sorted((tmpdir/"output").rglob("*.jpg"))
        if not imgs:
            imgs = sorted((tmpdir/"output").rglob("*.png"))
        if not imgs:
            print("  ✗ 没有生成图片")
            return False

        PREVIEW_OUT.mkdir(exist_ok=True)
        for i, src in enumerate(imgs[:PAGES], 1):
            dst = PREVIEW_OUT / f"{layout_id}_{i:02d}.jpg"
            shutil.copy2(src, dst)
            print(f"  ✓ {dst.name}  ({src.stat().st_size//1024} KB)")
        return True

    except subprocess.TimeoutExpired:
        print(f"  ✗ 超时(180s)")
        return False
    except Exception as e:
        print(f"  ✗ 异常: {e}")
        return False
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    layouts = sorted(LAYOUTS_DIR.glob("*.yaml"))
    print(f"找到 {len(layouts)} 个 layout，开始批量生成预览（每个 {PAGES} 页）…")

    ok, fail = [], []
    for yp in layouts:
        lid = yp.stem
        if run_layout(lid, yp):
            ok.append(lid)
        else:
            fail.append(lid)

    print(f"\n{'='*50}")
    print(f"完成！成功 {len(ok)} 个: {ok}")
    if fail:
        print(f"失败 {len(fail)} 个: {fail}")
    print(f"预览图已存入: {PREVIEW_OUT}")
