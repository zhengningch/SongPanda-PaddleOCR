#!/usr/bin/env python3
"""
评估脚本
================================

- 一期 / 二期：算法精度 = (0.8*NED + 0.2*NER-F1) * 100
- 三期：严格/宽松 = (0.5*NED + 0.3*圈点 + 0.1*眉批NER + 0.1*夹注NER) * 100

用法：
    python3 src/simple_eval.py                     # 全部模型
    python3 src/simple_eval.py --models stage8     # 指定模型
    python3 src/simple_eval.py --skip-realphoto    # 只跑原始 6 个模型

输出：
    bench/data/eval_output/simple_eval_summary.csv  +  控制台表格
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import editdistance
import pandas as pd

BENCH_DIR = Path(__file__).resolve().parent.parent
GT_FINAL = BENCH_DIR / "gt" / "groundtruth_index_answer_from.csv"
OUT_CSV = BENCH_DIR / "data" / "eval_output" / "simple_eval_summary.csv"


# ============================================================================
# 一二期算法
# ============================================================================
def remove_tags_keep_content(text: str) -> str:
    """去 <footnote>/<head>/【】/＜＞ 标签符号 + 圈点 + 标点，保留文字。"""
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"<footnote>|</footnote>|<head>|</head>", "", text)
    text = text.replace("【", "").replace("】", "")
    text = text.replace("＜", "").replace("＞", "")
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[●◎○☆◇◆△▲▽▼※]", "", text)
    text = re.sub(
        r"[。，、；：？！「」『』《》（）—…·～]"
        r"|[,.!?;:'\"\[\]{}()<>/\\|@$%^&*+=`~]+",
        "", text,
    )
    return text


def calc_ned_yiqi(gt: str, pred: str) -> float:
    """一二期 NED：去标签/圈点/标点后字符级 1 - edit_dist/max(len)。"""
    g = remove_tags_keep_content(gt).replace("\n", "").replace("\r", "")
    p = remove_tags_keep_content(pred).replace("\n", "").replace("\r", "")
    if "#" in g:
        # 通配符编辑距离
        m, n = len(g), len(p)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1): dp[i][0] = i
        for j in range(n + 1): dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if g[i-1] == p[j-1] or g[i-1] == "#":
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        edit_dist = dp[m][n]
    else:
        edit_dist = editdistance.eval(g, p)
    mx = max(len(g), len(p))
    return 0.0 if mx == 0 else 1 - edit_dist / mx


def extract_entities(text: str) -> List[str]:
    """提取 <footnote>/<head>/【】 实体。"""
    if pd.isna(text): return []
    text = str(text)
    ents = re.findall(r"<footnote>(.*?)</footnote>", text, re.DOTALL)
    ents += re.findall(r"<head>(.*?)</head>", text, re.DOTALL)
    ents += re.findall(r"【(.*?)】", text, re.DOTALL)
    return ents


def _entity_match_score(gt_e: str, pred_e: str, threshold: float = 0.5) -> float:
    gc, pc = set(gt_e), set(pred_e)
    if not gc and not pc: return 1.0
    if not gc or not pc: return 0.0
    inter = len(gc & pc)
    p = inter / len(pc) if pc else 0.0
    r = inter / len(gc) if gc else 0.0
    if p + r == 0: return 0.0
    f1 = 2 * p * r / (p + r)
    return f1 if f1 >= threshold else 0.0


def _find_best_matches(gt_ents, pred_ents, threshold=0.5):
    scores = []
    for i, g in enumerate(gt_ents):
        for j, p in enumerate(pred_ents):
            s = _entity_match_score(g, p, threshold)
            if s > 0: scores.append((i, j, s))
    scores.sort(key=lambda x: x[2], reverse=True)
    mg, mp, matches = set(), set(), []
    for i, j, s in scores:
        if i not in mg and j not in mp:
            matches.append((i, j, s)); mg.add(i); mp.add(j)
    return matches


def calc_ner_f1(gt: str, pred: str, threshold: float = 0.5) -> float:
    """一二期 NER-F1：实体级 F1。"""
    ge = extract_entities(gt)
    pe = extract_entities(pred)
    if not ge and not pe: return 1.0
    if not ge: return 0.0
    if not pe: return 0.0
    matches = _find_best_matches(ge, pe, threshold)
    tot = sum(s for _, _, s in matches)
    P = tot / len(pe)
    R = tot / len(ge)
    return 2 * P * R / (P + R) if (P + R) > 0 else 0.0


# ============================================================================
# 三期算法
# ============================================================================
CIRC_RAW = set("○●◎〇△▲▽▼※☆◇◆。、")
CIRC_EQUIV_TO_PERIOD = set("○〇。")


def is_cjk(ch: str) -> bool:
    o = ord(ch)
    return (0x4E00 <= o <= 0x9FFF) or (0x3400 <= o <= 0x4DBF) or (0x20000 <= o <= 0x2A6DF)


def _norm_circ(ch: str) -> str:
    return "。" if ch in CIRC_EQUIV_TO_PERIOD else ch


def _norm_brackets(s: str) -> str:
    return s.replace("＜", "<").replace("＞", ">")


def extract_meipi(s: str) -> List[str]:
    return re.findall(r"<(.*?)>", _norm_brackets(s), re.DOTALL)


def extract_jiajian(s: str) -> List[str]:
    return re.findall(r"【(.*?)】", s, re.DOTALL)


def extract_circ_pairs(s: str) -> List[Tuple[str, str]]:
    pairs, last_cjk, cnt = [], None, Counter()
    for ch in s:
        if is_cjk(ch):
            last_cjk = ch
        elif ch in CIRC_RAW:
            if last_cjk is not None and cnt[last_cjk] < 2:
                pairs.append((last_cjk, _norm_circ(ch)))
                cnt[last_cjk] += 1
    return pairs


def extract_circled_chars(s: str) -> set:
    return {p[0] for p in extract_circ_pairs(s)}


def strip_to_cjk(s: str) -> str:
    return "".join(ch for ch in s if is_cjk(ch))


def main_text_ned(gt: str, pred: str) -> float:
    """三期正文 NED：去眉批/夹注内容 + 圈点，只留汉字。"""
    g = _norm_brackets(gt); p = _norm_brackets(pred)
    g = re.sub(r"<.*?>", "", g, flags=re.DOTALL)
    p = re.sub(r"<.*?>", "", p, flags=re.DOTALL)
    g = re.sub(r"【.*?】", "", g, flags=re.DOTALL)
    p = re.sub(r"【.*?】", "", p, flags=re.DOTALL)
    g = strip_to_cjk(g); p = strip_to_cjk(p)
    mx = max(len(g), len(p))
    return 1.0 if mx == 0 else 1 - editdistance.eval(g, p) / mx


def _prf_from_sets(gs, ps):
    if not gs and not ps: return 1.0
    if not gs: return 0.0
    if not ps: return 0.0
    inter = len(gs & ps)
    P = inter / len(ps); R = inter / len(gs)
    return 2 * P * R / (P + R) if (P + R) > 0 else 0.0


def circ_strict_f1(gt: str, pred: str) -> float:
    g = Counter(map(tuple, extract_circ_pairs(gt)))
    p = Counter(map(tuple, extract_circ_pairs(pred)))
    if not g and not p: return 1.0
    if not g or not p: return 0.0
    inter = sum((g & p).values())
    P = inter / sum(p.values()); R = inter / sum(g.values())
    return 2 * P * R / (P + R) if (P + R) > 0 else 0.0


def circ_lenient_f1(gt: str, pred: str) -> float:
    return _prf_from_sets(extract_circled_chars(gt), extract_circled_chars(pred))


def ner_f1_entities(gt_ents, pred_ents, threshold=0.5) -> float:
    if not gt_ents and not pred_ents: return 1.0
    if not gt_ents: return 0.0
    if not pred_ents: return 0.0
    matches = _find_best_matches(gt_ents, pred_ents, threshold)
    tot = sum(s for _, _, s in matches)
    P = tot / len(pred_ents); R = tot / len(gt_ents)
    return 2 * P * R / (P + R) if (P + R) > 0 else 0.0


# ============================================================================
# 模型配置
# ============================================================================
# 模型配置：数据文件按约定存放于 data/jsonl/{模型短名}/{期别}.jsonl
# （实拍图为 data/jsonl/{模型短名}/实拍.jsonl）。仅保留模型短名，不硬编码内部路径。
MODEL_CONFIGS = {
    "paddleocrvl1.6_base": ["data/jsonl/paddleocrvl1.6_base/一期.jsonl", "data/jsonl/paddleocrvl1.6_base/二期.jsonl", "data/jsonl/paddleocrvl1.6_base/三期.jsonl"],
    "v2_from_base": ["data/jsonl/v2_from_base/一期.jsonl", "data/jsonl/v2_from_base/二期.jsonl", "data/jsonl/v2_from_base/三期.jsonl"],
    "stage6": ["data/jsonl/stage6/一期.jsonl", "data/jsonl/stage6/二期.jsonl", "data/jsonl/stage6/三期.jsonl"],
    "stage7": ["data/jsonl/stage7/一期.jsonl", "data/jsonl/stage7/二期.jsonl", "data/jsonl/stage7/三期.jsonl"],
    "stage8": ["data/jsonl/stage8/一期.jsonl", "data/jsonl/stage8/二期.jsonl", "data/jsonl/stage8/三期.jsonl"],
    "grpo_v5_step1500": ["data/jsonl/grpo_v5_step1500/一期.jsonl", "data/jsonl/grpo_v5_step1500/二期.jsonl", "data/jsonl/grpo_v5_step1500/三期.jsonl"],
    # ===== 实拍图 =====
    "realphoto_paddleocrvl1.6_base": ["data/jsonl/realphoto_paddleocrvl1.6_base/实拍.jsonl"],
    "realphoto_stage5": ["data/jsonl/realphoto_stage5/实拍.jsonl"],
    "realphoto_stage6_cpu": ["data/jsonl/realphoto_stage6_cpu/实拍.jsonl"],
    "realphoto_stage6": ["data/jsonl/realphoto_stage6/实拍.jsonl"],
    "realphoto_stage7": ["data/jsonl/realphoto_stage7/实拍.jsonl"],
    "realphoto_stage8": ["data/jsonl/realphoto_stage8/实拍.jsonl"],
    "realphoto_v5_cpu": ["data/jsonl/realphoto_v5_cpu/实拍.jsonl"],
    "realphoto_grpo_v5_step1500": ["data/jsonl/realphoto_grpo_v5_step1500/实拍.jsonl"],
}


# ============================================================================
# GT 与图片路径
# ============================================================================
def load_gt() -> dict:
    """读统一 GT → {index: (answer, from)}。"""
    df = pd.read_csv(GT_FINAL, dtype={"index": int, "answer": str, "from": str})
    return {int(r["index"]): (str(r["answer"]), str(r["from"])) for _, r in df.iterrows()}


def detect_period(image_path: str) -> str | None:
    m = re.search(r"实拍图/[^/]+/([一二]期|三期)/", image_path)
    if m: return m.group(1)
    if "【一期】" in image_path: return "一期"
    if "【二期】" in image_path: return "二期"
    if "【三期】" in image_path: return "三期"
    return None


def extract_seq(image_path: str) -> int | None:
    stem = Path(image_path).stem
    return int(stem) if stem.isdigit() else None


# ============================================================================
# 单样本评估
# ============================================================================
def eval_sample(period: str, gt_text: str, pred_text: str, status: str) -> dict | None:
    """返回单样本分项 dict，失败返回 None。"""
    if status == "失败": return None
    if period in ("一期", "二期"):
        ned = calc_ned_yiqi(gt_text, pred_text)
        ner = calc_ner_f1(gt_text, pred_text)
        return {"NED": ned, "NER_F1": ner}
    # 三期
    ned = main_text_ned(gt_text, pred_text)
    cs = circ_strict_f1(gt_text, pred_text)
    cl = circ_lenient_f1(gt_text, pred_text)
    mf = ner_f1_entities(extract_meipi(gt_text), extract_meipi(pred_text), 0.3)
    jf = ner_f1_entities(extract_jiajian(gt_text), extract_jiajian(pred_text), 0.5)
    return {"NED": ned, "圈点严格": cs, "圈点宽松": cl, "眉批NER": mf, "夹注NER": jf}


# ============================================================================
# 模型评估
# ============================================================================
def evaluate_model(model_key: str, jsonls: list, gt: dict) -> list:
    rows = []
    by_period = {"一期": [], "二期": [], "三期": []}
    for jp in jsonls:
        p = BENCH_DIR / jp
        if not p.exists(): continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                d = json.loads(line)
                image = d.get("image", "")
                period = detect_period(image)
                if not period: continue
                seq = extract_seq(image)
                if seq is None or seq not in gt: continue
                gt_text, _ = gt[seq]
                pred = d.get("ocr", "")
                finish = d.get("finish_reason", "")
                ok = d.get("ok", False)
                status = "成功" if (ok or finish == "stop") else "失败"
                by_period[period].append((gt_text, pred, status))

    for period, samples in by_period.items():
        if not samples: continue
        ok_samples = [eval_sample(period, *s) for s in samples if s[2] == "成功"]
        ok_samples = [s for s in ok_samples if s is not None]
        total = len(samples)
        success = len(ok_samples)
        if not ok_samples:
            continue

        if period in ("一期", "二期"):
            avg_ned = sum(s["NED"] for s in ok_samples) / success
            avg_ner = sum(s["NER_F1"] for s in ok_samples) / success
            score = (0.8 * avg_ned + 0.2 * avg_ner) * 100
            rows.append({
                "模型": model_key,
                "期别": period,
                "成功": success,
                "失败": total - success,
                "NED": round(avg_ned, 4),
                "NER_F1": round(avg_ner, 4),
                "分数": round(score, 2),
            })
        else:
            avg_ned = sum(s["NED"] for s in ok_samples) / success
            avg_cs = sum(s["圈点严格"] for s in ok_samples) / success
            avg_cl = sum(s["圈点宽松"] for s in ok_samples) / success
            avg_mf = sum(s["眉批NER"] for s in ok_samples) / success
            avg_jf = sum(s["夹注NER"] for s in ok_samples) / success
            strict = (0.5 * avg_ned + 0.3 * avg_cs + 0.1 * avg_mf + 0.1 * avg_jf) * 100
            lenient = (0.5 * avg_ned + 0.3 * avg_cl + 0.1 * avg_mf + 0.1 * avg_jf) * 100
            rows.append({
                "模型": model_key, "期别": "三期-严格", "成功": success, "失败": total - success,
                "NED": round(avg_ned, 4), "圈点": round(avg_cs, 4), "眉批NER": round(avg_mf, 4),
                "夹注NER": round(avg_jf, 4), "分数": round(strict, 2),
            })
            rows.append({
                "模型": model_key, "期别": "三期-宽松", "成功": success, "失败": total - success,
                "NED": round(avg_ned, 4), "圈点": round(avg_cl, 4), "眉批NER": round(avg_mf, 4),
                "夹注NER": round(avg_jf, 4), "分数": round(lenient, 2),
            })
    return rows


# ============================================================================
# 主函数
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="统一评估（自包含，直接算分）")
    parser.add_argument("--models", nargs="*", default=None, help="指定模型，默认全部")
    parser.add_argument("--skip-realphoto", action="store_true", help="跳过实拍图")
    args = parser.parse_args()

    if not GT_FINAL.exists():
        sys.exit(f"GT 不存在: {GT_FINAL}")

    gt = load_gt()
    print(f"[+] GT: {GT_FINAL} ({len(gt)} 条)")

    models = args.models or list(MODEL_CONFIGS.keys())
    if args.skip_realphoto:
        models = [m for m in models if not m.startswith("realphoto_")]

    all_rows = []
    for mk in models:
        if mk not in MODEL_CONFIGS:
            print(f"[!] 未知模型: {mk}"); continue
        print(f"[>] 评估: {mk}")
        rows = evaluate_model(mk, MODEL_CONFIGS[mk], gt)
        all_rows.extend(rows)

    if not all_rows:
        print("[!] 无结果"); return

    df = pd.DataFrame(all_rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")

    # 控制台表格：每期排序
    print(f"\n{'='*90}")
    print("  统一评估汇总（按最终统一 GT）")
    print(f"{'='*90}")
    for label in ["一期", "二期", "三期-严格", "三期-宽松"]:
        sub = df[df["期别"] == label].sort_values("分数", ascending=False)
        if sub.empty: continue
        print(f"\n  --- {label} ---")
        for i, r in enumerate(sub.itertuples(), 1):
            print(f"  {i:2d}. {r.模型:<35} 分数={r.分数:6.2f}  成功={r.成功}/{r.成功+r.失败}")

    # 分类汇总
    print(f"\n{'='*90}\n  原始模型 / 实拍图 分类\n{'='*90}")
    for kind in ["原始", "实拍"]:
        ms = [m for m in models if (m.startswith("realphoto_") == (kind == "实拍"))]
        if not ms: continue
        print(f"\n  【{kind}】")
        print(f"  {'模型':<35} {'一期':>8} {'二期':>8} {'三期严格':>8} {'三期宽松':>8}")
        for m in ms:
            cells = []
            for per in ["一期", "二期", "三期-严格", "三期-宽松"]:
                s = df[(df["模型"] == m) & (df["期别"] == per)]
                cells.append(f"{s['分数'].iloc[0]:.2f}" if len(s) else "  --")
            print(f"  {m:<35} {cells[0]:>8} {cells[1]:>8} {cells[2]:>8} {cells[3]:>8}")

    print(f"\n[+] 已保存: {OUT_CSV}")


if __name__ == "__main__":
    main()
