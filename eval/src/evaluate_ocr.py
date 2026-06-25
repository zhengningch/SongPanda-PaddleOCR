#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SongPanda-Bench 古籍OCR评测脚本

评估指标（与论文一致）：
  A = 0.8 × NED_score + 0.2 × NER_F1
  鲁棒性 R_i = 1 / (1 + log2(1 + r))   （失败样本 r 记为 20）

标签体系：
  <footnote>...</footnote>  双行小字夹注
  <head>...</head>          眉批
  【...】                    与 footnote 等价的备用标签
  #                         漫漶不清的通配符，可匹配任意字符

用法示例
-----------------------------------------------------------------
python evaluate_ocr.py \
    --groundtruth ../groundtruth.csv \
    --prediction  ./songpanda_pred.xlsx \
    --model-name  SongPanda \
    --output-dir  ./report
"""

import os
import re
import sys
import math
import argparse
from typing import List, Tuple, Set

import pandas as pd

try:
    import editdistance
except ImportError:
    print("❌ 未安装 editdistance，请先执行: pip install editdistance")
    sys.exit(1)


# =========================================================================
# 文本预处理 & 实体提取
# =========================================================================
def remove_tags_keep_content(text) -> str:
    """去掉所有标注标签但保留标签内内容，用于 NED 计算。"""
    if pd.isna(text):
        return ''
    text = str(text)
    text = re.sub(r'<footnote>|</footnote>', '', text)
    text = re.sub(r'<head>|</head>', '', text)
    text = re.sub(r'【|】', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    return text


def extract_entities(text) -> List[str]:
    """从文本中提取 <footnote> / <head> / 【】 包裹的实体。"""
    if pd.isna(text):
        return []
    text = str(text)
    annotations = []
    annotations += re.findall(r'<footnote>(.*?)</footnote>', text, re.DOTALL)
    annotations += re.findall(r'<head>(.*?)</head>', text, re.DOTALL)
    annotations += re.findall(r'【(.*?)】', text, re.DOTALL)
    return annotations


# =========================================================================
# 支持 # 通配符的编辑距离
# =========================================================================
def edit_distance_with_wildcard(truth: str, prediction: str) -> int:
    """
    truth 中 '#' 作为通配符，可匹配任何单一字符。
    """
    m, n = len(truth), len(prediction)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if truth[i - 1] == prediction[j - 1] or truth[i - 1] == '#':
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


# =========================================================================
# NED 得分
# =========================================================================
def calculate_ned(truth, prediction) -> float:
    t = remove_tags_keep_content(truth).replace('\n', '').replace('\r', '')
    p = remove_tags_keep_content(prediction).replace('\n', '').replace('\r', '')
    if '#' in t:
        dist = edit_distance_with_wildcard(t, p)
    else:
        dist = editdistance.eval(t, p)
    max_len = max(len(t), len(p))
    if max_len == 0:
        return 0.0
    return 1 - dist / max_len


# =========================================================================
# NER F1 得分（基于 footnote/head/【】 实体集合）
# =========================================================================
def _entity_f1_pair(gt_entity: str, pred_entity: str,
                    threshold: float) -> float:
    gt_chars = set(gt_entity)
    pred_chars = set(pred_entity)
    if not gt_chars and not pred_chars:
        return 1.0
    if not gt_chars or not pred_chars:
        return 0.0
    overlap = gt_chars & pred_chars
    precision = len(overlap) / len(pred_chars)
    recall = len(overlap) / len(gt_chars)
    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return f1 if f1 >= threshold else 0.0


def _greedy_match(gt_ents, pred_ents, threshold):
    scores = []
    for i, g in enumerate(gt_ents):
        for j, p in enumerate(pred_ents):
            s = _entity_f1_pair(g, p, threshold)
            if s > 0:
                scores.append((i, j, s))
    scores.sort(key=lambda x: x[2], reverse=True)
    matched_g, matched_p = set(), set()
    matches = []
    for i, j, s in scores:
        if i not in matched_g and j not in matched_p:
            matches.append((i, j, s))
            matched_g.add(i)
            matched_p.add(j)
    return matches


def calculate_ner_f1(truth, prediction,
                     threshold: float = 0.5) -> Tuple[float, float, float]:
    gt = extract_entities(truth)
    pred = extract_entities(prediction)
    if not gt and not pred:
        return 1.0, 1.0, 1.0
    if not gt:
        return 0.0, 0.0, 1.0
    if not pred:
        return 0.0, 1.0, 0.0
    matches = _greedy_match(gt, pred, threshold)
    total = sum(s for _, _, s in matches)
    precision = total / len(pred) if pred else 0.0
    recall = total / len(gt) if gt else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall


# =========================================================================
# 鲁棒性得分
# =========================================================================
def robustness_score(retry_count: int) -> float:
    """R_i = 1 / (1 + log2(1 + r))，首次成功 r=0 → 1.0"""
    return 1.0 / (1.0 + math.log2(1.0 + retry_count))


# =========================================================================
# 主评测逻辑
# =========================================================================
def load_groundtruth(path: str) -> pd.DataFrame:
    """支持 CSV / Excel 格式，列需含 [序号, ground truth] 或 [次序, answer]"""
    if path.lower().endswith('.csv'):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    rename = {}
    if '次序' in df.columns and '序号' not in df.columns:
        rename['次序'] = '序号'
    if 'answer' in df.columns and 'ground truth' not in df.columns:
        rename['answer'] = 'ground truth'
    if rename:
        df = df.rename(columns=rename)
    if '序号' not in df.columns or 'ground truth' not in df.columns:
        raise ValueError(
            f"groundtruth 文件必须包含列 [序号, ground truth]（或 [次序, answer]），"
            f"当前列: {list(df.columns)}"
        )
    return df


def load_prediction(path: str) -> pd.DataFrame:
    """加载 OCR 推理结果 Excel，必需列: 序号, OCR结果, 处理状态, 重试次数"""
    df = pd.read_excel(path)
    required = ['序号', 'OCR结果', '处理状态', '重试次数']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"推理结果文件缺少必需列: {missing}；当前列: {list(df.columns)}"
        )
    return df


def evaluate_dataset(gt_path: str, pred_path: str, model_name: str,
                     failed_retry_count: int = 20,
                     ner_weight: float = 0.2,
                     ner_threshold: float = 0.5) -> Tuple[pd.DataFrame, dict]:
    gt_df = load_groundtruth(gt_path)
    pred_df = load_prediction(pred_path)

    print(f"=== 评估模型: {model_name} ===")
    print(f"Ground truth 样本数: {len(gt_df)}")
    print(f"推理结果样本数: {len(pred_df)}")

    successful = pred_df[pred_df['处理状态'] == '成功'].copy()
    failed = pred_df[pred_df['处理状态'] == '失败'].copy()
    print(f"  ✅ 成功: {len(successful)}   ❌ 失败: {len(failed)}")

    rows = []

    for _, r in successful.iterrows():
        idx = r['序号']
        m = gt_df[gt_df['序号'] == idx]
        if m.empty:
            print(f"⚠️  序号 {idx} 未在 groundtruth 中找到")
            continue
        gt_text = m.iloc[0]['ground truth']
        pred_text = r['OCR结果']
        retry = int(r['重试次数']) if pd.notna(r['重试次数']) else 0

        ned = calculate_ned(gt_text, pred_text)
        ner_f1, ner_p, ner_r = calculate_ner_f1(
            gt_text, pred_text, threshold=ner_threshold
        )
        total = (1 - ner_weight) * ned + ner_weight * ner_f1
        rob = robustness_score(retry)

        rows.append({
            '模型名称': model_name,
            '序号': idx,
            '图片名称': r.get('图片名称', ''),
            'Ground Truth': gt_text,
            'OCR结果': pred_text,
            '处理状态': '成功',
            '重试次数': retry,
            'NED_score': ned,
            'NER_F1': ner_f1,
            'NER_Precision': ner_p,
            'NER_Recall': ner_r,
            'A（总得分）': total,
            '鲁棒性': rob,
            '处理时间(秒)': r.get('处理时间(秒)', None),
        })

    for _, r in failed.iterrows():
        idx = r['序号']
        m = gt_df[gt_df['序号'] == idx]
        if m.empty:
            continue
        rows.append({
            '模型名称': model_name,
            '序号': idx,
            '图片名称': r.get('图片名称', ''),
            'Ground Truth': m.iloc[0]['ground truth'],
            'OCR结果': '处理失败',
            '处理状态': '失败',
            '重试次数': failed_retry_count,
            'NED_score': None,
            'NER_F1': None,
            'NER_Precision': None,
            'NER_Recall': None,
            'A（总得分）': None,
            '鲁棒性': robustness_score(failed_retry_count),
            '处理时间(秒)': r.get('处理时间(秒)', None),
        })

    detail = pd.DataFrame(rows)

    # 统计：NED / NER / A 只基于成功样本；鲁棒性基于全部样本
    ok = detail[detail['处理状态'] == '成功']
    stats = {
        '模型名称': model_name,
        '总样本数': len(detail),
        '成功样本数': len(ok),
        '失败样本数': len(detail) - len(ok),
        'NED_score（均值）': ok['NED_score'].mean() if len(ok) else 0.0,
        'NER_F1（均值）': ok['NER_F1'].mean() if len(ok) else 0.0,
        'A 总得分（均值）': ok['A（总得分）'].mean() if len(ok) else 0.0,
        '鲁棒性（均值）': detail['鲁棒性'].mean() if len(detail) else 0.0,
        '平均重试次数（成功样本）': ok['重试次数'].mean() if len(ok) else 0.0,
        '平均处理时间（成功样本）': ok['处理时间(秒)'].mean() if len(ok) else None,
    }

    print("\n=== 评估结果 ===")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    return detail, stats


# =========================================================================
# CLI
# =========================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="SongPanda-Bench 古籍OCR评测脚本（NED + NER + 鲁棒性）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--groundtruth', required=True,
                        help='标注文件路径（CSV 或 Excel）')
    parser.add_argument('--prediction', required=True, action='append',
                        help='OCR 推理结果 Excel 文件路径（可指定多次以评测多个模型）')
    parser.add_argument('--model-name', action='append', default=None,
                        help='与 --prediction 一一对应的模型名；'
                             '若不传将以文件名作为默认模型名')
    parser.add_argument('--output-dir', default='./report',
                        help='输出目录')
    parser.add_argument('--ner-weight', type=float, default=0.2,
                        help='A 中 NER_F1 权重（NED 权重自动取 1 - 该值）')
    parser.add_argument('--ner-threshold', type=float, default=0.5,
                        help='NER 实体匹配 F1 阈值')
    parser.add_argument('--failed-retry-count', type=int, default=20,
                        help='失败样本鲁棒性计算时使用的重试次数')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    predictions = args.prediction
    model_names = args.model_name or []
    if len(model_names) < len(predictions):
        model_names += [
            os.path.splitext(os.path.basename(p))[0]
            for p in predictions[len(model_names):]
        ]

    all_detail = []
    all_stats = []

    for pred_path, model_name in zip(predictions, model_names):
        print(f"\n{'=' * 60}")
        print(f"处理: {pred_path}")
        print(f"{'=' * 60}")
        detail, stats = evaluate_dataset(
            gt_path=args.groundtruth,
            pred_path=pred_path,
            model_name=model_name,
            failed_retry_count=args.failed_retry_count,
            ner_weight=args.ner_weight,
            ner_threshold=args.ner_threshold,
        )
        all_detail.append(detail)
        all_stats.append(stats)

    if not all_detail:
        print("❌ 未产出任何评测结果")
        return

    detail_all = pd.concat(all_detail, ignore_index=True)
    stats_df = pd.DataFrame(all_stats)

    detail_path = os.path.join(args.output_dir, 'evaluation_detail.xlsx')
    stats_path = os.path.join(args.output_dir, 'evaluation_summary.xlsx')
    detail_all.to_excel(detail_path, index=False, engine='openpyxl')
    stats_df.to_excel(stats_path, index=False, engine='openpyxl')

    print(f"\n📝 详细评估结果: {os.path.abspath(detail_path)}")
    print(f"📊 汇总对比:     {os.path.abspath(stats_path)}")
    print("\n=== 模型对比 ===")
    print(stats_df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
