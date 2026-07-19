#!/usr/bin/env python3
"""
Step 4 v2: GRPO 无 ground truth 训练脚本（reward 信号强化版）

相对 v1 (step4_grpo_harvard_bert_reward.py) 的核心修改：
  1) 格式奖励 v2：纯文字（无标签）= 1.0；任一类型不闭合 = 0.0；
                   只用一种且闭合 = 0.5；都闭合 = 1.0
  2) 重复惩罚 v2：增加 4字/8字片段重复 + 整行重复检测
  3) num_generations 4 → 8（per_device_bs 4 → 1），advantage 方差减半
  4) BERT 区分度 v2：分桶奖励（按高置信错误字符绝对数），4 个 rollout 能拉开差距
  5) 权重微调：bert 0.5 + fmt 0.2 - rep 0.3（让格式塌陷的影响变小，重复的影响变大）

运行:
    cd /mnt/private_rainchzheng/paddleocr
    CUDA_VISIBLE_DEVICES=0,1,2,3 .venv_vllm310/bin/torchrun \
        --nproc_per_node=4 --master_port=29501 \
        --tee 3 --redirects 3 --log_dir /tmp/torchrun_log \
        training_stage3_4/scripts/grpo/step4_grpo_harvard_bert_reward_v2.py \
        2>&1 | tee training_stage3_4/output/grpo_harvard_bert_v2/run.log
"""

import os, re, json, random, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import torch.nn as nn
from datasets import Dataset
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    BertForMaskedLM,
    BertTokenizerFast,
)
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig, PeftModel, TaskType

ROOT         = "/mnt/private_rainchzheng/paddleocr"
POLICY_DIR   = f"{ROOT}/training_stage3_4/output/paddleocr_vl_1_6_v2_stage7"
DATA_JSONL   = f"{ROOT}/training_stage3_4/data/harvard_grpo_5k.jsonl"
OUTPUT_DIR   = os.environ.get("OUTPUT_DIR", f"{ROOT}/training_stage3_4/output/grpo_harvard_bert_v2")
LOG_DIR      = f"{OUTPUT_DIR}/visualdl_logs"

# 快速验证可设置 MAX_STEPS>0；正式全量跑时设置 MAX_STEPS=0
MAX_STEPS = int(os.environ.get("MAX_STEPS", "20"))
NUM_TRAIN_EPOCHS = float(os.environ.get("NUM_TRAIN_EPOCHS", "2"))
LOGGING_STEPS = int(os.environ.get("LOGGING_STEPS", "5"))
SAVE_STEPS = int(os.environ.get("SAVE_STEPS", "100"))
_save_total_limit_env = os.environ.get("SAVE_TOTAL_LIMIT", "")
SAVE_TOTAL_LIMIT = int(_save_total_limit_env) if _save_total_limit_env.strip() else None
MAX_COMPLETION_LENGTH = int(os.environ.get("MAX_COMPLETION_LENGTH", "640"))
RESUME_FROM_CHECKPOINT = os.environ.get("RESUME_FROM_CHECKPOINT") or None
# 仅加载 LoRA adapter 权重继续训练，不恢复 optimizer/scheduler；用于绕开 Trainer resume 的 PEFT/Transformers 兼容问题。
LOAD_LORA_ADAPTER = os.environ.get("LOAD_LORA_ADAPTER") or None
LOG_COMPLETIONS = os.environ.get("LOG_COMPLETIONS", "0") == "1"
NUM_COMPLETIONS_TO_PRINT = int(os.environ.get("NUM_COMPLETIONS_TO_PRINT", "4"))
DEBUG_COMPLETIONS = os.environ.get("DEBUG_COMPLETIONS", "0") == "1"
DEBUG_COMPLETIONS_EVERY = int(os.environ.get("DEBUG_COMPLETIONS_EVERY", "10"))
DEBUG_COMPLETION_CHARS = int(os.environ.get("DEBUG_COMPLETION_CHARS", "500"))

# Reward 权重：v3 默认把满分拉回 1.0，并让重复惩罚只针对明确退化模式。
BERT_REWARD_WEIGHT = float(os.environ.get("BERT_REWARD_WEIGHT", "0.7"))
FORMAT_REWARD_WEIGHT = float(os.environ.get("FORMAT_REWARD_WEIGHT", "0.3"))
REPETITION_REWARD_WEIGHT = float(os.environ.get("REPETITION_REWARD_WEIGHT", "0.35"))

# 降低视觉 token：默认 28*28*768 = 602112 像素，约 768 image tokens。
# 原始配置 28*28*1280 = 1003520 像素，约 1280 image tokens。
# 先降视觉输入，不先降 completion 到 512，避免无 GT reward 对尾部截断失明。
IMAGE_MAX_PIXELS = int(os.environ.get("IMAGE_MAX_PIXELS", str(28 * 28 * 768)))

# LoRA：默认开启。GRPO reward 还在调参阶段，LoRA 更省显存/磁盘，也更稳。
USE_LORA = os.environ.get("USE_LORA", "1") != "0"
LORA_R = int(os.environ.get("LORA_R", "16"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "32"))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0.05"))
LORA_TARGET_MODULES = [
    x.strip() for x in os.environ.get(
        "LORA_TARGET_MODULES",
        "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    ).split(",") if x.strip()
]

# vLLM rollout（默认关闭；用于小步 smoke 验证 .venv_vllm310 的 colocate 是否可用）
USE_VLLM = os.environ.get("USE_VLLM", "0") == "1"
VLLM_MODE = os.environ.get("VLLM_MODE", "colocate")
VLLM_MODEL_IMPL = os.environ.get("VLLM_MODEL_IMPL", "vllm")
VLLM_GPU_MEMORY_UTILIZATION = float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.25"))
VLLM_MAX_MODEL_LENGTH = int(os.environ.get("VLLM_MAX_MODEL_LENGTH", "2048"))
VLLM_TENSOR_PARALLEL_SIZE = int(os.environ.get("VLLM_TENSOR_PARALLEL_SIZE", "1"))
VLLM_ENABLE_SLEEP_MODE = os.environ.get("VLLM_ENABLE_SLEEP_MODE", "0") == "1"

# BERT reward model
BERT_CKPT    = f"{ROOT}/training_stage3_4/bert_grpo/model/ocr_error_detector_best.pt"
# 当前机器没有 /root/.cache 下的 SIKU-BERT/sikubert 缓存；这里用本地 continued-pretrain 目录提供 config/tokenizer。
# 随后会立刻 load BERT_CKPT 的完整 state_dict，BERT 权重会被 ocr_error_detector_best.pt 覆盖。
SIKUBERT_ID  = os.environ.get("SIKUBERT_ID", "/mnt/private_rainchzheng/gufeng/sikubert_continued_pretrain/final")

# ──────────────────────────────────────────────────────
# v2 新增：BERT 错误字符分桶阈值（高/中/低三档）
# ──────────────────────────────────────────────────────
BERT_HI_THRESHOLD = 0.95   # 高置信错误：模型非常确定的错字
BERT_LO_THRESHOLD = 0.80   # 弱置信错误：可能是错字
# 分桶区间改为 **错误率**（hi_count / hanzi_count），长短文本公平
# 设计：用比例而非绝对数，避免短文本天然 0 错误 → 满分 hack
BERT_RATE_BUCKETS = [
    (0.000, 0.000, 1.00),   # 0% 错误：完美
    (0.001, 0.020, 0.90),   # <2% 错误：偶发
    (0.021, 0.060, 0.72),   # 2~6%：一些错
    (0.061, 0.120, 0.45),   # 6~12%：明显有错
    (0.121, 0.250, 0.22),   # 12~25%：大量错
    (0.251, 1.000, 0.05),   # >25%：错到没法用
]

# GRPO prompt（v2 格式，与训练数据一致）
OCR_PROMPT   = "OCR:【】<>、.。"


# ──────────────────────────────────────────────────────
# 1. BERT reward model 定义（与 v1 完全一致）
# ──────────────────────────────────────────────────────

SIKUBERT_CACHED = None

class SikuBertMlmAwareClassifier(nn.Module):
    def __init__(self, model_name: str = SIKUBERT_ID, num_labels: int = 2, dropout: float = 0.15):
        super().__init__()
        self.bert_mlm = BertForMaskedLM.from_pretrained(model_name)
        hidden = self.bert_mlm.config.hidden_size
        vocab_size = self.bert_mlm.config.vocab_size
        self.vocab_size = vocab_size
        extra_feat_dim = 3
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden + extra_feat_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_labels),
        )

    def forward(self, input_ids, attention_mask):
        out = self.bert_mlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_state = out.hidden_states[-1]
        mlm_logits = out.logits

        with torch.no_grad():
            logp = torch.log_softmax(mlm_logits.float(), dim=-1)
            p = logp.exp()
            self_logp = logp.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)
            self_p = p.gather(-1, input_ids.unsqueeze(-1))
            rank = (p >= self_p).float().sum(-1) - 1.0
            rank_feat = (rank / self.vocab_size).clamp(0, 1)
            entropy = -(p * logp).sum(-1)
            extra = torch.stack([self_logp, rank_feat, entropy], dim=-1).to(hidden_state.dtype)

        feat = torch.cat([hidden_state, extra], dim=-1)
        feat = self.dropout(feat)
        logits = self.classifier(feat)
        return logits


def _load_bert_model():
    global SIKUBERT_CACHED
    if SIKUBERT_CACHED is not None:
        return SIKUBERT_CACHED

    print("[BERT reward] 加载 SikuBertMlmAwareClassifier ...")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    bert_model = SikuBertMlmAwareClassifier(model_name=SIKUBERT_ID).to(device)
    state_dict = torch.load(BERT_CKPT, map_location=device)
    bert_model.load_state_dict(state_dict)
    bert_model.eval()

    tokenizer = BertTokenizerFast.from_pretrained(SIKUBERT_ID)
    SIKUBERT_CACHED = (bert_model, tokenizer, device)
    print(f"[BERT reward] 加载完成，设备 {device}")
    return SIKUBERT_CACHED


# ──────────────────────────────────────────────────────
# 2. 文本提取与 reward 工具函数
# ──────────────────────────────────────────────────────

def _extract_text_from_completion(comp) -> str:
    if isinstance(comp, str):
        return comp
    if isinstance(comp, list):
        comp = comp[0] if comp else ""
        return _extract_text_from_completion(comp)
    if isinstance(comp, dict):
        content = comp.get("content", "")
        if isinstance(content, list):
            for c in content:
                if isinstance(c, dict) and c.get("type") == "text":
                    return c.get("text", "")
        return str(content)
    return str(comp)


def _extract_for_bert(generation: str) -> str:
    """只保留 CJK 汉字字符串，剔除所有圈点标点、ASCII、括号等。
    目的：让 BERT 专注于字符级错误，不被标点噪声影响；
    同时调用方可用返回的汉字数作为"有效文本长度"。
    """
    text = generation.strip()
    text = re.sub(r'^OCR:\s*', '', text)
    # 只保留 CJK 统一汉字（\u4e00-\u9fff）+ 扩展A（\u3400-\u4dbf）+ 扩展B（\U00020000-\U0002a6df）
    hanzi_only = re.sub(r'[^\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df]', '', text)
    return hanzi_only


def _count_closed(text: str, open_ch: str, close_ch: str) -> tuple:
    return text.count(open_ch), text.count(close_ch)


# ──────────────────────────────────────────────────────
# 3. Reward Functions (v2)
# ──────────────────────────────────────────────────────

@torch.no_grad()
def bert_reward_fn_v2(completions, **kwargs) -> list:
    """
    v3: 连续奖励 — 用 hi_rate + mean_prob 两个连续信号替代离散分桶。
    ──────────────────────────────────────────
    v2 问题：6 档分桶导致 54% completion 挤在 0.90 档（0.1%~2% 错误率），
    同一 image 的 4 个 rollout bert 完全相同 → GRPO advantage ≈ 0。

    v3 设计：
      - hi_rate = hi_count / actual_len （高置信错误率，连续值）
      - mean_prob = inner_probs.mean() （平均错误概率，连续值，捕获阈值下的细微差异）
      - 连续公式：reward = max(0.05, 1.0 - hi_rate * 4.0 - mean_prob * 1.5)
      - 4 个 rollout 几乎不可能得到完全相同的 bert 分数 → 始终有 advantage
    """
    bert_model, tokenizer, device = _load_bert_model()
    rewards = []

    for comp in completions:
        raw = _extract_text_from_completion(comp).strip()
        text = _extract_for_bert(raw)  # 此时 text 仅含汉字
        hanzi_count = len(text)

        # 极短兜底：不足 5 汉字，给中性低分而非完全归零（短页保留信号）
        if hanzi_count < 5:
            rewards.append(0.2)
            continue

        text = text[:510]
        enc = tokenizer(
            list(text),
            is_split_into_words=True,
            add_special_tokens=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        logits = bert_model(input_ids, attention_mask)
        probs = torch.softmax(logits.float(), dim=-1)[0, :, 1]

        inner_probs = probs[1:-1]
        actual_len = inner_probs.numel()
        if actual_len == 0:
            rewards.append(0.0)
            continue

        hi_count = (inner_probs >= BERT_HI_THRESHOLD).sum().item()

        # 连续奖励：hi_rate 为主信号，mean_prob 为辅信号（打破并列）
        hi_rate = hi_count / actual_len
        mean_prob = inner_probs.mean().item()
        reward = max(0.05, 1.0 - hi_rate * 4.0 - mean_prob * 1.5)

        rewards.append(float(reward))

    return rewards


def format_reward_fn_v2(completions, **kwargs) -> list:
    """
    v3: 符号序列栈式校验 — 只要出现的标签正确配对就 1.0，
    容忍尾部截断（末尾连续的开括号未闭合视为被截断，不算错）。
    ──────────────────────────────────────────
    规则：
      - 完全无任何标签（纯文字）            → 1.0
      - 出现的标签全部正确配对               → 1.0  （只用一种也 1.0）
      - 尾部截断：符号序列末尾连续的开括号
        未闭合（可能是生成被截断）           → 1.0
      - 中间未闭合、交叉嵌套（如 【<】>）    → 0.0
    """
    PAIRS = {'【': '】', '<': '>'}
    OPENERS = set(PAIRS.keys())
    CLOSERS = set(PAIRS.values())
    CLOSER_TO_OPENER = {v: k for k, v in PAIRS.items()}

    rewards = []
    for comp in completions:
        text = _extract_text_from_completion(comp).strip()
        body = re.sub(r'^OCR:\s*', '', text)

        # 提取所有标签符号，保持出现顺序
        symbols = [ch for ch in body if ch in OPENERS or ch in CLOSERS]

        if not symbols:
            rewards.append(1.0)  # 纯文字，无标签
            continue

        # 栈式校验：开括号入栈，闭括号检查栈顶是否匹配
        stack = []  # 存放 (opener, position_in_symbols)
        mismatch = False
        for idx, ch in enumerate(symbols):
            if ch in OPENERS:
                stack.append((ch, idx))
            else:  # closer
                if stack and stack[-1][0] == CLOSER_TO_OPENER[ch]:
                    stack.pop()
                else:
                    mismatch = True
                    break

        if mismatch:
            rewards.append(0.0)  # 交叉嵌套或多余闭括号
            continue

        if not stack:
            rewards.append(1.0)  # 全部配对
            continue

        # 栈中剩余的都是未闭合的开括号
        # 截断容忍：剩余开括号的位置构成符号序列的连续后缀
        n = len(symbols)
        k = len(stack)
        remaining_positions = [pos for _, pos in stack]
        if remaining_positions == list(range(n - k, n)):
            rewards.append(1.0)  # 尾部截断，容忍
        else:
            rewards.append(0.0)  # 中间未闭合

    return rewards


def repetition_penalty_fn_v2(completions, **kwargs) -> list:
    """
    v3: 只惩罚明确的退化重复，避免把古籍中正常复现的人名/官名/格式短语误杀。

    重点覆盖用户提供的坏例：
      - 【上】【上】【上】【上】 这类连续重复短标签
      - 【勅勅呂呂】连续刷屏
      - 大夫\n大夫\n大夫... 这类整行循环
      - 任意 2-12 字短片段连续重复 4 次以上

    不再使用 v2 的"任意位置 4字出现3次 / 8字出现2次"，因为它会误伤正常古籍复现。
    返回值为 penalty severity，范围 [0, 1]。
    """
    punct_pattern = re.compile(r'[○●□■△▲◎〇。、，；：！？…—～\u25CB\u25CF]{5,}')
    char_repeat = re.compile(r'(.)\1{4,}')
    tag_repeat = re.compile(r'(【[^】]{1,8}】|<[^>]{1,8}>)(?:\1){2,}')
    phrase_loop = re.compile(r'([^\s【】<>]{2,12})(?:\1){3,}')

    rewards = []
    for comp in completions:
        text = _extract_text_from_completion(comp).strip()
        text_for_line = text.replace('\\n', '\n').replace('\\f', '\n')
        text_clean = text.replace('【', '').replace('】', '').replace('<', '').replace('>', '')

        penalty = 0.0

        if tag_repeat.search(text):
            penalty += 0.60
        if char_repeat.search(text):
            penalty += 0.25
        if punct_pattern.search(text):
            penalty += 0.20
        if phrase_loop.search(text_clean):
            penalty += 0.60

        lines = [ln.strip() for ln in text_for_line.split('\n') if ln.strip()]
        if len(lines) >= 2:
            # 连续相同行检测（不限行数）
            run = 1
            max_run = 1
            for prev, cur in zip(lines, lines[1:]):
                if prev == cur:
                    run += 1
                    max_run = max(max_run, run)
                else:
                    run = 1
            if max_run >= 3:   # 连续 3 行一样即惩罚
                penalty += 0.80

        if len(lines) >= 6:
            from collections import Counter
            counter = Counter(lines)
            max_repeat = max(counter.values())
            if max_repeat >= 5 and max_repeat / len(lines) >= 0.35:
                penalty += 0.80

        rewards.append(min(penalty, 1.0))
    return rewards


REWARD_CALL_COUNT = 0


def combined_reward_v2(completions, **kwargs) -> list:
    """
    v3 总 reward：
      reward = (BERT_REWARD_WEIGHT * bert + FORMAT_REWARD_WEIGHT * format - REPETITION_REWARD_WEIGHT * repetition)
               * length_bonus

    length_bonus（基于 completion 汉字数）：
      hanzi < 5             → LB_MIN (0.2)  （极短但保留信号）
      5 <= hanzi < MIN_HANZI → 线性从 LB_MIN → 1  （写了一些但不够）
      hanzi >= MIN_HANZI     → 1.0  （正常长度，不惩罚不奖励）
    MIN_HANZI 默认 60（约 2~3 行古文），可用 REWARD_MIN_HANZI 覆盖。

    目的：彻底堵死"输出越少 BERT 越容易满分"的 hack。
    短页面（图中本来字就少）BERT 也能给高分，length_bonus 不影响；
    只要输出字数远低于正常 OCR 页面，就被整体压低。
    """
    global REWARD_CALL_COUNT
    REWARD_CALL_COUNT += 1

    MIN_HANZI = int(os.environ.get("REWARD_MIN_HANZI", "60"))

    bert_r = bert_reward_fn_v2(completions, **kwargs)
    fmt_r  = format_reward_fn_v2(completions, **kwargs)
    rep_p  = repetition_penalty_fn_v2(completions, **kwargs)

    results = []
    length_bonuses = []
    LB_MIN = 0.2   # 最低乘数：保留短页学习信号，同时让 hack 输出 vs 正常输出仍有 5x 差距
    for comp, b, f, p in zip(completions, bert_r, fmt_r, rep_p):
        raw = _extract_text_from_completion(comp).strip()
        hanzi_count = len(re.sub(r'[^\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df]', '', raw))

        if hanzi_count < 5:
            lb = LB_MIN          # 极短但保留信号，不完全归零
        elif hanzi_count < MIN_HANZI:
            # 线性从 LB_MIN → 1.0
            lb = LB_MIN + (1.0 - LB_MIN) * (hanzi_count - 5) / max(1, MIN_HANZI - 5)
        else:
            lb = 1.0
        length_bonuses.append(lb)

        # bert-rep 耦合：重复退化时 bert 分数不可信（重复的正确字≠真正 OCR 正确）
        # rep=0 时 bert 不变；rep=1.0 时 bert 打 7 折
        effective_bert = b * (1.0 - p * 0.3)

        r = (BERT_REWARD_WEIGHT * effective_bert + FORMAT_REWARD_WEIGHT * f - REPETITION_REWARD_WEIGHT * p) * lb
        results.append(float(max(0.0, min(1.0, r))))

    print(f"  [reward] bert={[f'{x:.3f}' for x in bert_r]} "
          f"fmt={[f'{x:.2f}' for x in fmt_r]} "
          f"pen={[f'{x:.2f}' for x in rep_p]} "
          f"lb={[f'{x:.2f}' for x in length_bonuses]} "
          f"→ final={[f'{x:.3f}' for x in results]}")

    if DEBUG_COMPLETIONS and int(os.environ.get("LOCAL_RANK", "0")) == 0:
        if REWARD_CALL_COUNT % max(1, DEBUG_COMPLETIONS_EVERY) == 0:
            for i, (comp, b, f, p, lb, r) in enumerate(zip(completions, bert_r, fmt_r, rep_p, length_bonuses, results)):
                if i >= NUM_COMPLETIONS_TO_PRINT:
                    break
                raw = _extract_text_from_completion(comp).strip()
                hanzi_c = len(re.sub(r'[^\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df]', '', raw))
                snippet = raw[:DEBUG_COMPLETION_CHARS].replace('\n', '\\n').replace('\f', '\\f')
                print(f"  [completion_sample #{i}] bert={b:.3f} fmt={f:.2f} pen={p:.2f} lb={lb:.2f} final={r:.3f} len={len(raw)} hanzi={hanzi_c} text={snippet}", flush=True)

    return results


# ──────────────────────────────────────────────────────
# 4. 数据集构建（与 v1 完全一致）
# ──────────────────────────────────────────────────────

def build_dataset(jsonl_path: str, max_samples: int = None, seed: int = 42) -> Dataset:
    from PIL import Image as PILImage

    lines = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)

    random.seed(seed)
    random.shuffle(lines)
    if max_samples is not None:
        lines = lines[:max_samples]

    prompts, img_paths = [], []
    for line in lines:
        d = json.loads(line)
        p = d["images"][0]
        if not os.path.exists(p):
            continue
        prompts.append([{"role": "user", "content": d["messages"][0]["content"]}])
        img_paths.append(p)

    print(f"[数据] {len(img_paths)} 条样本（秒级加载，collate 时读图）")

    ds = Dataset.from_dict({"prompt": prompts, "image_path": img_paths})

    def load_images(batch):
        batch["images"] = [
            [PILImage.open(p).convert("RGB")] for p in batch["image_path"]
        ]
        return batch

    return ds.with_transform(load_images)


# ──────────────────────────────────────────────────────
# 5. 主训练流程
# ──────────────────────────────────────────────────────

def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    print("=" * 70)
    print(" GRPO v2 — 哈佛善本，无 GT，BERT reward（信号强化版）")
    print("=" * 70)
    print(f" Policy:  {POLICY_DIR}")
    print(f" Data:    {DATA_JSONL}")
    print(f" Output:  {OUTPUT_DIR}")
    print(f" Prompt:  {OCR_PROMPT}")
    print(f" BERT ckpt: {BERT_CKPT}")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    _load_bert_model()

    print("\n[1] 加载 processor ...")
    processor = AutoProcessor.from_pretrained(POLICY_DIR, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    # 训练脚本内直接覆盖 image_processor 的 max_pixels，避免改模型目录配置文件。
    # PaddleOCRVLProcessor.__call__ 会调用 self.image_processor(images=...)，该处理器使用自身 max_pixels。
    processor.image_processor.max_pixels = IMAGE_MAX_PIXELS
    processor.image_processor.size = {
        "min_pixels": processor.image_processor.min_pixels,
        "max_pixels": IMAGE_MAX_PIXELS,
    }
    est_img_tokens = IMAGE_MAX_PIXELS // (28 * 28)
    print(f"  image max_pixels: {IMAGE_MAX_PIXELS} (~{est_img_tokens} image tokens, default=1280)")

    print("\n[2] 加载数据集 ...")
    dataset = build_dataset(DATA_JSONL, max_samples=None)
    print(f"  dataset size: {len(dataset)}")

    if len(dataset) == 0:
        print("[!] 数据集为空，退出")
        sys.exit(1)

    print("\n[3] 加载 policy model ...")
    model = AutoModelForCausalLM.from_pretrained(
        POLICY_DIR,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    # checkpoint 里 generation_config 默认 use_cache=False，会显著拖慢 autoregressive generate。
    # GRPO rollout 阶段在 no_grad 下生成，可强制打开 KV cache；训练 forward 仍由 Trainer 正常处理。
    model.config.use_cache = True
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = True

    manual_lora_loaded = False
    if LOAD_LORA_ADAPTER:
        print(f"  加载已有 LoRA adapter: {LOAD_LORA_ADAPTER}")
        model = PeftModel.from_pretrained(model, LOAD_LORA_ADAPTER, is_trainable=True)
        manual_lora_loaded = True

    print(f"  模型类型: {type(model).__name__}")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    print(f"  use_cache: {getattr(model.generation_config, 'use_cache', None)}")

    print("\n[4] 配置 GRPO (v2: num_generations=8, per_device_bs=4) ...")
    grpo_kwargs = {}
    if MAX_STEPS > 0:
        grpo_kwargs["max_steps"] = MAX_STEPS

    grpo_config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        logging_dir=LOG_DIR,

        # 训练超参
        num_train_epochs=NUM_TRAIN_EPOCHS,
        learning_rate=float(os.environ.get("LEARNING_RATE", "1e-6")),
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        bf16=True,

        # v2 关键改动：num_generations 4 → 8；per_device_bs 保持 4。
        # TRL 要求 generation_batch_size (= per_device_bs * world_size) 能被 num_generations 整除。
        # 4卡时 global batch=16，可被 8 整除；每组 8 个 rollout，同时每步仍是 16 completions。
        per_device_train_batch_size=4,
        num_generations=8,
        max_completion_length=MAX_COMPLETION_LENGTH,
        gradient_accumulation_steps=1,

        # 生成多样性（v1: temperature=1.0, top_p=0.9, top_k=50）
        temperature=0.9,                 # v2 略降，避免多样性反而变成乱码
        top_p=0.95,
        top_k=50,

        # 多卡
        ddp_find_unused_parameters=True,

        # vLLM rollout（用于 smoke；server mode 需要 TRL vllm-serve，不是普通 OpenAI api_server）
        use_vllm=USE_VLLM,
        vllm_mode=VLLM_MODE,
        vllm_model_impl=VLLM_MODEL_IMPL,
        vllm_gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
        vllm_max_model_length=VLLM_MAX_MODEL_LENGTH,
        vllm_tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE,
        vllm_enable_sleep_mode=VLLM_ENABLE_SLEEP_MODE,

        # 日志与保存
        logging_steps=LOGGING_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        report_to=["tensorboard"],
        log_completions=LOG_COMPLETIONS,
        num_completions_to_print=NUM_COMPLETIONS_TO_PRINT,

        remove_unused_columns=False,
        dataloader_num_workers=0,
        **grpo_kwargs,
    )

    peft_config = None
    if manual_lora_loaded:
        print("\n[5] 初始化 GRPOTrainer (manual LoRA adapter loaded) ...")
        print(f"  adapter={LOAD_LORA_ADAPTER}")
    elif USE_LORA:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGET_MODULES,
            bias="none",
        )
        print("\n[5] 初始化 GRPOTrainer (LoRA) ...")
        print(f"  LoRA r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
        print(f"  target_modules={LORA_TARGET_MODULES}")
    else:
        print("\n[5] 初始化 GRPOTrainer (full fine-tune) ...")

    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=[combined_reward_v2],
        args=grpo_config,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    if USE_LORA and hasattr(trainer.model, "print_trainable_parameters"):
        trainer.model.print_trainable_parameters()

    # 估算：TRL GRPO 的 step 计数与普通 DDP 不完全一样；历史 v1 实测 19686 样本、per_device_bs=4、2 epoch → 9842 steps。
    total_samples = len(dataset)
    world_size = int(os.environ.get("WORLD_SIZE", "4"))
    global_batch = grpo_config.per_device_train_batch_size * world_size
    prompts_per_generation = global_batch // grpo_config.num_generations
    trl_steps_per_epoch = total_samples / grpo_config.per_device_train_batch_size
    trl_total_steps = trl_steps_per_epoch * grpo_config.num_train_epochs

    print("\n[6] 开始 GRPO 训练 ...")
    print(f"  总样本:        {total_samples}")
    print(f"  num_generations: {grpo_config.num_generations}")
    print(f"  per_device_bs:   {grpo_config.per_device_train_batch_size}")
    print(f"  global_batch:    {global_batch}")
    print(f"  prompts/gen:     {prompts_per_generation}")
    print(f"  epochs:          {NUM_TRAIN_EPOCHS}")
    print(f"  use_lora:        {USE_LORA}")
    print(f"  use_vllm:        {USE_VLLM}")
    if USE_VLLM:
        print(f"  vllm_mode:       {VLLM_MODE}")
        print(f"  vllm_impl:       {VLLM_MODEL_IMPL}")
        print(f"  vllm_mem_util:   {VLLM_GPU_MEMORY_UTILIZATION}")
        print(f"  vllm_max_len:    {VLLM_MAX_MODEL_LENGTH}")
    print(f"  max_steps:       {MAX_STEPS if MAX_STEPS > 0 else 'full'}")
    print(f"  logging_steps:   {LOGGING_STEPS}")
    print(f"  save_steps:      {SAVE_STEPS}")
    print(f"  save_total_limit:{SAVE_TOTAL_LIMIT}")
    print(f"  log_completions: {LOG_COMPLETIONS}")
    print(f"  debug_completions:{DEBUG_COMPLETIONS}")
    print(f"  resume_ckpt:     {RESUME_FROM_CHECKPOINT}")
    print(f"  load_lora_adapter:{LOAD_LORA_ADAPTER}")
    print(f"  reward_weights:  bert={BERT_REWARD_WEIGHT}, fmt={FORMAT_REWARD_WEIGHT}, rep={REPETITION_REWARD_WEIGHT}")
    print(f"  max_completion:  {MAX_COMPLETION_LENGTH}")
    print(f"  估算 TRL steps/epoch: {trl_steps_per_epoch:.0f}")
    print(f"  估算 TRL total_steps: {trl_total_steps:.0f}")
    print()

    trainer.train(resume_from_checkpoint=RESUME_FROM_CHECKPOINT)

    print(f"\n[✓] 训练完成！模型保存于: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
