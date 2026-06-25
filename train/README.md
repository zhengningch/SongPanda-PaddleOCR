# SongPanda 训练流程

本目录提供 **SongPanda** 模型的训练代码与配置。训练分为两条路线：

| 路线 | 基座 | 框架 | 用途 |
|---|---|---|---|
| **A（比赛主线）** | **PaddleOCR-VL-1.5** | **PaddleFormers** | 发布模型 `ningzhuo/SongPanda2.0` |
| B（方法论验证） | Qwen2.5-VL / Qwen3-VL | LLaMA-Factory | 横向对比、合成数据 vs 真实数据实验 |

---

## 目录结构

```
train/
├── README.md                       # 本文档
├── requirements.txt                # 训练环境依赖（PaddleFormers 栈）
├── finetune_paddleocrvl.yaml       # 路线 A：PaddleOCR-VL-1.5 全参微调配置
└── prepare_dataset.py              # 数据转换：vRain 输出 → PaddleFormers 训练格式
```

> **关于合成数据工具 vRain（方案 b 引用模式）**：本仓库**不内嵌** vRain 源码（原作者为兀雨书屋，请直接使用其开源仓库）。本项目仅提供「**使用 vRain 的具体配置与参数**」用以复现 SongPanda-Dataset，见下节「合成数据配置」。

---

## 合成数据配置（方案 b：仅提供配置，不内嵌源码）

### 上游工具

- **工具**：vRain（Perl）——由兀雨书屋开源的古籍刻本扫描流水线框架
- **获取**：请前往上游仓库下载（搜索 “vRain 兀雨书屋” 即可找到最新版本）
- **许可**：合成图像保留原作者“兀雨书屋”篆刻水印，表示致谢

### 本项目对 vRain 的关键参数选择

| 项 | 值 / 说明 |
|---|---|
| 字体 | `HanaMinA.ttf` / `HanaMinB.ttf` / `qiji-combo.ttf` / `WenYue_GuTiFangSong_F.otf` 混用（确保盖全复杂古字 + 字形风格多样性） |
| 画布 | `24_paper` / `28_paper` / `20_paper` / `vintage` / `mr_4` / `mr_5`（模拟不同纸张 + 摩尔纹 / 旧书效果） |
| 文本来源 | 基于《四库全书总目》四部分类的 56 本真实古籍文本（总计 1400w 字符） |
| 小字夹注标记 | 原文以 `【 】` 包裹，经由 vRain 渲染为双行小字格式 |
| Pure 策略 | 统一字体 + 统一画布，约 **2.2w 张** |
| Mix 策略 | 20% 书目换字体 / 换画布，插入换列 `\n` / 换页 `\f` / 眉批 `<head>` 结构标笾，并重复添加含水印 / 摩尔纹的噪音组，约 **2w 张** |

完整的数据构建详细流程见 [`../docs/training_data_report.md`](../docs/training_data_report.md)。

---

## 路线 A（比赛主线）：PaddleOCR-VL-1.5 全参微调

### 参考项目

本路线实现直接参考 AI Studio 上的官方最佳实践项目：

> [PaddleOCR-VL 微调项目 #10127933](https://aistudio.baidu.com/projectdetail/10127933)

### 训练配置速览

| 配置项 | 值 |
|---|---|
| 基座模型 | `PaddlePaddle/PaddleOCR-VL-1.5` |
| 微调方式 | 全参微调（Full Fine-tuning） |
| GPU | 1 × NVIDIA RTX 4090 48GB |
| 学习率 | 1e-5 |
| Epoch | 2 |
| 训练时长 | 约 8 小时 |
| 训练数据 | SongPanda-Dataset（Mix 集，约 2w 张） |
| 输出模型 | [`ningzhuo/SongPanda2.0`](https://huggingface.co/ningzhuo/SongPanda2.0) |

完整超参数见 [`finetune_paddleocrvl.yaml`](./finetune_paddleocrvl.yaml)。

### 环境准备

```bash
# 建议使用 Python 3.10+，CUDA 12.x
pip install -r requirements.txt
```

### Step 1：准备训练数据

1. 使用上游开源的 **vRain** 项目生成 Mix 策略合成数据（具体字体/画布/文本切分配置见本文档「合成数据配置」一节 + [`../docs/training_data_report.md`](../docs/training_data_report.md) 第 3-4 节）；
2. 将生成的图像 + 标签（`<footnote>` 标签格式）输出到一个目录；
3. 运行数据转换脚本，将其转为 PaddleFormers 所需的 JSONL 格式：

```bash
python prepare_dataset.py \
    --images-dir ./synthetic_images \
    --labels-dir ./synthetic_labels \
    --output ./data/train.jsonl
```

### Step 2：启动训练

```bash
python -m paddleformers.trl.run_finetune \
    --config ./finetune_paddleocrvl.yaml
```

或者直接在 AI Studio 上 fork 参考项目，将 `finetune_paddleocrvl.yaml` 中的数据路径改为自己的 JSONL 即可。

### Step 3：导出与上传

训练完成后权重输出至 `./output/all_full_epoch2`，包含：

```
all_full_epoch2/
├── config.json
├── model-*.safetensors
├── tokenizer.json / tokenizer_config.json / tokenizer.model
├── chat_template.jinja
├── preprocessor_config.json
└── generation_config.json
```

使用 `hf upload` 推送至 Hugging Face（排除中间 checkpoint）：

```bash
hf upload <your-repo> ./output/all_full_epoch2 . \
    --repo-type model \
    --exclude "checkpoint-*/*" "visualdl_logs/*" "training_args.bin"
```

---

## 路线 B（方法论验证）：Qwen 系列 LoRA 微调

该路线为**对比实验用途**，不作为比赛提交模型。关键配置：

| 配置项 | 值 |
|---|---|
| 基座模型 | Qwen2.5-VL-7B / Qwen3-VL-8B |
| 训练框架 | LLaMA-Factory |
| 微调方式 | LoRA |
| GPU | 2 × NVIDIA 5090 |
| 学习率 | 1e-5 |
| Epoch | 1 |

三轮实验分别产出 `Qwen2.5-VL-7B-Pure`、`Qwen2.5-VL-7B-Mix`、`Qwen3-VL-8B-Mix`，用于验证：

1. 合成数据在不同基座上的泛化增益；
2. Mix 策略相对 Pure 策略的鲁棒性优势；
3. 合成数据与等量真实数据的效能对比（见 [`../docs/training_data_report.md`](../docs/training_data_report.md) 第 8 节）。

如需复现，将 `prepare_dataset.py` 的 `--format llama_factory` 选项打开即可输出 LLaMA-Factory 兼容格式。

---

## 任务模板（Prompt 约定）

无论哪条路线，训练样本的 prompt 统一使用以下模板：

```
请对这张古籍图像进行 OCR：
- 自动删去版心无关正文的字段；
- 识别正文；
- 将双行小字夹注以 <footnote></footnote> 标签标出（Mix 集中 <head> 表示眉批、\n 表示列末换列、\f 表示换半页）。
```

模型响应即为完整的标注字符串。

---

## 致谢

- [PaddleFormers](https://github.com/PaddlePaddle/PaddleFormers) 以及 [AI Studio PaddleOCR-VL 微调项目 #10127933](https://aistudio.baidu.com/projectdetail/10127933) 提供的训练基础设施；
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 提供的对比实验框架；
- [vRain](https://github.com/fxliang/vRain)（兀雨书屋）提供的古籍合成工具。本项目仅引用其上游仓库与配置，合成图像保留原作者“兀雨书屋”篆刻水印。
