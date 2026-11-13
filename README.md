# SciChart-Evidence-VL

[![CI](https://github.com/Free1740/SciChart-Evidence-VL/actions/workflows/ci.yml/badge.svg)](https://github.com/Free1740/SciChart-Evidence-VL/actions/workflows/ci.yml)

面向科研图表理解的视觉证据驱动多模态大模型框架，基于 **Qwen2.5-VL**。

## 背景与问题

科研文献阅读、实验结果分析与模型性能对比等场景中，多模态大模型在复杂科研图表理解上存在三个核心问题：

1. **语言先验干扰**：模型依赖语言先验与表面线索，而非图表中的真实视觉证据；
2. **关键视觉信息遗漏**：关键曲线、实验指标与局部数据变化被忽略，导致错误结论；
3. **长链推理中的证据遗忘**：推理链越长，模型越难持续利用最初的关键视觉证据。

## 方法

### 1. 科研数据构建与反事实增强

- 基于约 5 万条可追溯绘图代码的图表问答数据，构建 Teacher 评审、答案校验及难度筛选流程，保留约 2 万条高难度样本；
- 通过程序化修改绘图代码，构造两类反事实视图，形成约 2 万组配对样本：
  - **EP（Evidence Preserved，关键证据保留）**：保留关键证据，仅做与答案无关的扰动；
  - **EA（Evidence Ablated，关键证据删除）**：删除关键曲线 / 实验指标 / 局部数据。
- 反事实配对模拟科研图表中关键曲线、实验指标及局部数据变化，强化模型对有效实验信息的辨别能力。

### 2. 视觉证据驱动的强化学习

双阶段 GRPO 优化策略：

- **阶段一（Stage 1）**：KL 一致性约束强化原图与 EP 图的预测一致性；反事实分离目标拉大原图与 EA 图的预测分布差异，抑制语言先验和表面线索依赖；
- **阶段二（Stage 2）**：构建 **Evidence Token Mask**，结合关键区域 Attention 轨迹设计 **Evidence Persistence Reward**，减少长链推理中的视觉证据遗忘。

### 3. 效果验证

围绕实验曲线解读、多指标对比、趋势变化及消融实验分析等任务，提升模型对关键视觉证据的提取与持续利用能力，降低因遗漏图表信息导致的错误结论风险。

在图表理解及通用视觉推理的 8 项公开基准上评测，平均得分由 **44.3 提升至 52.1（+7.8）**。

| 评测 | 基线 Qwen2.5-VL | 本方法 |
| --- | --- | --- |
| 8 项基准平均分 | 44.3 | **52.1** |

各基准的详细分数见 `src/eval/evaluate.py` 与评测输出的 JSON 报告。

## 目录结构

```text
SciChart-Evidence-VL/
├── configs/                  # 训练与评测配置
│   ├── base.yaml             # 模型 / 数据 / 训练公共配置
│   ├── grpo_stage1.yaml      # GRPO 阶段一：KL 一致性 + 反事实分离
│   ├── grpo_stage2.yaml      # GRPO 阶段二：证据持久性奖励
│   └── eval.yaml             # 8 项基准评测配置
├── data/
│   └── README.md             # 数据说明（真实数据不入库）
├── scripts/
│   ├── build_counterfactual.py  # 构造 EP / EA 反事实配对
│   ├── train.py              # 双阶段 GRPO 训练入口
│   └── evaluate.py           # 8 项基准评测入口
└── src/
    ├── data/
    │   ├── schema.py         # ChartQASample / CounterfactualPair 等数据结构
    │   └── counterfactual.py # 程序化修改绘图代码，生成 EP / EA 视图
    ├── model/
    │   └── qwen_vl.py        # Qwen2.5-VL 加载与 Attention 轨迹提取
    ├── training/
    │   ├── evidence.py       # Evidence Token Mask 与 Evidence Persistence Reward
    │   ├── grpo.py           # 双阶段 GRPO 损失
    │   └── rewards.py        # 答案正确性 / 证据一致性等奖励
    └── eval/
        └── evaluate.py       # 基准评测与汇总
```

## 快速开始

```bash
# 1. 创建环境（Python 3.10+）
pip install -r requirements.txt

# 2. 构造反事实配对（示例：不依赖真实数据）
python scripts/build_counterfactual.py \
    --input data/examples/sample.jsonl \
    --output data/counterfactual_pairs.jsonl

# 3. 训练
python scripts/train.py --config configs/grpo_stage1.yaml
python scripts/train.py --config configs/grpo_stage2.yaml

# 4. 评测
python scripts/evaluate.py --config configs/eval.yaml
```

## 配置说明

- `configs/base.yaml`：模型名称、最大图像 token 数、学习率、批量大小等公共项；
- `configs/grpo_stage1.yaml`：继承 base，开启 KL 一致性与反事实分离损失；
- `configs/grpo_stage2.yaml`：继承 base，开启 Evidence Token Mask 与 Evidence Persistence Reward；
- `configs/eval.yaml`：8 项基准列表、生成参数与输出路径。

## 数据说明

本仓库**不包含任何真实数据文件**。数据构建流程的接口与数据结构见 `src/data/`，真实数据（约 5 万条绘图代码问答、约 2 万条高难度样本、约 2 万组 EP/EA 配对）由数据构建流水线在本地生成，见 `data/README.md`。

## 许可

本项目代码以 MIT License 发布（见 `LICENSE`）。
