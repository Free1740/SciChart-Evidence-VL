# 方法：视觉证据驱动的科研图表理解

本文详细介绍 SciChart-Evidence-VL 的方法。项目基于 Qwen2.5-VL，面向科研文献阅读、实验结果分析及模型性能对比等场景。

## 1. 问题

多模态大模型在复杂科研图表理解中存在三个核心问题：

### 1.1 语言先验干扰

模型在回答图表问题时，往往依赖语言先验与表面线索（例如常见的实验结论表述、图例文字的出现频率），而不是图表中真实的视觉证据。即使关键曲线与问题结论矛盾，模型也可能给出符合语言习惯、但与图表不符的答案。

### 1.2 关键视觉信息遗漏

科研图表信息密度高，关键证据往往只占图表的一小部分：一根关键曲线、一个消融实验的柱子、一个局部数据点。模型在编码整张图表时容易遗漏这些关键视觉信息，从而得出错误结论。

### 1.3 长链推理中的证据遗忘

在多步推理中，模型一开始可能注意到了关键证据，但随着推理链变长，对关键视觉证据的注意力逐渐下降，最终的结论不再以最初的证据为依据。推理越长，证据遗忘越明显。

## 2. 数据构建与反事实增强

### 2.1 从绘图代码出发的可追溯数据

数据的基础是约 5 万条可追溯绘图代码的图表问答数据。每条样本都对应一段真实的绘图代码（例如 matplotlib 代码），因此图表中的每个元素——曲线、柱子、标注——都能在代码中找到来源，关键证据也可以被精确定位。

### 2.2 Teacher 评审、答案校验与难度筛选

原始数据经过三步筛选，只保留约 2 万条高难度样本：

- **Teacher 评审**：由更强的教师模型评审样本的问题质量、证据充分性与标注合理性；
- **答案校验**：校验答案是否确实能从图表证据中得出，剔除答案与图表不一致的样本；
- **难度筛选**：保留基线模型容易答错、需要真正理解图表才能答对的高难度样本。

### 2.3 EP / EA 反事实配对

对保留下来的样本，通过程序化修改绘图代码构造反事实视图，形成约 2 万组配对样本：

- **EP（Evidence Preserved，关键证据保留）**：保留关键证据，只做与答案无关的扰动，例如改变一条非关键曲线的颜色。正确答案在 EP 视图下应当不变。
- **EA（Evidence Ablated，关键证据删除）**：删除关键曲线、实验指标或局部数据。正确答案在 EA 视图下不再成立，模型应当改变答案或降低置信度。

这种配对模拟了科研图表中关键曲线、实验指标及局部数据的变化，迫使模型学会辨别哪些视觉信息是真正有效的实验信息，而不是依赖语言先验或表面线索。

实现见 `src/data/counterfactual.py`（`CounterfactualBuilder`、`make_evidence_preserved`、`make_evidence_ablated`）和 `src/data/schema.py`（`ChartQASample`、`CounterfactualPair`）。

## 3. 视觉证据驱动的强化学习

训练采用双阶段 GRPO（Group Relative Policy Optimization）优化策略，基座模型为 Qwen2.5-VL。

### 3.1 阶段一：一致性与分离

阶段一的目标是让模型的预测以视觉证据为依据，而不是语言先验：

- **KL 一致性约束**：强化原图与 EP 图的预测一致性。既然 EP 视图保留了关键证据，模型在两张图上的预测分布应当接近。
- **反事实分离目标**：拉大原图与 EA 图的预测分布差异。既然 EA 视图删除了关键证据，模型在两张图上的预测应当不同。

两者结合，抑制模型对语言先验和表面线索的依赖。相关奖励见 `src/training/rewards.py` 中的 `evidence_consistency_reward`，损失组合见 `src/training/grpo.py`。

### 3.2 阶段二：证据持久性

阶段二解决长链推理中的证据遗忘问题：

- **Evidence Token Mask**：标记属于关键证据区域的图像 token（`src/training/evidence.py` 中的 `build_evidence_token_mask`）。在解码时可以对这些 token 的 logit 做增强（`apply_evidence_token_mask`），让解码过程保持以证据为依据。
- **Evidence Persistence Reward**：沿着关键区域的 Attention 轨迹，衡量推理每一步对证据 token 的注意力占比（`evidence_attention_mass`），并对推理后半段相对前半段的注意力下降施加遗忘惩罚（`evidence_persistence_reward`，`forgetting_penalty` 参数）。奖励鼓励模型在整个推理链中持续利用关键视觉证据。

组合奖励（答案正确性、证据一致性、格式、证据持久性）见 `src/training/rewards.py` 中的 `combined_reward`。

### 3.3 训练流程

```bash
# 阶段一：KL 一致性 + 反事实分离
python scripts/train.py --config configs/grpo_stage1.yaml

# 阶段二：Evidence Token Mask + Evidence Persistence Reward
python scripts/train.py --config configs/grpo_stage2.yaml
```

## 4. 评测

### 4.1 评测范围

在图表理解及通用视觉推理的 8 项公开基准上进行评测：ChartQA、PlotQA、ChartBench、CharXiv、FigureQA、DVQA、ChartX、MathVista（其中 7 项为图表理解基准，1 项为通用视觉推理基准）。评测入口见 `src/eval/evaluate.py` 与 `scripts/evaluate.py`：

```bash
python scripts/evaluate.py --config configs/eval.yaml
```

### 4.2 结果

8 项基准的平均得分由 **44.3 提升至 52.1（+7.8）**。这里只报告已知的平均分；各基准的具体分数以评测输出的 JSON 报告为准，本文不编造未给出的单项分数。

这一提升验证了方法对复杂科研图表推理的优化效果：围绕实验曲线解读、多指标对比、趋势变化及消融实验分析等任务，模型对关键视觉证据的提取与持续利用能力得到提升，因遗漏图表信息导致错误结论的风险随之降低。

## 5. 代码结构速览

| 模块 | 作用 |
| --- | --- |
| `src/data/schema.py` | `ChartQASample`、`EvidenceRegion`、`CounterfactualPair` |
| `src/data/counterfactual.py` | 程序化修改绘图代码，生成 EP / EA 视图 |
| `src/model/qwen_vl.py` | Qwen2.5-VL 加载与关键区域 Attention 轨迹提取 |
| `src/training/grpo.py` | 双阶段 GRPO 损失 |
| `src/training/evidence.py` | Evidence Token Mask 与 Evidence Persistence Reward |
| `src/training/rewards.py` | 答案正确性、证据一致性、格式等奖励 |
| `src/eval/evaluate.py` | 8 项基准评测与汇总 |
