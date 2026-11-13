# 评测说明

## 评测基准

本项目在图表理解及通用视觉推理的 8 项公开基准上进行评测
（基准列表与 `src/eval/evaluate.py` 中的 `BENCHMARKS` 一致）：

| 基准 | 类型 |
| --- | --- |
| ChartQA | 图表问答 |
| PlotQA | 图表问答 |
| ChartBench | 图表理解 |
| CharXiv | 科研图表理解 |
| FigureQA | 图表问答 |
| DVQA | 图表问答 |
| ChartX | 图表理解 |
| MathVista | 通用视觉推理 |

## 评测流程

1. 准备各基准的数据文件，放在评测数据目录下，命名为
   `<基准名>.jsonl`，每行一条 `{image, question, answer}` 记录。
2. 运行评测入口：

```bash
python scripts/evaluate.py --config configs/eval.yaml
```

3. 评测脚本调用 `src/eval/evaluate.py` 中的 `run_evaluation`：
   对每个基准用模型预测答案并按答案正确性计分，汇总平均分，
   写入 JSON 报告（包含各基准分数、平均分、基线平均分和提升）。

## 结果

8 项基准的平均得分由基线 Qwen2.5-VL 的 **44.3 提升至 52.1
（+7.8）**。

各基准的单项分数如下（单项分数以评测输出的 JSON 报告为准，
此处不另行编造）：

| 基准 | 分数 |
| --- | --- |
| ChartQA | 待补充 / 见评测输出 JSON |
| PlotQA | 待补充 / 见评测输出 JSON |
| ChartBench | 待补充 / 见评测输出 JSON |
| CharXiv | 待补充 / 见评测输出 JSON |
| FigureQA | 待补充 / 见评测输出 JSON |
| DVQA | 待补充 / 见评测输出 JSON |
| ChartX | 待补充 / 见评测输出 JSON |
| MathVista | 待补充 / 见评测输出 JSON |

## 注意事项

- 评测数据不入库：各基准的数据文件由使用者自行准备，放在本地
  评测数据目录，不要提交到仓库。
- 评测关注的任务类型包括实验曲线解读、多指标对比、趋势变化
  及消融实验分析，重点考察模型对关键视觉证据的提取与持续
  利用能力。
