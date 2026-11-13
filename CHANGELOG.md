# 更新日志

## Unreleased

- 新增数据构建流水线：`src/data/teacher_review.py`（Teacher 评审）、
  `src/data/answer_check.py`（答案校验）、`src/data/difficulty.py`
  （难度评分与高难度筛选），以及入口 `scripts/build_dataset.py`。
- 新增示例：`examples/sample_qa.jsonl`（2 条合成样本）、
  `examples/quickstart.py`（难度评分 + EP/EA 配对演示）和
  `examples/README.md`。
- 新增 `CONTRIBUTING.md`、`CHANGELOG.md` 和 `docs/EVALUATION.md`。
- 新增开发工具：`Dockerfile`、`.dockerignore`、
  `.pre-commit-config.yaml`。

## 0.1.0 — 2026-11-13

- 初始框架：基于 Qwen2.5-VL 的视觉证据驱动框架，面向科研图表理解。
- 双阶段 GRPO：阶段一 KL 一致性约束 + 反事实分离目标；阶段二
  Evidence Token Mask + Evidence Persistence Reward。
- EP/EA 反事实数据构建：通过程序化修改绘图代码，构造关键证据保留
  （EP）与关键证据删除（EA）配对样本。
- 评测：8 项公开基准，平均得分由 44.3 提升至 52.1（+7.8）。
- MIT LICENSE。
- 单元测试：奖励函数、证据持久性、反事实生成（26 passed、1 skipped）。
- CI 工作流：编译检查、单元测试、反事实 demo。
- 方法文档：`docs/METHOD.md`。
