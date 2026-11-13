# 贡献指南

欢迎为 SciChart-Evidence-VL 贡献代码、文档和测试！请按以下流程操作。

## 开发流程

1. Fork 本仓库，在本地 clone。
2. 新建分支：`git checkout -b feature/<简短描述>` 或 `fix/<简短描述>`。
3. 只改与本次任务相关的文件；不要把真实数据、密钥或大文件提交进来。
4. 本地检查通过后再提交 Pull Request，PR 描述里写清楚改了什么、为什么。

## 代码风格

- Python 3.10+，使用类型注解；行宽建议不超过 100 字符。
- 模块、函数和类都要有简短的 docstring，说明用途和关键参数。
- 新增的公开函数请同时补单元测试（见下）。
- 保持接口稳定：`src/data/schema.py` 里的数据结构和 `src/training/rewards.py`
  等核心函数的签名如需改动，请先在 PR 里说明影响。

## 测试要求

- 测试放在 `tests/`，用 pytest 编写，命名为 `test_*.py`。
- 运行全部测试：

```bash
pip install pytest
pytest tests/ -v
```

- 缺少重型依赖（如 torch）时，相关测试应使用
  `pytest.importorskip` 跳过，而不是失败。
- 新功能至少覆盖主路径和一个边界情况（空输入、缺失证据、非法参数等）。

## 提交规范

- 一个提交只做一件事，提交信息用英文、祈使句开头，例如：
  `Add unit tests for rewards`、`Fix EP builder for missing evidence`。
- 不要把 `__pycache__`、`.pytest_cache`、真实数据文件（`data/*.jsonl`）
  提交进来；`.gitignore` 已排除它们。
- 涉及 `.github/workflows/` 的文件无法通过 GitHub App 的 API 推送，
  请通过网页端添加或修改。

## 数据说明

本仓库不包含任何真实数据。示例数据只放在 `examples/`，且必须明确
标注为合成示例。真实数据由数据构建流水线在本地生成，见
`scripts/build_dataset.py` 和 `data/README.md`。
