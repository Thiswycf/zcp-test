# 分析、可视化与监控

## 一键报告

```bash
zcp-test report bundle RUN_A RUN_B --output reports/compare
```

根据输入自动生成 CSV、PNG、SVG 和静态 `index.html`。原始 JSONL 始终是唯一真源。

## 常用研究

```bash
# 相关性、bootstrap CI、scatter、rank、heatmap、top-k
zcp-test analyze correlation --scores RUN/scores.jsonl --output reports/correlation

# top-k overlap、验证集 rank aggregation、跨数据集相关性、耗时/显存 Pareto
zcp-test analyze compare --scores RUN/scores.jsonl --output reports/compare

# seed/batch/input/source 等敏感性和 sample-size convergence
zcp-test analyze sensitivity --scores RUNS.jsonl --parameter seed --output reports/seed

# 进化搜索 best/mean/分位数/diversity/cache/budget 曲线
zcp-test analyze search --source RUN/search.jsonl --output RUN/reports/search

# 训练 loss、top-1/top-5、LR 与耗时曲线
zcp-test analyze training --source RUN/training.jsonl --output RUN/reports/training
```

schema 2 的原始 `scores.jsonl` 每个架构/代理一行；分析 CSV 会按组件展开，因而 ER 等多组件代理会增加明细行。这是派生视图，不是重复评估。

## 实时监控

```bash
zcp-test monitor RUN --interval 5
zcp-test monitor RUN --once
```

监控器只读 JSONL，容忍尚未写完的最后一行，原子刷新 `RUN/reports/monitor.html`。HTML 每 5 秒自动刷新，不需要 Jupyter、TensorBoard 或后台 Web 服务。

