# 严格 1% 数据协议修复证据

日期：2026-07-31。

## 协议规则

- 每个 train/valid split 独立计算全局目标 `round(N * 0.01)`，再按类别理想配额的最大余数分配；固定 seed 用于余数相同时的可复现破同分。
- 不再逐类执行 `max(1, round(...))`。当类别数大于目标条数时，只覆盖可由全局目标容纳的类别，不把 ImageNet-1k validation 的 500 条扩大为 1000 条。
- 若 `round(N * fraction) == 0`，协议拒绝运行，不把空目标静默扩大为 1 条。
- acceptance 入口只接受浮点值精确等于 `0.01`；`0.0100001`、`0.010001` 与 `0.0099999` 均拒绝。
- checkpoint 的 `run_identity` 同时记录 `acceptance_protocol=one_percent_data_protocol` 与 `data_fraction=0.01`，恢复时参与严格身份比较。

## 规范 split 断言

| 数据集 | split | 原始条数 | 严格 1% 条数 | 类别配额断言 |
|---|---:|---:|---:|---|
| CIFAR-10 | train | 50,000 | 500 | 每类 50 |
| CIFAR-10 | valid | 10,000 | 100 | 每类 10 |
| CIFAR-100 | train | 50,000 | 500 | 每类 5 |
| CIFAR-100 | valid | 10,000 | 100 | 每类 1 |
| ImageNet-1k | train | 1,281,167 | 12,812 | 每个已覆盖类别 12 或 13 |
| ImageNet-1k | valid | 50,000 | 500 | 固定 seed 选择 500 类，每类 1 条 |

对应测试同时断言相同 seed 的索引完全一致，并检查 `last.pt` 保存上述协议身份。
