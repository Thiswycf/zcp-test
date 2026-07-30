# NATS-SSS 1% 跨数据集正式验收证据

本文记录 NATS-SSS 在同一份 1% 分层架构样本上的 CIFAR-10-valid、CIFAR-100 与
ImageNet16-120 对比。大型 `scores.jsonl`、CSV、HTML 和图像继续保存在 audit 目录；仓库只提交
本说明和机器可读摘要
[`nats_sss_cross_dataset_summary.json`](nats_sss_cross_dataset_summary.json)。

## 结论

- CIFAR-100：328 个架构 × 22 个代理 = 7,216 行，`ok=7,216`、失败 0、unsupported 0、
  duplicate key 0。
- ImageNet16-120：328 × 22 = 7,216 行，`ok=7,216`、失败 0、unsupported 0、duplicate key 0。
- CIFAR-10-valid 复用已验收的同一份 328 架构、22 代理、seed 2026 分片；三数据集共 12 个
  `scores.jsonl` 分片进入统一 size study。
- Dataset-specific 与 target-only 已由独立表表达，不再共用一个含糊的“cross-dataset”数字。

这是一项 **完成的 1% 范围验收**，不是全体 32,768 个 NATS-SSS 架构的结论，也不是多输入或
多初始化 seed 稳定性验收。

## 原始运行验收

原始验收摘要：

```text
$AUDIT/preflight/nats_sss_cross_dataset_raw_acceptance.json
SHA-256 96f83e82ddda9d12a2123c8bee3d13b8ef3074fb0ba2f4dabe5f4b7efc02e707
```

共同协议：

| 字段 | 值 |
|---|---|
| benchmark | `nats_sss@1.0` / `nats_size` |
| population | 32,768 |
| sample | 328，`proportional_feature_stratified`，seed 2026 |
| sample manifest SHA | `07767985afbad7d498acf062620aea5ef7b66b2bfc8b3db3fea3a0fc768e1992` |
| target | validation accuracy，90 epoch，repeat `mean`，maximize |
| evaluation/input/初始化 seed | 2026，单 seed |

输入 fingerprint 分别为：

- CIFAR-10-valid：`5de04e6a61157306b00dc80b673de420ecd45a843b1564f78acaa8bbfab1ceaf`；
- CIFAR-100：`ede1d9035544056b23859a9f1339b8c7990026b7c81bd1c74f041511b536240c`；
- ImageNet16-120：`b3df614ef5496e9eea73b5ce8b76cf52eb11c64ad0186b7eaae7219a637e10fe`。

三个 fingerprint 不同，证明 dataset-specific ZCP 是按各自真实输入重新计算；target-only 表则固定
source dataset 的 score/fingerprint，仅按 canonical architecture ID 连接另一数据集 target。

## 三数据集分析

正式命令使用 12 个分片，而不是只向命令传入父目录：

```bash
AUDIT=/path/to/audit
mapfile -t SCORES < <(find \
  "$AUDIT/h1-nats-sss-seed2026" \
  "$AUDIT/h1-nats-sss-cifar100-seed2026" \
  "$AUDIT/h1-nats-sss-imagenet16-seed2026" \
  -name scores.jsonl -type f | sort)
test "${#SCORES[@]}" -eq 12
zcp-test analyze benchmark \
  --scores "${SCORES[@]}" \
  --benchmark nats_sss --view size \
  --output "$AUDIT/h1-nats-sss-cross-dataset-analysis"
```

报告索引：

```text
$AUDIT/h1-nats-sss-cross-dataset-analysis/study.json
SHA-256 0bc3734543ac77b80c40008285abbe4938668fa052620127b3c58a3a8638c954
```

四张跨数据集表：

| 表 | 行数 | 作用 |
|---|---:|---|
| `dataset_proxy_target_matrix.csv` | 594 | source dataset ZCP × target dataset truth；对角线是 dataset-specific，非对角线是 target-only。 |
| `proxy_dataset_stability.csv` | 186 | 同一代理在不同输入数据集间的 score 排名稳定性。 |
| `target_dataset_transfer.csv` | 9 | 三对 target truth 的 Spearman/Kendall/Pearson。 |
| `controlled_proxy_target_transfer.csv` | 1,188 | 控制 size 特征后的 source-proxy→target 相关性。 |

表 SHA-256 已逐项写入机器可读摘要。报告目录还包含常规 size 架构、stage、strata、相关性和图形
产物；本文不复制这些大型派生产物。

## 关键结果

同一 328 架构上的 target rank Spearman：

| Target pair | n | Spearman |
|---|---:|---:|
| ImageNet16-120 ↔ CIFAR-10-valid | 328 | 0.869439 |
| ImageNet16-120 ↔ CIFAR-100 | 328 | 0.924531 |
| CIFAR-10-valid ↔ CIFAR-100 | 328 | 0.772225 |

Dataset-specific 对角线的最高 Spearman：

| Dataset | 代理/组件 | 版本 | n | Spearman |
|---|---|---|---:|---:|
| CIFAR-10-valid | Params / `score` | `count-v2` | 328 | 0.876103 |
| CIFAR-100 | SynFlow / `score` | `double-v2` | 328 | 0.832093 |
| CIFAR-100 | TE-NAS / `synflow` | `portable-v2` | 328 | 0.832093 |
| ImageNet16-120 | Params / `score` | `count-v2` | 328 | 0.835932 |

TE-NAS 的结果仍按 `portable-v2` 组件命名，不能称为官方完整 TE-NAS。Params、SynFlow 等高相关
也不自动证明代理捕获了因果机制；应结合 `controlled_proxy_target_transfer.csv` 和 size-controlled
结果判断宽度混杂。

## 解释边界

1. 样本是 32,768 个架构中的 328 个确定性分层样本；不能将相关系数直接外推到全搜索空间。
2. 每个数据集只覆盖单个输入/初始化 seed 2026；没有证明多输入 batch 或多初始化 seed 稳定性。
3. Dataset-specific 对角线允许 ZCP 随输入数据集变化；target-only 非对角线固定 source score，二者
   回答不同问题，不应合并排名。
4. 所有相关性均为观察性统计；target rank 一致、控制变量相关或代理高相关都不是因果结论。
5. 本验收完成的是 NATS-SSS 跨数据集 1% 扩展，不改变其他 benchmark 尚未完成的验收状态。
