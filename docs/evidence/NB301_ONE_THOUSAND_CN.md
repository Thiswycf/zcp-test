# NAS-Bench-301 一千候选正式验收证据

## 判定

NAS-Bench-301 surrogate 的锁定候选协议已完成：从 seed 2026 生成的 11,221 个 DARTS corpus 中，
按 operation/topology 特征分层抽取 1,000 个候选。22 个代理单 seed 共 22,000 行，全部成功；
核心 11 个代理在 seed 2026/2027/2028 共 33,000 行，全部成功、无重复任务键。机器可读摘要见
[`nb301_one_thousand_summary.json`](nb301_one_thousand_summary.json)。大型 JSONL、CSV、图片和 HTML
只保存在外部 audit root，不进入 Git。

本判定是 **deterministic surrogate association**，不是候选架构的真实训练精度，也不是 DARTS
搜索空间的完整穷举。H1 整体仍等待 TransNAS-Bench-101 和 ViT-Bench-101，不能据此标记完成。

## 抽样与真值协议

- sample manifest：`$ZCP_TEST_AUDIT_ROOT/sampling/nb301-1000-seed2026.json`；SHA-256：
  `aff2d3ea7b2cb38954d59c37a4a807137ccdd9a57772565a2a82621bb6eb53b5`。
- population 11,221、sample 1,000、strata 8,628、4 个 shard；抽样 seed 为 2026。
- 11,221 只是锁定的确定性 genotype 生成 corpus，用于给 1,000 抽样定义可复现分母；它不是有限
  DARTS 搜索空间的“标准答案全集”。
- benchmark/space：`nasbench301_surrogate@1.0` / `darts`；目标为 CIFAR-10 test accuracy、repeat
  mean、`with_noise=False`。
- 每个 ZCP 使用真实 CIFAR-10 dataset batch；模型 fidelity 为 `reference_model`，权重协议为
  `independent_scratch`。surrogate target 与 ZCP 模型权重不是同一 fidelity。

## 22 代理单 seed

有效集位于 `$ZCP_TEST_AUDIT_ROOT/effective/nb301-1k-seed2026/scores.jsonl`，SHA-256 为
`0c27d4bb4011a0be8c6c5a626f41bcfce4b05c84dd66bd2a0ba0ee4d5226c8d1`。1,000 个架构 ×
22 个代理严格得到 22,000 行：成功 22,000、失败 0、重复键 0。

主组件与 surrogate accuracy 的 Spearman 最高项为：AZ-NAS expressivity/NASWOT `0.480561`、
Params `0.425214`、FLOPs `0.392485`。Near 与 SWAP 为常数输出，相关性保持未定义，不伪造为 0。
负相关代理仍按原结果报告；这些数值只描述当前 surrogate/corpus/input/model/proxy 版本下的排序关联。

AZ-NAS 的主组件与 NASWOT 同秩，TE-NAS 的主组件与 SynFlow 同秩，MeCo 与 MeCo-opt 相同；名称不同
不构成独立证据，rank aggregation 不得重复计权。测试 split 不能学习融合权重，因此
`rank_aggregation.csv` 只有表头是正确的 validation-only 保护结果。

## 核心 11 代理三 seed

核心代理为 `params,flops,gradnorm,jacob_cov,naswot,synflow,zen,zico,meco,te_nas,az_nas`。
合并文件为 `$ZCP_TEST_AUDIT_ROOT/effective/nb301-1k-core11-three-seed/scores.jsonl`，SHA-256：
`feecdedbddc6d12230833d62000751a3938864d2b2b07c609dc222fd624ae172`。

- 每个 seed 为 11,000 行，总计 33,000；成功 33,000、失败 0、重复键 0。
- 三个 seed 的 1,000 个 canonical architecture ID 完全一致；deterministic surrogate target 对每个
  architecture 跨 seed 完全一致。
- 按 architecture ID inner join 的 33 个跨 seed proxy-pair 均覆盖 1,000 个架构。
- Params/FLOPs 的跨 seed score Spearman 均为 `1.0`；SynFlow/TE-NAS 均值 `0.999989`；
  AZ-NAS/NASWOT `0.997046`；Jacobian covariance 均值 `-0.007574`，说明当前协议下跨 seed 排名不稳。
- sample-size convergence 包含 `10,25,50,100,250,500,1000` 七个点、11 代理、3 seed，共 231 行；
  小样本相关性与完整 1,000 点差异明显，不得以 n=10 或 n=100 代替正式结果。

标准公共命令生成的报告目录为
`$ZCP_TEST_AUDIT_ROOT/reports/nb301-1k-core11-three-seed`。其中：

- `correlations.csv`：33 行，SHA-256
  `d8fd03d958f60d6e3f2f72e805c0afcbdcfe3cd70d204e52d5652e7c545fdb3f`；
- `sensitivity_rank.csv`：33 行，SHA-256
  `8b270df4787090a349f5ac226058005998e6dc540ba49955ef408c81f63c7be4`；
- `sample_size_convergence.csv`：231 行，SHA-256
  `685c62a390fd25a5069062e37732f8e03c11ae24b8cb834c1e7b462212bb88ac`；
- `index.html`：同时链接 CSV、PNG 和 SVG，SHA-256
  `cd2530b5857fc7d067074965d1f97dfe20e87e29de2a6b1381ea548033309d87`。

## DARTS operation × topology 定制研究

单 seed 的 DARTS 专属报告位于 `$ZCP_TEST_AUDIT_ROOT/reports/nb301-1k-darts`，含 1,000 条
architecture、352,000 条 edge-level 长表、7,524 条条件相关性和 2,156 条
operation-topology interaction。normal/reduction cell、operation、source node、edge span 与 topology
特征分别保留。该分解是根据 DARTS 编码特点的项目推广，不是论文因果结论；所有统计仍是
surrogate association。

通用多代理报告位于 `$ZCP_TEST_AUDIT_ROOT/reports/nb301-1k-multi-proxy`，含 22 条主组件相关性、
231 条 proxy-pair correlation、693 条 proxy-pair top-k 和 693 条 complementarity。低互相关本身
不等于互补，只有 validation holdout 上的检索/融合增益才能支持组合结论。

## Params/FLOPs 方向修复

旧 artifact 把资源约束 `minimize` 错当成 accuracy 相关性方向并取负。当前 reader 保持 raw JSONL
不可变，将 legacy `version=1` 显式派生为 Params `count-v2` / FLOPs `thop-v2`，使用
`direction=maximize` 计算原始规模—accuracy 关联，并另存 `resource_direction=minimize`。三 seed
相关表用 `legacy_direction_migrated_count` 记录每组迁移行数，不再把新旧 shard 拆成重复 heatmap 行。

## 运行时与实现边界

- 运行栈为 `xgboost==2.1.4`、`nasbench301==0.3`。官方旧二进制 ensemble 会发出跨版本兼容警告；
  已验证锁定运行时内重复预测逐值一致，但尚无旧版官方 XGBoost golden 对照，因此 fidelity 只能写
  `deterministic_on_locked_runtime`。
- NAS-Bench-301 是 surrogate prediction；不得称为 1,000 个候选均完成真实训练。
- TE-NAS `portable-v2` 与 AZ-NAS `portable-v1` 是可移植/组合近似，不能冒充官方完整实现。
- 所有结论仅覆盖该 1,000 分层样本、当前 deterministic surrogate、CIFAR-10 input、模型实现、代理
  版本和三个 seed，不外推到完整 DARTS 空间、noisy surrogate、真实训练或其他数据协议。

## 复现命令

```bash
zcp-test analyze sensitivity \
  --scores "$ZCP_TEST_AUDIT_ROOT/effective/nb301-1k-core11-three-seed/scores.jsonl" \
  --parameter seed \
  --sample-sizes 10 25 50 100 250 500 1000 \
  --bootstrap-samples 100 \
  --top-k 10 50 100 \
  --title "NB301 deterministic surrogate: core 11, three-seed stability" \
  --output "$ZCP_TEST_AUDIT_ROOT/reports/nb301-1k-core11-three-seed"
```
