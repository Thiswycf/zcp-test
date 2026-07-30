# NB201 核心 11 代理三种子验收

## 结论

NAS-Bench-201 的核心 11 个代理已在同一份 feature-stratified 1% manifest 上完成 seed
`2026/2027/2028`：157 个架构、5,181 个“seed × 架构 × 代理”键，5,172 条成功、9 条失败、
0 个重复键。该结果只关闭 **NB201 核心代理三 seed** 子项；H1 整体仍进行中，因为 NB101、
NATS-TSS/SSS、NB301、TransNAS-Bench-101 和 ViT-Bench-101 尚未完成各自协议。

机器可读摘要见
[`nb201_core_three_seed_summary.json`](nb201_core_three_seed_summary.json)。合并后的外部审计文件
SHA-256 为 `66ef0f353810f26472684ad70452f698a64b1ae2ee64209c2aad7302bebb38d3`；抽样 manifest
SHA-256 仍为 `9b9e7b0e8b7e59b76cee386cf6221bdac3f9b463a9a4729f68faffcd671391bc`。

## 协议与 GPU

- benchmark：`nasbench201` v1.1；空间：`nb201_topology`。
- 真值：`cifar10-valid / valid-accuracy / valid / 200 epoch / repeat mean`。
- 核心代理：`params,flops,gradnorm,jacob_cov,naswot,synflow,zen,zico,meco,te_nas,az_nas`。
- seed 2027 与 2028 各拆为四个互斥 shard；八个新增 run 均为 `completed`。
- 每个 seed 使用四张自动选择且加锁的 4090/4090D；manifest 均记录 GPU UUID、PCI Bus ID、
  `CUDA_DEVICE_ORDER=PCI_BUS_ID` 和内部逻辑设备 `cuda:0`。

## 主组件结果

| 代理 | 主组件 | 三 seed Spearman | 均值 ± 标准差 | 平均跨 seed score Spearman |
|---|---|---|---:|---:|
| `az_nas` | `expressivity` | 0.606714 / 0.599469 / 0.708571 | 0.638251 ± 0.049811 | 0.947487 |
| `flops` | `score` | 0.619446 / 0.619446 / 0.619446 | 0.619446 ± 0 | 1.000000 |
| `gradnorm` | `score` | 0.348334 / 0.323346 / 0.365820 | 0.345833 ± 0.017430 | 0.954752 |
| `jacob_cov` | `score` | 0.609405 / 0.519034 / 0.562087 | 0.563509 ± 0.036908 | 0.509573 |
| `meco` | `score` | 0.143196 / 0.031893 / 0.170927 | 0.115339 ± 0.060081 | 0.492219 |
| `naswot` | `score` | 0.606714 / 0.599469 / 0.708571 | 0.638251 ± 0.049811 | 0.947487 |
| `params` | `score` | 0.654632 / 0.654632 / 0.654632 | 0.654632 ± 0 | 1.000000 |
| `synflow` | `score` | 0.341958 / 0.343977 / 0.341276 | 0.342404 ± 0.001147 | 0.912331 |
| `te_nas` | `synflow` | 0.329766 / 0.331444 / 0.328937 | 0.330049 ± 0.001043 | 0.910519 |
| `zen` | `score` | 0.616712 / 0.605313 / 0.623739 | 0.615254 ± 0.007593 | 0.937376 |
| `zico` | `score` | 0.471291 / 0.471849 / 0.438842 | 0.460661 ± 0.015430 | 0.977346 |

这里的第一组 Spearman 是“代理与 NB201 真值”的 accuracy 相关性；最后一列是同一代理在不同
初始化/输入 seed 下的逐架构 score 排名一致性。二者回答不同问题，不能互相替代。Params/FLOPs
已按每个 seed 的原始资源值重新计算并使用 `identity`，不是对旧负号作盲目翻转。旧 schema 错把
资源优化方向 `minimize` 用作 accuracy 方向；当前只读派生语义为 `direction=maximize`、
`resource_direction=minimize`，并将 legacy `version=1` 映射为 Params `count-v2`、FLOPs `thop-v2`。
原始验收 scores 保持不变。

`jacob_cov` 与 `meco` 的平均跨 seed 排名一致性分别只有约 0.510 和 0.492，明显低于其他随机性
代理；这只是本协议上的稳定性观察，不应外推成跨 benchmark 结论。`az_nas` 的主组件仍是
portable approximation 的 NASWOT expressivity，因此它与 `naswot` 数值一致并非独立复现。

## 失败与覆盖率修复

三个 seed 都在 benchmark index `3943`、架构
`nb201_topology:839da408774c5a50b88c` 上产生 `az_nas/naswot/te_nas` 非有限失败，每个相关组件
的有效覆盖率为 `156/157 = 0.993631`。失败没有删除或改写。

真实三种子报告同时暴露并修复了一个程序缺陷：CLI 曾在进入 bundle 前过滤 failed 行，使
`correlations.csv` 错写为 coverage 1.0。现在 CLI 与 bundle 都把完整调用记录传给
`correlation_table`，相关系数只使用有限成功对，但 `total_count/failed_count/invalid_count/coverage`
保留真实调用分母。多组件代理的一次调用失败会进入其各组件覆盖率分母；失败路径也保留声明的
`primary_component`。

## 边界

- 这是 1% 分层样本，不是 NB201 全空间或论文数值复现。
- NB201 与 NATS-TSS 即使共享 topology codec，也必须分别运行、分别查询真值、分别报告。
- 22 名称单 seed与核心 11 代理三 seed是两个口径；alias 和 approximation 不能算作独立论文公式。
- 大型 JSONL、PNG/SVG 和 HTML 保留在外部审计目录，不进入 Git；仓库只保存摘要、命令语义和哈希。
