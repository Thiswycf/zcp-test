# NB101 1% 正式验收证据

## 判定

NAS-Bench-101 `full` 搜索空间的 1% 既定协议验收通过：seed 2026 的 22 个代理有效集共
93,214 条，全部成功；核心 11 个代理在 seed 2026/2027/2028 的合并集共 139,821 条，全部成功。
机器可读摘要见 [`nb101_one_percent_summary.json`](nb101_one_percent_summary.json)。本证据只提交摘要、
SHA-256 和报告目录引用，不复制外部 audit 目录中的大型 JSONL、CSV、图片或 HTML 产物。

## 抽样与协议

- sample manifest：`$ZCP_TEST_AUDIT_ROOT/sampling/nb101-1pct-seed2026.json`；
  SHA-256 为 `e54cba029c74197037f1268f1689ec2b198261641133fccd6acda4a89c67c347`。
- 抽样为 seed 2026 的比例特征分层无放回抽样：population 423,624，sample 4,237，250 个
  strata；验收范围严格限定为该 manifest 选中的架构。
- benchmark/space：`nasbench101@full` / `nb101_dag`；数据集和真值为 CIFAR-10、valid、
  `final_accuracy`、maximize。
- 预算研究使用 4/12/36/108 epoch，并分别对 benchmark repeats 使用 `mean`、`min`、`max`
  聚合；三种聚合不可互换，正文结论必须带上对应聚合协议。
- 代理输入来自 dataset；模型 fidelity 为 `reference_topology_pytorch_port`。它是依据 NB101 DAG
  构建的 PyTorch 参考拓扑端口，不是 NAS-Bench-101 原始训练 checkpoint 的逐权重复现。

## 22 代理单 seed 与失败闭环

旧 22 代理原始运行中，SynFlow `v1` 与 TE-NAS `portable-v1` 合计有 2,224 条失败。原始产物保持
不可变；有效集明确排除这两个旧版本的全部行，改用 SynFlow `double-v2` 与 TE-NAS
`portable-v2` 的完整补跑。补跑应有/成功为 `8,474/8,474`，即两个代理各覆盖 4,237 个样本。

替换后的 seed 2026 有效集覆盖 `4,237 × 22 = 93,214` 个稳定任务键，成功 93,214、失败 0、
重复任务键 0。外部有效文件为
`$ZCP_TEST_AUDIT_ROOT/effective/nb101-1pct-seed2026/scores.jsonl`，SHA-256 为
`d45116653a2556216190f73cd7ff11160d137ad5c3ddf882465ec05406fb93e1`。

这里的 TE-NAS `portable-v2` 是本仓库的可移植近似实现；**不得称为官方完整 TE-NAS**，也不得用
该标签暗示已逐项复现官方训练、扰动、停止判据或完整实现细节。

## 核心 11 代理三 seed

核心代理为 `az_nas`、`flops`、`gradnorm`、`jacob_cov`、`meco`、`naswot`、`params`、
`synflow`、`te_nas`、`zen`、`zico`。seed 2026/2027/2028 的有效合并集覆盖
`4,237 × 11 × 3 = 139,821` 个任务键，成功 139,821、失败 0、重复任务键 0。

三 seed 报告中每个 seed 的 11 个主组件均覆盖 4,237 个架构；proxy–proxy CSV 使用 55 个无序
非对角 pair 的紧凑长表，加入对角线并镜像后才是 11×11 热力图，不应把 55 行误判为缺失矩阵。
`az_nas/expressivity` 与 `naswot/score` 在当前 portable 实现中严格同秩，`te_nas/synflow` 与
`synflow/score` 近乎同秩；这些组件复用关系禁止在 rank aggregation 中作为相互独立证据重复计权。

外部有效文件为
`$ZCP_TEST_AUDIT_ROOT/effective/nb101-1pct-core11-three-seed/scores.jsonl`，SHA-256 为
`794535f4e6a33e8ce3f13c29c8eca2d27386f42207025ef75fe024ddef20ab0b`。三 seed 结果用于当前
协议下的稳定性检查，不把 seed 2026 的 22 代理覆盖误写成全部 22 代理均已完成三 seed。

## 预算与结构分析

预算报告对 4/12/36/108 epoch 分别计算代理—真值相关性、预算间 rank stability、top-k retrieval、
结构控制相关性和邻域一致性。结构控制使用 vertices、edges、longest-path depth、`conv3` 数、
`conv1` 数和 max-pool 数；它只能降低这些已观测结构特征的混杂，不能证明因果关系。

样本内共找到 306 对一编辑邻居（214 对 operation edit、92 对 edge edit）。这些配对仅来自 4,237
个抽中架构之间恰好仍落在样本内的一编辑关系，不是全空间邻域枚举，也不能代表未抽中邻居。

外部报告目录：

- 22 代理：`$ZCP_TEST_AUDIT_ROOT/reports/nb101-1pct-multi-proxy`；
- 核心三 seed：`$ZCP_TEST_AUDIT_ROOT/reports/nb101-1pct-core11-three-seed`；
- repeat mean：`$ZCP_TEST_AUDIT_ROOT/reports/nb101-1pct-budget-mean`；
- repeat min：`$ZCP_TEST_AUDIT_ROOT/reports/nb101-1pct-budget-min`；
- repeat max：`$ZCP_TEST_AUDIT_ROOT/reports/nb101-1pct-budget-max`。

`ZCP_TEST_AUDIT_ROOT` 是机器本地、不得提交 Git 的审计产物根目录；报告目录只用于审计引用，不纳入
本次小型证据提交。

## Fidelity、近似与外推边界

- benchmark 真值来自 NAS-Bench-101 已记录的训练重复与预算；ZCP 则在
  `reference_topology_pytorch_port`、当前 dataset input 和代理版本下计算。两者 fidelity 不同。
- `portable-*`、`double-v2` 和其他版本标签是本仓库实现协议的一部分，不自动等同于原论文或官方
  实现；尤其 TE-NAS `portable-v2` 只能称为可移植近似版本。
- 部分代理存在大量 ties，相关结论需同时查看 Kendall tau-b、唯一分数数和 ties，不得只报告
  Spearman。样本规模收敛曲线必须包含完整 4,237 样本点，最大只到 100 的旧图不构成收敛证据。
- 统计只支持该 1% 分层样本、当前 CIFAR-10 valid 真值、预算、repeat 聚合、输入、模型端口和 seed
  协议。结构控制与一编辑邻居分析也受相同范围约束。
- **不得把本验收结果外推到全部 423,624 个架构、完整搜索过程、其他数据集、其他 fidelity、其他
  代理实现或未运行 seed。** 1% 抽样验收不是全搜索空间穷举验收。
