# 分析、可视化与监控

需要执行跨 benchmark 的 1% 正式相关性验收时，先阅读
[`ONE_PERCENT_ACCEPTANCE_CN.md`](ONE_PERCENT_ACCEPTANCE_CN.md)，使用持久化 feature-stratified
manifest，而不是连续 `--start/--count` 代替随机/分层样本。

当前真实高成本实例包括 NB201 与 NATS-TSS，各自保留独立 manifest、真值和报告。NATS-TSS 的
22 代理、核心三 seed、proxy–proxy、top-k 和 topology 典型输出见
[`evidence/NATS_TSS_ONE_PERCENT_CN.md`](evidence/NATS_TSS_ONE_PERCENT_CN.md)；不得把 NB201 图表
改标题后当作 NATS-TSS 结果。

NATS-SSS 的 size 专属实例见
[`evidence/NATS_SSS_ONE_PERCENT_CN.md`](evidence/NATS_SSS_ONE_PERCENT_CN.md)。多文件输入会合并
互斥 shard，但不同 evaluation seed 必须保持独立；`run_id` 只属于 provenance，不能作为科学协议
分组，否则会把总体样本量错误缩小为单 shard。

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
zcp-test analyze sensitivity --scores RUNS.jsonl --parameter seed \
  --sample-sizes 10 25 50 100 250 500 1000 2000 4237 \
  --title "NB101 core proxies · three seeds" \
  --output reports/seed

# 进化搜索 best/mean/分位数/diversity/cache/budget 曲线
zcp-test analyze search --source RUN/search.jsonl --output RUN/reports/search

# 训练 loss、top-1/top-5、LR 与耗时曲线
zcp-test analyze training --source RUN/training.jsonl --output RUN/reports/training
```

`sensitivity` 同时输出两类不同证据：`sample_size_convergence.csv` 研究样本数变化；
`sensitivity_rank.csv` 按 canonical architecture ID 对齐两个 seed/condition，报告 score 的
Spearman、Kendall tau-b、Pearson、共同样本数和覆盖率，并生成 `sensitivity_rank.png/svg`。
`sensitivity.png/svg` 仅展示每个条件下的原始 score 均值/标准差；不同代理量纲不同，不能把该图
当成跨代理排名稳定性证据。分片的 `run_id/source_run` 只作 provenance，不拆分科学协议；同一条件
内重复 architecture×proxy 键仍明确报错。

`correlation`、`compare` 和 `sensitivity` 现在 fail closed：必须至少存在一条成功记录，且
`architecture_id/proxy_id/component/score/target_value` 不能全空；非有限 score/真值不能组成
报告。`sensitivity --parameter seed` 还要求至少两个非空 seed 值。验证失败时不会先创建空报告目录。
`--sample-sizes` 应覆盖小样本区间并包含本次研究的完整架构数；若最大值仅为 100，不能据此声明
数千架构规模已经收敛。请求值超过可用架构数时使用全部可用架构，并在 `sample_count` 中记录实际数量。

schema 2 的原始 `scores.jsonl` 每个架构/代理一行；reader 会在内存中展开具名组件，但
`analyze correlation|compare|sensitivity`、`analyze benchmark` 和 `report bundle` 默认只保留
每行声明的 `primary_component`。因此 ER 默认使用 `mean`，不会同时把 `sum` 当成第二个 ZCP。
只有显式传入 `--component sum` 时才研究该辅助分量。旧 schema 若没有
`primary_component` 无法可靠推断主分量，reader 会保留其全部组件；此时应先检查字段并显式指定
`--component`。组件展开是派生视图，不是重复评估。

validation-only rank aggregation 默认排除记录中声明了 `proxy_alias_of` 的显式别名，避免同一公式
重复计权。组合代理若主组件复用了其他代理但未声明为完整别名，工具不会仅凭数值相同自动删除；此时
必须结合 `proxy matrix`、实现 fidelity 和 proxy–proxy 热力图人工审计后再解释聚合结果。

### 多协议 × 多 ZCP 研究

`compare` 现在接受一个或多个 JSONL，并严格按 benchmark/version/variant/protocol、搜索空间、
dataset、metric、split、budget、target seed reduction、代理版本、输入来源与指纹、模型保真度和
初始化 seed 分组：

```bash
zcp-test analyze compare \
  --scores RUN_A/scores.jsonl RUN_B/scores.jsonl \
  --top-k 1 5 10 50 \
  --bootstrap-samples 2000 \
  --output reports/multi-proxy
```

主要产物：

- `correlations.csv`：proxy-target Spearman/Kendall/Pearson 与 bootstrap CI；
- `proxy_target_long.csv`、`proxy_target_matrix.csv`：不折叠协议的长表和热力图矩阵；
- `proxy_proxy_correlations.csv`：按 `architecture_id` 交集计算的代理互相关、共同样本数和覆盖率；
- `proxy_proxy_top_k.csv`：代理两两 top-k Jaccard、交集和并集；
- `complementarity.csv`：真值 rank 残差相关、top-k union recall、边际增益和 validation holdout
  rank-fusion 增益；
- `proxy_cost_pareto.csv`：每个协议内的相关性—耗时—峰值显存 Pareto；
- `proxy_target_protocol_heatmap.*`、`proxy_proxy_heatmap.*`：静态 PNG/SVG。

`correlations.csv` 不再用空系数掩盖统计原因，并为每个代理组件写出：

- `total_count`、`successful_count`、`failed_count`、`sample_count`、`invalid_count` 与 `coverage`；
- `target_unique_count`、`score_unique_count`、`target_tied_observations`、
  `score_tied_observations`；
- `correlation_status`：`ok`、`insufficient_samples`、`constant_target`、
  `constant_score` 或 `constant_target_and_score`；
- `score_direction` / `target_direction` 以及对应的 `*_direction_transform`。其中
  `negated` 表示在计算 rank/correlation 前按 `minimize` 语义取负，`identity` 表示未反向。

例如，代理在全部架构上输出同一个值时，相关系数是未定义值，CSV 会记录
`correlation_status=constant_score`，而不是伪造为 `0`。同一协议、代理和组件中出现重复
`architecture_id` 会直接报错，禁止后写记录静默覆盖前一条记录。`failed_count` 按代理调用统计；
默认主分量报告中，一次失败只计入该代理声明的主分量。显式研究辅助组件时，同一失败会进入该
组件的覆盖率分母。

`params` 与 `flops` 同时具有两种不同语义：作为 accuracy 预测基线时
`direction=maximize`，按文献直接研究原始规模—精度关联；作为资源约束时
`resource_direction=minimize`。旧 `version=1` 记录曾错误地把资源方向用于相关性，reader 会显式
写入 `direction_migration=legacy-resource-direction-to-accuracy-v2`，保留
`reported_proxy_version=1`，并派生为 Params `count-v2` 或 FLOPs `thop-v2` 后恢复原始方向；raw 文件
不改写。迁移 provenance 不会把同一科学协议拆成多个相关性分组；汇总表通过
`legacy_direction_migrated_count`、`direction_migrations` 和
`legacy_reported_proxy_versions` 记录混合新旧 shard 的只读迁移情况。
代理注册名不同也不代表算法独立：CSV 中的
`proxy_alias_of` 与 `proxy_implementation_fidelity` 用于识别显式 alias 和未核验移植。

不得把 `proxy_proxy` 低相关直接称为互补：两个随机代理也可能低相关。只有当 union recall 或严格
validation holdout 上的 fusion 增益为正时，才有“候选互补”的观察性证据。融合权重只在明确
标记为 `valid|validation|val` 的真值上学习；输入 test 真值时 `fusion_status` 为
`unsupported_target_split`。validation/evaluation 子集分别计算 rank transform，evaluation
代理分布不会反向改变已选权重。相关性、残差和条件均值都不是因果效应。

### 典型可复现实例

仓库保留小型确定性输入，不保留可重建的临时 PNG/缓存：

```bash
zcp-test analyze compare \
  --scores examples/studies/data/generic_multi_proxy.jsonl \
  --top-k 1 3 \
  --bootstrap-samples 100 \
  --output /tmp/zcp-test-examples/generic
```

验收时应看到 2 个 budget 协议、3 个代理、3 组 proxy-pair；不得把两个 budget 平均成一列。

真实 NB201/CIFAR 输入的小样本工作流证据、原始 JSONL SHA-256、显式 failure 和完整命令见
[`evidence/E2_E3_NB201_REAL_CN.md`](evidence/E2_E3_NB201_REAL_CN.md)。该 20 架构连续样本只验证
通用分析接线，不是 1% 科学样本或论文相关性结果。

NB201 核心 11 代理在同一 1% 分层样本上的三 seed 稳定性证据见
[`evidence/NB201_CORE_THREE_SEED_CN.md`](evidence/NB201_CORE_THREE_SEED_CN.md)。该真实运行发现 CLI
曾在 bundle 前过滤 failed 行；现已修复为“有限成功对用于系数、全部调用用于覆盖率分母”。因此
验收时应同时检查 `sample_count` 与 `total_count/failed_count/coverage`，不能只看系数列。

## 论文依据与推广边界

- NASWOT、Zero-Cost Proxies for Lightweight NAS：全局 rank、top-region、局部邻域、初始化和
  batch 稳健性；
- NAS-Bench-Suite-Zero：跨 benchmark/task、代理互相关、结构偏置、成本与代理组合；
- ZiCo、MeCo：NB101、NATS-TSS/SSS、TNB101、NB301 上的 Spearman/Kendall 与搜索验证；
- AZ-NAS：多代理非线性 rank aggregation、相关性—成本比较和 AutoFormer 搜索。

原始链接与“论文直接实验/同空间部分直接/本项目推广”的逐项登记见
`docs/RESEARCH_EVIDENCE_CN.md`。项目不会把 PiT、NATS 的 matched-pair 或 DARTS
operation×topology 分析伪称为论文原实验。

## 实时监控

```bash
zcp-test monitor RUN --interval 5
zcp-test monitor RUN --once
```

监控器只读 JSONL，容忍尚未写完的最后一行，原子刷新 `RUN/reports/monitor.html`。HTML 每 5 秒自动刷新，不需要 Jupyter、TensorBoard 或后台 Web 服务。
