# Benchmark 定制研究操作手册

通用相关性只能回答“某个 ZCP 与某个真值是否同序”。不同 benchmark 还需要检查不同的结构
因素：NAS-Bench-101 的训练预算、NAS-Bench-201/NATS-TSS 的固定边操作、NATS-SSS 的逐 stage
通道、NAS-Bench-301 的 DARTS operation×topology、TransNAS-Bench-101 的任务与架构因素，
以及 ViT-Bench-101 的深度、维度、head 和 MLP ratio。

统一入口为：

```bash
zcp-test analyze benchmark \
  --scores /path/to/run/scores.jsonl \
  --benchmark auto \
  --view auto \
  --output /path/to/run/reports/benchmark
```

`--benchmark auto` 要求过滤后只有一个 `benchmark_id`。自动视图映射如下：

| Benchmark | 自动视图 | 核心问题 |
|---|---|---|
| `nasbench101` | `budget` | ZCP 对 4/12/36/108 epoch 排名的敏感性 |
| `nasbench201` | `topology` | 六条固定边上的操作分布与结构偏置 |
| `nats_tss` | `topology` | 与 NB201 共享 topology，但真值来源保持 NATS-TSS 身份 |
| `nats_sss` | `size` | 五个 stage 的 channel 与整体宽度结构 |
| `nasbench301_surrogate` | `darts` | normal/reduce cell 的操作、来源节点和位置交互 |
| `transnasbench101` | `transfer` | micro/macro 内不同任务的 ZCP transfer |
| `vitbench101` | `architecture` | AutoFormer/PiT 的结构特征相关性 |

## 1. 先生成协议完整的 score

新运行写入 score schema `2.1`，新增：

- `target_direction`：真值是 `maximize` 还是 `minimize`；
- `target_epoch_budget`：真值训练预算，而不是 ZCP 计算预算；
- `target_seed`、`target_seed_reduction`：benchmark repeat 及聚合方式；
- `benchmark_variant`：`micro/macro` 或 `autoformer_main/autoformer_ext/pit`；
- `benchmark_protocol`：native API 或安全转换格式的协议身份。

例：NAS-Bench-101 的 108-epoch validation accuracy：

```bash
zcp-test evaluate \
  --benchmark nasbench101 \
  --benchmark-version full \
  --proxies er,naswot,synflow \
  --count 100 \
  --dataset cifar10 \
  --target-metric final_accuracy \
  --target-split valid \
  --target-direction maximize \
  --epoch-budget 108 \
  --metric-seed-reduction mean \
  --input-source dataset \
  --data-root /path/to/data/cifar10 \
  --catalog ~/.config/zcp-test/data.json \
  --output /path/to/runs/nb101
```

NAS-Bench-101 和安全 JSONL benchmark 在一个指标匹配多个 epoch budget 时，缺少
`--epoch-budget` 会明确失败，不再把不同预算静默平均。NB201/NATS 使用 adapter 的发布默认
预算时，解析后的有效预算仍会写进 JSONL。

`--target-direction auto` 会把名称包含 `loss/error/time/latency` 的指标视为 minimize；正式研究
建议显式传值，不依赖名称推断。

## 2. NAS-Bench-101 多预算研究

预算视图需要读取安全转换后的 benchmark，以同一组 architecture ID 查询多个预算真值：

```bash
zcp-test analyze benchmark \
  --scores /path/to/shard-0/scores.jsonl /path/to/shard-1/scores.jsonl \
           /path/to/shard-2/scores.jsonl /path/to/shard-3/scores.jsonl \
  --benchmark nasbench101 \
  --view budget \
  --benchmark-path /path/to/data/nasbench101/converted/full/manifest.json \
  --benchmark-version full \
  --budgets 4 12 36 108 \
  --study-dataset cifar10 \
  --study-split valid \
  --study-metric final_accuracy \
  --seed-reduction mean \
  --study-target-direction maximize \
  --top-k 5 10 50 \
  --bootstrap-samples 1000 \
  --output /path/to/reports/nb101-budget
```

产物：

- `detailed.csv`：architecture × proxy/component × budget 的显式 join；
- `score_coverage.csv`：每个 proxy/component/evaluation seed 的总架构数、成功数、失败数和覆盖率；
- `correlations.csv`：每个预算独立的 Spearman、Kendall tau-b、Pearson 和 bootstrap CI；
- `rank_stability.csv`：预算两两之间的真值 rank correlation 和 top-k Jaccard；
- `top_k_retrieval.csv`：每个 ZCP 在各预算的 precision@k、Jaccard、选中集合真值和 regret；
- `architecture_features.csv`：顶点数、边数、最长输入输出路径，以及 3×3 conv、1×1 conv、
  3×3 max-pool 数量；
- `feature_strata.csv`：按上述结构特征取值分层后的 ZCP 与真值均值/中位数；
- `structure_controlled_correlations.csv`：固定六项结构计数后，对组内去均值残差计算的
  Spearman 与 Kendall tau-b；
- `edit_neighbors.csv`：当前样本内合法 canonical DAG 的一编辑邻居，不对全体 423,624 个
  架构做平方复杂度枚举；
- `neighborhood_differences.csv`：每对邻居的方向修正后 ZCP 差值与预算真值差值；
- `neighborhood_correlations.csv`：operation/edge 邻居对数量、差值相关性和方向一致率；
- `budget_correlation.png/svg`：相关性随预算变化曲线；
- `budget_top_k_retrieval.png/svg`：不同预算下的 top-k 检索能力；
- `budget_structure_controlled.png/svg`：固定结构计数后的相关性随预算变化；
- `budget_neighborhood_agreement.png/svg`：一编辑邻域排序方向一致率随预算变化；
- `index.html`：静态预览。

多个 `scores.jsonl` 会按科学协议合并：`run_id` 和读取时生成的 `source_run` 只用于溯源，不能把
四个 shard 拆成四个相关性；evaluation `seed` 仍是协议字段，因此不同初始化 seed 独立报告。
失败和非有限分数不会被填成 0；`score_coverage.csv` 与 `correlations.csv` 中的 `failed_count`、
`coverage` 必须与原始调用数一致。若只研究 ER 的主分量，可显式添加 `--component mean`；不要在
全代理命令中添加该过滤器，因为多数代理的主分量名为 `score` 或其他具名组件。

一编辑邻居的定义是：canonical 邻接矩阵相同且仅一个中间操作不同，或 canonical 操作序列相同
且严格上三角邻接仅一条边不同。它是当前分层样本内的局部敏感性分析；缺少邻居对时只能报告
覆盖不足，不能外推到完整搜索空间。结构控制与邻域分析属于基于 NB101 特征的项目推广，不应写成
原论文已经报告的因果结论。

解释时应区分两种现象：ZCP 与 4 epoch 高相关但与 108 epoch 低相关，可能表示 ZCP 更接近早期
优化速度；4 与 108 epoch 真值本身不稳定，则不应把下降全部归因于 ZCP。

## 3. NAS-Bench-201 与 NATS-TSS topology

```bash
zcp-test analyze benchmark \
  --scores /path/to/nb201/scores.jsonl \
  --benchmark nasbench201 \
  --view topology \
  --output /path/to/reports/nb201-topology
```

NATS-TSS 只需把 benchmark 改为 `nats_tss`。两者可以共享 topology 解析器，但不得合并真值或
benchmark 标签。

产物：

- `architectures.csv`：六条固定边的 one-hot 操作特征和各操作计数；
- `edges.csv`：每个 architecture 的 `0->1`、`0->2`、`1->2`、`0->3`、`1->3`、`2->3`；
- `operations.csv`：操作的 edge/architecture 覆盖率；
- `correlations.csv`：操作计数、操作比例和六条边 one-hot 特征分别与真值/ZCP 的相关性；
- `operation_effects.csv`：按 `proxy/component × edge × operation` 分层的样本数、条件均值、
  中位数及相对该 edge 总体均值的偏移；
- `topology_operations.png/svg`：操作分布。
- `topology_feature_correlations.png/svg`：结构特征相关性预览。
- `matched_pairs.csv`、`matched_pair_summary.csv`：固定其余五条边、仅一条边操作不同的局部
  对比、方向一致率与 ties；这是观察性 matched contrast，不是因果替换实验。

`operation_effects.csv` 可以直接检查某个 ZCP 是否系统性偏好特定边上的 `skip_connect`、卷积或
`none`，无需手工 join。但它仍是观察性条件统计：操作之间存在强组合约束，不能把 delta 解释为
独立替换该操作后的因果增益。建议同时查看 `sample_count`，并在相同 dataset、split、budget、seed
聚合协议内比较 NB201 或 NATS-TSS；两者的表不能直接拼接。

## 4. NATS-SSS size

```bash
zcp-test analyze benchmark \
  --scores /path/to/nats-sss/scores.jsonl \
  --benchmark nats_sss \
  --view size \
  --output /path/to/reports/nats-size
```

产物：

- `architectures.csv`：`stage_0_channel...stage_4_channel`、总/均值/范围/标准差、首尾宽度、扩张比；
- `stages.csv`：architecture × stage × channel 长表；
- `summary.csv`：全部 size 特征的描述统计；
- `correlations.csv`：总宽度、扩张比、各 stage channel/delta 与真值/ZCP 的 Spearman、Kendall
  tau-b 和 Pearson；
- `stage_sensitivity.csv`：只保留逐 stage channel，便于比较早期与后期 stage 的敏感性；
- `size_stages.png/svg`：逐 stage channel 分布。
- `size_feature_correlations.png/svg`：宽度特征相关性预览。
- `size_controlled_correlations.csv`：控制总 channel sum 后的 partial Spearman；
- `size_strata.csv`：总宽度分位区间内的 ZCP—真值相关性，用于检查规模混杂。

先比较 `outcome=score` 与 `outcome=target`：若某个宽度特征与 ZCP 高相关、与真值低相关，ZCP
可能主要编码模型规模；若两者都高，还需在 `size_channel_sum` 分位区间内重复通用相关性，排除
规模混杂。stage channel 是离散变量且 ties 较多，优先解读 Spearman/Kendall，不应只看 Pearson。
不要把 NATS-SSS 的 90-epoch 结果与 TSS 的 200-epoch 结果放在同一相关性组。

## 5. NAS-Bench-301 DARTS 联合研究

```bash
zcp-test analyze benchmark \
  --scores /path/to/nb301/scores.jsonl \
  --benchmark nasbench301_surrogate \
  --view darts \
  --component score \
  --output /path/to/reports/nb301-darts
```

产物包括 `architectures.csv`、`edges.csv`、`correlations.csv`、
`operation_topology_interactions.csv` 和交互热力图。特征分开记录 normal/reduce cell 的 operation
count、source node、cell input/intermediate edge、edge span、normal-reduce balance 和 cell
interaction。条件效应按 `cell × node × source_class × operation` 计算，并保留样本量。

NB301 真值是 surrogate prediction；`with_noise=False` 与 noisy repeat 必须分协议。当前视图支持
deterministic surrogate 结果；如果输入含不同 surrogate seed/noise 协议，必须先过滤或分别报告。
NAS-Bench-Suite-Zero 和 MeCo 对 NB301 有直接 ZCP 研究依据；本项目的 operation×topology
条件分解是依据 DARTS 编码特点的推广，不是上述论文原表复刻。

典型实例：

```bash
zcp-test analyze benchmark \
  --scores examples/studies/data/nasbench301_darts.jsonl \
  --benchmark nasbench301_surrogate \
  --output /tmp/zcp-test-examples/nb301
```

## 6. TransNAS-Bench-101 任务与架构

先为多个任务分别 evaluate，保证相同 space、architecture ID 集合、metric 和 split。然后：

```bash
zcp-test analyze benchmark \
  --scores /path/to/class-object/scores.jsonl /path/to/segment/scores.jsonl \
  --benchmark transnasbench101 \
  --view transfer \
  --component score \
  --target-metric test_top1 \
  --benchmark-variant micro \
  --output /path/to/reports/transnas-transfer
```

产物：`task_quality.csv`、`task_transfer.csv`、`space_summary.csv`、`architecture_features.csv`、
`architecture_factors.csv`、`feature_correlations.csv`、`factor_effects.csv` 和 task-transfer heatmap。
micro 编码拆分六条 edge 的 `none/skip/conv1x1/conv3x3`；macro 编码拆分 module 的 normal、
channel×2、resolution/2 和联合缩放。micro 与 macro
必须分开；同一 task/architecture/proxy/component 存在重复行时命令会失败，要求先明确选择
metric、split、budget 或 run，而不是任意取第一条。

搜索和聚合权重只能由 validation 协议确定。test task/metric 只能用于最终报告。

典型实例：

```bash
zcp-test analyze benchmark \
  --scores examples/studies/data/transnas_tasks.jsonl \
  --benchmark transnasbench101 \
  --output /tmp/zcp-test-examples/tnb101
```

## 7. ViT-Bench-101 结构研究

```bash
zcp-test analyze benchmark \
  --scores /path/to/vit/scores.jsonl \
  --benchmark vitbench101 \
  --view architecture \
  --benchmark-variant autoformer_main \
  --component score \
  --output /path/to/reports/vit-architecture
```

`features.csv` 包含总 depth、stage 数、hidden/base dimension、head dimension、MLP ratio、
MLP width 以及明确标为 proxy 的 attention/MLP block parameter 指标；`layers.csv` 展开每层/stage；
`correlations.csv` 分别报告特征与 target、特征与 ZCP score 的相关性。AutoFormer 的
`hidden_dim` 是总 embedding width；PiT 发布编码中的 stage embedding width 为
`base_dim × stage_num_heads`，因此不能把 PiT 的 `base_dim` 直接当作各 stage 总宽度。

`autoformer_main`、`autoformer_ext` 和 `pit` 必须独立报告。vanilla、KD 与 inherited-supernet
accuracy 是不同协议；跨指标比较只能按 architecture ID 取交集，并报告交集样本量。

典型实例：

```bash
zcp-test analyze benchmark \
  --scores examples/studies/data/vit_autoformer.jsonl \
  --benchmark vitbench101 \
  --output /tmp/zcp-test-examples/vit
```

AZ-NAS 对 AutoFormer/ImageNet 属于同搜索空间的部分直接依据，但不是 ViT-Bench-101 发布 GT
切片实验；PiT 尚无对应直接依据，当前分析属于按 stage/downsampling 特点推广。

## 8. NAS-Bench-101/NATS 典型实例

```bash
zcp-test analyze benchmark \
  --scores examples/studies/data/nasbench101_budget.jsonl \
  --benchmark nasbench101 \
  --benchmark-path /path/to/data/nasbench101/converted/full \
  --budgets 4 12 36 108 --top-k 2 \
  --output /tmp/zcp-test-examples/nb101

zcp-test analyze benchmark \
  --scores examples/studies/data/nats_tss_topology.jsonl \
  --benchmark nats_tss --output /tmp/zcp-test-examples/nats-tss

zcp-test analyze benchmark \
  --scores examples/studies/data/nats_sss_size.jsonl \
  --benchmark nats_sss --output /tmp/zcp-test-examples/nats-sss
```

## 9. 过滤与防止协议混合

统一入口支持：

```text
--component
--dataset
--target-metric
--target-split
--target-epoch-budget
--benchmark-variant
```

通用 `correlation_table` 与 top-k 报告也会自动按 benchmark、variant、space、dataset、metric、
split、target direction、epoch budget 和 seed reduction 分组。不同协议不会再被 pooled 成一个
相关性数值。

旧 schema 仍可做通用分析；若缺少专属视图需要的 `architecture` 或协议字段，定制分析会明确
失败。不要为了得到图表而补造 budget、task 或 direction。

## 10. 验收与清理

```bash
pytest -q tests/test_proxy_studies.py tests/test_analysis.py \
  tests/test_benchmark_budget.py tests/test_benchmark_studies.py
ruff check src/zcp_test/reporting tests/test_proxy_studies.py \
  tests/test_benchmark_budget.py tests/test_benchmark_studies.py
```

验收还会执行 `examples/studies/data/*.jsonl` 的七个典型命令。仓库保留输入 fixture、命令和预期
表名；`/tmp/zcp-test-examples`、pytest cache、`__pycache__`、临时 PNG/HTML 均删除，不进入 Git。

此外，NAS-Bench-201 topology 视图已用真实 benchmark 真值和真实 CIFAR 输入完成 20 架构工作流
验收，生成 20 条 architecture、120 条 edge、5 类 operation、720 条 feature correlation 和
90 条 operation effect。精简证据见
[`evidence/E2_E3_NB201_REAL_CN.md`](evidence/E2_E3_NB201_REAL_CN.md)。由于连续样本未形成合法
matched pair，matched-pair 表为空；这不是错误，也不能用来支持 matched-contrast 结论。其余
benchmark 仍需分别生成真实 score → 专属表/图证据，不能由 NB201 结果外推。

在上述工作流 smoke 之外，NB201 seed 2026 已完成正式 feature-stratified 1% × 22 ZCP：157 个
架构、3,454 行、3,451 成功、3 条明确失败、0 个重复架构—代理键。修复把 `run_id` 错当协议的
shard grouping 缺陷后，专属 topology 输出为 157 条 architecture、942 条 edge、5 类 operation、
6,720 条 correlation、840 条 operation effect、588 条 matched pair 和 504 条 matched-pair
summary。完整摘要见
[`evidence/NB201_ONE_PERCENT_22ZCP_CN.md`](evidence/NB201_ONE_PERCENT_22ZCP_CN.md)。H1 仍只能标为
核心 11 代理的另外两个 seed 现已补齐，三 seed 稳定性见
[`evidence/NB201_CORE_THREE_SEED_CN.md`](evidence/NB201_CORE_THREE_SEED_CN.md)。当前状态更新为
**“NB201 既定 seed 协议完成”**；`params`/`flops` 的负号来自
`minimize → negated` 方向转换，资源方向与原始规模—精度关联仍须分开解释；疑似相同算法结果不能
用于独立性结论，且该结果绝不能外推为 NATS-TSS 证据。

NATS-TSS 随后已用独立 API 和真值完成相同最低规模：22 代理单 seed、核心 11 代理三 seed、
topology operation effect 与 matched-pair 报告均已生成。157 个共同 topology 中有 31 个 NATS
与 NB201 target 不同，直接证明共享 codec 不能替代独立查询。详见
[`evidence/NATS_TSS_ONE_PERCENT_CN.md`](evidence/NATS_TSS_ONE_PERCENT_CN.md)。H1 整体仍进行中。

NATS-SSS 已完成 CIFAR-10-valid/90-epoch 的 328 架构最低规模、22 代理单 seed 与核心三 seed。
修复 shard grouping 后，size 视图以完整 n=328 生成 stage、总通道、stage sensitivity、
size-controlled correlation 和 strata 表。详见
[`evidence/NATS_SSS_ONE_PERCENT_CN.md`](evidence/NATS_SSS_ONE_PERCENT_CN.md)。跨 CIFAR-100 与
ImageNet16-120 的 rank transfer 尚未完成，不能由当前结果外推。

## 11. 常见错误

- `multiple epoch budgets`：evaluate 或查询缺少 `--epoch-budget`。
- `auto requires exactly one benchmark_id`：输入混有多个 benchmark；先用参数或拆分文件过滤。
- `map to multiple specifications`：同一 architecture ID 对应不同 spec，数据不可直接合并。
- `requires one row per ...`：TransNAS 输入有重复协议；显式选择 metric/split/budget/run。
- `unsupported space`：ViT 行的 `search_space_id` 不是 `autoformer` 或 `pit`。
- 只有一个样本或常数特征时相关性为空，这是统计不可辨识，不应填成 0。
- `precision_at_k` 使用实际可用样本数截断 k，并在 `effective_k` 中记录；比较不同 run 时应先保证
  architecture 候选集一致。
- `mean_regret` 基于方向调整后的 benchmark 真值，越小越好；它不是训练 loss，也不能跨 metric
  直接比较绝对值。
