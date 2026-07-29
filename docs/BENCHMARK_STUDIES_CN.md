# Benchmark 定制研究操作手册

通用相关性只能回答“某个 ZCP 与某个真值是否同序”。不同 benchmark 还需要检查不同的结构
因素：NAS-Bench-101 的训练预算、NAS-Bench-201/NATS-TSS 的固定边操作、NATS-SSS 的逐 stage
通道、TransNAS-Bench-101 的任务迁移，以及 ViT-Bench-101 的深度、维度、head 和 MLP ratio。

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
  --scores /path/to/run/scores.jsonl \
  --benchmark nasbench101 \
  --view budget \
  --component mean \
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
- `correlations.csv`：每个预算独立的 Spearman、Kendall tau-b、Pearson 和 bootstrap CI；
- `rank_stability.csv`：预算两两之间的真值 rank correlation 和 top-k Jaccard；
- `budget_correlation.png/svg`：相关性随预算变化曲线；
- `index.html`：静态预览。

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
- `topology_operations.png/svg`：操作分布。

该视图用于发现样本分布和结构偏置，不应把条件均值解释为某个操作的因果贡献。需要比较 ZCP
偏置时，可将 `architectures.csv` 与 `detailed scores` 按 `architecture_id` join，再按 edge/op
分层运行通用相关性。

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
- `size_stages.png/svg`：逐 stage channel 分布。

建议在 size quantile 内分别计算相关性，检查 ZCP 是否只是参数量或宽度的替代指标。不要把
NATS-SSS 的 90-epoch 结果与 TSS 的 200-epoch 结果放在同一相关性组。

## 5. TransNAS-Bench-101 任务迁移

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

产物：`task_quality.csv`、`task_transfer.csv`、`space_summary.csv` 和静态图。micro 与 macro
必须分开；同一 task/architecture/proxy/component 存在重复行时命令会失败，要求先明确选择
metric、split、budget 或 run，而不是任意取第一条。

搜索和聚合权重只能由 validation 协议确定。test task/metric 只能用于最终报告。

## 6. ViT-Bench-101 结构研究

```bash
zcp-test analyze benchmark \
  --scores /path/to/vit/scores.jsonl \
  --benchmark vitbench101 \
  --view architecture \
  --benchmark-variant autoformer_main \
  --component score \
  --output /path/to/reports/vit-architecture
```

`features.csv` 包含总 depth、stage 数、hidden/base dimension、head 与 MLP ratio 的均值/范围/
标准差和派生宽度；`correlations.csv` 分别报告特征与 target、特征与 ZCP score 的相关性。

`autoformer_main`、`autoformer_ext` 和 `pit` 必须独立报告。vanilla、KD 与 inherited-supernet
accuracy 是不同协议；跨指标比较只能按 architecture ID 取交集，并报告交集样本量。

## 7. 过滤与防止协议混合

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

## 8. 常见错误

- `multiple epoch budgets`：evaluate 或查询缺少 `--epoch-budget`。
- `auto requires exactly one benchmark_id`：输入混有多个 benchmark；先用参数或拆分文件过滤。
- `map to multiple specifications`：同一 architecture ID 对应不同 spec，数据不可直接合并。
- `requires one row per ...`：TransNAS 输入有重复协议；显式选择 metric/split/budget/run。
- `unsupported space`：ViT 行的 `search_space_id` 不是 `autoformer` 或 `pit`。
- 只有一个样本或常数特征时相关性为空，这是统计不可辨识，不应填成 0。

