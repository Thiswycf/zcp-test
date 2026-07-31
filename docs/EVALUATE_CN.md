# ZCP 评估

## 真实数据默认协议

`evaluate` 默认 `--input-source dataset`。CIFAR-10/100 和 ImageNet 路径可通过 `--data-root` 指定，也可在 data catalog 中注册为 `dataset_cifar10`、`dataset_cifar100`、`dataset_imagenet1k`。

```bash
zcp-test evaluate \
  --space nb201_topology \
  --proxies er,naswot,synflow \
  --count 10 \
  --data-root /path/to/cifar10
```

预期终端摘要为 10 个架构、30 次代理调用、30 行 `scores.jsonl`。ER 的记录示意：

```json
{"proxy_id":"er","score":1.23,"primary_component":"mean","components":{"mean":1.23,"sum":8.61}}
```

`score` 是排序和默认相关性使用的规范值；`components` 保留完整研究数据。需要研究 ER sum 时使用：

```bash
zcp-test correlate \
  --scores RUN/scores.jsonl \
  --targets /path/to/targets.jsonl \
  --output RUN/correlations-sum.jsonl \
  --target-field accuracy --target-direction maximize --component sum
zcp-test analyze correlation --scores RUN/scores.jsonl --component sum --output REPORT
```

兼容入口 `correlate` 的 target 文件至少需要一行一个
`{"architecture_id":"...","accuracy":...}`；它按 canonical ID inner join、拒绝重复 ID，并把
`direction=minimize` 的代理和 `--target-direction minimize` 的真值统一转换为“越大越优”后计算。
输出每个实际有 join 的 proxy 一行，同时记录 paired count、score/target coverage 和方向。该入口不
自动混合或推断 dataset/split/budget；多协议输入应先拆分，正式研究优先使用
`analyze correlation` 的协议分组报告。

显式随机消融：

```bash
zcp-test evaluate ... --input-source random
zcp-test evaluate ... --input-source noise
```

随机输入、真实数据、样本 ID、transform 和 SHA-256 指纹都会写入记录。缺失真实数据时命令失败，不自动切换到随机输入。

## Benchmark 真值

搜索或代理开发使用 validation。NATS-TSS 的 CIFAR-10 validation 协议应使用 `cifar10-valid`，不能把 `cifar10` test accuracy 当 validation：

```bash
zcp-test evaluate \
  --benchmark nats_tss --benchmark-path DATA/NATS-tss-v1_0-3ffb9-simple \
  --trusted --dataset cifar10-valid --target-split valid \
  --target-metric accuracy --epoch-budget 200 \
  --proxies er,naswot,synflow --count 100
```

普通 evaluate 只创建实际文件，不创建空的 `checkpoints/`、`parts/` 或 `reports/`。这些目录分别在保存 checkpoint、写分片和生成报告时按需创建。
