# Benchmark-Specific Research Guide

Use the unified dispatcher after producing protocol-complete score records:

```bash
zcp-test analyze benchmark \
  --scores /path/to/run/scores.jsonl \
  --benchmark auto --view auto \
  --output /path/to/run/reports/benchmark
```

Automatic views are: NAS-Bench-101 `budget`, NAS-Bench-201/NATS-TSS `topology`, NATS-SSS
`size`, TransNAS-Bench-101 `transfer`, and ViT-Bench-101 `architecture`.

## Target protocol schema

New evaluations write schema `2.1` fields `target_direction`, `target_epoch_budget`,
`target_seed`, `target_seed_reduction`, `benchmark_variant`, and `benchmark_protocol`.
Correlation and top-k reports group by these fields and never silently pool heterogeneous
protocols. A JSONL metric matching multiple epoch budgets now requires an explicit budget.

## NAS-Bench-101 budgets

```bash
zcp-test analyze benchmark \
  --scores /path/to/run/scores.jsonl \
  --benchmark nasbench101 --view budget --component mean \
  --benchmark-path /path/to/data/nasbench101/converted/full/manifest.json \
  --benchmark-version full --budgets 4 12 36 108 \
  --study-dataset cifar10 --study-split valid --study-metric final_accuracy \
  --seed-reduction mean --top-k 5 10 50 --bootstrap-samples 1000 \
  --output /path/to/reports/nb101-budget
```

The report contains per-budget proxy correlations and pairwise ground-truth rank stability. A
drop in proxy correlation is not attributable to the proxy alone when early- and late-budget
ground-truth rankings are themselves unstable.

## Topology and size

```bash
zcp-test analyze benchmark --scores /path/to/nb201/scores.jsonl \
  --benchmark nasbench201 --view topology --output /path/to/reports/nb201
zcp-test analyze benchmark --scores /path/to/nats-sss/scores.jsonl \
  --benchmark nats_sss --view size --output /path/to/reports/nats-size
```

Topology reports expose the six fixed edges, per-edge operations, and operation coverage. Size
reports expose per-stage channels and aggregate width statistics. These are descriptive bias
diagnostics, not causal operation or channel effects. NAS-Bench-201 and NATS-TSS may share a
topology parser but never share benchmark identity or targets.

## TransNAS and ViT

```bash
zcp-test analyze benchmark \
  --scores /path/to/task-a/scores.jsonl /path/to/task-b/scores.jsonl \
  --benchmark transnasbench101 --view transfer --benchmark-variant micro \
  --target-metric test_top1 --output /path/to/reports/transnas

zcp-test analyze benchmark --scores /path/to/vit/scores.jsonl \
  --benchmark vitbench101 --view architecture --benchmark-variant autoformer_main \
  --output /path/to/reports/vit
```

TransNAS micro and macro runs must remain separate. Duplicate task/architecture/proxy/component
rows require explicit metric, split, budget, or run filtering. ViT main, extension, and PiT slices
remain independent, as do vanilla, KD, and inherited-supernet metrics.

Only validation protocols may determine search or aggregation weights. Test targets are reserved
for final reporting. See the Chinese guide for complete output-table interpretation, filters, and
troubleshooting: [BENCHMARK_STUDIES_CN.md](BENCHMARK_STUDIES_CN.md).

