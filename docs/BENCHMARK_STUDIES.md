# Benchmark-Specific Research Guide

Use the unified dispatcher after producing protocol-complete score records:

```bash
zcp-test analyze benchmark \
  --scores /path/to/run/scores.jsonl \
  --benchmark auto --view auto \
  --output /path/to/run/reports/benchmark
```

Automatic views are: NAS-Bench-101 `budget`, NAS-Bench-201/NATS-TSS `topology`, NATS-SSS
`size`, NAS-Bench-301 `darts`, TransNAS-Bench-101 `transfer`, and ViT-Bench-101 `architecture`.

## Target protocol schema

New evaluations write schema `2.1` fields `target_direction`, `target_epoch_budget`,
`target_seed`, `target_seed_reduction`, `benchmark_variant`, and `benchmark_protocol`.
Correlation and top-k reports group by these fields and never silently pool heterogeneous
protocols. A JSONL metric matching multiple epoch budgets now requires an explicit budget.

## NAS-Bench-101 budgets

```bash
zcp-test analyze benchmark \
  --scores /path/to/shard-0/scores.jsonl /path/to/shard-1/scores.jsonl \
           /path/to/shard-2/scores.jsonl /path/to/shard-3/scores.jsonl \
  --benchmark nasbench101 --view budget \
  --benchmark-path /path/to/data/nasbench101/converted/full/manifest.json \
  --benchmark-version full --budgets 4 12 36 108 \
  --study-dataset cifar10 --study-split valid --study-metric final_accuracy \
  --seed-reduction mean --top-k 5 10 50 --bootstrap-samples 1000 \
  --output /path/to/reports/nb101-budget
```

The report contains per-budget proxy correlations, pairwise ground-truth rank stability,
`score_coverage.csv`, and `top_k_retrieval.csv`. Coverage retains failed and non-finite calls
instead of replacing them with zero. Multiple shard files are pooled by scientific protocol;
`run_id` and `source_run` are provenance rather than grouping fields, while evaluation `seed`
remains separate. The retrieval table records precision/Jaccard, selected versus oracle target
quality, and direction-adjusted regret for every proxy/component/budget/k combination.
`budget_top_k_retrieval.png/svg` complements the correlation curve. A drop in proxy correlation
is not attributable to the proxy alone when early- and late-budget ground-truth rankings are
themselves unstable.

NB101-specific outputs also include `architecture_features.csv`, `feature_strata.csv`, and
`structure_controlled_correlations.csv` for vertex/edge/depth and operation-count controls.
`edit_neighbors.csv`, `neighborhood_differences.csv`, and `neighborhood_correlations.csv` study
sample-local one-operation or one-edge edits using indexed signatures rather than an all-pairs
scan. `budget_structure_controlled.png/svg` and `budget_neighborhood_agreement.png/svg` make the
controlled and local-edit results directly comparable across budgets. These controls and neighborhood contrasts are benchmark-driven project extensions, not
causal claims reproduced directly from the NAS-Bench-101 paper. Analysis defaults to each proxy's
declared `primary_component`; current ER exposes only `score`. Historical ER `mean/sum` records are
read-only superseded artifacts and are excluded from current formal summaries.

## Topology and size

```bash
zcp-test analyze benchmark --scores /path/to/nb201/scores.jsonl \
  --benchmark nasbench201 --view topology --output /path/to/reports/nb201
zcp-test analyze benchmark --scores /path/to/nats-sss/scores.jsonl \
  --benchmark nats_sss --view size --output /path/to/reports/nats-size
```

Topology reports expose the six fixed edges, per-edge operations, operation coverage, numeric
feature correlations, `operation_effects.csv`, and one-edge `matched_pairs.csv`. The effect table stratifies target and
direction-adjusted proxy values by edge and operation, including deltas from each edge baseline.
It diagnoses structural preference but is observational, not a causal replacement effect.

Size reports expose per-stage channels, aggregate width statistics, feature correlations, and a
focused `stage_sensitivity.csv`. `size_controlled_correlations.csv` reports partial Spearman after
controlling channel sum; `size_strata.csv` reports quality within total-size quantiles. Compare `outcome=score` with `outcome=target` to detect proxies
that mostly encode width. Prefer rank correlations for discrete channel choices with many ties.
NAS-Bench-201 and NATS-TSS may share a topology parser but never share benchmark identity or
targets; NATS-SSS results must not be pooled with TSS.

For NATS-SSS cross-dataset work, distinguish two protocols. Dataset-specific ZCP recomputes scores
on CIFAR-100 and ImageNet16-120 inputs and correlates each against the matching 90-epoch validation
accuracy. Target-only transfer keeps the source score and `input_fingerprint` unchanged and joins
only another dataset's target by architecture ID. The accepted 12-shard size study now emits the
594-row dataset/target matrix, 186-row proxy-stability table, 9-row target-rank table, and 1,188-row
controlled-transfer table separately. Report coverage, input fingerprints, and size controls; do
not collapse dataset-specific and target-only results. See the
[evidence](evidence/NATS_SSS_CROSS_DATASET_CN.md).

## NAS-Bench-301 DARTS

```bash
zcp-test analyze benchmark \
  --scores examples/studies/data/nasbench301_darts.jsonl \
  --benchmark nasbench301_surrogate --view darts \
  --output /tmp/zcp-test-examples/nb301
```

The report separates normal/reduction cell operation counts, source nodes, edge spans, cell
balance, and `cell × node × source-class × operation` effects. Deterministic and noisy surrogate
protocols must remain separate. NAS-Bench-Suite-Zero and MeCo directly study ZCPs on NB301; this
fine-grained interaction decomposition is a benchmark-driven extension, not a reproduced paper
table.

The locked 1,000-candidate protocol has completed 22 proxies at seed 2026 and the core 11 proxies
at three seeds. The standard sensitivity bundle now includes per-seed correlations, canonical-ID
cross-seed rank stability and sample-size convergence. This remains surrogate association on the
locked XGBoost 2.1.4 / nasbench301 0.3 runtime; see
[`evidence/NB301_ONE_THOUSAND_CN.md`](evidence/NB301_ONE_THOUSAND_CN.md). The 11,221-candidate
generation corpus is a reproducible sampling denominator, not the full DARTS search space.

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
rows require explicit metric, split, budget, or run filtering. Reports include a task-transfer
matrix and official micro operation/macro module factors. ViT reports include per-layer/stage
tables, head dimension, MLP width, and explicitly named parameter proxies. ViT main, extension, and PiT slices
remain independent, as do vanilla, KD, and inherited-supernet metrics.
AutoFormer `hidden_dim` is the embedding width. For PiT, the released encoding uses stage width
`base_dim * stage_num_heads`; feature and parameter proxies apply that stage-specific rule.

Only validation protocols may determine search or aggregation weights. Test targets are reserved
for final reporting. See the Chinese guide for complete output-table interpretation, filters, and
troubleshooting: [BENCHMARK_STUDIES_CN.md](BENCHMARK_STUDIES_CN.md).

## Reproducible examples and evidence

Small deterministic JSONL examples for all six benchmark families and the generic multi-proxy
study are retained under `examples/studies/data/`. Generated reports belong in `/tmp` or a run
directory and are not committed. The complete evidence boundary—direct paper experiment,
same-space partial evidence, or project extension—is documented in
[RESEARCH_EVIDENCE.md](RESEARCH_EVIDENCE.md).
