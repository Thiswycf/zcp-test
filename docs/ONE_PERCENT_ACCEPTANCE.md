# One-percent benchmark correlation acceptance

This document is the English companion to `ONE_PERCENT_ACCEPTANCE_CN.md`. The Chinese document is
the authoritative detailed runbook and contains the retained run IDs and checksums.

## Protocol

1. Generate a deterministic stratified architecture manifest; never use list-position alignment.
2. Evaluate all 22 registered proxies at seed 2026.
3. Evaluate the core 11 proxies (`params,flops,gradnorm,jacob_cov,naswot,synflow,zen,zico,meco,te_nas,az_nas`)
   at seeds 2026, 2027, and 2028 where the model family supports them.
4. Join proxy and ground-truth rows by canonical architecture ID.
5. Report dataset, split, budget/task, metric direction, coverage, failures, unsupported calls,
   ties, Spearman, Kendall tau-b, Pearson, nDCG, and bootstrap confidence intervals separately.
6. Search and aggregation may use validation only; test metrics enter final reporting only.

## Minimum sample sizes

| Benchmark/protocol | Minimum |
|---|---:|
| NAS-Bench-101 | 4,237 |
| NAS-Bench-201 | 157 |
| NATS-TSS | 157 |
| NATS-SSS | 328 |
| NAS-Bench-301 deterministic surrogate | 1,000 |
| TransNAS micro/macro | 41 / 33 per task protocol |
| ViT public release slice | 5 per 100-record slice, preacceptance only |

The ViT minimum-five rule is 5% of each public file, not one percent of the paper's 500-candidate
protocol. It must not be labelled formal H1 until the complete candidate set and disjoint 60/40
identities are available.

## Example

```bash
zcp-test benchmark sample nasbench101 --catalog ~/.config/zcp-test/data.json \
  --count 4237 --seed 2026 --output /path/to/audit/nb101-seed2026.json
zcp-test evaluate --benchmark nasbench101 --catalog ~/.config/zcp-test/data.json \
  --sample-manifest /path/to/audit/nb101-seed2026.json --sample-shard 0 \
  --dataset cifar10 --target-split valid --target-metric final_accuracy \
  --target-budget 108 --proxies params,flops,naswot,synflow,zen,zico \
  --input-source dataset --data-root /path/to/cifar10 --gpu auto \
  --output /path/to/audit/runs/nb101
```

Use the exact timestamped run printed by `evaluate`, not the output parent directory. Validate row
counts, unique task keys, finite values, manifest/checksum provenance, and explicit failure records
before running `analyze correlation` or `report bundle`.

## Current status

The scoped NB101, NB201, NATS-TSS, NATS-SSS including the accepted CIFAR-100/ImageNet16-120
cross-dataset extension, and locked deterministic NB301 protocols are accepted. Formal TransNAS input acceptance
is blocked by the unpublished author split/config and licensed Taskonomy data. ViT has only the
three public-slice minimum-five preacceptance runs; it is not formal paper-level H1.

The two new dataset-specific sweeps each completed 7,216/7,216 rows with zero failures and duplicate
keys. The 12-shard size study separates dataset-specific and target-only results into four tables
with 594/186/9/1,188 rows. This is evidence for the 328-architecture stratified sample at one
input/initialization seed only, not the full search space, a multi-seed result, or a causal claim. See
the [human-readable evidence](evidence/NATS_SSS_CROSS_DATASET_CN.md) and
[machine-readable summary](evidence/nats_sss_cross_dataset_summary.json).
