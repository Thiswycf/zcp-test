# Acceptance Report

This report separates lightweight software acceptance from high-cost scientific validation. Unit
tests, an executable proxy, an adapter smoke or one synthetic epoch do not establish paper-level
reproduction or formal benchmark accuracy.

## Evidence status

| Area | Recorded evidence | Status | What it establishes |
|---|---|---|---|
| Unit/integration baseline | Current 2026-07-30 tree: **216 tests passed** | Passed | Small fixtures, schemas, adapters, reporting, GPU selection, reference construction and workflow contracts; not high-cost scientific validation |
| Coverage | First-party source **86%**; CLI 80%, reports 96%/100%, converters 98%, doctor/legacy 100% | Passed | Meets the planned aggregate 85% and listed critical-module 80% gates; native-data contracts still require separate smoke evidence |
| Proxy sweep | Acceptance sweep included **22 registered proxies** | Partial evidence | Registry coverage and explicit status handling, not numerical reproduction on every model family |
| DARTS smoke | `runs/training/20260729T055707Z_6737dcdb935c`: `completed`, one synthetic epoch, checkpoints written | Passed smoke | DARTS construction, optimizer/AMP path, training JSONL and checkpoint writing on an RTX 4090 |
| Evaluation smoke | `runs/evaluate/20260729T055018Z_aa69ffaeb008`: `completed` | Historical smoke | A 10-architecture, three-proxy pipeline completed; it is not the 22-proxy sweep artifact |
| Search smoke | One failed and one completed AutoFormer ER search under `runs/search/` | Partial evidence | Historical search plumbing only; the old manifest cannot reconstruct current model fidelity, and the failed run must not be hidden |

The 216-test run, Ruff, compileall, pip check, diff check and 86% coverage are the current
low-cost software baseline. The repository retains DARTS/evaluation/search manifests but no
dedicated 22-proxy sweep manifest, so that sweep remains only partially reconstructable. Under
Conda, coverage is invoked as `python -m coverage`; a host `coverage` entry point may carry the
wrong Python shebang.

## Proxy sweep scope

The 22 names are `az_nas`, `er`, `er_conn`, `er_deg`, `er_dist`, `er_pr`, `flops`, `gradnorm`,
`jacob_cov`, `meco`, `meco_opt`, `naswot`, `near`, `ntkt`, `params`, `swap`, `synflow`, `te_nas`,
`ter`, `vkdnw`, `zen`, and `zico`.

A sweep means each name was exercised through the common evaluator and produced an explicit
`ok`, `unsupported`, or `failed` outcome. It does not mean every proxy supports every model
family, every `portable-v1` implementation matches its paper numerically, or all scores have been
validated against standard answers.

## DARTS smoke boundary

The retained run executed:

```bash
zcp-test train --config configs/training/darts_cifar10.yaml --epochs 1 --smoke
```

It wrote `best.pt`, `last.pt`, `training.jsonl` and a completed manifest. This validates pipeline
wiring only. It does not validate CIFAR-10 test accuracy, 600-epoch convergence, augmentation
fidelity, multi-GPU scaling, arbitrary-epoch recovery or cross-hardware reproducibility.

## Fidelity and result protocols

| Fidelity | Spaces | Acceptance consequence |
|---|---|---|
| `reference_model` | `darts`, `autoformer`, `pit`, `zennas_plainnet_mbv2`, `ofa_proxyless_mbv2`, `ofa_mbv3` | Static model structure is implemented; formal training additionally requires `formal_training_ready: true` |
| `reference_topology_pytorch_port` | `nb101_dag`, `nb201_topology`, `nats_size` | Topology is represented by a port; ZCP values are not automatically identical to the original training implementation |
| `reference_topology_pytorch_port` | `transnas_micro`, `transnas_macro` | Official encoder topology port; seven Taskonomy task heads are not yet fully implemented |
| `proxy_approximation` | legacy toy spaces | Explicit opt-in method smoke only; formal training and reference conclusions are prohibited |

Static model fidelity does not grant formal-training readiness. Only DARTS profiles currently set
`formal_training_ready: true`; AutoFormer and Proxyless-MBV2 are explicitly blocked, while PiT and
PlainNet-MBV2 do not yet have accepted formal profiles.
The boolean is not self-authorizing: a non-smoke run must match a code-owned approved protocol and
its critical fields, including the accepted batch and input size.

NAS-Bench-101/201, NATS and converted TransNAS records are **standard answers** only for their
explicit dataset/split/budget/seed protocol. NAS-Bench-301 is a **surrogate** prediction, and its
deterministic/noisy modes are distinct. ViT-Bench metrics may be **scratch**, distillation, or
**inherited-supernet** results. These protocols must not be pooled.

## Known partial acceptance

- The retained evaluation covers 3 proxies, not 22, and uses an older 40-row component-long score
  schema rather than the current one-row-plus-components layout.
- Only the registry count, not the dedicated 22-proxy sweep artifact, is independently reproducible
  from the current tree.
- The completed AutoFormer search validates mechanics, not scientific fidelity; one failed search is
  also retained.
- Some upstream native assets lack pinned checksums. Path existence is not authenticity.
- Bootstrap and index-0 adapter smokes do not establish full-record, all-budget/split or cross-machine
  coverage.
- Manual `--start/--count` partitioning has no accepted launcher or merge CLI. Multi-file analysis is
  supported, but end-to-end multi-GPU orchestration is not accepted.
- `portable-v1` proxies and topology ports need numerical comparison with official implementations
  before paper-reproduction claims.
- The released PiT example completes `load → build → forward`; its 893,828 parameters and parameter
  shape multiset match Auto-Prox `90ed458`. MAC validation, formal training and KD reproduction are
  still outstanding, and vanilla/KD standard answers remain separate query protocols.
- The OFA-MBV3 all-3×3, expand-3, depth-2, width-1.0 subnet matches official commit `f03b267`
  at 3,410,792 parameters and the complete parameter-shape multiset; BN recalibration is implemented.
  Official inherited checkpoints, active-weight export and formal training remain outstanding.

## High-cost acceptance not completed

The following work is explicitly **not accepted** and must not be reported as completed:

1. Full 600-epoch DARTS CIFAR-10/CIFAR-100 and 250-epoch DARTS ImageNet training.
2. AutoFormer 500-epoch and Proxyless-MBV2 150-epoch formal protocols; static reference models do
   not replace the missing sampler, distributed-batch semantics, augmentation validation or fixtures.
3. Full benchmark download, checksum and provenance validation on a clean second machine.
4. Full-scale 22-proxy evaluation across supported benchmark datasets, splits, budgets and seeds.
5. Exhaustive NAS-Bench-101 evaluation or theoretical NAS-Bench-301 DARTS-space traversal.
6. Multi-GPU launch, duplicate-safe consolidation, restart and failure-injection acceptance.
7. Paper-number reproduction, independent-seed confidence intervals and official-code cost/accuracy
   comparison.

Formal acceptance requires retained manifests, resolved configs, commit identity, environment,
input hashes, explicit result type, failure rows and exact commands. Until then, the project may
claim lightweight software acceptance and smoke coverage only.
