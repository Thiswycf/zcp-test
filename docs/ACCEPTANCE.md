# Acceptance Report

This report separates lightweight software acceptance from high-cost scientific validation. Unit
tests, an executable proxy, an adapter smoke or one synthetic epoch do not establish paper-level
reproduction or formal benchmark accuracy.

## Evidence status

| Area | Recorded evidence | Status | What it establishes |
|---|---|---|---|
| Unit/integration baseline | Current 2026-07-31 tree: **380 tests passed** | Passed | Small fixtures, schemas, adapters, reporting, GPU selection, reference construction and workflow contracts; not high-cost scientific validation |
| Coverage | First-party source **87%**; CLI 81%, analysis 93%, proxy studies 94%, and the ImageNet16 converter 83% | Passed | Meets the planned aggregate 85% and critical-module gates; native-data contracts still require separate real-data evidence |
| Proxy sweep | Acceptance sweep included **22 registered proxies** | Partial evidence | Registry coverage and explicit status handling, not numerical reproduction on every model family |
| H1 one-percent correlations | NB101, NB201, NATS-TSS, three-dataset NATS-SSS and the NB301 deterministic surrogate completed their scoped protocols | In progress overall | NATS-SSS transfer covers one stratified sample and one input/initialization seed; TNB101 and formal ViT-Bench remain |
| DARTS smoke | `runs/training/20260729T055707Z_6737dcdb935c`: `completed`, one synthetic epoch, checkpoints written | Passed smoke | DARTS construction, optimizer/AMP path, training JSONL and checkpoint writing on an RTX 4090 |
| Evaluation smoke | `runs/evaluate/20260729T055018Z_aa69ffaeb008`: `completed` | Historical smoke | A 10-architecture, three-proxy pipeline completed; it is not the 22-proxy sweep artifact |
| Search smoke | One failed and one completed AutoFormer ER search under `runs/search/` | Partial evidence | Historical search plumbing only; the old manifest cannot reconstruct current model fidelity, and the failed run must not be hidden |

The 380-test run, Ruff, compileall, pip check, diff check and 87% coverage are the current
low-cost software baseline. Machine-readable summaries and checksums for the real NB201 and
NATS-TSS sweeps are tracked under `docs/evidence/`; raw JSONL, plots and checkpoints remain in the
external audit root. Under Conda, coverage is invoked as `python -m coverage`; a host `coverage`
entry point may carry the wrong Python shebang.

## Proxy sweep scope

The 22 names are `az_nas`, `er`, `er_conn`, `er_deg`, `er_dist`, `er_pr`, `flops`, `gradnorm`,
`jacob_cov`, `meco`, `meco_opt`, `naswot`, `near`, `ntkt`, `params`, `swap`, `synflow`, `te_nas`,
`ter`, `vkdnw`, `zen`, and `zico`.

A sweep means each name was exercised through the common evaluator and produced an explicit
`ok`, `unsupported`, or `failed` outcome. It does not mean every proxy supports every model
family, every `portable-v1` implementation matches its paper numerically, or all scores have been
validated against standard answers.

The NATS-TSS H1 evidence is recorded in
[`evidence/NATS_TSS_ONE_PERCENT_CN.md`](evidence/NATS_TSS_ONE_PERCENT_CN.md) and the language-neutral
[`evidence/nats_tss_one_percent_summary.json`](evidence/nats_tss_one_percent_summary.json). It uses
`nats_bench.create(..., "tss")`, not NAS-Bench-201 truth. Among the 157 shared topology IDs, 31
target values differ, which directly rejects treating the two benchmark adapters as interchangeable.

The NATS-SSS cross-dataset evidence is recorded in
[`evidence/NATS_SSS_CROSS_DATASET_CN.md`](evidence/NATS_SSS_CROSS_DATASET_CN.md) and
[`evidence/nats_sss_cross_dataset_summary.json`](evidence/nats_sss_cross_dataset_summary.json).
CIFAR-100 and ImageNet16-120 each add 328 architectures × 22 proxies with no failed or duplicate
stable keys. Dataset-conditioned proxy correlations, fixed-source-score target-only transfer,
proxy rank stability and size/stage-controlled correlations are separate tables. This closes only
the locked one-percent, single-input/initialization-seed extension, not the full 32,768-architecture
space or independent-seed confidence intervals.

The scoped NAS-Bench-101 H1 evidence is recorded in
[`evidence/NB101_ONE_PERCENT_CN.md`](evidence/NB101_ONE_PERCENT_CN.md) and
[`evidence/nb101_one_percent_summary.json`](evidence/nb101_one_percent_summary.json). It covers
4,237 stratified architectures, all 22 proxies at seed 2026, the core 11 proxies at three seeds,
and 4/12/36/108-epoch repeat `mean/min/max` studies. TE-NAS `portable-v2` remains an approximation,
and the evidence must not be extrapolated to all 423,624 architectures.

The locked NAS-Bench-301 evidence is recorded in
[`evidence/NB301_ONE_THOUSAND_CN.md`](evidence/NB301_ONE_THOUSAND_CN.md) and
[`evidence/nb301_one_thousand_summary.json`](evidence/nb301_one_thousand_summary.json). It covers
1,000 stratified candidates, all 22 proxies at seed 2026 and the core 11 proxies at three seeds.
It is deterministic-surrogate association on the locked XGBoost 2.1.4 / nasbench301 0.3 runtime,
not real-training truth, an exhaustive DARTS space or a cross-XGBoost-version guarantee.

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
| `reference_topology_pytorch_port` | `transnas_micro`, `transnas_macro` | Official encoder and seven task-head PyTorch ports; a safe Taskonomy contract provider exists, but the formal 24-building split/config is unpublished and licensed data is unavailable here |
| `proxy_approximation` | legacy toy spaces | Explicit opt-in method smoke only; formal training and reference conclusions are prohibited |

Static model fidelity does not grant formal-training readiness. Only DARTS profiles currently set
`formal_training_ready: true`; AutoFormer and Proxyless-MBV2 are explicitly blocked, while PiT and
PlainNet-MBV2 do not yet have accepted formal profiles.
The boolean is not self-authorizing: a non-smoke run must match a code-owned approved protocol and
its critical fields, including the accepted batch and input size.

AutoFormer now has the AZ-NAS three-repeat sampler, exact linear LR rule, and six exact
Cream/AZ-NAS parameter-count goldens. Mixed 4090D/4090 two-rank DARTS and AutoFormer smokes validate
torchrun/DDP wrapping, cross-rank metric reduction, a shared run and rank-zero artifact ownership.
All six upstream complexity values are reproduced under the explicit `official_complexity_ops`
field. For AZ-NAS Tiny, THOP reports 1,100,420,352 MACs versus 1,380,128,376 upstream operations and
also omits relative-position parameters; the two measures are therefore reported separately rather
than forced into a false FLOPs equivalence. Two-rank interruption and new-run recovery on a tiny
fixture made from real ImageNet images now validates an `interrupted` manifest, checkpoint SHA-256/
source-run lineage, continuous epoch 0–4 logs and `.tmp` cleanup. This is recovery-mechanism evidence,
not either full ImageNet 1% protocol.

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
- TransNAS now separates all seven task heads against upstream commit `6d4231b`; one micro fixture
  matches official parameter counts and complete parameter-shape multisets for every task. A safe
  manifest, seven-task real input/target loader, final5k masks and deterministic jigsaw protocol are
  implemented. The 105 MB raw standard answer and both converted tables are checksum-locked;
  4,096/3,256 records and all selected validation targets are complete, and 41/33-architecture 1%
  manifests are frozen. The paper's 24-building/120K split and final transform/config were not
  released, and separately licensed Taskonomy data is not present on this machine. Formal real-input
  GPU ZCP therefore remains blocked. Fixtures, arbitrary Taskonomy splits, and random inputs are not
  accepted as formal substitutes; label-dependent regression/dense proxies remain explicitly
  unsupported pending a source-backed loss contract.
- The released PiT example completes `load → build → forward`; its 893,828 parameters and parameter
  shape multiset match Auto-Prox `90ed458`. MAC validation, formal training and KD reproduction are
  still outstanding, and vanilla/KD standard answers remain separate query protocols.
- The OFA-MBV3 all-3×3, expand-3, depth-2, width-1.0 subnet matches official commit `f03b267`
  at 3,410,792 parameters and the complete parameter-shape multiset; BN recalibration is implemented.
  Official inherited checkpoints, active-weight export and formal training remain outstanding.
- OFA-Proxyless-MBV2 now uses the official 21-dynamic-block positional encoding: five searchable
  max-depth-4 stages plus one fixed final stage. The registered space fixes width 1.3 to the released
  supernet and accepts resolutions 128–224 in steps of four. The width-1.0 all-3×3, expansion-3,
  searchable-depth-2 fixture has 2,500,632 parameters in both this port and official commit
  `f03b267`, with identical parameter-shape multisets; the released width-1.3 fixture has 3,718,832
  parameters in both. The official 32,202,338-byte supernet checkpoint is now bootstrapped under a
  fixed SHA-256. A mixed `k/e/d` subnet exported with active channel selection and learned kernel
  transforms matches official `get_active_subnet` parameter counts and has about `1.9e-6` maximum
  absolute output error on the same input. Real `evaluate` and short `search` smokes record inherited
  mode, checkpoint digest, active positions and `bn_recalibration_required`. A one-batch deterministic
  BN pipeline smoke now passes on the local real ImageNet-1k tree and records sample IDs, transform
  and fingerprint. This project protocol explicitly says `official_protocol_match=false`; numerical
  comparison with the official data provider, inherited accuracy, MAC golden values and formal
  training remain unvalidated.

## High-cost acceptance not completed

The following work is explicitly **not accepted** and must not be reported as completed:

1. Full 600-epoch DARTS CIFAR-10/CIFAR-100 and 250-epoch DARTS ImageNet training.
2. AutoFormer 500-epoch and Proxyless-MBV2 150-epoch formal protocols; AutoFormer sampler, LR rule,
   parameter/complexity fixtures, two-rank smoke and real-image-fixture recovery are accepted, but
   full ImageNet × 1%-epoch and 1%-ImageNet × full-schedule runs remain missing.
3. Full benchmark download, checksum and provenance validation on a clean second machine.
4. Remaining benchmark protocols beyond the accepted scoped NB101, NB201, NATS-TSS,
   three-dataset NATS-SSS and NB301 deterministic-surrogate runs, including TNB101 and formal
   ViT-Bench identity/split acceptance.
5. Exhaustive NAS-Bench-101 evaluation or theoretical NAS-Bench-301 DARTS-space traversal.
6. Built-in multi-GPU evaluate launch and duplicate-safe consolidation; training DDP launch and
   fixture-level restart/failure injection are accepted, but full-data-level acceptance is not.
7. Paper-number reproduction, independent-seed confidence intervals and official-code cost/accuracy
   comparison.

Formal acceptance requires retained manifests, resolved configs, commit identity, environment,
input hashes, explicit result type, failure rows and exact commands. Until then, the project may
claim lightweight software acceptance and smoke coverage only.

## Real standard-answer index-0 acceptance (2026-07-30)

The machine-local catalog successfully queried NB101 full (423,624 architectures), NB201 v1.1,
NATS-TSS/SSS, TransNAS micro/macro, deterministic NB301, and all three ViT release slices. Assets
reported as `external catalog` are usable on this machine but are not contained by the inspected data
root; another machine must bootstrap or register them again.

Key values were NB101 CIFAR-10 valid/108 mean `0.9264155825`, NB201 and NATS-TSS CIFAR-10-valid
accuracy `81.98266666`, NATS-SSS 90-epoch accuracy `76.88799999`, TNB101 class-scene valid top-1
`7.48407650` (micro) and `52.97074127` (macro), and deterministic NB301 `93.45854187`. ViT main,
extension and PiT returned CIFAR-100 `68.66` vanilla, `78.07` KD and `68.33` vanilla respectively.
The extension slice has no vanilla metric; an explicit vanilla query correctly fails. NB201 and
NATS-TSS share the index-0 topology ID but retain distinct benchmark IDs, API sources and protocols.

A matching ten-slice index-0 `build_model → params proxy` sweep completed with one successful row and
no failure per slice. It used explicit random input and the data-independent `params` proxy, so it
proves adapter-to-model-to-evaluator wiring only, not dataset-input or correlation validity.

## ViT-Bench release-slice preacceptance

The three pinned public files were bootstrapped, converted to safe JSONL, checksum-locked, queried,
and exercised with deterministic real CIFAR-100 inputs. Each slice has 100 records. A fixed
minimum-five manifest produces 110 rows for 22 proxies: 80 supported results, 30 explicit
Transformer `unsupported` results, and zero failures. Correlation/report/architecture-study bundles
were generated, but `n=5` is execution evidence only.

This does not close formal ViT-Bench H1. The paper describes 500 candidates per AutoFormer/PiT space
and a disjoint 60/40 development/test split; neither the complete candidates nor split identities are
published in the pinned repository. AutoFormer main, extension, and PiT remain separate, as do
vanilla, KD, and inherited-supernet metrics. PiT is now conservatively classified as
`reference_topology_pytorch_port`: structural, parameter, and MAC fixtures pass, but checkpoint and
layerwise numerical parity are unavailable.
