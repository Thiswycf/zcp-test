# Acceptance Report

This report separates lightweight software acceptance from high-cost scientific validation. Unit
tests, an executable proxy, an adapter smoke or one synthetic epoch do not establish paper-level
reproduction or formal benchmark accuracy.

## Evidence status

| Area | Recorded evidence | Status | What it establishes |
|---|---|---|---|
| Unit/integration baseline | Current 2026-08-04 tree: **565 tests passed** across 38 files | Passed | Full pytest, Ruff, compileall, pip check, Bash syntax, JSON validation, and diff checks passed; panel consistency is rechecked separately after its parallel update; four upstream THOP `distutils` deprecation warnings remain non-failing |
| Static quality gates | Ruff, compileall, pip check, repository hygiene, panel check, and `git diff --check` all passed | Passed | Syntax, dependencies, panel validation, and basic repository hygiene; not scientific correctness |
| Coverage | First-party source **87%**; CLI **82%**, analysis 93%, proxy studies 94%, and the ImageNet16 converter 83% | Passed | Meets the planned aggregate 85% and critical-module gates; native-data contracts still require separate real-data evidence |
| Proxy sweep | Acceptance sweep included **22 registered proxies** | Partial evidence | Registry coverage and explicit status handling, not numerical reproduction on every model family |
| H1 one-percent correlations | NB101, NB201, NATS-TSS, three-dataset NATS-SSS and the NB301 deterministic surrogate completed their scoped protocols | In progress overall | NATS-SSS transfer covers one stratified sample and one input/initialization seed; TNB101 and formal ViT-Bench remain |
| DARTS smoke | `runs/training/20260729T055707Z_6737dcdb935c`: `completed`, one synthetic epoch, checkpoints written | Passed smoke | DARTS construction, optimizer/AMP path, training JSONL and checkpoint writing on an RTX 4090 |
| DARTS CIFAR dual one-percent | Three candidates on CIFAR-10/100: six full-data × 6-epoch runs and six 1%-data × 600-epoch runs; deterministic preflight, two recovery audits, and reporting complete | Scoped protocol passed | Engineering/scoped acceptance only; not 600-epoch full-data accuracy reproduction or multi-seed search gain, and the protocols must not be averaged |
| Evaluation smoke | `runs/evaluate/20260729T055018Z_aa69ffaeb008`: `completed` | Historical smoke | A 10-architecture, three-proxy pipeline completed; it is not the 22-proxy sweep artifact |
| AutoFormer search and freeze | The AZ-NAS `3×8,000` cohort is reconciled: 24,000 candidates, 23,999 unique evaluations, and one cache hit; three candidates are frozen | Search/freeze passed | The post-completion supervisor failure remains visible but does not overwrite validated scientific artifacts; supporting seeds are provenance-only |
| AutoFormer candidate smokes | All three frozen candidates completed batch-256 synthetic memory, atomic-checkpoint, and trusted checkpoint-load/resume smokes | Smoke passed | Construction, memory, optimizer/checkpoint, and recovery plumbing only; random-input accuracy is meaningless and is not ImageNet evidence |
| AutoFormer selected-candidate real dual one-percent V2 | `zcp-selected` completed full-data 5 epochs and one-percent-data 500 epochs with 5/500 rows, terminal manifests, and last/best checkpoints | **Scoped protocol passed** | Baseline tasks 5/6 were policy-interrupted and excluded; this proves implementation readiness, not full-data paper accuracy or search gain |

The current full gate executes and passes 565 tests across 38 files. First-party source coverage of
87% and CLI coverage of 82% remain from the latest retained coverage run. Ruff, compileall, pip check,
Bash syntax, panel validation, JSON validation, and `git diff --check` pass; four THOP `distutils`
deprecation warnings are non-failing. Machine-readable summaries and checksums for the real NB201 and
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

## DARTS CIFAR dual one-percent boundary

DARTS CIFAR-10/100 dual one-percent acceptance is complete for the frozen ER-selected, fixed-random,
and parameter-matched-random candidates: six full-data × 6-epoch runs and six exactly-1%-data ×
600-epoch runs. The deterministic real-data preflight, one trusted-checkpoint recovery audit per
protocol, and reporting are complete. See the
[evidence report](evidence/DARTS_CIFAR_DUAL_ONE_PERCENT_CN.md) and
[machine-readable summary](evidence/darts_cifar_dual_one_percent_summary.json).

This is not a 600-epoch full-data accuracy reproduction or a multi-seed search-gain result. All
training uses seed `20260731`, and candidate selection uses one fixed CIFAR-10 batch and one
initialization seed. The protocols test different questions, rank candidates differently, and must
not be averaged. The ImageNet-1k asset now passes a 1,000-class, 1,281,167/50,000-image structural
audit and a real-loader decode check. The first detached DARTS ImageNet run used commit `78d8118`
and the `/public` spinning disk; after three hours it had no completed epoch, and its unused logger
left `run.log` empty. It is now explicitly `interrupted`, not accepted. Commit `c0c7815` mirrors
events to both `events.jsonl` and `run.log`, adds a portable four-GPU launcher, and restarts from a
verified local NVMe copy. The new run reached about 7.2 rank-zero batches/s with a roughly 22-minute
first-epoch ETA in its first minute. All six scoped DARTS ImageNet runs completed at 2026-07-31
16:52 Asia/Shanghai, producing 759 epoch rows and a CSV/PNG/SVG/HTML bundle. Final valid top-1 for
the three full-data × 3-epoch candidates is `39.528/38.624/29.852`, and for the three 1%-data ×
250-epoch candidates is `9.6/10.6/5.0` (ZCP-selected/fixed-random/params-matched in both cases).
The first full-data run used four-GPU DDP while the remaining runs used one GPU, so ordinary
BatchNorm sees different per-device batch statistics. This accepts implementation and recovery,
not a topology-identical search-gain conclusion. The throughput, packing, checkpoint, and six-run
summary is in [`evidence/gpu_throughput_optimization.json`](evidence/gpu_throughput_optimization.json).
AutoFormer selected-candidate dual one-percent acceptance is complete; PlainNet-MBV2 and
Proxyless-MBV2 remain incomplete.

For future launches after 2026-08-04, dual-one-percent **engineering acceptance** is reduced to one
ZCP-selected architecture under both protocols: full data × at least 1% of epochs, and exact 1% data
× the complete schedule. Historical three-candidate runs are retained as immutable evidence but do
not define the new gate. Short training is not a valid test of ZCP search gain, so fixed-random and
parameter/FLOPs-matched candidates are no longer multiplied into this engineering gate. Any
superiority claim requires a separate predeclared, sufficiently trained, multi-seed experiment.

The retained historical synthetic smoke run executed:

```bash
zcp-test train --config configs/training/darts_cifar10.yaml --epochs 1 --smoke
```

It wrote `best.pt`, `last.pt`, `training.jsonl` and a completed manifest. This validates pipeline
wiring only. It does not validate CIFAR-10 test accuracy, 600-epoch convergence, augmentation
fidelity, multi-GPU scaling, arbitrary-epoch recovery or cross-hardware reproducibility.

## Fidelity and result protocols

| Fidelity | Spaces | Acceptance consequence |
|---|---|---|
| `reference_topology_pytorch_port` | `nb101_dag`, `nb201_topology`, `nats_size` | Topology is represented by a port; ZCP values are not automatically identical to the original training implementation |
| `reference_topology_pytorch_port` | `transnas_micro`, `transnas_macro` | Official encoder and seven task-head PyTorch ports; a safe Taskonomy contract provider exists, but the formal 24-building split/config is unpublished and licensed data is unavailable here |
| `reference_topology_pytorch_port` | `pit` | Published topology port; it is not an official numerical reference or an accepted formal-training implementation |
| `reference_model` | `darts`, `autoformer`, `zennas_plainnet_mbv2`, `ofa_proxyless_mbv2`, `ofa_mbv3` | Static model structure is implemented; formal training additionally requires `formal_training_ready: true` |
| `proxy_approximation` | legacy toy spaces | Explicit opt-in method smoke only; formal training and reference conclusions are prohibited |

The upstream comparison and reference-upgrade requirements are recorded in
[`evidence/PLAINNET_MBV2_FIDELITY_AUDIT_CN.md`](evidence/PLAINNET_MBV2_FIDELITY_AUDIT_CN.md).
The formal controller now locks the pinned upstream 100k run, population 1,024, batch 64, 1/2 block
replacement, top-1,023 parent pool, no crossover, and per-insertion full-history four-component
log-rank. The official launch script overrides the parser's default 512 with 1,024; pinned source
hashes are recorded in `evidence/plainnet_source_protocol_20260804.json`. The host-specific
conservative cumulative CPU rerank estimate is 15,165.56 seconds (about 4.21 hours). The GPU
batch64/224 preflight is complete. Formal 450M/600M/1G 100k runs started at 2026-08-04 13:50+08
from commit `0d86588` on three independent 4090D GPUs; all remain `running` with
`formal_search_completed=false`, so search acceptance is not yet granted. See
`evidence/plainnet_source_aligned_100k_launch_20260804.json`.

Static model fidelity does not grant formal-training readiness. DARTS profiles and the accepted
AutoFormer AZ-NAS scratch profile set `formal_training_ready: true`; PlainNet-MBV2 and Proxyless-MBV2
remain explicitly blocked, while PiT is not a formal-training target. PlainNet-MBV2 has a locked 150-epoch candidate profile
but remains blocked pending dual one-percent GPU acceptance.
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

### AutoFormer search, frozen candidates, and V2 training state

The AutoFormer AZ-NAS search cohort is complete and reconciled: three seeds processed 8,000
candidates each, for 24,000 candidate rows, 23,999 unique evaluations, and one cache hit. The
historical supervisor failed only after all seed artifacts completed. That post-completion
orchestration state remains auditable but does not relabel completed manifests, generation summaries,
or the reconciled cohort as a scientific failure. Launcher protection uses `git archive` to pin all
tracked Shell, Python, and configuration files at the launch commit into a read-only
`launcher-snapshots` tree. Later lanes execute from that tree and do not import newer main-worktree
code. Current launchers use task-scoped locks for serial work and packed-scope locks only while all
packed workloads remain active. Older immutable snapshots retain their original scope until exit and
must be audited, not hot-patched or bypassed by deleting lock files. See
[`evidence/gpu_launcher_snapshot_fix.json`](evidence/gpu_launcher_snapshot_fix.json).

The predeclared primary/supporting protocol froze three unique candidates:
`zcp_selected=42e6457ccb580a092454`, `fixed_random=d904aacf51d2b0867df6`, and
`params_flops_matched=41b5e6d4dc3279909487`. Supporting seeds are provenance-only and cannot be
averaged, substituted for the primary winner, or cherry-picked. The frozen manifest SHA-256 is
`42dc72f29e141fa97c042c1979f390486962a97fa34cdbcd3394b556148bdb4a`; see
[`evidence/autoformer_frozen_candidates.json`](evidence/autoformer_frozen_candidates.json).

All three candidates completed configured micro-batch-256 synthetic memory smokes, atomic
`last.pt`/`best.pt` writes, and trusted checkpoint-load/resume smokes. This validates memory,
training control flow, and checkpoint identity/recovery only. Random-input accuracy is meaningless;
see [`evidence/autoformer_frozen_candidate_smokes.json`](evidence/autoformer_frozen_candidate_smokes.json).

Real dual-one-percent V2 started at 2026-08-04 11:07+08 from commit/source snapshot `76a0fcd`.
The selected candidate completed full-data 5 epochs at 12:52 and one-percent-data 500 epochs at
13:18, retaining 5/500 continuous rows, finite metrics, and last/best checkpoints. The immutable
pre-policy supervisor auto-started not-yet-started baseline
tasks 5/6. Continuing them after the single-candidate policy took effect was a scheduling error, even
though no new command was submitted. Both baseline tasks were terminated with `interrupted`
manifests; GPU0/1 dropped to about 89/15 MiB at 0% utilization and both kernel locks were released.
The watcher removed the old main supervisor after the selected candidate became terminal.
Data preflight records 1,000 classes, 1,281,167 training files, and 50,000 validation files.
Gradient accumulation preserves global batch 2,048 and the LR protocol from micro-batch 256.
This scoped gate permits the repository profile to set `formal_training_ready: true`, but it is not
a 500-epoch full-data paper-accuracy reproduction or evidence of ZCP superiority. See
[`evidence/autoformer_dual_one_percent_launch.json`](evidence/autoformer_dual_one_percent_launch.json)
and [`evidence/autoformer_single_candidate_policy_intervention_20260804.json`](evidence/autoformer_single_candidate_policy_intervention_20260804.json),
[`evidence/autoformer_single_candidate_dual_one_percent_completion_20260804.json`](evidence/autoformer_single_candidate_dual_one_percent_completion_20260804.json).

NAS-Bench-101/201, NATS and converted TransNAS records are **standard answers** only for their
explicit dataset/split/budget/seed protocol. NAS-Bench-301 is a **surrogate** prediction, and its
deterministic/noisy modes are distinct. ViT-Bench metrics may be **scratch**, distillation, or
**inherited-supernet** results. These protocols must not be pooled.

## Known partial acceptance

- The retained evaluation covers 3 proxies, not 22, and uses an older 40-row component-long score
  schema rather than the current one-row-plus-components layout.
- Only the registry count, not the dedicated 22-proxy sweep artifact, is independently reproducible
  from the current tree.
- The source-pinned `az_nas_autoformer` port captures attention/MLP residual features and computes
  expressivity, trainability, official complexity, and three-component log-rank aggregation. Its
  two-candidate ImageNet-224 GPU smoke writes two candidates plus one summary and resumable component
  cache. Architecture-hash initialization makes two independent same-seed GPU runs identical after
  timing fields are removed; see `docs/evidence/aznas_autoformer_rank_smoke.json`. The stabilized covariance clamp is
  explicitly versioned, and the project evolution controller is not a line-for-line upstream
  candidate controller. The `3×8,000` cohort and candidate freeze are complete; the post-completion
  supervisor failure remains as orchestration evidence. Real dual-one-percent V2 is non-terminal and
  must not be reported as accepted training.
- Explicit `--full-batch-smoke` completed one synthetic epoch for a sampled reference AutoFormer at
  the configured micro-batch 256. Peak allocated/reserved memory was `8920/10390 MiB`; see
  `docs/evidence/autoformer_full_batch_memory_smoke.json`. This proves only the single-process memory
  and training-step path, not ImageNet accuracy or the final frozen candidates. Another user's roughly
  10 GiB process shared the physical GPU, reinforcing that project locks are not system-wide reservations.
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

1. DARTS CIFAR-10/CIFAR-100 600-epoch **full-data accuracy reproduction** and multi-seed search-gain
   validation remain incomplete; only the scoped dual one-percent protocol above is accepted. The
   six scoped DARTS ImageNet dual one-percent runs are complete, with the documented DDP-versus-
   single-GPU BatchNorm caveat. Full-data 250-epoch training is outside this scoped acceptance and
   has not run.
2. PlainNet-MBV2 and Proxyless-MBV2 dual one-percent acceptance remains incomplete. AutoFormer's
   selected-candidate dual one-percent gate is complete and its profile launch gate is released, but
   full-data 500-epoch accuracy reproduction and multi-seed search-gain validation remain incomplete.
   PlainNet has only a three-accepted-candidate GPU preflight with `formal_search_completed=false`;
   Proxyless-MBV2 formal 150-epoch training is not released.
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
input hashes, explicit result type, failure rows and exact commands. For the open items above, claims
must remain limited to the software, scoped-protocol, or smoke evidence actually retained.

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

The three pinned public files were bootstrapped, converted to safe JSONL, checksum-locked, and
queried. Each slice has 100 records. The historical minimum-five × 22-proxy workflow produced the
expected 110 rows, but its AutoFormer scores used a Cream/AZ-NAS-style static model rather than the
pinned Auto-Prox static model. Those scores are retained only as legacy cross-implementation workflow
evidence and must be recomputed with `vitbench-autoprox-90ed458`; see
[`evidence/AUTOFORMER_FIDELITY_AUDIT_CN.md`](evidence/AUTOFORMER_FIDELITY_AUDIT_CN.md).

This does not close formal ViT-Bench H1. The paper describes 500 candidates per AutoFormer/PiT space
and a disjoint 60/40 development/test split; neither the complete candidates nor split identities are
published in the pinned repository. AutoFormer main, extension, and PiT remain separate, as do
vanilla, KD, and inherited-supernet metrics. PiT is now conservatively classified as
`reference_topology_pytorch_port`: structural, parameter, and MAC fixtures pass, but checkpoint and
layerwise numerical parity are unavailable.
