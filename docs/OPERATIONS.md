# Operations and Research Guide

## Trust boundary

`--trusted` acknowledges that the operator has independently verified a serialized input. It does
not calculate a checksum, sandbox deserialization, or make pickle/PyTorch files safe. Native
NAS-Bench-201, NATS-TSS/SSS and NAS-Bench-301 queries, checkpoint resume, ViT conversion and
legacy pickle import require the acknowledgement on the command line:

```bash
zcp-test evaluate --config configs/benchmarks/nasbench201.yaml --trusted \
  --proxies params --count 1 --input-source random --device cpu
zcp-test train --config configs/training/darts_cifar10.yaml \
  --resume "$RUN/checkpoints/last.pt" --trusted
```

A config file cannot enable trusted execution by itself. Verify provenance and checksum first, then
pass `--trusted` for that invocation only. Do not use it in a shared shell alias.

## Configuration precedence

`evaluate`, `correlate`, `search` and legacy `report` accept `--config`. The file may either contain
the command mapping directly or place it below a command-named section:

```yaml
evaluate:
  benchmark: nasbench101
  benchmark_version: full
  proxies: params,naswot
  count: 10
```

The effective order is parser defaults, then matching config values, then options explicitly present
on the command line. Both `--count 20` and standard argparse spelling `--count=20` are recognized as
explicit overrides. Every command rejects unknown keys. `train` additionally accepts only the
model, optimizer, augmentation and protocol fields declared by the versioned training-profile
schema, so misspellings such as `learnng_rate` fail before a run starts. Training configs remain
subject to protocol validation; inspect each run's resolved `config.yaml` before treating it as
research evidence. In particular, `trusted: true` in YAML is rejected unless `--trusted` is also
present on the CLI.

## GPU selection and locking

The Conda environment persists `CUDA_DEVICE_ORDER=PCI_BUS_ID`. Run `zcp-test gpu list` to see
the `nvidia-smi` index, PCI order, UUID, bus ID, model, free memory, utilization and visible logical
index. GPU commands default to `--gpu auto`; selection ranks free memory descending, utilization
ascending and bus ID ascending.

```bash
zcp-test evaluate ... --gpu auto --min-free-memory 20480
zcp-test evaluate ... --gpu GPU-... --gpu-lock-timeout 30
zcp-test evaluate ... --device cpu
```

`--gpu-lock-timeout 0` is fail-fast. A positive value is the total number of seconds available for
acquiring an eligible same-user lock; negative values are invalid. Automatic selection tries the next
eligible GPU while time remains. An explicit index, UUID or bus ID never changes devices. The lock
under `~/.cache/zcp-test/gpu-locks/` coordinates only same-user processes following this protocol;
it is not a system-wide reservation. `--device` bypasses physical GPU selection and locking.

## Evaluation inputs and result types

Dataset input is the default and requires `--data-root` or a valid `dataset_<name>` catalog asset.
`--input-source random|noise` is an explicit ablation and is fingerprinted separately. Missing real
data is an error and never causes a synthetic fallback.

```bash
zcp-test evaluate --space nb201_topology --proxies er,naswot,synflow \
  --count 10 --data-root /path/to/data/cifar10 --output /path/to/runs/evaluate
```

Ten architectures and three proxies produce exactly 30 schema-2 score records; a multi-component
proxy stores one primary `score` and its complete `components` mapping in the same record. Keep
the result protocol explicit:

- **standard answer**: a published benchmark observation for a specified dataset, split, budget and
  seed/reduction;
- **surrogate**: a model prediction such as NAS-Bench-301, not a fully trained observation;
- **inherited**: a subnet metric evaluated with supernet weights;
- **scratch**: an independently trained architecture metric.

Do not pool these result types or substitute NAS-Bench-201 truth for NATS-TSS truth.

## Run directories

`--output` is a parent directory. Every command creates
`<output>/YYYYMMDDTHHMMSS+0800_<run-id>/` in `Asia/Shanghai` and prints that exact path as `run`. Use the printed path
for reports, monitoring and resume:

```bash
RUN=/path/to/runs/evaluate/YYYYMMDDTHHMMSS+0800_runid
zcp-test report bundle "$RUN" --output "$RUN/reports/bundle"
zcp-test monitor "$RUN" --interval 5
```

`report bundle` expands one level below a parent without direct artifacts and processes all
recognizable timestamped runs. `monitor` enters a parent only when it contains exactly one
recognizable run; pass the exact `RUN` when multiple runs exist. During training it prefers
`events.jsonl`: rank 0 writes `training_batch_progress` about every 30 seconds and at each split's
last batch, plus `training_epoch_completed` at epoch completion. `training.jsonl` remains one row
per completed epoch and is the canonical curve source. `rank_local_samples` is rank 0 local
progress, not an exact distributed sample count. Runs created before heartbeat support are not
backfilled. The same events are flushed to human-readable `run.log`; for a newly created run,
non-empty `events.jsonl` with a persistently empty `run.log` is a logging regression. For ImageNet's
1.28 million small files, inspect the actual mount with `findmnt -T`, run a full-epoch preflight and
point `--data-root` to a verified local SSD/NVMe copy when available. The CLI never hard-codes,
copies or silently switches between `/home`, `/public` or other machine-specific roots.
All new manifest, event, status and quarantine timestamps use explicit Beijing offsets
(`+08:00` in ISO fields and `+0800` in filenames). Historical `...Z_...` runs remain read-only
compatible and are not rewritten.
The four-GPU DARTS ImageNet acceptance launcher is
`tools/acceptance/run-darts-imagenet-dual-one-percent.sh`. It requires explicit
`ZCP_IMAGENET_ROOT`, `ZCP_DARTS_CANDIDATES` and four UUIDs in `ZCP_GPU_UUIDS`, validates the
1,000/1,281,167/50,000 layout, and writes under project-local `runs/acceptance` by default.

## Range partitioning and merge

`evaluate --start/--count` supports disjoint manual partitions. There is no built-in process launcher
or JSONL merge CLI. Keep each partition as an independent run with its manifest. Analysis accepts
multiple score files, and reports accept multiple run directories, so merging is normally unnecessary:

```bash
zcp-test analyze compare \
  --scores "$RUN_A/scores.jsonl" "$RUN_B/scores.jsonl" \
  --output /path/to/reports/partitions
zcp-test report bundle "$RUN_A" "$RUN_B" --output /path/to/reports/bundle
```

If a downstream consumer requires one file, merge only partitions with identical resolved protocol
fields and non-overlapping ranges. Use `zcp_test.artifacts.merge_jsonl` with an explicit unique key,
then verify the output count; do not use `cat`, because it cannot detect duplicate evaluations or a
partial trailing record. The merged file is a derived artifact and does not replace source manifests.

## Data registry fetch

`data fetch` downloads one catalog asset from its declared `source_url`; it is not the benchmark
bootstrap workflow:

```bash
zcp-test data fetch ASSET_ID \
  --catalog /path/to/data/catalog.json \
  --destination /path/to/data/file
```

The command writes `<destination>.part`, verifies the catalog SHA-256 when one is present, and
atomically replaces the destination. It fails if `source_url` is absent. Unlike `data bootstrap`, this
command does not expand benchmark groups, extract archives, convert native formats, register a new
path, or provide resumable range downloads. Without a declared checksum, completion does not
establish authenticity.

## Legacy pickle import

Legacy import is an explicit one-way conversion for already verified pickle files:

```bash
zcp-test legacy import --source verified.pkl --output converted.jsonl --trusted
```

Python pickle can execute code while loading. Run this only on a trusted, checksum-verified source
in an isolated environment. A list becomes one JSONL record per item; a mapping becomes
`{"key": ..., "value": ...}` records; other objects become one `{"value": ...}` record. This is a
shape-preserving migration, not schema validation. Inspect the JSONL before using it as scores or
targets, and never overwrite the source file.

## NATS-SSS CIFAR-100 and ImageNet16-120

Bootstrap NATS-SSS, create the deterministic 328-of-32,768 sample, and keep native benchmark access
explicitly trusted:

```bash
zcp-test benchmark sample nats_sss --catalog /path/to/data/catalog.json --trusted \
  --fraction 0.01 --seed 2026 --shards 4 --output /path/to/audit/nats-sss-1pct.json

zcp-test evaluate --benchmark nats_sss --catalog /path/to/data/catalog.json --trusted \
  --sample-manifest /path/to/audit/nats-sss-1pct.json --sample-shard 0 \
  --dataset ImageNet16-120 --target-metric accuracy --target-split valid \
  --epoch-budget 90 --metric-seed-reduction mean --target-direction maximize \
  --input-source dataset --input-size 16 --classes 120 --batch-size 16 \
  --proxies params,naswot,synflow --seed 2026 --gpu auto --output /path/to/runs
```

For dataset-specific CIFAR-100, use `--dataset cifar100 --data-root /path/to/cifar100
--input-size 32 --classes 100`. These runs recompute ZCPs on each dataset. The accepted target-only
study passes all 12 CIFAR-10-valid/CIFAR-100/ImageNet16 score shards to
`analyze benchmark --benchmark nats_sss --view size`; analysis preserves each source score and
`input_fingerprint` while joining new targets by architecture ID. It emits separate matrix,
stability, target-rank, and controlled-transfer tables. See the
[cross-dataset evidence](evidence/NATS_SSS_CROSS_DATASET_CN.md).
Missing `--trusted`, an official MD5 mismatch, a corrupt safe shard, a missing
`dataset_imagenet16_120` catalog entry, or an incorrect `ImageNet16-120` spelling fails explicitly.
See the [Chinese operations guide](OPERATIONS_CN.md) for full commands and
troubleshooting.

## Adding a proxy

Run `zcp-test proxy scaffold my_proxy` only from a writable source checkout or editable install. It
writes both `src/zcp_test/proxies/custom/my_proxy.py` and `tests/test_proxy_my_proxy.py`; a normal
read-only wheel/site-packages installation is not a supported scaffold target. Implement a
`ProxyCapability`, then run `zcp-test proxy validate my_proxy`. Multi-component formulas should
return `ProxyOutput(score=..., primary_component=..., components=...)`. Validation checks a small
synthetic model for finite output, model-state isolation and hook cleanup; it is not a benchmark-wide
scientific validation.

## Training and architecture files

Static reference models exist for `darts`, AutoFormer, PiT, PlainNet-MBV2 and Proxyless-MBV2, but
model fidelity and training-protocol readiness are separate gates. Only DARTS configurations
currently set `formal_training_ready: true`; all other spaces must remain smoke-only until their
listed protocol blockers are closed. This flag is not self-authorizing: non-smoke runs must also
match a code-owned approved protocol and its critical fields, and cannot override accepted batch or
input size. `--smoke` uses tiny synthetic loaders and validates plumbing, not accuracy.

`--acceptance-smoke` is mutually exclusive with `--smoke`, uses real data, and accepts only two
code-locked modes: full data with at least 1% of the formal epochs and no more than the complete
schedule (five epochs is the minimum for the 500-epoch AutoFormer profile), or exactly 1%
deterministic stratified data with the complete 500-epoch
schedule. It does not grant formal readiness. Batch and input size overrides are rejected, and a
real `--data-root` is mandatory:

The second mode computes an exact global target of `round(N * 0.01)` per split and allocates class
quotas with a largest-remainder rule; a fixed seed breaks equal-remainder ties. If the target is
smaller than the number of classes (for example, 500 samples from the 50,000-image ImageNet-1k
validation split), covering every class is mathematically impossible. The tool does not silently
inflate the subset to 2% by forcing one sample per class.

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=GPU-UUID-0,GPU-UUID-1
torchrun --standalone --nproc-per-node=2 -m zcp_test.cli train \
  --config configs/training/autoformer_imagenet.yaml \
  --acceptance-smoke --epochs 5 --data-fraction 1.0 \
  --architecture /path/to/autoformer-architecture.json \
  --data-root /path/to/imagenet1k --output /path/to/runs/acceptance
```

The second mode requires `--epochs 500 --data-fraction 0.01`. A tiny fixture made from real images
may validate DDP interruption and recovery mechanics, but must not be labelled as the full-data 1%
protocol. The retained two-rank fixture evidence has one interrupted run and one completed new run;
the recovered log contains epochs 0–4, `runtime.resume` records the checkpoint SHA-256 and source
run ID, and no `.tmp` checkpoint remains. Full ImageNet execution is still pending.
The checkpoint also embeds the small epoch-level `training_history`; if the original absolute log
path is unavailable after moving the checkpoint, a new run can still reconstruct a continuous log.
When the source JSONL exists, it remains the preferred record source.

Before launching a 6/3-epoch or full-schedule job, run one complete real-data epoch for every
profile and candidate:

```bash
zcp-test train --config configs/training/darts_cifar10.yaml \
  --real-data-preflight --epochs 1 --data-fraction 1.0 \
  --architecture ARCH.json --data-root DATA/cifar10 --output RUNS/preflight
```

This mode uses real data, the accepted batch, and the reference model, but records
`training_mode=real_data_preflight`. It never counts as either one-percent acceptance protocol and
requires exactly one epoch over the complete dataset. Each training row records train/validation
duration, sample throughput, peak allocated memory, and peak reserved memory for resource estimates.
For multiple training runs, `report bundle RUN...` writes `training.csv` with `source_run` labels
and compares validation accuracy, validation loss, epoch duration, and peak memory in separate
panels. Its result reports score and training row counts independently.
Training-only bundles do not create an empty `scores.csv`; derived files are created only for data
that is actually present.

The AutoFormer profile pins AZ-NAS commit `5e6683a2cfa5c6d0dc34a1317a842497ba7eae47`.
Repeated augmentation uses three repeats, and the effective LR follows
`base_lr * per_device_batch * world_size * accumulation / 512`; the published 8×256 launch therefore
uses `0.002`, not `0.0005`. Exact parameter-count and `official_complexity_ops` fixtures cover Cream
T/S/B and AZ-NAS Tiny/Small/Base. AZ-NAS Tiny reports 1,380,128,376 upstream operations, while THOP
reports 1,100,420,352 MACs and omits relative-position parameters; these remain separate columns and
the upstream value is not relabelled generic FLOPs. Multi-GPU training uses launcher-managed UUID
ordering and must not also pass `--device`:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-UUID-0,GPU-UUID-1,GPU-UUID-2,GPU-UUID-3 \
torchrun --standalone --nproc-per-node=4 -m zcp_test.cli train \
  --config configs/training/autoformer_imagenet.yaml \
  --smoke --epochs 1 --batch-size 2 --output /path/to/runs/training
```

Each process uses `cuda:LOCAL_RANK`; metrics are reduced across ranks, and only rank zero owns the
run manifest, JSONL and checkpoints. Auto accumulation preserves target global batch 2048, so four
GPUs at batch 256 use two micro-steps. Real mixed-4090D/4090 two-rank DARTS and AutoFormer smokes
passed, as did interruption and new-run recovery on a real-image fixture. The two full ImageNet 1%
protocols remain unaccepted, so AutoFormer formal training stays disabled.
The resolved config stores the Cream static-model commit
`b799630a29995163f282b15e2f38701160272fd1` separately from the AZ-NAS training-recipe commit;
one ambiguous implementation field must not overwrite either provenance.
This is an executable DDP plumbing smoke. Removing `--smoke` is intentionally rejected until the
remaining formal gate closes; the manual does not present a future formal command as currently usable.

Without `--architecture`, training samples an architecture from the configured space.
`--architecture` accepts either an existing JSON file or inline JSON; either a top-level `spec`
envelope or the compatible spec object itself is accepted:

```json
{
  "spec": {
    "normal": [["sep_conv_3x3", 0], ["skip_connect", 1]],
    "normal_concat": [2, 3, 4, 5],
    "reduce": [["max_pool_3x3", 0], ["sep_conv_5x5", 1]],
    "reduce_concat": [2, 3, 4, 5]
  }
}
```

The abbreviated example illustrates the envelope only; a DARTS genotype must contain the complete
valid edge list. Architecture specs are not portable across spaces. Formal training requires a real
`--data-root` or dataset catalog asset. Resume only the exact compatible run checkpoint and pass
`--trusted` explicitly.

## Analysis and monitoring

`zcp-test report bundle RUN... --output REPORT` creates rebuildable CSV, PNG, SVG and static
HTML. `zcp-test analyze correlation`, `zcp-test analyze compare` and
`zcp-test analyze sensitivity` cover bootstrap correlations, top-k overlap, validation-only rank
aggregation, transfer, cost/memory Pareto and sample-size convergence. `zcp-test analyze search`
and `zcp-test analyze training` render progression curves. `zcp-test monitor RUN --interval 5`
tolerates an incomplete JSONL tail and atomically refreshes an auto-reloading HTML page. For a
live training run it reads batch heartbeat events first; completed-epoch plots still read
`training.jsonl`.

## DARTS

`darts` is a standard normal/reduce genotype and cell implementation; the previous placeholder is
named `darts_toy_legacy`. ZCP evaluation uses the lightweight `zcp` profile. Formal profiles are
`configs/training/darts_cifar10.yaml`, `darts_cifar100.yaml`, `darts_imagenet.yaml` and
`tenas_imagenet.yaml`. Original DARTS ImageNet uses SGD 0.1 with per-epoch StepLR 0.97; the TE-NAS
retrain profile uses SGD 0.5 with cosine scheduling, so their results must not be pooled. Use
`--smoke --epochs 1` only for pipeline validation. Full 250/600-epoch training remains a separate,
high-cost acceptance stage.

## OFA-Proxyless inherited evaluation

The registered Proxyless spec follows the official supernet positional schema: 21 kernel and
expansion entries, five searchable stage depths, fixed width 1.3, and resolution 128–224 in steps of
four. Bootstrap the official model asset explicitly, then opt in to trusted inherited weights:

```bash
zcp-test data bootstrap --root /path/to/data \
  --benchmarks ofa_proxyless_supernet --catalog /path/to/data/catalog.json --yes
zcp-test evaluate --space ofa_proxyless_mbv2 --weight-mode ofa_inherited --trusted \
  --catalog /path/to/data/catalog.json --classes 1000 --proxies params,naswot \
  --count 2 --input-source dataset --dataset imagenet1k --data-root /path/to/imagenet1k \
  --bn-recalibration-batches 20 --bn-recalibration-batch-size 64 \
  --input-size 224 --gpu auto --output /path/to/runs/evaluate
```

The checkpoint is loaded once per command. Static subnets select active channels and apply the
official learned 7→5→3 kernel transforms. Score and search records identify inherited mode,
checkpoint digest, active positions and BN status. Current records deliberately state
`bn_recalibration_required=true` and `bn_recalibrated_batches=0` when calibration is omitted. When
enabled, deterministic non-overlapping real-data batches record every sample ID, transform, batch
count and SHA-256 fingerprint; missing data fails rather than falling back to random input. The
current `zcp-test-deterministic-v1` resize/center-crop protocol explicitly records
`official_protocol_match=false`. It supports reproducible ZCP comparisons but is not a claim of
published inherited accuracy until numerically compared with the official OFA data provider.

## TransNAS-Bench-101 task model contracts

For TransNAS, `dataset` selects a Taskonomy task rather than a generic classification dataset. The
current PyTorch port follows upstream commit `6d4231b`: scene/object classification produce 47/75
logits, room layout produces 9 regression values, jigsaw consumes `[B,9,3,64,64]` and produces 1000
logits, semantic segmentation produces `[B,17,256,256]`, and normal/autoencoder produce
`[B,3,256,256]`. Official parameter counts and complete parameter-shape multisets match for all seven
heads.

The tabular benchmark and Taskonomy inputs are separate assets. Bootstrap downloads the public
105 MB standard-answer file; it does not download Taskonomy. New Taskonomy access must follow the
[official distribution method](https://docs.omnidata.vision/starter_dataset_download.html#Examples)
and its [dataset EULA](https://github.com/StanfordVL/taskonomy/blob/master/data/LICENSE). After lawful
access, create a safe relative-path manifest and register one shared root:

The paper used a random 24-building, 120K-image split (80K/20K/20K), but the public release does not
contain a verifiable building split, final training config, or complete per-task transforms. A
user-supplied Taskonomy split therefore supports a real-data **contract protocol**, not an accepted
TransNAS reference-input reproduction. Formal H1 remains blocked unless the author split/config is
obtained and verified.

```bash
zcp-test data prepare-transnas-input \
  --data-root /path/to/taskonomy-transnas5k \
  --split-json /path/to/taskonomy-train-split.json \
  --split train --verify-files

zcp-test data register dataset_transnas_taskonomy /path/to/taskonomy-transnas5k \
  --version taskonomy-contract-v1 \
  --protocol licensed-external-taskonomy-manifest-v1 --trusted --replace
```

The loader rejects absolute/escaping paths and missing task assets, never substitutes CIFAR or
random inputs, and records sample IDs, upstream commit, manifest checksum, transform fidelity and
license boundary. Class-object/scene use the official final5k masks, and jigsaw uses the official
1,000-permutation `[B,9,3,64,64]` protocol. The deterministic evaluation transform is explicitly not
claimed to reproduce upstream training augmentation.

Formal validation targets are: class scene/object `valid_top1@25`, room layout `valid_loss@25`
(minimize), jigsaw `valid_top1@10`, segmentation `valid_mIoU@30`, and normal/autoencoder
`valid_ssim@30`. Micro and macro are finite populations of 4,096 and 3,256 architectures, so 1%
means 41 and 33 architectures. Analyze every task and space separately. Label-dependent proxies are
enabled for the three classification tasks; regression and dense tasks remain explicitly
`unsupported` until a source-backed ZCP loss contract is implemented. The transfer report includes
`score_coverage.csv` with ok/failed/unsupported/skipped counts and paired coverage.

## ViT-Bench-101 release-slice research

ViT-Bench fixed candidates and open AutoFormer search are different protocols. The fixed benchmark
queries released ground truth and does not retrain candidates. The public pinned Auto-Prox commit
contains three 100-record files: AutoFormer main, an insufficiently documented AutoFormer extension,
and PiT. They must remain separate, as must vanilla, KD, and inherited-supernet metrics. The paper
instead describes 500 AutoFormer and 500 PiT candidates with a disjoint 60/40 proxy-development/test
split; the public files do not identify that split. Therefore the current run is a
`partial_release_slice_preacceptance`, not formal paper-level H1 acceptance.

```bash
CATALOG=~/.config/zcp-test/data.json
DATA=/path/to/data
zcp-test data bootstrap --root "$DATA" --benchmarks vitbench101 --catalog "$CATALOG" --yes
zcp-test data checklist --root "$DATA" --catalog "$CATALOG" --json
zcp-test benchmark inspect vitbench101 --catalog "$CATALOG" \
  --slice-id autoformer_main --start 0 \
  --dataset cifar100 --split test --metric-name accuracy_vanilla
```

Generate a deterministic minimum-five manifest per public slice, then evaluate with a real registered
dataset. Five architectures × 22 proxies must produce 110 rows; the current Transformer capability
matrix yields 80 `ok`, 30 `unsupported`, and zero `failed` rows. A five-candidate correlation only
validates the execution path and must not be reported as a stable scientific result.

```bash
zcp-test benchmark sample vitbench101 --catalog "$CATALOG" \
  --slice-id autoformer_main --count 5 --seed 2026 \
  --output /path/to/audit/vit-main-minimum5.json
zcp-test evaluate --benchmark vitbench101 --slice-id autoformer_main \
  --catalog "$CATALOG" --sample-manifest /path/to/audit/vit-main-minimum5.json \
  --sample-shard 0 --dataset cifar100 --target-metric accuracy_vanilla \
  --target-split test --proxies params,flops,naswot,synflow,zen,zico \
  --seed 2026 --input-source dataset --data-root /path/to/cifar100 \
  --batch-size 2 --input-size 224 --classes 100 --gpu auto \
  --output /path/to/runs/evaluate
```

The PiT builder is `reference_topology_pytorch_port`, not `reference_model`. Patch embedding, QKV,
pooling, `LayerNorm(eps=1e-6)`, and the upstream drop-path schedule are covered by structural,
parameter-count, and MAC fixtures, but no official checkpoint or layerwise numerical parity test is
available. Use it for ZCP structure studies; do not describe it as official numerical reproduction.
Catalog-backed benchmark opening re-checks SHA-256, version, and protocol. Explicit
`--benchmark-path` is a caller-managed trust boundary.

## AutoFormer and MobileNetV2 dual-one-percent acceptance

AutoFormer scratch, ZenNAS PlainNet-MBV2, and Proxyless-MBV2 scratch use separate candidate and
training identities. Their launchers are respectively
`run-autoformer-imagenet-dual-one-percent.sh`,
`run-plainnet-mbv2-imagenet-dual-one-percent.sh`, and
`run-proxyless-mbv2-imagenet-dual-one-percent.sh`. Full-data minimum schedules are 5/500, 2/150,
and 2/150 epochs; the companion protocol runs the complete 500/150/150 schedule on an exact,
deterministic 1% data subset.

Each candidate directory must contain `zcp_selected.json`, `fixed_random.json`, and
`params_flops_matched.json`. Those labels require, respectively, a provenance-recorded ZCP search,
a fixed-seed sample, and an independently sampled candidate matched on both parameters and FLOPs.
Do not relabel published, hand-picked, or parameter-only candidates.

Freeze candidates from a completed search run, not from an arbitrary architecture file:

```bash
zcp-test acceptance freeze-candidates \
  --search-run /path/to/timestamped/search-run \
  --training-config configs/training/autoformer_imagenet.yaml \
  --seed 20260731 --pool-size 32 \
  --output /path/to/frozen-candidates/autoformer
```

The command requires a versioned search identity containing space, proxy/version, dataset, input
fingerprint, and seed, and verifies that the best architecture occurs in `search.jsonl`. The output
manifest locks all source and candidate checksums. MobileNet uses THOP MACs as its explicit compute
convention. AutoFormer instead uses the Cream/AZ-NAS `get_complexity` protocol and records
`generic_flops=false`; it must not be relabelled as generic FLOPs. Resource matching means log-ratio
proximity in parameters and the declared compute metric, not equivalent accuracy or latency.

```bash
export TZ=Asia/Shanghai
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export ZCP_IMAGENET_ROOT=/path/to/imagenet1k
export ZCP_TRAINING_CANDIDATES=/path/to/frozen-candidates/autoformer
export ZCP_GPU_UUIDS=GPU-...,GPU-...,GPU-...,GPU-...
setsid -f env ZCP_START_AT=1 \
  bash tools/acceptance/run-autoformer-imagenet-dual-one-percent.sh
```

The common launcher verifies the canonical ImageNet file counts, the config space/schedule, four
GPU UUID locks, candidate staging, and a clean Git worktree. New workflow times use
`Asia/Shanghai`. After interruption, audit the latest manifest/checkpoint and resume at the first
unfinished task with `ZCP_START_AT=2..6`; never relabel `interrupted` as `completed`. A passing dual
1% run validates the implementation and recovery protocol, not full-data/full-schedule paper
accuracy reproduction.

The common launcher defaults to `sequential_ddp`. After a real single-GPU memory smoke proves that
the profile's unchanged batch fits, set `ZCP_EXECUTION_STRATEGY=parallel_single_gpu` and
`ZCP_PARALLEL_SINGLE_GPU_ACCEPTED=yes` to schedule the six runs over four independent candidate
lanes. This never overrides batch, accumulation, or LR. The acceptance flag is a manual gate, not
automatic memory evidence; AutoFormer, PlainNet, and Proxyless require separate smokes.

After a real two-process-per-GPU forward/backward smoke also passes, the six runs may start at once:

```bash
export ZCP_EXECUTION_STRATEGY=packed_single_gpu
export ZCP_PACKED_SINGLE_GPU_ACCEPTED=yes
export ZCP_DATA_WORKERS=4
export ZCP_CPU_AFFINITIES='32-63,96-127;32-63,96-127;32-63,96-127;32-63,96-127'
```

This raises aggregate project throughput without changing any run's batch or LR. Verify combined
peak memory first and reduce workers to avoid CPU decode contention. The optional four affinity
lists correspond to the four GPU UUIDs and must be derived from `nvidia-smi topo -m`; never copy
host-specific CPU IDs to another machine or mechanically split a NUMA node into undersized groups.
On this host, a 16-logical-CPU-per-run trial reduced throughput by about 6–8%, while a short trial
with the complete shared NUMA-1 list did not establish a gain over baseline. The live jobs were
therefore fully reverted to the system affinity. Affinity remains an opt-in experiment, not a
default recommendation. Without packed-memory evidence, use `parallel_single_gpu`.

Explicit runtime tuning keys are `prefetch_factor`, `pin_memory`, `persistent_workers`,
`non_blocking_transfer`, `memory_format: channels_last`, `cudnn_benchmark`, and `allow_tf32`.
Defaults retain the prior protocol. Enable channels-last only after a CNN-specific smoke.
`cudnn_benchmark: true` is rejected with deterministic training; TF32 or nondeterministic settings
must define a new versioned protocol and must not silently resume or merge with older runs.

The formal DARTS ImageNet global batch is 128. Four-way DDP reduces this to 32 images per GPU,
which under-fills 4090-class devices; increasing the scientific batch merely to raise utilization
is not allowed. After the already completed first task, use
`resume-darts-imagenet-parallel-from-task2.sh` to run tasks 2–6 as four independent one-GPU lanes.
Every run retains global batch 128 and the locked LR, while different candidate/protocol runs execute
concurrently. Lane zero chains task 2 then task 6; the other lanes run tasks 3, 4, and 5.
