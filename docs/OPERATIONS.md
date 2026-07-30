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
on the command line. Use the separated spelling `--count 20`, rather than `--count=20`, so the
current explicit-option detector records the override. Config keys that are not parser attributes are
ignored; inspect each run's resolved `config.yaml` before treating it as research evidence. In
particular, `trusted: true` in YAML is rejected unless `--trusted` is also present on the CLI.

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
`<output>/YYYYMMDDTHHMMSSZ_<run-id>/` and prints that exact path as `run`. Use the printed path
for reports, monitoring and resume:

```bash
RUN=/path/to/runs/evaluate/YYYYMMDDTHHMMSSZ_runid
zcp-test report bundle "$RUN" --output "$RUN/reports/bundle"
zcp-test monitor "$RUN" --interval 5
```

Neither `report bundle` nor `monitor` recursively discovers timestamped runs below a parent
folder.

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
code-locked modes: full data with at most 1% of the formal epochs (at most five for the 500-epoch
AutoFormer profile), or at most 1% deterministic stratified data with the complete 500-epoch
schedule. It does not grant formal readiness. Batch and input size overrides are rejected, and a
real `--data-root` is mandatory:

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
tolerates an incomplete JSONL tail and atomically refreshes an auto-reloading HTML page.

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
