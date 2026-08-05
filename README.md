# zcp-test

## 中文主指引

`zcp-test` 是独立的零成本代理（ZCP）评估、相关性研究、进化搜索和搜索后训练工具。项目严格区分
`benchmark_id`、`search_space_id`、模型 fidelity 和指标协议；原始 benchmark、训练数据、run、
checkpoint 与转换后大文件均保存在仓库外部，不修改原有 `TER-Score` 项目。

### 研究边界

| 类型 | 对象 | 主要实验 | 禁止混用 |
|---|---|---|---|
| 发布标准答案 benchmark | NB101、NB201、NATS-TSS/SSS、TNB101 | ZCP—发布真值相关性、top-k 检索、结构偏置、预算/任务迁移 | 不对 benchmark 全候选重复完整训练；dataset/split/budget/task 必须分开 |
| 明确 surrogate benchmark | NB301 | ZCP—surrogate prediction 相关性、deterministic/noisy 敏感性 | 只能称 surrogate association，不能称真实训练标准答案 |
| 发布指标但 fidelity 混合 | ViT-Bench-101 | AutoFormer/PiT 分开，vanilla/KD/inherited 分协议相关性 | 不把 inherited、KD、vanilla 或扩展切片合成单一 accuracy |
| 无完整 tabular 真值的开放空间 | DARTS、AutoFormer、PlainNet-MBV2、Proxyless-MBV2 | validation-only ZCP 搜索、随机/资源匹配基线、选中架构从头训练 | 不进行伪造的“全空间相关性”；不得用 test 指标选架构 |
| inherited supernet | OFA 或 ViT inherited protocol | active subnet、静态导出、BN recalibration、inherited accuracy | inherited、scratch 与 predictor 指标不得合并 |

PiT 已按 Auto-Prox `90ed458` 发布编码实现三阶段静态参考拓扑；OFA-MBV3 也已按官方
Once-for-All 五阶段/20-block 编码实现独立静态子网和 BatchNorm recalibration。它们与
AutoFormer、两类 MBV2 虽已有静态参考模型，但正式训练或 inherited-supernet 协议仍有 blocker。
是否允许完整训练以配置中的 `formal_training_ready` 为准，而不是只看 `reference_model` 标签。
OFA Proxyless 的官方 active-subnet 域是 21 个 `ks/e` 位置、width 1.3、resolution 128–224；OFA
tutorial 的 20-block/5-stage 域属于 MobileNetV3，不能移植后称为官方 Proxyless tutorial。项目在
Proxyless 域上运行 ZCP + 通用进化控制器时固定标记为 `project_zcp_transfer`，不是官方 OFA 搜索协议。

Engineering search acceptance is capped at 1%: PlainNet evaluates 1,000 of the upstream 100,000
valid-candidate budget, while Proxyless evaluates 1,000 of a predeclared 100,000-evaluation project
budget. The latter is not 1% of the combinatorial search-space cardinality.

### 首次使用 checklist

```bash
conda env create -f environment.yml
conda activate zcp-test

zcp-test doctor --catalog ~/.config/zcp-test/data.json
zcp-test data checklist --root /path/to/data
zcp-test data bootstrap --root /path/to/data \
  --benchmarks nasbench101,nasbench201 --yes
zcp-test data checklist --root /path/to/data --json
```

普通 `evaluate` 不会自动下载大型标准答案。`ready` 可能表示数据位于所选 root，也可能表示本机 catalog
指向经过检查的外部路径；必须查看 checklist 的 `location`，并执行对应 benchmark 的 index-0 smoke。
完整数据自举、Google Drive 配额、checksum、转换恢复和离线迁移见
[`docs/DATA_BOOTSTRAP_CN.md`](docs/DATA_BOOTSTRAP_CN.md)。

### 常用研究流程

```bash
# 1. 确认 GPU 的物理 UUID、PCI Bus ID 与逻辑 cuda:0 映射
zcp-test gpu list

# 2. 真实 benchmark 小样本 ZCP；原生序列化资产必须显式 --trusted
zcp-test evaluate --config configs/benchmarks/nasbench201.yaml --trusted \
  --proxies params,naswot,synflow --count 10 \
  --input-source dataset --data-root /path/to/cifar10 \
  --output /path/to/runs/evaluate

# 3. 使用命令返回的准确 timestamp run，而不是 --output 父目录
RUN=/path/to/runs/evaluate/YYYYMMDDTHHMMSS+0800_runid
zcp-test analyze correlation --scores "$RUN/scores.jsonl" --output "$RUN/reports/correlation"
zcp-test analyze compare --scores "$RUN/scores.jsonl" --output "$RUN/reports/compare"
zcp-test report bundle "$RUN" --output "$RUN/reports/bundle"

# 4. 开放空间只用 validation 协议搜索
zcp-test search --space darts --proxy er --population 20 --generations 5 \
  --input-source dataset --data-root /path/to/cifar10 \
  --output /path/to/runs/search

# 5. 对搜索架构执行已放行的训练 profile
zcp-test train --config configs/training/darts_cifar10.yaml \
  --architecture /path/to/best_architecture.json \
  --data-root /path/to/cifar10 --output /path/to/runs/training
```

GPU、评估、分析、benchmark 定制研究、训练、代理扩展和验收状态分别见：

- [`docs/GPU_CN.md`](docs/GPU_CN.md)
- [`docs/EVALUATE_CN.md`](docs/EVALUATE_CN.md)
- [`docs/ANALYSIS_CN.md`](docs/ANALYSIS_CN.md)
- [`docs/BENCHMARK_STUDIES_CN.md`](docs/BENCHMARK_STUDIES_CN.md)
- [`docs/TRAINING_CN.md`](docs/TRAINING_CN.md)
- [`docs/PROXIES_CN.md`](docs/PROXIES_CN.md)
- [`docs/PROXY_OFFICIAL_AUDIT.md`](docs/PROXY_OFFICIAL_AUDIT.md) / [`docs/PROXY_OFFICIAL_AUDIT_CN.md`](docs/PROXY_OFFICIAL_AUDIT_CN.md)
- [`docs/ADD_PROXY.md`](docs/ADD_PROXY.md) / [`docs/ADD_PROXY_CN.md`](docs/ADD_PROXY_CN.md)
- [`docs/ACCEPTANCE_CN.md`](docs/ACCEPTANCE_CN.md)

实时任务与验收看板位于 [`panel/index.html`](panel/index.html)。推荐运行
`python -m http.server 8000 --directory panel` 后访问 `http://127.0.0.1:8000/`；页面会每 30 秒拉取
最新数据，也可手动刷新，不需要按 F5。

## English Quick Reference

`zcp-test` is an independent, reproducible zero-cost proxy evaluation and neural architecture search toolkit. It separates benchmark identity from search-space identity, stores append-only JSONL artifacts, and never modifies the source projects used for reference.

See the [benchmark data bootstrap guide](docs/DATA_BOOTSTRAP.md) before the first benchmark
evaluation. The [operations and research guide](docs/OPERATIONS.md) covers GPU selection,
evaluation semantics, custom proxies, analysis, monitoring, and DARTS training.
Benchmark-specific budget, topology, size, task-transfer, and ViT structure studies are documented
in the [benchmark research guide](docs/BENCHMARK_STUDIES.md).

## Environment

```bash
conda env create -f environment.yml
conda activate zcp-test
zcp-test doctor --catalog configs/data.example.json
```

The environment intentionally excludes TensorFlow, torchaudio, PyG, Jupyter, ROCm packages, and frozen transitive dependencies. Large benchmark files remain external assets.

Native serialized benchmarks and resumed checkpoints require an explicit `--trusted` acknowledgement. Use it only after verifying the source and checksum; the flag does not perform verification itself.

## First use: prepare benchmark data explicitly

`evaluate` never downloads benchmark or training data implicitly. Run data preparation as a
separate, auditable workflow. `ready` means either that the selected root passes the available
raw/runtime checks or that a valid machine-local catalog points to all required runtime assets.
In the latter case, checklist reports `catalog_state=external_ready` and
`location=catalog_external`; it does not imply that files were copied below `--root`. Native
pickle/PyTorch files are not deserialized by a read-only checklist, so run the documented adapter
smoke before research use.
For a file, `runtime_integrity=verified` means its file SHA-256 matches. For a directory, it means a
deterministic directory-tree digest matches; that digest is locked locally during a trusted
bootstrap and is not presented as an upstream-published checksum. Rerunning the same explicit
bootstrap preserves a valid external catalog path, skips ready downloads, and pins an `unpinned`
runtime before the next checklist.

```bash
# 1. Plan and inspect source, size, paths, partial downloads, and free space.
zcp-test data checklist --root /path/to/data

# 2. Download and convert only the benchmark needed.
zcp-test data bootstrap \
  --root /path/to/data \
  --benchmarks nasbench101 \
  --catalog /path/to/data/catalog.json \
  --yes

# 3. Verify installation state, then run the benchmark-specific smoke.
zcp-test data checklist --root /path/to/data --json \
  > /path/to/data/checklist-after.json
zcp-test benchmark inspect nasbench101 \
  --path /path/to/data/nasbench101/converted/full/manifest.json \
  --version full

# 4. Hash runtime data before an offline transfer.
zcp-test data export-manifest \
  --root /path/to/data \
  --benchmarks nasbench101 \
  --output /path/to/data/transfer/manifest.json

# 5. On the destination, verify files copied under the same relative paths.
zcp-test data import-manifest \
  --root /path/to/data/offline \
  --manifest /path/to/data/transfer/manifest.json
```

`export-manifest` does not copy data, and `import-manifest` only verifies the transferred tree;
it does not copy files or register a catalog. The detailed guide documents per-benchmark sources,
planning sizes, pinned and missing checksums, protocol boundaries, Google Drive quota and resume,
corruption and disk recovery, offline transfer, and the dedicated NAS-Bench-101 safe interface.

## Key commands

Research manuals:

- [Interactive task and acceptance panel](panel/index.html) — open directly in a browser, or run
  `python -m http.server 8000 --directory panel` and visit `http://127.0.0.1:8000/`.
- [CLI operations and safety boundaries](docs/OPERATIONS.md) / [中文](docs/OPERATIONS_CN.md)
- [搜索后完整训练操作手册（中文）](docs/TRAINING_CN.md)
- [Generic multi-proxy analysis](docs/ANALYSIS_CN.md)
- [Benchmark-specific studies](docs/BENCHMARK_STUDIES.md)
- [Paper evidence and extension boundaries](docs/RESEARCH_EVIDENCE.md)
- [Retained reproducible examples](examples/studies/README_CN.md)
- [Acceptance status](docs/ACCEPTANCE.md) / [中文验收状态](docs/ACCEPTANCE_CN.md)

The PlainNet-MBV2 engineering-acceptance search must not be replaced by a generic
`population × generations` example. Its stable one-percent entry point is:

```bash
zcp-test search --config configs/search/plainnet_mbv2_source_aligned.yaml \
  --flops-target 450m --gpu auto \
  --output /path/to/runs/search/plainnet-aznas-450m
```

This evaluates exactly 1,000 valid candidates, or 1% of the upstream 100,000-candidate budget,
with batch 64/224, source-aligned four-component log-rank, no crossover, and the explicit fidelity
`source_aligned_control_flow_port_truncated_one_percent_budget`. It is an engineering acceptance,
not a completed 100k AZ-NAS reproduction. The retained 450M/600M/1G runs each contain 1,000
candidate rows plus one summary row and are already complete; do not restart them merely to repeat
acceptance. See the [operations guide](docs/OPERATIONS.md#az-nas-plainnet-mbv2-search) for the
separate upstream 100k protocol and the one-percent boundary.

Before engineering training acceptance, freeze the selected architecture from a completed,
versioned one-percent search run:

```bash
zcp-test acceptance freeze-candidates \
  --search-run /path/to/timestamped/search-run \
  --training-config configs/training/autoformer_imagenet.yaml \
  --output /path/to/frozen-candidates/autoformer
```

The freeze utility may also emit research baselines, but the engineering gate reads and trains only
`zcp_selected.json`. It runs two jobs total for that architecture: full data with 1% of the formal
epochs, and exactly 1% data with the complete schedule. The operations manual documents provenance
checks, resource matching, dual-one-percent launchers, and interruption recovery.

```bash
zcp-test data list --catalog configs/data.example.json
zcp-test data verify vitbench101_0 --catalog configs/data.example.json
zcp-test benchmark list
zcp-test space list
zcp-test proxy list
zcp-test gpu list

zcp-test evaluate --space darts --proxies er,naswot,synflow,gradnorm \
  --count 5 --data-root /path/to/data/cifar10 --output /path/to/data/runs/evaluate
zcp-test search --space darts --proxy er --population 20 --generations 10 \
  --data-root /path/to/data/cifar10 --output /path/to/data/runs/search
zcp-test train --config configs/training/darts_cifar10.yaml --epochs 1 --smoke \
  --output /path/to/data/runs/training

# Copy the exact "run" value printed by evaluate; --output is only the parent directory.
RUN=/path/to/data/runs/evaluate/YYYYMMDDTHHMMSS+0800_runid
zcp-test report --source "$RUN/scores.jsonl" --output "$RUN/reports/scores.csv"
zcp-test report bundle "$RUN" --output "$RUN/reports/bundle"
zcp-test monitor "$RUN" --interval 5
```

`search` and `evaluate` default to `--input-source dataset`; they fail rather than silently using
synthetic input when `--data-root` or the corresponding `dataset_<name>` catalog asset is absent.
Native serialized benchmarks and checkpoint or legacy-pickle loading must be acknowledged with
`--trusted` on the command line. A YAML file is not allowed to enable trusted execution by itself.

## Benchmark identities

| Benchmark | Search space | Result type | Ground-truth source |
|---|---|---|---|
| `nasbench201` | `nb201_topology` | standard answer | NAS-Bench-201 API records |
| `nats_tss` | `nb201_topology` | standard answer | NATS-Bench TSS API records |
| `nats_sss` | `nats_size` | standard answer | NATS-Bench SSS API records |
| `nasbench101` | `nb101_dag` | standard answer | Safely converted official records |
| `nasbench301_surrogate` | `darts` | surrogate | Performance surrogate; deterministic unless noise is requested |
| `transnasbench101` | `transnas_micro` / `transnas_macro` | standard answer | Safely converted task-specific records |
| `vitbench101` | `autoformer` / `pit` | mixed | Auto-Prox release: scratch, distillation, or inherited-supernet metrics |

NAS-Bench-201 and NATS-TSS share an architecture codec only. Their adapter, version, budget,
split, and metrics are never substituted. A **standard answer** is a published benchmark record for
an explicit dataset/split/budget/seed protocol. A **surrogate** is a model prediction, not a fully
trained observation. An **inherited** metric evaluates a subnet using supernet weights. A **scratch**
metric comes from independently training that architecture. These result types must not be pooled.

## Safe data conversion

Runtime adapters read JSONL. Pickle/PyTorch benchmark conversion is opt-in and requires `--trusted`:

```bash
zcp-test data convert-vit \
  --source /path/to/data/Auto-Prox-AAAI24/gt_results/gt_autoformer.pth \
  --output /path/to/data/vitbench101/autoformer-main.jsonl \
  --slice-id autoformer_main --trusted
```

The main and extension AutoFormer slices remain distinct. Vanilla, knowledge-distillation, and inherited-supernet accuracy are separate metric protocols.

For NATS-SSS/ImageNet16-120, never use the release pickle as a runtime dataset. Verify the official
per-file MD5 values and convert it explicitly:

```bash
zcp-test data convert-imagenet16 \
  --source /path/to/raw/ImageNet16 \
  --output /path/to/data/datasets/ImageNet16-120-safe \
  --trusted --register --catalog /path/to/data/catalog.json
zcp-test data verify dataset_imagenet16_120 --catalog /path/to/data/catalog.json
```

The runtime is `npy-shards-v1`, loaded with `allow_pickle=False` and per-shard SHA-256 verification.
Dataset-specific CIFAR-100 and ImageNet16-120 ZCPs require separate evaluations. The accepted
12-shard `analyze benchmark --benchmark nats_sss --view size` study then keeps source scores and
input fingerprints fixed for target-only joins and emits separate matrix, stability, target-rank,
and controlled-transfer tables. Both new datasets completed 7,216/7,216 rows with zero duplicate
keys. See [operations](docs/OPERATIONS.md), [one-percent status](docs/ONE_PERCENT_ACCEPTANCE.md), and
the [cross-dataset evidence](docs/evidence/NATS_SSS_CROSS_DATASET_CN.md).

## Artifacts

Each new run creates `YYYYMMDDTHHMMSS+0800_<run-id>/` in the fixed `Asia/Shanghai` timezone, with `manifest.json`, resolved `config.yaml`, `events.jsonl`, human-readable `run.log`, command-specific JSONL files, checkpoints, and derived reports. ISO timestamps carry an explicit `+08:00` offset. Historical `...Z_...` runs remain readable and are not rewritten. JSONL is the source of truth; CSV and HTML are rebuildable views.

## Current boundaries

- `evaluate` supports range partitioning with `--start/--count`. There is no built-in multi-process
  launcher or merge CLI. Launch disjoint ranges explicitly, retain each run manifest, and prefer
  multi-file analysis/report inputs. Merge only same-protocol partitions with a declared unique key;
  never concatenate heterogeneous JSONL and call it one run.
- Real index-0 queries completed for NAS-Bench-101, NAS-Bench-201, NATS-TSS/SSS, TransNAS micro/
  macro, NAS-Bench-301 performance surrogate, and ViT-Bench AutoFormer/PiT using the machine-local
  catalog. Other machines must bootstrap or register their own paths; repository configuration
  never stores host paths.
- TransNAS tabular answers and Taskonomy inputs are separate assets. The raw/converted answers and
  41/33-architecture micro/macro 1% manifests are checksum-locked, and a safe seven-task input
  contract provider is implemented. The paper's 24-building/120K split and final config are not
  public, and separately licensed Taskonomy data is not present on this machine. The formal real-input
  22-proxy sweep therefore remains blocked; arbitrary Taskonomy splits and random/CIFAR fixtures are
  not substitutes.
  See the [TransNAS preflight evidence](docs/evidence/TRANSNAS_PREFLIGHT_CN.md).
- The formal NAS-Bench-101 1% scoped protocol is accepted on 4,237 stratified architectures:
  all 22 proxies at seed 2026 completed 93,214/93,214 task keys, and the core 11 proxies at seeds
  2026/2027/2028 completed 139,821/139,821 task keys. Budget-repeat analyses for `mean`, `min`, and
  `max` cover epochs 4/12/36/108. TE-NAS `portable-v2` remains a repository approximation rather
  than the complete official TE-NAS method. See the
  [human-readable evidence](docs/evidence/NB101_ONE_PERCENT_CN.md) and
  [machine-readable summary](docs/evidence/nb101_one_percent_summary.json).
- The locked NAS-Bench-301 deterministic-surrogate protocol is accepted on 1,000 stratified
  candidates from a reproducible 11,221-candidate generation corpus: all 22 proxies at seed 2026
  completed 22,000/22,000 task keys, and the core 11 proxies at three seeds completed
  33,000/33,000. This is `deterministic_on_locked_runtime` association under XGBoost 2.1.4 and
  nasbench301 0.3, not real-training ground truth or an exhaustive DARTS space. See the
  [human-readable evidence](docs/evidence/NB301_ONE_THOUSAND_CN.md) and
  [machine-readable summary](docs/evidence/nb301_one_thousand_summary.json).
- DARTS CIFAR-10/100 dual one-percent acceptance is complete for three frozen candidates: six
  full-data × 6-epoch runs and six exactly-1%-data × 600-epoch runs. The deterministic preflight,
  one trusted-checkpoint recovery audit per protocol, and reporting are complete; see the
  [evidence report](docs/evidence/DARTS_CIFAR_DUAL_ONE_PERCENT_CN.md) and
  [machine-readable summary](docs/evidence/darts_cifar_dual_one_percent_summary.json). This is not a
  600-epoch full-data accuracy reproduction or a multi-seed search-gain result. The protocols rank
  candidates differently and must not be averaged. The local ImageNet-1k asset now passes a structural
  audit (1,000 classes, 1,281,167 training images, and 50,000 validation images) plus a real-loader
  decode check. All six scoped DARTS ImageNet runs completed on 2026-07-31 with 759 epoch rows and a
  CSV/PNG/SVG/HTML bundle. The first candidate used four-GPU DDP while the other five used one GPU,
  so their per-device BatchNorm statistics are not strictly comparable. This remains scoped dual
  one-percent engineering evidence, not a 250-epoch full-data reproduction. The first attempt exposed
  slow-disk small-file I/O and an empty legacy `run.log`; current runs mirror events to both
  `events.jsonl` and `run.log`, and operators should explicitly select a verified fast local
  `--data-root`. AutoFormer and Proxyless-MBV2 selected-candidate dual one-percent acceptance are
  complete; PlainNet-MBV2 remains incomplete. Proxyless releases only the versioned candidate-resolution
  scratch profile, not an official 224 reproduction. For launches after 2026-08-04, engineering acceptance trains only the ZCP-selected
  architecture under the two 1% protocols (two runs total). Historical completed artifacts remain
  unchanged; queued baseline tasks that have not started must be cancelled rather than inherited from
  an old supervisor. Short acceptance training must not be used to claim
  superiority over random or parameter/FLOPs baselines; that requires a separate sufficiently
  trained, multi-seed research protocol.
- MobileNetV3 now has an official-structure static subnet and BN-recalibration utility, but inherited
  OFA checkpoints and a formal training profile remain unaccepted. AutoFormer now has a real repeated-
  augmentation sampler, the AZ-NAS linear LR rule (`base_lr * global_batch / 512`), and six exact
  Cream/AZ-NAS parameter-count goldens. Mixed 4090D/4090 two-rank DARTS and AutoFormer smokes now
  validate DDP wrapping, reduced metrics, one shared run and rank-zero-only artifacts. UUID-ordered
  `CUDA_VISIBLE_DEVICES` is mandatory, and AutoFormer can derive accumulation to retain global batch
  2048 (four GPUs × 256 uses two micro-steps). Six upstream `get_complexity` goldens are retained
  under the explicit `official_complexity_ops` name; a THOP cross-check proves that this is not a
  generic MAC/FLOPs value. A two-rank interruption/new-run recovery test using a tiny fixture made
  from real ImageNet images now validates interrupted manifests, checkpoint lineage, epoch
  de-duplication and temporary-file cleanup. It validates recovery mechanics, not full ImageNet
  training. Formal readiness remains blocked by the two ImageNet 1% acceptance protocols.

## Proxy capability policy

Every proxy is registered with model-family, label, device, component, direction, and dependency metadata. Accuracy-prediction direction and resource-constraint direction are separate fields; Params/FLOPs use `direction=maximize` for raw size-accuracy association and `resource_direction=minimize` for constraints. Calls run in a model-state isolation context. Unsupported combinations return `unsupported`; failures remain failures and are never replaced by fabricated losses or values.

## Validation scope

Unit tests use small fixtures. GPU smoke uses synthetic batches and short epochs. The scoped 1%
proxy sweeps for NB101, NB201, NATS-TSS, and NATS-SSS/CIFAR-10-valid are accepted under their
documented protocols; the scoped NB301 deterministic-surrogate protocol is also accepted. Project-wide
H1 remains incomplete: TransNAS-Bench-101, ViT-Bench-101, and other remaining protocols are pending.
The DARTS CIFAR dual one-percent protocol is accepted, but full-data 600-epoch CIFAR accuracy
reproduction and 250-epoch ImageNet training remain unaccepted; see the
[acceptance report](docs/ACCEPTANCE.md).

### ViT-Bench public-release boundary

The pinned Auto-Prox repository exposes three 100-record slices, not the paper's complete 500
AutoFormer plus 500 PiT candidate protocol or its disjoint 60/40 split identities. The project keeps
AutoFormer main, AutoFormer extension, and PiT separate and labels current evidence
`partial_release_slice_preacceptance`. PiT model construction is a
`reference_topology_pytorch_port`, not an official numerical reproduction. See
[`docs/OPERATIONS.md`](docs/OPERATIONS.md) and
[`docs/evidence/VITBENCH_PREFLIGHT_CN.md`](docs/evidence/VITBENCH_PREFLIGHT_CN.md).

Catalog-backed benchmark opening revalidates file SHA-256, version, and protocol. An explicit
`--benchmark-path` remains a caller-managed trust boundary.
