# zcp-test

## 中文主指引

`zcp-test` 是独立的零成本代理（ZCP）评估、相关性研究、进化搜索和搜索后训练工具。项目严格区分
`benchmark_id`、`search_space_id`、模型 fidelity 和指标协议；原始 benchmark、训练数据、run、
checkpoint 与转换后大文件均保存在仓库外部，不修改原有 `TER-Score` 项目。

### 研究边界

| 类型 | 对象 | 主要实验 | 禁止混用 |
|---|---|---|---|
| 有标准答案 benchmark | NB101、NB201、NATS-TSS/SSS、NB301 surrogate、TNB101、ViT-Bench-101 | ZCP—真值相关性、top-k 检索、结构偏置、预算/任务迁移 | 不对 benchmark 全候选重复完整训练；NB301 只能称 surrogate association |
| 无完整 tabular 真值的开放空间 | DARTS、AutoFormer、PlainNet-MBV2、Proxyless-MBV2 | validation-only ZCP 搜索、随机/资源匹配基线、选中架构从头训练 | 不进行伪造的“全空间相关性”；不得用 test 指标选架构 |
| inherited supernet | OFA 或 ViT inherited protocol | active subnet、静态导出、BN recalibration、inherited accuracy | inherited、scratch 与 predictor 指标不得合并 |

PiT 已按 Auto-Prox `90ed458` 发布编码实现三阶段静态参考拓扑；OFA-MBV3 也已按官方
Once-for-All 五阶段/20-block 编码实现独立静态子网和 BatchNorm recalibration。它们与
AutoFormer、两类 MBV2 虽已有静态参考模型，但正式训练或 inherited-supernet 协议仍有 blocker。
是否允许完整训练以配置中的 `formal_training_ready` 为准，而不是只看 `reference_model` 标签。

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
RUN=/path/to/runs/evaluate/YYYYMMDDTHHMMSSZ_runid
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
- [`docs/ADD_PROXY_CN.md`](docs/ADD_PROXY_CN.md)
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
RUN=/path/to/data/runs/evaluate/YYYYMMDDTHHMMSSZ_runid
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

## Artifacts

Each run creates `YYYYMMDDTHHMMSSZ_<run-id>/` with `manifest.json`, resolved `config.yaml`, `events.jsonl`, human-readable `run.log`, command-specific JSONL files, checkpoints, and derived reports. JSONL is the source of truth; CSV and HTML are rebuildable views.

## Current boundaries

- `evaluate` supports range partitioning with `--start/--count`. There is no built-in multi-process
  launcher or merge CLI. Launch disjoint ranges explicitly, retain each run manifest, and prefer
  multi-file analysis/report inputs. Merge only same-protocol partitions with a declared unique key;
  never concatenate heterogeneous JSONL and call it one run.
- Real index-0 queries completed for NAS-Bench-101, NAS-Bench-201, NATS-TSS/SSS, TransNAS micro/
  macro, NAS-Bench-301 performance surrogate, and ViT-Bench AutoFormer/PiT using the machine-local
  catalog. Other machines must bootstrap or register their own paths; repository configuration
  never stores host paths.
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

Every proxy is registered with model-family, label, device, component, direction, and dependency metadata. Calls run in a model-state isolation context. Unsupported combinations return `unsupported`; failures remain failures and are never replaced by fabricated losses or values.

## Validation scope

Unit tests use small fixtures. GPU smoke uses synthetic batches and short epochs. Full DARTS
250/600-epoch training and all high-cost benchmark/proxy sweeps remain unaccepted; see the
[acceptance report](docs/ACCEPTANCE.md).
