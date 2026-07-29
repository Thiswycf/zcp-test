# zcp-test

`zcp-test` is an independent, reproducible zero-cost proxy evaluation and neural architecture search toolkit. It separates benchmark identity from search-space identity, stores append-only JSONL artifacts, and never modifies the source projects used for reference.

See the [operations and research guide](docs/OPERATIONS.md) for GPU selection, evaluation semantics, custom proxies, analysis, monitoring and DARTS training.

## Environment

```bash
conda env create -f environment.yml
conda activate zcp-test
zcp-test doctor --catalog configs/data.example.json
```

The environment intentionally excludes TensorFlow, torchaudio, PyG, Jupyter, ROCm packages, and frozen transitive dependencies. Large benchmark files remain external assets.

Native serialized benchmarks and resumed checkpoints require an explicit `--trusted` acknowledgement. Use it only after verifying the source and checksum; the flag does not perform verification itself.

## Key commands

```bash
zcp-test data list --catalog configs/data.example.json
zcp-test data verify vitbench101_autoformer_main --catalog configs/data.example.json
zcp-test benchmark list
zcp-test space list
zcp-test proxy list
zcp-test gpu list

zcp-test evaluate --space darts --proxies er,naswot,synflow,gradnorm --count 5 --data-root DATA/cifar10
zcp-test search --space ofa_proxyless_mbv2 --proxy er --population 20 --generations 10
zcp-test train --config configs/training/darts_cifar10.yaml --epochs 1 --smoke
zcp-test report --source RUN/scores.jsonl --output RUN/reports/scores.csv
zcp-test report bundle RUN --output RUN/reports/bundle
zcp-test monitor RUN --interval 5
```

## Benchmark identities

| Benchmark | Search space | Ground-truth source |
|---|---|---|
| `nasbench201` | `nb201_topology` | NAS-Bench-201 API |
| `nats_tss` | `nb201_topology` | NATS-Bench TSS API |
| `nats_sss` | `nats_size` | NATS-Bench SSS API |
| `nasbench101` | `nb101_dag` | Converted safe JSONL |
| `nasbench301_surrogate` | `darts` | Deterministic surrogate by default |
| `transnasbench101` | `transnas_micro` / `transnas_macro` | Converted safe JSONL |
| `vitbench101` | `autoformer` / `pit` | Auto-Prox release slices |

NAS-Bench-201 and NATS-TSS share an architecture codec only. Their adapter, version, budget, split, and metrics are never substituted.

## Safe data conversion

Runtime adapters read JSONL. Pickle/PyTorch benchmark conversion is opt-in and requires `--trusted`:

```bash
zcp-test data convert-vit \
  --source /path/to/Auto-Prox-AAAI24/gt_results/gt_autoformer.pth \
  --output data/vitbench101/autoformer-main.jsonl \
  --slice-id autoformer_main --trusted
```

The main and extension AutoFormer slices remain distinct. Vanilla, knowledge-distillation, and inherited-supernet accuracy are separate metric protocols.

## Artifacts

Each run creates `YYYYMMDDTHHMMSSZ_<run-id>/` with `manifest.json`, resolved `config.yaml`, `events.jsonl`, human-readable `run.log`, command-specific JSONL files, checkpoints, and derived reports. JSONL is the source of truth; CSV and HTML are rebuildable views.

## Current boundaries

- `evaluate` supports range partitioning with `--start/--count`. Multi-GPU execution currently uses one process and disjoint range per GPU followed by JSONL merge; there is no built-in process launcher yet.
- NAS-Bench-101 download, conversion and adapter support are implemented, but the official TFRecord is not present on this host, so no real NB101 integration smoke was run.
- MobileNetV3 remains an optional adapter. Formal 150/300/600-epoch profiles are provided, while this build validation runs only short GPU smoke and checkpoint resume.

## Proxy capability policy

Every proxy is registered with model-family, label, device, component, direction, and dependency metadata. Calls run in a model-state isolation context. Unsupported combinations return `unsupported`; failures remain failures and are never replaced by fabricated losses or values.

## Validation scope

Unit tests use small fixtures. GPU smoke uses synthetic batches and short epochs. Full 150/300/600-epoch profiles are supplied but are not automatically launched.
