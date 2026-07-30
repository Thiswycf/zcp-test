# Benchmark data bootstrap and offline transfer

This guide covers benchmark ground-truth assets. Training datasets such as CIFAR-10,
CIFAR-100, and ImageNet-1k are separate: place them under `/path/to/data` yourself and pass
the corresponding `--data-root`. The runtime loaders use `download=False`, and `evaluate`
does not silently download either benchmark assets or training data.

## Five-step workflow

Use an explicit data root and catalog throughout a deployment. The examples below use
`/path/to/data`; replace its suffixes as needed, but keep the same root when running
`checklist`, `bootstrap`, `export-manifest`, and `import-manifest`.

### 1. Checklist before downloading

```bash
zcp-test data checklist --root /path/to/data
zcp-test data checklist --root /path/to/data --json \
  > /path/to/data/checklist-before.json
```

The table is a plan, not proof that a remote download works. JSON output additionally exposes
the expected raw paths, runtime paths, source pages or URLs, planning size, partial bytes,
free bytes at the filesystem probe, and a remediation command.

When using a non-default catalog, pass the same `--catalog /path/to/data/catalog.json` to
checklist, bootstrap, inspect, and evaluate. Files in `configs/benchmarks/*.yaml` are executable
evaluate configurations that resolve through `~/.config/zcp-test/data.json`; they contain no
developer-machine path. For example:

```bash
zcp-test evaluate --config configs/benchmarks/nasbench201.yaml --trusted \
  --proxies params --count 1 --input-source random --device cpu
```

The config identifies a native serialized benchmark, but it cannot acknowledge trust on the
operator's behalf. Native benchmark, checkpoint and pickle loading always requires `--trusted` on
the command line after independent provenance and checksum verification.

Possible states are:

| State | Exact meaning | Next action |
|---|---|---|
| `missing` | At least one expected raw asset is absent and no `.part` file was found. | Bootstrap that benchmark or place the official file at the listed raw path. |
| `partial` | At least one raw asset is absent and one or more download `.part` files exist. | Rerun the same bootstrap command to resume. |
| `corrupt` | A raw file with a built-in SHA-256 exists but does not match it. | Quarantine or remove it, then bootstrap again. |
| `conversion_required` | All raw assets satisfy the available checks, but at least one runtime path is absent. | Rerun bootstrap; it performs the required conversion. |
| `ready` | Either the selected root passes raw/runtime checks, or every required catalog entry passes path and declared SHA-256 verification. | Check `location`, then run a benchmark smoke test before a formal evaluation. |

### 2. Bootstrap only the benchmarks needed

Review the source, expected size, checksum coverage, upstream terms, and disk headroom before
confirming. Interactive use asks for confirmation; automation must pass `--yes` explicitly.

```bash
zcp-test data bootstrap \
  --root /path/to/data \
  --benchmarks nasbench101,nasbench201 \
  --catalog /path/to/data/catalog.json
```

```bash
zcp-test data bootstrap \
  --root /path/to/data \
  --benchmarks nasbench101,nasbench201 \
  --catalog /path/to/data/catalog.json \
  --yes
```

`--all` requests every built-in bootstrap group. It is usually slower and requires much more
disk than selecting one benchmark at a time:

```bash
zcp-test data bootstrap \
  --root /path/to/data \
  --all \
  --catalog /path/to/data/catalog.json \
  --yes
```

Bootstrap downloads to a `.part` path, publishes files by rename, converts formats when needed,
and registers ready runtime paths in the selected catalog. A JSON response with `"ok": true`
means that invocation completed its implemented checks; this document does not assert that any
remote asset has been downloaded successfully in your environment.

### 3. Verify readiness and run a smoke test

Run checklist again after bootstrap:

```bash
zcp-test data checklist --root /path/to/data --json \
  > /path/to/data/checklist-after.json
```

`verify --all` is intentionally strict: it exits with an error unless every built-in benchmark
group is `ready`. Do not use it to verify only one selected benchmark.

```bash
zcp-test data verify --all --root /path/to/data
```

For one catalog entry, `data verify` checks existence and a catalog SHA-256 when one was
registered:

```bash
zcp-test data list --catalog /path/to/data/catalog.json
zcp-test data verify nasbench101 --catalog /path/to/data/catalog.json
```

Catalog verification is not a substitute for checklist or a query smoke. Bootstrap-generated
runtime catalog entries currently have no SHA-256, so their registry verification proves path
existence only. `benchmark inspect` resolves paths from `--catalog`; native serialized formats
still require explicit `--trusted`. Minimal index-0 query smokes are:

```bash
CATALOG=/path/to/data/catalog.json
zcp-test benchmark inspect nasbench101 --catalog "$CATALOG" --dataset cifar10 --split valid --metric-name final_accuracy --epoch-budget 108
zcp-test benchmark inspect nasbench201 --trusted --catalog "$CATALOG" --dataset cifar10-valid --split valid --metric-name accuracy --epoch-budget 200
zcp-test benchmark inspect nats_tss --trusted --catalog "$CATALOG" --dataset cifar10-valid --split valid --metric-name accuracy --epoch-budget 200
zcp-test benchmark inspect nats_sss --trusted --catalog "$CATALOG" --dataset cifar10-valid --split valid --metric-name accuracy --epoch-budget 90
zcp-test benchmark inspect transnasbench101 --catalog "$CATALOG" --transnas-space micro --dataset class_object --split valid --metric-name valid_top1 --epoch-budget 25
zcp-test benchmark inspect transnasbench101 --catalog "$CATALOG" --transnas-space macro --dataset class_object --split valid --metric-name valid_top1 --epoch-budget 25
zcp-test benchmark inspect nasbench301_surrogate --trusted --catalog "$CATALOG" --dataset cifar10 --split test --metric-name accuracy
zcp-test benchmark inspect vitbench101 --catalog "$CATALOG" --slice-id autoformer_main --dataset cifar100 --split test --metric-name accuracy_vanilla
zcp-test benchmark inspect vitbench101 --catalog "$CATALOG" --slice-id pit --dataset cifar100 --split test --metric-name accuracy_vanilla

zcp-test evaluate \
  --benchmark nasbench101 \
  --catalog "$CATALOG" \
  --benchmark-version full \
  --proxies params \
  --count 1 \
  --input-source random \
  --device cpu \
  --output /path/to/data/smoke/nasbench101
```

Treat the smoke as successful only if each command exits zero, the adapter reports the expected
identity/version, and `query.value` is finite. The optional evaluation must write a non-failed
score record. These are wiring checks, not scientific validation of the complete benchmarks.

### 4. Export an offline-transfer manifest

`export-manifest` does not copy or archive data. It hashes each selected **runtime** file or
directory, records a root-relative path, and writes a manifest. Raw downloads and conversion
scratch state are not included.

```bash
zcp-test data export-manifest \
  --root /path/to/data/source \
  --benchmarks nasbench101,vitbench101 \
  --output /path/to/data/transfer/manifest.json
```

Copy the manifest and runtime data while preserving their paths relative to the source root.
For NAS-Bench-101, copy the entire `converted/full` directory, not only `manifest.json`:

```bash
mkdir -p /path/to/data/offline/nasbench101/converted
mkdir -p /path/to/data/offline/vitbench101
rsync -a \
  /path/to/data/source/nasbench101/converted/full \
  /path/to/data/offline/nasbench101/converted/
rsync -a \
  /path/to/data/source/vitbench101/converted \
  /path/to/data/offline/vitbench101/
```

Move `/path/to/data/transfer/manifest.json` and the copied data to the offline machine using your
approved transfer mechanism. The manifest contains checksums, not benchmark payloads.

`export-manifest` hashes the complete NAS-Bench-101 `converted/full` tree, including
`manifest.json`, `hash-index.json`, `offsets.bin`, and all `architectures-*.jsonl` shards. It also
hashes the complete converted directories for TransNAS-Bench-101 and ViT-Bench-101. Run the
adapter smoke after transfer as a semantic check in addition to the byte-level tree digest.

### 5. Import by verifying the transferred tree

Despite its name, `import-manifest` does not copy files and does not update the data catalog. It
verifies that every safe relative path exists under the destination root and that its digest
matches the export manifest.

```bash
zcp-test data import-manifest \
  --root /path/to/data/offline \
  --manifest /path/to/data/transfer/manifest.json

zcp-test data checklist --root /path/to/data/offline
```

Use explicit `--benchmark-path` values after transfer, or register verified runtime paths:

```bash
zcp-test data register \
  nasbench101 \
  /path/to/data/offline/nasbench101/converted/full/manifest.json \
  --version full \
  --protocol official-tfrecord-converted \
  --catalog /path/to/data/offline/catalog.json
```

### Fetching one registered asset

`data fetch` is a lower-level single-asset operation, not a replacement for bootstrap:

```bash
zcp-test data fetch ASSET_ID \
  --catalog /path/to/data/catalog.json \
  --destination /path/to/data/file
```

The asset must declare `source_url`. The command writes a `.part` file, verifies the catalog
SHA-256 when present, and atomically publishes the destination. It does not expand benchmark
groups, extract archives, convert native data, register a path, or provide bootstrap's resumable
download workflow. Without a declared checksum, successful transfer does not prove authenticity.

## What `ready` does and does not guarantee

A benchmark group is exactly `ready` through one of two routes:

1. Every built-in raw asset's installed path exists. If that asset has a built-in SHA-256 and
   is a file, the digest must match. Raw assets without a pinned digest pass this stage by
   existence alone; extracted archive directories also pass by existence.
2. Every runtime path declared for the benchmark exists.

Alternatively, all expected machine-local catalog IDs may resolve to existing runtime paths and
pass each catalog entry's declared SHA-256 check. This route reports
`catalog_state=external_ready` and `location=catalog_external`; it does not claim that raw files
exist below `--root`. Catalog entries without SHA-256 are existence-checked only.

The checklist deliberately does not deserialize native `.pth`/pickle benchmark files merely to
claim readiness: doing so would violate its read-only safety boundary and can be expensive.
Therefore `ready` is followed by the documented adapter smoke. NAS-Bench-101 conversion and
offline transfer manifests additionally bind the index, offsets, and shards with SHA-256.

State priority is `corrupt`, then `partial`/`missing`, then `conversion_required`, then `ready`.
Consequently, `ready` is a precise installation-state predicate, not proof of upstream
authenticity, license compliance, complete semantic integrity, or reproducibility of a target
paper result. Preserve provenance and perform the smoke appropriate to your protocol.

## Per-benchmark download plan

Planning sizes below are the estimates exposed by checklist. They are not exact transfer sizes
or sufficient-space guarantees. Conversion output, extracted archives, retained downloads,
`.part` files, and temporary databases can coexist.

| Benchmark group | Version / protocol boundary | Planning size | Built-in raw SHA-256 | Source and runtime result |
|---|---|---:|---|---|
| `nasbench101` | Official `full` TFRecord; budgets 4/12/36/108; converted to safe sharded JSONL | 2,085,986,016 B (about 1.94 GiB) | `3d64db8180fb1b0207212f9032205064312b6907a3bbc81eabea10db2f5c7e9c` | [Google NASBench](https://github.com/google-research/nasbench); runtime manifest at `/path/to/data/nasbench101/converted/full/manifest.json` |
| `nasbench201` | Native v1.1 `096897`; 12/200-epoch protocols | 4,700,000,000 B (about 4.38 GiB) | Not pinned | [NAS-Bench-201](https://github.com/D-X-Y/NAS-Bench-201); native `.pth`, therefore query only with explicit `--trusted` after independent verification |
| `nats_tss` | Topology search space, v1.0 `3ffb9`, 12/200 epochs | 1,100,000,000 B (about 1.02 GiB) | Not pinned | [NATS-Bench](https://github.com/D-X-Y/NATS-Bench); Google Drive tar, extracted native API directory |
| `nats_sss` | Size search space, v1.0 `50262`, 12/90 epochs | 1,100,000,000 B (about 1.02 GiB) | Not pinned | [NATS-Bench](https://github.com/D-X-Y/NATS-Bench); separate Google Drive tar and API directory from TSS |
| `transnasbench101` | Official v10141024; micro and macro are separate runtime tables | 105,000,000 B (about 100 MiB) | Not pinned | [Upstream Drive folder](https://drive.google.com/drive/folders/1HlLr2ihZX_ZuV3lJX_4i7q4w-ZBdhJ6o); trusted `.pth` conversion produces `transnas_micro.jsonl` and `transnas_macro.jsonl` |
| `nasbench301_surrogate` | Official surrogate models v1.0; performance/runtime remain separate | 1,848,669,012 B (about 1.72 GiB) | `e807411d6a454841965d3157a977896683b716dc48743049bd6be0ce94210824` | [Official Figshare](https://figshare.com/articles/software/nasbench301_models_v1_0_zip/13061510); safely extracted `xgb_v1.0` and `lgb_runtime_v1.0` |
| `vitbench101` | Auto-Prox commit `90ed458`; main AutoFormer, extension AutoFormer, and PiT remain separate | 62,925 B planning estimate | Three pinned hashes listed below | [Auto-Prox source](https://github.com/lliai/Auto-Prox-AAAI24/tree/90ed458); trusted `.pth` conversion produces three JSONL tables |

Run one group at a time when quota or storage is constrained:

```bash
zcp-test data bootstrap --root /path/to/data --benchmarks nasbench101 --catalog /path/to/data/catalog.json --yes
zcp-test data bootstrap --root /path/to/data --benchmarks nasbench201 --catalog /path/to/data/catalog.json --yes
zcp-test data bootstrap --root /path/to/data --benchmarks nats_tss --catalog /path/to/data/catalog.json --yes
zcp-test data bootstrap --root /path/to/data --benchmarks nats_sss --catalog /path/to/data/catalog.json --yes
zcp-test data bootstrap --root /path/to/data --benchmarks transnasbench101 --catalog /path/to/data/catalog.json --yes
zcp-test data bootstrap --root /path/to/data --benchmarks nasbench301_surrogate --catalog /path/to/data/catalog.json --yes
zcp-test data bootstrap --root /path/to/data --benchmarks vitbench101 --catalog /path/to/data/catalog.json --yes
```

### TransNAS Taskonomy input is not a bootstrap asset

The 105 MB TransNAS file above is the tabular standard answer only. It does not include RGB images
or seven-task labels. Taskonomy has a separate EULA that requires new access through its official
distribution method, so `zcp-test data bootstrap` deliberately does not download or redistribute it.
After lawful access, build a safe input manifest and register one shared root:

```bash
zcp-test data prepare-transnas-input \
  --data-root /path/to/taskonomy-transnas5k \
  --split-json /path/to/taskonomy-train-split.json \
  --split train --verify-files

zcp-test data register dataset_transnas_taskonomy /path/to/taskonomy-transnas5k \
  --version taskonomy-contract-v1 \
  --protocol licensed-external-taskonomy-manifest-v1 --trusted --replace
```

See [the Taskonomy download instructions](https://docs.omnidata.vision/starter_dataset_download.html#Examples),
[the dataset EULA](https://github.com/StanfordVL/taskonomy/blob/master/data/LICENSE), and the detailed
[TransNAS operations section](OPERATIONS.md#transnas-bench-101-task-model-contracts). A missing
licensed input root is an explicit blocker for formal TransNAS ZCP evaluation, not a reason to use
random or CIFAR data silently.
The published benchmark describes a 24-building, 120K-image split but does not release a verifiable
split or final transform configuration. Registered Taskonomy data is therefore contract-smoke input
unless those author artifacts are independently supplied and checked.

ViT-Bench-101 pins these source-file hashes:

| Slice | SHA-256 | Metric-protocol rule |
|---|---|---|
| `autoformer_main` | `712ad277546d9f7f565ce07885be7e0b98dcd8d0724fdd1120f595b517436eca` | Main AutoFormer slice; do not merge with the extension slice. |
| `autoformer_ext` | `05f5df6a41f338fb5f47eafebfc8758c75e451606856b278ccda1c60b26e7bca` | Extension slice; preserve its identity. |
| `pit` | `bdda89841d4105f99ab759e3243e7a2402929ba7a8430dac12a50256aa533bb2` | PiT search space. |

Vanilla, knowledge-distillation, and inherited-supernet accuracy are distinct metric protocols.
Do not combine them under one target metric.

NAS-Bench-301 surrogate models v1.0 are a built-in bootstrap group sourced from official
Figshare. Performance and runtime ensembles remain separate. Deterministic DARTS sampling is the
default candidate source; a fixed safe architecture JSONL remains optional. The `nb301` dependency
group pins the legacy-compatible `ConfigSpace==0.4.21` and supplies the official package's omitted
inference imports. It does not install PyG; unused GNN-surrogate training modules are not loaded.

Existing data may stay outside `--root`. Register absolute runtime paths in the machine-local
catalog; checklist then reports `catalog_state=external_ready` and `location=catalog_external`.
Repository configuration keeps `/path/to/data` placeholders and never stores host-specific paths.

## Source, checksum, trust, and upstream terms

- `checklist --json` is the authoritative view of URLs/source pages compiled into the installed
  version of `zcp-test`. Record that output with each acquisition.
- A missing built-in checksum means **not pinned**, not “checksum passed.” Obtain an upstream
  digest or generate and record an organization-approved digest before adding `--trusted`.
- `--trusted` permits loading native Python/PyTorch serialization. It does not verify origin,
  checksum, benchmark protocol, or safety.
- Bootstrap does not accept licenses or grant redistribution rights. Review the upstream source
  page, dataset/repository license, access policy, and citation requirements before downloading
  or transferring any asset. If terms are absent or ambiguous, resolve that with the upstream
  owner; do not infer rights from successful HTTP access.
- Preserve the source URL, retrieval date, exact byte size, SHA-256, benchmark version, split,
  epoch budget, seed reduction, and metric protocol in experiment provenance.

## Google Drive quota and interrupted downloads

NAS-Bench-201, both NATS groups, and TransNAS-Bench-101 use Google Drive URLs. The bootstrap path
uses `gdown` with resume enabled and writes `<destination>.part` (NATS tar files live under
`/path/to/data/.downloads`).

- On interruption, rerun the identical bootstrap command. Do not rename `.part` to the final
  name manually.
- If Drive reports quota exceeded, wait for quota recovery or download from the source page using
  an approved authenticated workflow. Place the result at the exact `raw_paths` location shown
  by checklist, then rerun bootstrap for conversion and registration.
- The tool does not bypass Drive quotas, HTML confirmation pages, or upstream access controls.
- If the server ignores a standard HTTP Range request, the HTTP downloader restarts the `.part`
  file instead of appending incompatible bytes.
- For assets without a pinned checksum, resumed completion cannot establish authenticity. Record
  an independently obtained digest before trusting the native format.

## Corruption and disk-space recovery

1. Save `checklist --json` output and inspect `state`, `raw_paths`, `runtime_paths`,
   `partial_bytes`, and `disk_probe`.
2. For `corrupt`, bootstrap renames the bad file to `.invalid-<timestamp>` before downloading a
   replacement. Remove quarantined files only after they are no longer needed for diagnosis. Do
   not use `--trusted` to suppress corruption.
3. For a suspicious asset without a pinned checksum, compare it with an upstream or locally
   approved digest. If it differs, remove both the final file and its `.part` before retrying.
4. For `partial`, preserve `.part` when it is a legitimate interrupted download; rerun to resume.
   Remove it only when its provenance is wrong or repeated resume attempts fail consistently.
5. For `conversion_required`, keep the raw source and rerun bootstrap. NAS-Bench-101 conversion
   uses a temporary SQLite state while ingesting and publishes a manifest only after output is
   written.

The checklist reports free space but does not reserve space or reject a bootstrap in advance.
Provision room for the raw file, `.part`, retained archive, extracted tree, converted runtime
data, and conversion temporary state at the same time. On `ENOSPC`, stop other writers, free
space without deleting the only verified raw source, and rerun checklist before resuming.

## NAS-Bench-101 dedicated safe interface

NAS-Bench-101 does not require TensorFlow at runtime. Its dedicated converter:

- streams official TFRecord framing and verifies CRC-32C by default;
- parses the benchmark's `ModelMetrics` protobuf with the project parser;
- records the source SHA-256 and size;
- uses restart offsets in a temporary SQLite database during ingestion;
- writes `architectures-*.jsonl`, `hash-index.json`, fixed-width `offsets.bin`, per-shard hashes,
  and an atomic manifest;
- removes conversion state after successful publication.

Bootstrap is the normal interface. For controlled conversion or debugging, the Python API is:

```python
from zcp_test.data.nasbench101 import convert_nasbench101, read_indexed_record

manifest = convert_nasbench101(
    "/path/to/data/nasbench101/nasbench_full.tfrecord",
    "/path/to/data/nasbench101/converted/full",
    benchmark_version="full",
)
record = read_indexed_record(
    "/path/to/data/nasbench101/converted/full",
    "<official-module-hash>",
)
```

Do not disable CRC verification merely to make a damaged TFRecord convert. A complete conversion
still needs the adapter smoke shown earlier. Release validation on 2026-07-30 completed the
documented index-0 queries for all listed adapters; every new machine must repeat them against its
own catalog because that result does not validate copied files or a future dependency stack.

## Why `evaluate` never bootstraps implicitly

Benchmark evaluation resolves an existing explicit `--benchmark-path` first, then a matching
catalog entry. If neither exists, it raises `FileNotFoundError` with checklist/bootstrap guidance.
It does not invoke bootstrap. `ZCP_DATA_ROOT` is used in that remediation message; it is not an
implicit benchmark-path resolver.

This separation prevents an evaluation job from unexpectedly consuming network quota, filling a
shared filesystem, accepting upstream terms, loading a newly downloaded trusted format, or
changing the data version during a run. Prepare and verify data as a separate auditable step.

## OFA-Proxyless supernet model asset

`ofa_proxyless_supernet` is an official model asset for an open search space, **not benchmark ground
truth**. The legacy `--benchmarks` option name is retained for compatibility but also accepts this
versioned asset group:

```bash
zcp-test data bootstrap --root /path/to/data \
  --benchmarks ofa_proxyless_supernet --catalog /path/to/data/catalog.json --yes
zcp-test data checklist --root /path/to/data --catalog /path/to/data/catalog.json
```

The asset comes from the Once-for-All release associated with commit `f03b267`, is `32,202,338`
bytes, and has SHA-256
`10ce40eec63dd020b4fa0096b1ff3c1e81e5b740446ddef6a59651bb36e6b907`. Runtime loading uses PyTorch
`weights_only=True` and additionally requires explicit `--trusted --weight-mode ofa_inherited`.
Ordinary evaluation never downloads it implicitly. Re-run checklist/bootstrap after cross-machine
copying so the new absolute path is registered locally.
