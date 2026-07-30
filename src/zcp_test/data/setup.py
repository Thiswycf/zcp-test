from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterable

from zcp_test.data.assets import DataAsset, DataRegistry, sha256_file
from zcp_test.data.bootstrap import (
    BENCHMARK_ASSETS,
    BUILTIN_ASSETS,
    AssetState,
    BootstrapResult,
    bootstrap_data,
    quarantine_file,
    resumable_http_download,
)
from zcp_test.data.converters import (
    convert_transnasbench101,
    convert_vitbench101,
    vitbench101_release_parser,
)
from zcp_test.data.nasbench101 import convert_nasbench101


BENCHMARK_SIZES = {
    "nasbench101": 2_085_986_016,
    "nasbench201": 4_700_000_000,
    "nats_tss": 1_100_000_000,
    "nats_sss": 1_100_000_000,
    "transnasbench101": 105_000_000,
    "nasbench301_surrogate": 1_848_669_012,
    "vitbench101": 62_925,
    "ofa_proxyless_supernet": 32_202_338,
}

CONVERTED_BENCHMARKS = {"nasbench101", "transnasbench101", "vitbench101"}

RUNTIME_PROTOCOLS: dict[str, tuple[str, ...]] = {
    "nasbench101": ("zcp-test-bootstrap",),
    "nasbench201": ("official-v1.1",),
    "nats_tss": ("tss-200",),
    "nats_sss": ("sss-90",),
    "transnasbench101": (
        "transnasbench101-micro-final",
        "transnasbench101-macro-final",
    ),
    "nasbench301_surrogate": (
        "deterministic-performance-surrogate",
        "deterministic-runtime-surrogate",
    ),
    "vitbench101": (
        "auto-prox-90ed458-autoformer-main",
        "auto-prox-90ed458-autoformer-ext",
        "auto-prox-90ed458-pit",
    ),
    "ofa_proxyless_supernet": ("official-inherited-supernet",),
}


def runtime_catalog_contract(benchmark: str, index: int) -> tuple[str, str | None]:
    version = str(BUILTIN_ASSETS[BENCHMARK_ASSETS[benchmark][0]].version)
    protocols = RUNTIME_PROTOCOLS.get(benchmark, ())
    protocol = protocols[index] if index < len(protocols) else None
    return version, protocol


def _runtime_paths(root: Path, benchmark: str) -> tuple[Path, ...]:
    paths = {
        "nasbench101": (root / "nasbench101/converted/full/manifest.json",),
        "nasbench201": (root / "nasbench201/NAS-Bench-201-v1_1-096897.pth",),
        "nats_tss": (root / "natsbench/NATS-tss-v1_0-3ffb9-simple",),
        "nats_sss": (root / "natsbench/NATS-sss-v1_0-50262-simple",),
        "transnasbench101": (
            root / "transnasbench101/converted/transnas_micro.jsonl",
            root / "transnasbench101/converted/transnas_macro.jsonl",
        ),
        "nasbench301_surrogate": (
            root / "nasbench301/nb_models/xgb_v1.0",
            root / "nasbench301/nb_models/lgb_runtime_v1.0",
        ),
        "vitbench101": (
            root / "vitbench101/converted/gt_autoformer.jsonl",
            root / "vitbench101/converted/gt_autoformer_2.jsonl",
            root / "vitbench101/converted/gt_pit.jsonl",
        ),
        "ofa_proxyless_supernet": (
            root / "ofa/ofa_proxyless_d234_e346_k357_w1.3",
        ),
    }
    return paths[benchmark]


def _transfer_paths(root: Path, benchmark: str) -> tuple[Path, ...]:
    directories = {
        "nasbench101": (root / "nasbench101/converted/full",),
        "transnasbench101": (root / "transnasbench101/converted",),
        "vitbench101": (root / "vitbench101/converted",),
    }
    return directories.get(benchmark, _runtime_paths(root, benchmark))


def _catalog_paths(
    catalog: Path, benchmark: str, runtime_paths: tuple[Path, ...]
) -> tuple[str, tuple[Path, ...], str]:
    if not catalog.is_file():
        return "missing", (), "missing"
    expected_ids = (
        (benchmark,)
        if len(runtime_paths) == 1
        else tuple(f"{benchmark}_{index}" for index in range(len(runtime_paths)))
    )
    registry = DataRegistry(catalog)
    try:
        registered = {asset.asset_id: asset for asset in registry.list()}
    except (OSError, TypeError, ValueError):
        return "corrupt", (), "corrupt"
    if not all(asset_id in registered for asset_id in expected_ids):
        return "missing", (), "missing"
    expected_paths = [path.resolve() for path in runtime_paths]
    actual_paths = [Path(registered[asset_id].path).expanduser().resolve() for asset_id in expected_ids]
    try:
        verified = []
        for index, asset_id in enumerate(expected_ids):
            expected_version, expected_protocol = runtime_catalog_contract(benchmark, index)
            registry.get_verified(
                asset_id,
                expected_version=expected_version,
                expected_protocol=expected_protocol,
            )
            verified.append({"valid": True})
    except (OSError, TypeError, ValueError):
        return "corrupt", tuple(actual_paths), "corrupt"
    if not all(item["valid"] for item in verified):
        return "corrupt", tuple(actual_paths), "corrupt"
    integrity = (
        "verified"
        if all(registered[asset_id].sha256 for asset_id in expected_ids)
        else "unpinned"
    )
    return (
        ("ready" if actual_paths == expected_paths else "external_ready"),
        tuple(actual_paths),
        integrity,
    )


def _aggregate_raw_state(asset_states: list[AssetState]) -> str:
    if any(state is AssetState.CORRUPT for state in asset_states):
        return "corrupt"
    if all(state is AssetState.READY for state in asset_states):
        return "ready"
    if any(state in {AssetState.READY, AssetState.PARTIAL} for state in asset_states):
        return "partial"
    return "missing"


def data_checklist(
    data_root: str | Path, catalog: str | Path | None = None
) -> list[dict[str, Any]]:
    root = Path(data_root).expanduser().resolve()
    catalog_path = Path(catalog).expanduser() if catalog else Path.home() / ".config/zcp-test/data.json"
    probe = root if root.exists() else root.parent
    free_bytes = shutil.disk_usage(probe).free if probe.exists() else None
    records = []
    for benchmark, asset_ids in BENCHMARK_ASSETS.items():
        asset_states = []
        partial_bytes = 0
        sources = []
        for asset_id in asset_ids:
            asset = BUILTIN_ASSETS[asset_id]
            installed = asset.installed_path(root)
            download = asset.download_path(root)
            partial = Path(f"{download}.part")
            partial_bytes += partial.stat().st_size if partial.is_file() else 0
            sources.extend(asset.urls)
            if asset.source_page:
                sources.append(asset.source_page)
            if installed.exists():
                if asset.sha256 and installed.is_file():
                    asset_states.append(
                        AssetState.READY
                        if sha256_file(installed) == asset.sha256
                        else AssetState.CORRUPT
                    )
                else:
                    asset_states.append(AssetState.READY)
            else:
                asset_states.append(AssetState.PARTIAL if partial.is_file() else AssetState.MISSING)
        runtime_paths = _runtime_paths(root, benchmark)
        catalog_state, catalog_paths, catalog_integrity = _catalog_paths(
            catalog_path, benchmark, runtime_paths
        )
        raw_state = _aggregate_raw_state(asset_states)
        local_runtime_ready = all(path.exists() for path in runtime_paths)
        if catalog_state == "corrupt":
            runtime_state = "corrupt"
            runtime_integrity = "corrupt"
        elif catalog_state in {"ready", "external_ready"}:
            runtime_state = "ready"
            runtime_integrity = catalog_integrity
            runtime_paths = catalog_paths
        elif local_runtime_ready:
            runtime_state = "ready"
            runtime_integrity = "unpinned"
        elif raw_state == "ready":
            runtime_state = "conversion_required"
            runtime_integrity = "missing"
        else:
            runtime_state = "missing"
            runtime_integrity = "missing"
        if (
            benchmark not in CONVERTED_BENCHMARKS
            and raw_state == "missing"
            and runtime_state == "ready"
            and catalog_state in {"ready", "external_ready"}
        ):
            raw_state = "ready"
        operational_ready = runtime_state == "ready"
        if raw_state == "corrupt" or runtime_state == "corrupt":
            state = AssetState.CORRUPT
        elif (
            benchmark in CONVERTED_BENCHMARKS
            and raw_state in {"missing", "partial"}
            and operational_ready
        ):
            state = AssetState.PARTIAL
        elif raw_state == "partial":
            state = AssetState.PARTIAL
        elif raw_state == "missing":
            state = AssetState.MISSING
        elif runtime_state != "ready":
            state = AssetState.CONVERSION_REQUIRED
        else:
            state = AssetState.READY
        if (
            benchmark not in CONVERTED_BENCHMARKS
            and state in {AssetState.MISSING, AssetState.CONVERSION_REQUIRED}
            and catalog_state in {
            "ready",
            "external_ready",
            }
        ):
            state = AssetState.READY
            runtime_paths = catalog_paths
        records.append(
            {
                "benchmark_id": benchmark,
                "version": BUILTIN_ASSETS[asset_ids[0]].version,
                "state": state.value,
                "raw_state": raw_state,
                "runtime_state": runtime_state,
                "runtime_integrity": runtime_integrity,
                "operational_ready": operational_ready,
                "raw_paths": [str(BUILTIN_ASSETS[item].installed_path(root)) for item in asset_ids],
                "runtime_paths": [str(path) for path in runtime_paths],
                "catalog_path": str(catalog_path),
                "catalog_state": catalog_state,
                "location": "catalog_external"
                if catalog_state == "external_ready"
                else "data_root",
                "estimated_bytes": BENCHMARK_SIZES[benchmark],
                "partial_bytes": partial_bytes,
                "sources": list(dict.fromkeys(sources)),
                "remediation": (
                    None
                    if state is AssetState.READY
                    else f"zcp-test data bootstrap --root {root} --benchmarks {benchmark}"
                ),
                "disk_probe": free_bytes,
            }
        )
    return records


def _expand_benchmarks(benchmarks: Iterable[str]) -> tuple[str, ...]:
    requested = tuple(dict.fromkeys(benchmarks))
    unknown = sorted(set(requested) - set(BENCHMARK_ASSETS))
    if unknown:
        raise KeyError(f"Unknown benchmark data groups: {unknown}")
    return tuple(asset for benchmark in requested for asset in BENCHMARK_ASSETS[benchmark])


def _convert(root: Path, benchmark: str) -> None:
    if benchmark == "nasbench101":
        convert_nasbench101(
            root / "nasbench101/nasbench_full.tfrecord",
            root / "nasbench101/converted/full",
            benchmark_version="full",
        )
    elif benchmark == "transnasbench101":
        source = root / "transnasbench101/transnas-bench_v10141024.pth"
        destination = root / "transnasbench101/converted"
        destination.mkdir(parents=True, exist_ok=True)
        for space in ("micro", "macro"):
            convert_transnasbench101(
                source, destination / f"transnas_{space}.jsonl", space=space, trusted=True
            )
    elif benchmark == "vitbench101":
        destination = root / "vitbench101/converted"
        destination.mkdir(parents=True, exist_ok=True)
        for source_name, output_name, slice_id in (
            ("gt_autoformer.pth", "gt_autoformer.jsonl", "autoformer_main"),
            ("gt_autoformer_2.pth", "gt_autoformer_2.jsonl", "autoformer_ext"),
            ("gt_pit.pth", "gt_pit.jsonl", "pit"),
        ):
            convert_vitbench101(
                root / "vitbench101" / source_name,
                destination / output_name,
                slice_id=slice_id,
                parser=vitbench101_release_parser,
                trusted=True,
            )


def _download(url: str, destination: Path, *, sha256: str | None) -> Path:
    if "drive.google.com" not in url and "drive.usercontent.google.com" not in url:
        return resumable_http_download(url, destination, sha256=sha256)
    import gdown

    partial = Path(f"{destination}.part")
    partial.parent.mkdir(parents=True, exist_ok=True)
    downloaded = gdown.download(url=url, output=str(partial), quiet=False, resume=True, fuzzy=True)
    if downloaded is None or not partial.is_file():
        raise OSError(
            "Google Drive download failed or quota was exceeded; use the source page shown by "
            "`zcp-test data checklist --json` and rerun bootstrap"
        )
    if sha256:
        from zcp_test.data.assets import sha256_file

        if sha256_file(partial) != sha256:
            quarantine_file(partial)
            raise ValueError(f"Checksum mismatch while downloading {destination.name}")
    partial.replace(destination)
    return destination


def _register_catalog(root: Path, catalog: Path, benchmarks: Iterable[str]) -> None:
    registry = DataRegistry(catalog)
    for benchmark in benchmarks:
        runtime_paths = _runtime_paths(root, benchmark)
        if not all(path.exists() for path in runtime_paths):
            continue
        for index, runtime_path in enumerate(runtime_paths):
            version, protocol = runtime_catalog_contract(benchmark, index)
            suffix = "" if len(runtime_paths) == 1 else f"_{index}"
            registry.register(
                DataAsset(
                    f"{benchmark}{suffix}",
                    str(runtime_path),
                    str(version),
                    sha256=sha256_file(runtime_path) if runtime_path.is_file() else None,
                    protocol=protocol,
                    trusted=benchmark
                    in {
                        "nasbench201",
                        "nats_tss",
                        "nats_sss",
                        "nasbench301_surrogate",
                        "ofa_proxyless_supernet",
                    },
                ),
                replace=True,
            )


def bootstrap_benchmarks(
    data_root: str | Path,
    benchmarks: Iterable[str],
    *,
    catalog: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(data_root).expanduser().resolve()
    selected = tuple(dict.fromkeys(benchmarks))
    catalog_path = Path(catalog).expanduser() if catalog else Path.home() / ".config/zcp-test/data.json"
    existing = {
        item["benchmark_id"]: item
        for item in data_checklist(root, catalog_path)
        if item["benchmark_id"] in selected
    }
    prepare = tuple(
        benchmark for benchmark in selected if existing[benchmark]["state"] != AssetState.READY.value
    )
    asset_ids = _expand_benchmarks(prepare)
    result: BootstrapResult = (
        bootstrap_data(root, asset_ids, downloader=_download)
        if asset_ids
        else BootstrapResult((), ())
    )
    failed = [asdict(item) for item in result.items if not item.ready]
    if failed:
        return {"ok": False, "downloads": [asdict(item) for item in result.items], "failed": failed}
    for benchmark in prepare:
        if any(not path.exists() for path in _runtime_paths(root, benchmark)):
            _convert(root, benchmark)
    _register_catalog(root, catalog_path, selected)
    checklist = [
        item for item in data_checklist(root, catalog_path) if item["benchmark_id"] in selected
    ]
    incomplete = [item for item in checklist if item["state"] != AssetState.READY.value]
    return {
        "ok": not incomplete,
        "benchmarks": selected,
        "catalog": str(catalog_path),
        "checklist": checklist,
        "failed": incomplete,
    }


def _path_digest(path: Path) -> str:
    digest = hashlib.sha256()
    files = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    for item in files:
        digest.update(item.relative_to(path.parent if path.is_file() else path).as_posix().encode())
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def export_data_manifest(
    data_root: str | Path,
    destination: str | Path,
    benchmarks: Iterable[str],
) -> Path:
    root = Path(data_root).expanduser().resolve()
    selected = tuple(dict.fromkeys(benchmarks))
    records = []
    for benchmark in selected:
        for path in _transfer_paths(root, benchmark):
            if not path.exists():
                raise FileNotFoundError(f"Cannot export missing benchmark data: {path}")
            records.append(
                {
                    "benchmark_id": benchmark,
                    "path": path.relative_to(root).as_posix(),
                    "kind": "file" if path.is_file() else "directory",
                    "sha256": _path_digest(path),
                }
            )
    target = Path(destination).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps({"schema_version": 1, "records": records}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


def verify_data_manifest(data_root: str | Path, manifest: str | Path) -> dict[str, Any]:
    root = Path(data_root).expanduser().resolve()
    payload = json.loads(Path(manifest).expanduser().read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("records"), list):
        raise ValueError("Invalid zcp-test data transfer manifest")
    records = []
    for record in payload["records"]:
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe data manifest path: {relative}")
        path = root / relative
        exists = path.exists()
        actual = _path_digest(path) if exists else None
        records.append(
            {
                **record,
                "exists": exists,
                "actual_sha256": actual,
                "valid": exists and actual == record["sha256"],
            }
        )
    return {"valid": all(record["valid"] for record in records), "records": records}


__all__ = [
    "BENCHMARK_SIZES",
    "bootstrap_benchmarks",
    "data_checklist",
    "export_data_manifest",
    "verify_data_manifest",
]
