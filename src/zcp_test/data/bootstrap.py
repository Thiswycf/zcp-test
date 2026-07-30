from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
import tempfile
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Protocol


def _validate_relative_path(value: str, field: str) -> None:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must be a safe relative path")


class AssetState(str, Enum):
    READY = "ready"
    MISSING = "missing"
    PARTIAL = "partial"
    CORRUPT = "corrupt"
    INVALID = "corrupt"
    CONVERSION_REQUIRED = "conversion_required"
    SOURCE_UNAVAILABLE = "source_unavailable"
    UNAVAILABLE = "source_unavailable"
    FAILED = "failed"


@dataclass(frozen=True)
class BootstrapAsset:
    asset_id: str
    version: str
    path: str
    urls: tuple[str, ...] = ()
    sha256: str | None = None
    archive: str | None = None
    archive_name: str | None = None
    extract_root: str | None = None
    trusted_format: bool = False
    source_page: str | None = None

    def __post_init__(self) -> None:
        _validate_relative_path(self.path, "path")
        if self.archive not in {None, "tar", "zip"}:
            raise ValueError(f"Unsupported archive type: {self.archive}")
        if self.archive_name is not None:
            _validate_relative_path(self.archive_name, "archive_name")
        if self.extract_root is not None:
            _validate_relative_path(self.extract_root, "extract_root")
        if self.sha256 is not None:
            checksum = self.sha256.lower()
            if len(checksum) != 64 or any(character not in "0123456789abcdef" for character in checksum):
                raise ValueError("sha256 must contain 64 hexadecimal characters")

    def installed_path(self, data_root: str | Path) -> Path:
        return Path(data_root).expanduser() / self.path

    def download_path(self, data_root: str | Path) -> Path:
        if not self.archive:
            return self.installed_path(data_root)
        filename = self.archive_name or f"{self.asset_id}.tar"
        return Path(data_root).expanduser() / ".downloads" / filename


@dataclass(frozen=True)
class ChecklistItem:
    asset_id: str
    state: AssetState
    path: Path
    version: str
    detail: str | None = None
    partial_bytes: int = 0

    @property
    def ready(self) -> bool:
        return self.state is AssetState.READY


@dataclass(frozen=True)
class BootstrapResult:
    requested: tuple[str, ...]
    items: tuple[ChecklistItem, ...]

    @property
    def ok(self) -> bool:
        return all(item.ready for item in self.items)

    def by_id(self) -> dict[str, ChecklistItem]:
        return {item.asset_id: item for item in self.items}


BUILTIN_ASSETS: Mapping[str, BootstrapAsset] = {
    "nasbench101": BootstrapAsset(
        asset_id="nasbench101",
        version="full",
        path="nasbench101/nasbench_full.tfrecord",
        urls=("https://storage.googleapis.com/nasbench/nasbench_full.tfrecord",),
        sha256="3d64db8180fb1b0207212f9032205064312b6907a3bbc81eabea10db2f5c7e9c",
        source_page="https://github.com/google-research/nasbench",
    ),
    "nasbench201": BootstrapAsset(
        asset_id="nasbench201",
        version="1.1-096897",
        path="nasbench201/NAS-Bench-201-v1_1-096897.pth",
        urls=(
            "https://drive.usercontent.google.com/download"
            "?id=16Y0UwGisiouVRxW-W5hEtbxmcHw_0hF_&export=download&confirm=t",
        ),
        trusted_format=True,
        source_page="https://github.com/D-X-Y/NAS-Bench-201",
    ),
    "nats_tss": BootstrapAsset(
        asset_id="nats_tss",
        version="1.0-3ffb9",
        path="natsbench/NATS-tss-v1_0-3ffb9-simple",
        urls=(
            "https://drive.usercontent.google.com/download"
            "?id=17_saCsj_krKjlCBLOJEpNtzPXArMCqxU&export=download&confirm=t",
        ),
        archive="tar",
        archive_name="NATS-tss-v1_0-3ffb9-simple.tar",
        extract_root="natsbench",
        trusted_format=True,
        source_page="https://github.com/D-X-Y/NATS-Bench",
    ),
    "nats_sss": BootstrapAsset(
        asset_id="nats_sss",
        version="1.0-50262",
        path="natsbench/NATS-sss-v1_0-50262-simple",
        urls=(
            "https://drive.usercontent.google.com/download"
            "?id=1scOMTUwcQhAMa_IMedp9lTzwmgqHLGgA&export=download&confirm=t",
        ),
        archive="tar",
        archive_name="NATS-sss-v1_0-50262-simple.tar",
        extract_root="natsbench",
        trusted_format=True,
        source_page="https://github.com/D-X-Y/NATS-Bench",
    ),
    "transnasbench101": BootstrapAsset(
        asset_id="transnasbench101",
        version="v10141024",
        path="transnasbench101/transnas-bench_v10141024.pth",
        urls=(
            "https://drive.usercontent.google.com/download"
            "?id=1BV_BRMsCUVBtSVj4SN4QmA9Pjd35B2M4&export=download&confirm=t",
        ),
        trusted_format=True,
        source_page="https://drive.google.com/drive/folders/1HlLr2ihZX_ZuV3lJX_4i7q4w-ZBdhJ6o",
    ),
    "nasbench301_models": BootstrapAsset(
        asset_id="nasbench301_models",
        version="1.0",
        path="nasbench301/nb_models/xgb_v1.0",
        urls=("https://ndownloader.figshare.com/files/24992018",),
        sha256="e807411d6a454841965d3157a977896683b716dc48743049bd6be0ce94210824",
        archive="zip",
        archive_name="nasbench301_models_v1.0.zip",
        extract_root="nasbench301",
        trusted_format=True,
        source_page="https://figshare.com/articles/software/nasbench301_models_v1_0_zip/13061510",
    ),
    "vitbench101_autoformer_main": BootstrapAsset(
        asset_id="vitbench101_autoformer_main",
        version="auto-prox-90ed458",
        path="vitbench101/gt_autoformer.pth",
        urls=(
            "https://raw.githubusercontent.com/lliai/Auto-Prox-AAAI24/90ed458/"
            "gt_results/gt_autoformer.pth",
        ),
        sha256="712ad277546d9f7f565ce07885be7e0b98dcd8d0724fdd1120f595b517436eca",
        trusted_format=True,
        source_page="https://github.com/lliai/Auto-Prox-AAAI24/tree/90ed458",
    ),
    "vitbench101_autoformer_ext": BootstrapAsset(
        asset_id="vitbench101_autoformer_ext",
        version="auto-prox-90ed458",
        path="vitbench101/gt_autoformer_2.pth",
        urls=(
            "https://raw.githubusercontent.com/lliai/Auto-Prox-AAAI24/90ed458/"
            "gt_results/gt_autoformer_2.pth",
        ),
        sha256="05f5df6a41f338fb5f47eafebfc8758c75e451606856b278ccda1c60b26e7bca",
        trusted_format=True,
        source_page="https://github.com/lliai/Auto-Prox-AAAI24/tree/90ed458",
    ),
    "vitbench101_pit": BootstrapAsset(
        asset_id="vitbench101_pit",
        version="auto-prox-90ed458",
        path="vitbench101/gt_pit.pth",
        urls=(
            "https://raw.githubusercontent.com/lliai/Auto-Prox-AAAI24/90ed458/"
            "gt_results/gt_pit.pth",
        ),
        sha256="bdda89841d4105f99ab759e3243e7a2402929ba7a8430dac12a50256aa533bb2",
        trusted_format=True,
        source_page="https://github.com/lliai/Auto-Prox-AAAI24/tree/90ed458",
    ),
}


BENCHMARK_ASSETS: Mapping[str, tuple[str, ...]] = {
    "nasbench101": ("nasbench101",),
    "nasbench201": ("nasbench201",),
    "nats_tss": ("nats_tss",),
    "nats_sss": ("nats_sss",),
    "transnasbench101": ("transnasbench101",),
    "nasbench301_surrogate": ("nasbench301_models",),
    "vitbench101": (
        "vitbench101_autoformer_main",
        "vitbench101_autoformer_ext",
        "vitbench101_pit",
    ),
}


class _Response(Protocol):
    headers: Mapping[str, str]

    def read(self, size: int = -1) -> bytes: ...

    def getcode(self) -> int | None: ...

    def __enter__(self) -> "_Response": ...

    def __exit__(self, *args: object) -> None: ...


HttpOpener = Callable[..., _Response]
Downloader = Callable[..., Path]
Extractor = Callable[[str | Path, str | Path], tuple[Path, ...]]


def asset_checklist(
    data_root: str | Path,
    assets: Mapping[str, BootstrapAsset] = BUILTIN_ASSETS,
) -> tuple[ChecklistItem, ...]:
    return tuple(_check_asset(asset, data_root) for asset in assets.values())


def resumable_http_download(
    url: str,
    destination: str | Path,
    *,
    sha256: str | None = None,
    chunk_size: int = 1024 * 1024,
    timeout: float = 60,
    opener: HttpOpener = urllib.request.urlopen,
) -> Path:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    target = Path(destination).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and _checksum_matches(target, sha256):
        return target
    partial = Path(f"{target}.part")
    offset = partial.stat().st_size if partial.is_file() else 0
    headers = {"User-Agent": "zcp-test/0.1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)

    try:
        response_context = opener(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        if error.code == 416 and offset and _range_is_complete(error, offset, partial, sha256):
            partial.replace(target)
            return target
        raise

    with response_context as response:
        status = response.getcode() or getattr(response, "status", 200)
        append = offset > 0 and status == 206
        if append:
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {offset}-"):
                raise OSError(f"Unexpected Content-Range while resuming {url}: {content_range!r}")
        mode = "ab" if append else "wb"
        with partial.open(mode) as output:
            copied = _copy_response(response, output, chunk_size)
            output.flush()
            os.fsync(output.fileno())
        content_length = response.headers.get("Content-Length")
        if content_length is not None and copied != int(content_length):
            raise OSError(
                f"Incomplete HTTP response from {url}: expected {content_length} bytes, got {copied}"
            )

    if not _checksum_matches(partial, sha256):
        quarantine_file(partial)
        raise ValueError(f"Checksum mismatch while downloading {url}")
    partial.replace(target)
    return target


def safe_extract_tar(archive: str | Path, destination: str | Path) -> tuple[Path, ...]:
    archive_path = Path(archive).expanduser()
    target = Path(destination).expanduser()
    if target.is_symlink():
        raise ValueError(f"Extraction destination is a symlink: {target}")
    _reject_symlinks(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []

    with tarfile.open(archive_path, mode="r:*") as source:
        members = source.getmembers()
        for member in members:
            relative = _safe_tar_path(member.name)
            if not (member.isdir() or member.isfile()):
                raise ValueError(f"Unsafe tar member type: {member.name}")
            if relative.parts:
                extracted.append(target / Path(*relative.parts))

        with tempfile.TemporaryDirectory(prefix=f".{target.name}-", dir=target.parent) as temporary:
            staging = Path(temporary)
            source.extractall(staging, members=members)
            target.mkdir(parents=True, exist_ok=True)
            shutil.copytree(staging, target, dirs_exist_ok=True)
    return tuple(extracted)


def safe_extract_zip(archive: str | Path, destination: str | Path) -> tuple[Path, ...]:
    archive_path = Path(archive).expanduser()
    target = Path(destination).expanduser()
    if target.is_symlink():
        raise ValueError(f"Extraction destination is a symlink: {target}")
    _reject_symlinks(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with zipfile.ZipFile(archive_path) as source:
        members = source.infolist()
        for member in members:
            relative = _safe_tar_path(member.filename)
            mode = member.external_attr >> 16
            if mode and (mode & 0o170000) == 0o120000:
                raise ValueError(f"Unsafe zip symlink: {member.filename}")
            if relative.parts:
                extracted.append(target / Path(*relative.parts))
        with tempfile.TemporaryDirectory(prefix=f".{target.name}-", dir=target.parent) as temporary:
            staging = Path(temporary)
            source.extractall(staging)
            target.mkdir(parents=True, exist_ok=True)
            shutil.copytree(staging, target, dirs_exist_ok=True)
    return tuple(extracted)


def bootstrap_data(
    data_root: str | Path,
    asset_ids: Iterable[str] | None = None,
    *,
    assets: Mapping[str, BootstrapAsset] = BUILTIN_ASSETS,
    downloader: Downloader = resumable_http_download,
    extractor: Extractor = safe_extract_tar,
    force: bool = False,
) -> BootstrapResult:
    root = Path(data_root).expanduser()
    requested = tuple(asset_ids) if asset_ids is not None else tuple(assets)
    unknown = sorted(set(requested) - set(assets))
    if unknown:
        raise KeyError(f"Unknown bootstrap assets: {unknown}")
    results: list[ChecklistItem] = []

    for asset_id in requested:
        asset = assets[asset_id]
        current = _check_asset(asset, root)
        if current.ready and not force:
            results.append(current)
            continue
        if current.state is AssetState.CORRUPT and current.path.is_file():
            quarantine_file(current.path)
        if not asset.urls:
            results.append(
                ChecklistItem(
                    asset_id,
                    AssetState.UNAVAILABLE,
                    asset.installed_path(root),
                    asset.version,
                    f"Manual download required; see {asset.source_page}",
                )
            )
            continue

        download_path = asset.download_path(root)
        download_path.parent.mkdir(parents=True, exist_ok=True)
        error: Exception | None = None
        for url in asset.urls:
            try:
                downloader(url, download_path, sha256=asset.sha256)
                if asset.archive == "tar":
                    extractor(download_path, root / asset.extract_root if asset.extract_root else root)
                elif asset.archive == "zip":
                    safe_extract_zip(
                        download_path, root / asset.extract_root if asset.extract_root else root
                    )
                error = None
                break
            except (OSError, ValueError, urllib.error.URLError) as candidate:
                error = candidate
        if error is not None:
            results.append(
                ChecklistItem(
                    asset_id,
                    AssetState.FAILED,
                    asset.installed_path(root),
                    asset.version,
                    str(error),
                    _partial_size(download_path),
                )
            )
            continue
        result = _check_asset(asset, root)
        if not result.ready:
            result = ChecklistItem(
                asset_id,
                AssetState.FAILED,
                result.path,
                asset.version,
                "Download completed but the expected installed path is missing",
            )
        results.append(result)
    return BootstrapResult(requested, tuple(results))


def _check_asset(asset: BootstrapAsset, data_root: str | Path) -> ChecklistItem:
    installed = asset.installed_path(data_root)
    download = asset.download_path(data_root)
    partial_bytes = _partial_size(download)
    if not installed.exists():
        state = AssetState.PARTIAL if partial_bytes else AssetState.MISSING
        return ChecklistItem(asset.asset_id, state, installed, asset.version, partial_bytes=partial_bytes)
    if not asset.archive and installed.is_file() and not _checksum_matches(installed, asset.sha256):
        return ChecklistItem(asset.asset_id, AssetState.INVALID, installed, asset.version)
    return ChecklistItem(asset.asset_id, AssetState.READY, installed, asset.version)


def _checksum_matches(path: Path, expected: str | None) -> bool:
    if not path.is_file():
        return False
    if expected is None:
        return True
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest() == expected.lower()


def _copy_response(response: _Response, output: BinaryIO, chunk_size: int) -> int:
    copied = 0
    while chunk := response.read(chunk_size):
        output.write(chunk)
        copied += len(chunk)
    return copied


def _range_is_complete(
    error: urllib.error.HTTPError,
    offset: int,
    partial: Path,
    sha256: str | None,
) -> bool:
    if sha256 is not None:
        return _checksum_matches(partial, sha256)
    content_range = error.headers.get("Content-Range", "") if error.headers else ""
    prefix = "bytes */"
    if not content_range.startswith(prefix):
        return False
    try:
        return int(content_range[len(prefix) :]) == offset
    except ValueError:
        return False


def _partial_size(path: Path) -> int:
    partial = Path(f"{path}.part")
    return partial.stat().st_size if partial.is_file() else 0


def quarantine_file(path: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    quarantine = path.with_name(f"{path.name}.invalid-{timestamp}")
    path.replace(quarantine)
    return quarantine


def _safe_tar_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe tar member path: {value}")
    return path


def _reject_symlinks(root: Path) -> None:
    if not root.exists():
        return
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Extraction destination contains a symlink: {path}")


download_http = resumable_http_download
bootstrap_checklist = asset_checklist
bootstrap = bootstrap_data


__all__ = [
    "AssetState",
    "BENCHMARK_ASSETS",
    "BUILTIN_ASSETS",
    "BootstrapAsset",
    "BootstrapResult",
    "ChecklistItem",
    "asset_checklist",
    "bootstrap",
    "bootstrap_checklist",
    "bootstrap_data",
    "download_http",
    "quarantine_file",
    "resumable_http_download",
    "safe_extract_tar",
    "safe_extract_zip",
]
