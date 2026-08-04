from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DataAsset:
    asset_id: str
    path: str
    version: str
    sha256: str | None = None
    source_url: str | None = None
    protocol: str | None = None
    trusted: bool = False

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DataAsset":
        return cls(**value)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    target = Path(path)
    if target.is_file():
        return sha256_file(target, chunk_size)
    if not target.is_dir():
        raise FileNotFoundError(target)
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in target.rglob("*") if candidate.is_file()):
        relative = item.relative_to(target).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(item.stat().st_size.to_bytes(8, "big"))
        with item.open("rb") as handle:
            while chunk := handle.read(chunk_size):
                digest.update(chunk)
    return digest.hexdigest()


class DataRegistry:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()

    def list(self) -> list[DataAsset]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("schema_version") != 1 or not isinstance(payload.get("assets"), list):
            raise ValueError(f"Invalid data registry schema: {self.path}")
        return [DataAsset.from_dict(item) for item in payload["assets"]]

    def get(self, asset_id: str) -> DataAsset:
        for asset in self.list():
            if asset.asset_id == asset_id:
                return asset
        raise KeyError(f"Unknown data asset: {asset_id}")

    def register(self, asset: DataAsset, *, replace: bool = False) -> None:
        assets = {item.asset_id: item for item in self.list()}
        if asset.asset_id in assets and not replace:
            raise KeyError(f"Data asset already registered: {asset.asset_id}")
        assets[asset.asset_id] = asset
        self._write(sorted(assets.values(), key=lambda item: item.asset_id))

    def verify(self, asset_id: str) -> dict[str, Any]:
        asset = self.get(asset_id)
        path = Path(asset.path).expanduser()
        result: dict[str, Any] = {"asset_id": asset_id, "path": str(path), "exists": path.exists()}
        if not path.exists():
            result["valid"] = False
            return result
        result["size"] = path.stat().st_size if path.is_file() else None
        if asset.sha256:
            actual = sha256_path(path)
            result.update(actual_sha256=actual, valid=actual == asset.sha256)
        else:
            result["valid"] = True
        if asset.protocol == "imagenet16-120-official-md5-safe-conversion-v1":
            from zcp_test.data.imagenet16 import verify_safe_imagenet16

            runtime_integrity = verify_safe_imagenet16(path)
            result["runtime_integrity"] = runtime_integrity
            result["valid"] = bool(result["valid"] and runtime_integrity["valid"])
        return result

    def get_verified(
        self,
        asset_id: str,
        *,
        expected_version: str | None = None,
        expected_protocol: str | None = None,
    ) -> DataAsset:
        asset = self.get(asset_id)
        if expected_version is not None and asset.version != expected_version:
            raise ValueError(
                f"Data asset {asset_id!r} has version {asset.version!r}; "
                f"expected {expected_version!r}"
            )
        if expected_protocol is not None and asset.protocol != expected_protocol:
            raise ValueError(
                f"Data asset {asset_id!r} has protocol {asset.protocol!r}; "
                f"expected {expected_protocol!r}"
            )
        verification = self.verify(asset_id)
        if not verification["valid"]:
            raise ValueError(f"Data asset {asset_id!r} failed integrity verification")
        return asset

    def fetch(self, asset_id: str, destination: str | Path | None = None) -> Path:
        asset = self.get(asset_id)
        if not asset.source_url:
            raise ValueError(f"No source URL configured for {asset_id}")
        target = Path(destination or asset.path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".part")
        request = urllib.request.Request(asset.source_url, headers={"User-Agent": "zcp-test/0.1"})
        with urllib.request.urlopen(request) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        if asset.sha256 and sha256_file(temporary) != asset.sha256:
            temporary.unlink(missing_ok=True)
            raise ValueError(f"Checksum mismatch while fetching {asset_id}")
        temporary.replace(target)
        return target

    def _write(self, assets: list[DataAsset]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        payload = {"schema_version": 1, "assets": [asdict(asset) for asset in assets]}
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(self.path)
