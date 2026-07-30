from __future__ import annotations

import hashlib
import json
import pickle
from bisect import bisect_right
from pathlib import Path
import shutil
from typing import Any

import numpy as np


IMAGENET16_FILES: dict[str, tuple[tuple[str, str], ...]] = {
    "train": (
        ("train_data_batch_1", "27846dcaa50de8e21a7d1a35f30f0e91"),
        ("train_data_batch_2", "c7254a054e0e795c69120a5727050e3f"),
        ("train_data_batch_3", "4333d3df2e5ffb114b05d2ffc19b1e87"),
        ("train_data_batch_4", "1620cdf193304f4a92677b695d70d10f"),
        ("train_data_batch_5", "348b3c2fdbb3940c4e9e834affd3b18d"),
        ("train_data_batch_6", "6e765307c242a1b3d7d5ef9139b48945"),
        ("train_data_batch_7", "564926d8cbf8fc4818ba23d2faac7564"),
        ("train_data_batch_8", "f4755871f718ccb653440b9dd0ebac66"),
        ("train_data_batch_9", "bb6dd660c38c58552125b1a92f86b5d4"),
        ("train_data_batch_10", "8f03f34ac4b42271a294f91bf480f29b"),
    ),
    "valid": (("val_data", "3410e3017fdaefba8d5073aaa65e4bd6"),),
}


def _digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class _RestrictedNumpyUnpickler(pickle.Unpickler):
    _allowed = {
        ("_codecs", "encode"),
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
    }

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in self._allowed:
            raise pickle.UnpicklingError(f"Forbidden ImageNet16 pickle global: {module}.{name}")
        return super().find_class(module, name)


def _load_trusted_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        payload = _RestrictedNumpyUnpickler(handle, encoding="latin1").load()
    if not isinstance(payload, dict):
        raise ValueError(f"ImageNet16 batch is not a mapping: {path}")
    data = payload.get("data", payload.get(b"data"))
    labels = payload.get("labels", payload.get(b"labels"))
    if not isinstance(data, np.ndarray) or data.dtype != np.uint8 or data.ndim != 2:
        raise ValueError(f"ImageNet16 batch has invalid uint8 data: {path}")
    if data.shape[1] != 3 * 16 * 16:
        raise ValueError(f"ImageNet16 batch has invalid flattened image shape: {path}")
    label_array = np.asarray(labels)
    if label_array.ndim != 1 or len(label_array) != len(data):
        raise ValueError(f"ImageNet16 batch labels do not align with images: {path}")
    if not np.issubdtype(label_array.dtype, np.integer):
        raise ValueError(f"ImageNet16 labels must be integers: {path}")
    return data, label_array.astype(np.int64, copy=False)


def _save_array(path: Path, values: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    temporary.replace(path)


def _runtime_path(root: Path, value: Any) -> Path:
    relative = Path(str(value))
    candidate = (root / relative).resolve()
    if relative.is_absolute() or not candidate.is_relative_to(root.resolve()):
        raise ValueError(f"ImageNet16 runtime path escapes its root: {value!r}")
    return candidate


def verify_safe_imagenet16(path: str | Path) -> dict[str, Any]:
    root = Path(path).expanduser().resolve()
    manifest_path = root if root.is_file() else root / "manifest.json"
    if not manifest_path.is_file():
        return {"valid": False, "reason": "manifest_missing", "path": str(manifest_path)}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {"valid": False, "reason": f"manifest_invalid:{error}", "path": str(manifest_path)}
    if manifest.get("schema_version") != 1 or manifest.get("dataset_id") != "ImageNet16-120":
        return {"valid": False, "reason": "manifest_contract_mismatch", "path": str(manifest_path)}
    data_root = manifest_path.parent
    for shard in manifest.get("shards", []):
        for key in ("images", "labels"):
            record = shard.get(key, {})
            try:
                candidate = _runtime_path(data_root, record.get("path", ""))
            except ValueError as error:
                return {
                    "valid": False,
                    "reason": f"{key}_path_escape",
                    "path": str(record.get("path", "")),
                    "error": str(error),
                }
            if not candidate.is_file() or _digest(candidate, "sha256") != record.get("sha256"):
                return {
                    "valid": False,
                    "reason": f"{key}_integrity_failed",
                    "path": str(candidate),
                }
    return {
        "valid": True,
        "path": str(manifest_path),
        "samples": int(manifest.get("samples", 0)),
        "classes": int(manifest.get("classes", 0)),
        "splits": manifest.get("splits", {}),
    }


def convert_imagenet16_120(
    source: str | Path,
    destination: str | Path,
    *,
    trusted: bool = False,
    replace: bool = False,
) -> Path:
    if not trusted:
        raise PermissionError("ImageNet16 conversion requires explicit --trusted")
    source_root = Path(source).expanduser().resolve()
    if (source_root / "ImageNet16").is_dir():
        source_root = source_root / "ImageNet16"
    destination_root = Path(destination).expanduser().resolve()
    if destination_root.exists():
        verification = verify_safe_imagenet16(destination_root)
        if verification["valid"] and not replace:
            return destination_root / "manifest.json"
        if not replace:
            raise FileExistsError(
                f"ImageNet16 destination already exists but is not reusable: {destination_root}"
            )
    temporary = destination_root.with_name(destination_root.name + ".conversion-part")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    shards: list[dict[str, Any]] = []
    split_counts: dict[str, int] = {}
    try:
        for split, files in IMAGENET16_FILES.items():
            split_count = 0
            for index, (filename, expected_md5) in enumerate(files, start=1):
                source_path = source_root / filename
                if not source_path.is_file():
                    raise FileNotFoundError(f"Missing ImageNet16 source batch: {source_path}")
                actual_md5 = _digest(source_path, "md5")
                if actual_md5 != expected_md5:
                    raise ValueError(f"ImageNet16 MD5 mismatch: {source_path}")
                data, labels = _load_trusted_batch(source_path)
                selected = (labels >= 1) & (labels <= 120)
                images = data[selected].reshape(-1, 3, 16, 16).transpose(0, 2, 3, 1)
                normalized_labels = labels[selected] - 1
                prefix = f"{split}-{index:02d}"
                image_path = temporary / f"{prefix}-images.npy"
                label_path = temporary / f"{prefix}-labels.npy"
                _save_array(image_path, np.ascontiguousarray(images))
                _save_array(label_path, np.ascontiguousarray(normalized_labels))
                count = int(len(normalized_labels))
                split_count += count
                shards.append(
                    {
                        "split": split,
                        "source": {
                            "filename": filename,
                            "md5": actual_md5,
                            "sha256": _digest(source_path, "sha256"),
                        },
                        "count": count,
                        "images": {
                            "path": image_path.name,
                            "sha256": _digest(image_path, "sha256"),
                        },
                        "labels": {
                            "path": label_path.name,
                            "sha256": _digest(label_path, "sha256"),
                        },
                    }
                )
            split_counts[split] = split_count
        manifest = {
            "schema_version": 1,
            "dataset_id": "ImageNet16-120",
            "runtime_format": "npy-shards-v1",
            "source_protocol": "downsampled-imagenet16-official-md5-trusted-conversion",
            "classes": 120,
            "image_shape": [16, 16, 3],
            "label_base": 0,
            "samples": sum(split_counts.values()),
            "splits": split_counts,
            "shards": shards,
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if destination_root.exists():
            backup = destination_root.with_name(destination_root.name + ".replacement-backup")
            if backup.exists():
                raise FileExistsError(f"ImageNet16 replacement backup already exists: {backup}")
            destination_root.replace(backup)
            try:
                temporary.replace(destination_root)
            except BaseException:
                backup.replace(destination_root)
                raise
            else:
                shutil.rmtree(backup)
        else:
            temporary.replace(destination_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination_root / "manifest.json"


class SafeImageNet16:
    def __init__(
        self,
        root: str | Path,
        *,
        train: bool = True,
        transform: Any = None,
        verify: bool = True,
    ) -> None:
        from PIL import Image

        self._image_type = Image
        root_path = Path(root).expanduser().resolve()
        self.manifest_path = root_path if root_path.is_file() else root_path / "manifest.json"
        if verify:
            verification = verify_safe_imagenet16(self.manifest_path)
            if not verification["valid"]:
                raise ValueError(f"Unsafe or corrupt ImageNet16 runtime: {verification}")
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        split = "train" if train else "valid"
        records = [record for record in manifest["shards"] if record["split"] == split]
        if not records:
            raise ValueError(f"ImageNet16 runtime has no {split!r} shards")
        self.transform = transform
        self._images = [
            np.load(
                _runtime_path(self.manifest_path.parent, record["images"]["path"]),
                mmap_mode="r",
                allow_pickle=False,
            )
            for record in records
        ]
        self._labels = [
            np.load(
                _runtime_path(self.manifest_path.parent, record["labels"]["path"]),
                mmap_mode="r",
                allow_pickle=False,
            )
            for record in records
        ]
        self._ends: list[int] = []
        total = 0
        for images, labels in zip(self._images, self._labels, strict=True):
            if images.shape != (len(labels), 16, 16, 3) or images.dtype != np.uint8:
                raise ValueError("ImageNet16 runtime shard shape or dtype is invalid")
            total += len(labels)
            self._ends.append(total)

    def __len__(self) -> int:
        return self._ends[-1]

    def __getitem__(self, index: int) -> tuple[Any, int]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard_index = bisect_right(self._ends, index)
        start = 0 if shard_index == 0 else self._ends[shard_index - 1]
        local_index = index - start
        image = self._image_type.fromarray(
            np.asarray(self._images[shard_index][local_index]).copy()
        )
        if self.transform is not None:
            image = self.transform(image)
        return image, int(self._labels[shard_index][local_index])


__all__ = [
    "IMAGENET16_FILES",
    "SafeImageNet16",
    "convert_imagenet16_120",
    "verify_safe_imagenet16",
]
