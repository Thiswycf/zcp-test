from __future__ import annotations

import hashlib
import json
import random
from importlib.resources import files
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

TRANSNAS_UPSTREAM_COMMIT = "6d4231b1eb04e95750a5b2b6cf391db770bc25d6"
TRANSNAS_MANIFEST_SCHEMA = "zcp-test.transnas-input-manifest.v1"
TRANSNAS_MANIFEST_NAMES = ("transnas-inputs.json", "transnas_inputs.json")
TRANSNAS_TASKS = frozenset(
    {
        "autoencoder",
        "class_object",
        "class_scene",
        "jigsaw",
        "normal",
        "room_layout",
        "segmentsemantic",
    }
)

_TARGET_DOMAINS = {
    "autoencoder": ("rgb", "png"),
    "class_object": ("class_object", "npy"),
    "class_scene": ("class_scene", "npy"),
    "jigsaw": ("rgb", "png"),
    "normal": ("normal", "png"),
    "room_layout": ("room_layout", "npy"),
    "segmentsemantic": ("segmentsemantic", "png"),
}
_RESOURCE_NAMES = {
    "class_object": "class_object_final5k.npy",
    "class_scene": "class_scene_final5k.npy",
    "jigsaw": "permutations_hamming_max_1000.npy",
}


def is_transnas_task(dataset: str) -> bool:
    return dataset in TRANSNAS_TASKS


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_path(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty relative POSIX path")
    if "\\" in value:
        raise ValueError(f"{field} must use POSIX separators")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Unsafe relative path in {field}: {value!r}")
    return path.as_posix()


def _resolve_asset(root: Path, relative: str, *, field: str) -> Path:
    safe = _safe_relative_path(relative, field=field)
    root_resolved = root.resolve(strict=True)
    candidate = (root_resolved / PurePosixPath(safe)).resolve(strict=False)
    if not candidate.is_relative_to(root_resolved):
        raise ValueError(f"Resolved path escapes Taskonomy root in {field}: {relative!r}")
    if not candidate.is_file():
        raise FileNotFoundError(f"TransNAS input asset is missing for {field}: {candidate}")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root_resolved):
        raise ValueError(f"Symlink escapes Taskonomy root in {field}: {relative!r}")
    return resolved


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON file {path}: {error}") from error


def _template_base(template: Any, *, field: str) -> str:
    value = _safe_relative_path(template, field=field)
    suffix = PurePosixPath(value).suffix.lower()
    if suffix in {".png", ".npy"}:
        value = value[: -len(suffix)]
    if "{domain}" not in value:
        raise ValueError(
            f"{field} must contain the official '{{domain}}' placeholder: {template!r}"
        )
    return value


def _domain_path(template: str, domain: str, extension: str) -> str:
    return _safe_relative_path(
        f"{template.replace('{domain}', domain)}.{extension}", field=f"target.{domain}"
    )


def generate_transnas_input_manifest(
    split_json: str | Path,
    dataset_root: str | Path,
    output_path: str | Path,
    *,
    split: str | None = None,
    verify_files: bool = False,
) -> dict[str, Any]:
    split_path = Path(split_json).expanduser().resolve(strict=True)
    root = Path(dataset_root).expanduser().resolve(strict=True)
    split_document = _read_json(split_path)
    if not isinstance(split_document, dict) or not isinstance(
        split_document.get("filename_list"), list
    ):
        raise ValueError("Official split JSON must contain a filename_list array")

    records: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    building_files: list[str] = []
    for building_index, building_value in enumerate(split_document["filename_list"]):
        building_file = _safe_relative_path(
            building_value, field=f"filename_list[{building_index}]"
        )
        building_files.append(building_file)
        building_path = _resolve_asset(
            root, building_file, field=f"filename_list[{building_index}]"
        )
        templates = _read_json(building_path)
        if isinstance(templates, dict):
            templates = templates.get("filename_list")
        if not isinstance(templates, list):
            raise ValueError(f"Per-building JSON must contain an array: {building_path}")
        for template_index, template_value in enumerate(templates):
            template = _template_base(
                template_value, field=f"{building_file}[{template_index}]"
            )
            sample_id = template.replace("{domain}", "*")
            if sample_id in sample_ids:
                raise ValueError(f"Duplicate Taskonomy sample template: {sample_id}")
            sample_ids.add(sample_id)
            rgb = _domain_path(template, "rgb", "png")
            targets = {
                task: _domain_path(template, domain, extension)
                for task, (domain, extension) in _TARGET_DOMAINS.items()
            }
            record = {"sample_id": sample_id, "rgb": rgb, "targets": targets}
            if verify_files:
                _resolve_asset(root, rgb, field=f"{sample_id}.rgb")
                for task, target in targets.items():
                    _resolve_asset(root, target, field=f"{sample_id}.targets.{task}")
            records.append(record)

    manifest = {
        "schema_version": TRANSNAS_MANIFEST_SCHEMA,
        "benchmark_id": "transnasbench101",
        "upstream": {
            "repository": "https://github.com/yawen-d/TransNASBench",
            "commit": TRANSNAS_UPSTREAM_COMMIT,
        },
        "license_requirement": (
            "Taskonomy/TransNAS files are external data; the user must obtain and use them "
            "under their applicable terms. zcp-test does not download or redistribute them."
        ),
        "split": split or split_path.stem,
        "split_protocol": {
            "fidelity": "user-supplied-taskonomy-split",
            "official_transnas_24_building_split": False,
            "transnas_final_training_config_available": False,
        },
        "source": {
            "split_json": split_path.name,
            "split_sha256": _sha256_file(split_path),
            "building_files": building_files,
        },
        "records": records,
    }
    destination = Path(output_path).expanduser()
    if destination.parent.resolve() != root:
        raise ValueError("TransNAS input manifest must be written directly inside dataset_root")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return manifest


def build_transnas_input_manifest(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return generate_transnas_input_manifest(*args, **kwargs)


def _manifest_path(data_root: str | Path) -> tuple[Path, Path]:
    supplied = Path(data_root).expanduser()
    if supplied.is_file():
        return supplied.resolve(strict=True), supplied.parent.resolve(strict=True)
    if not supplied.exists():
        raise FileNotFoundError(f"Taskonomy data root does not exist: {supplied}")
    if not supplied.is_dir():
        raise ValueError(f"Taskonomy data root is not a directory: {supplied}")
    for name in TRANSNAS_MANIFEST_NAMES:
        candidate = supplied / name
        if candidate.is_file():
            return candidate.resolve(strict=True), supplied.resolve(strict=True)
    names = ", ".join(TRANSNAS_MANIFEST_NAMES)
    raise FileNotFoundError(f"No TransNAS input manifest found in {supplied}; expected {names}")


def _validate_manifest(document: Any) -> list[dict[str, Any]]:
    if not isinstance(document, dict):
        raise ValueError("TransNAS input manifest must be a JSON object")
    if document.get("schema_version") != TRANSNAS_MANIFEST_SCHEMA:
        raise ValueError(
            f"Unsupported TransNAS input manifest schema: {document.get('schema_version')!r}"
        )
    if document.get("benchmark_id") != "transnasbench101":
        raise ValueError("TransNAS input manifest benchmark_id must be 'transnasbench101'")
    upstream = document.get("upstream")
    if not isinstance(upstream, dict) or upstream.get("commit") != TRANSNAS_UPSTREAM_COMMIT:
        raise ValueError(f"TransNAS input manifest must identify upstream commit {TRANSNAS_UPSTREAM_COMMIT}")
    if not isinstance(document.get("license_requirement"), str) or not document[
        "license_requirement"
    ].strip():
        raise ValueError("TransNAS input manifest must state its external-data license requirement")
    split_protocol = document.get("split_protocol")
    if not isinstance(split_protocol, dict) or split_protocol.get("fidelity") != (
        "user-supplied-taskonomy-split"
    ):
        raise ValueError("TransNAS input manifest must identify a user-supplied Taskonomy split")
    if split_protocol.get("official_transnas_24_building_split") is not False:
        raise ValueError(
            "The published TransNAS release does not provide a verifiable 24-building split"
        )
    records = document.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("TransNAS input manifest records must be a non-empty array")
    seen: set[str] = set()
    validated: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"records[{index}] must be an object")
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"records[{index}].sample_id must be a non-empty string")
        if sample_id in seen:
            raise ValueError(f"Duplicate sample_id in TransNAS input manifest: {sample_id}")
        seen.add(sample_id)
        rgb = _safe_relative_path(record.get("rgb"), field=f"records[{index}].rgb")
        targets = record.get("targets")
        if not isinstance(targets, dict) or set(targets) != TRANSNAS_TASKS:
            raise ValueError(
                f"records[{index}].targets must contain exactly {sorted(TRANSNAS_TASKS)}"
            )
        validated.append(
            {
                "sample_id": sample_id,
                "rgb": rgb,
                "targets": {
                    task: _safe_relative_path(
                        targets[task], field=f"records[{index}].targets.{task}"
                    )
                    for task in sorted(TRANSNAS_TASKS)
                },
            }
        )
    return validated


def _resource_array(task: str) -> np.ndarray:
    resource = files("zcp_test").joinpath("resources", "transnas", _RESOURCE_NAMES[task])
    with resource.open("rb") as handle:
        return np.load(handle, allow_pickle=False)


def _load_rgb(path: Path, size: int) -> Any:
    from PIL import Image
    from torchvision.transforms import functional as functional

    with Image.open(path) as image:
        image = image.convert("RGB")
        image = functional.resize(image, [size, size], antialias=True)
        tensor = functional.to_tensor(image)
    return functional.normalize(tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])


def _classification_label(path: Path, task: str) -> Any:
    import torch

    logits = np.load(path, allow_pickle=False)
    selection = _resource_array(task).astype(bool)
    if logits.ndim != 1 or logits.shape[0] != selection.shape[0]:
        raise ValueError(
            f"{task} target must have shape ({selection.shape[0]},), got {logits.shape}: {path}"
        )
    selected = logits[selection]
    expected = 75 if task == "class_object" else 47
    if selected.shape != (expected,):
        raise ValueError(f"{task} final5k selection must contain {expected} classes")
    return torch.tensor(int(selected.argmax()), dtype=torch.long)


def _dense_target(path: Path, task: str, input_size: int) -> Any:
    import torch
    from PIL import Image
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as functional

    if task == "room_layout":
        value = np.load(path, allow_pickle=False)
        if value.size != 9:
            raise ValueError(f"room_layout target must contain 9 values, got {value.shape}: {path}")
        return torch.from_numpy(np.asarray(value, dtype=np.float32).reshape(9))

    with Image.open(path) as image:
        if task == "segmentsemantic":
            array = np.asarray(image)
            if array.ndim == 3 and array.shape[2] == 1:
                array = array[:, :, 0]
            if array.ndim != 2:
                raise ValueError(f"segmentsemantic target must be single-channel: {path}")
            shifted = array.copy()
            shifted[shifted == 0] = 1
            shifted = shifted.astype(np.int64) - 1
            target = torch.from_numpy(shifted)
            return functional.resize(
                target.unsqueeze(0),
                [input_size, input_size],
                interpolation=InterpolationMode.NEAREST,
            ).squeeze(0).long()
        image = image.convert("RGB")
        image = functional.resize(image, [input_size, input_size], antialias=True)
        target = functional.to_tensor(image)
    if task == "autoencoder":
        target = functional.normalize(target, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    return target


def _stable_integer(seed: int, sample_id: str, field: str, upper: int) -> int:
    digest = hashlib.sha256(f"{seed}\0{sample_id}\0{field}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % upper


def _jigsaw_input(path: Path, sample_id: str, seed: int) -> tuple[Any, Any]:
    import torch
    from PIL import Image
    from torchvision.transforms import functional as functional

    permutations = _resource_array("jigsaw")
    if permutations.shape != (1000, 9):
        raise ValueError(f"Official jigsaw permutation resource has invalid shape {permutations.shape}")
    class_index = _stable_integer(seed, sample_id, "permutation", 1000)
    permutation = permutations[class_index]
    with Image.open(path) as source:
        image = functional.resize(source.convert("RGB"), [255, 255], antialias=True)
        tiles = []
        for tile_index, position_value in enumerate(permutation):
            position = int(position_value)
            row, column = divmod(position, 3)
            top_jitter = _stable_integer(seed, sample_id, f"tile-{tile_index}-top", 22)
            left_jitter = _stable_integer(seed, sample_id, f"tile-{tile_index}-left", 22)
            tile = functional.crop(
                image, row * 85 + top_jitter, column * 85 + left_jitter, 64, 64
            )
            tensor = functional.to_tensor(tile)
            mean = tensor.mean(dim=(-2, -1))
            standard_deviation = tensor.std(dim=(-2, -1)) + 0.0001
            tiles.append(functional.normalize(tensor, mean, standard_deviation))
    return torch.stack(tiles), torch.tensor(class_index, dtype=torch.long)


def load_transnas_input_batch(
    task: str,
    data_root: str | Path,
    batch_size: int,
    input_size: int,
    seed: int,
) -> tuple[Any, Any, dict[str, Any]]:
    import torch

    if task not in TRANSNAS_TASKS:
        raise ValueError(f"Unknown TransNAS task: {task!r}")
    if batch_size <= 0 or input_size <= 0:
        raise ValueError("batch_size and input_size must be positive")
    expected_input_size = 64 if task == "jigsaw" else 256
    if input_size != expected_input_size:
        raise ValueError(
            f"TransNAS {task} requires input_size={expected_input_size} for the "
            "reference task model"
        )

    manifest_path, root = _manifest_path(data_root)
    document = _read_json(manifest_path)
    records = _validate_manifest(document)
    if len(records) < batch_size:
        raise ValueError(
            f"TransNAS input manifest has {len(records)} samples, fewer than batch size {batch_size}"
        )
    selected_indices = random.Random(seed).sample(range(len(records)), batch_size)
    selected = [records[index] for index in selected_indices]
    inputs = []
    labels = []
    for record in selected:
        rgb_path = _resolve_asset(root, record["rgb"], field=f"{record['sample_id']}.rgb")
        target_path = _resolve_asset(
            root,
            record["targets"][task],
            field=f"{record['sample_id']}.targets.{task}",
        )
        if task == "jigsaw":
            input_tensor, label = _jigsaw_input(rgb_path, record["sample_id"], seed)
        else:
            input_tensor = _load_rgb(rgb_path, input_size)
            if task in {"class_object", "class_scene"}:
                label = _classification_label(target_path, task)
            else:
                label = _dense_target(target_path, task, input_size)
        inputs.append(input_tensor)
        labels.append(label)

    protocol = {
        "source": "dataset",
        "dataset": task,
        "benchmark_id": "transnasbench101",
        "task": task,
        "seed": seed,
        "sample_ids": [record["sample_id"] for record in selected],
        "batch_size": batch_size,
        "input_size": 64 if task == "jigsaw" else input_size,
        "label_protocol": (
            "transnas-final5k-hard-label"
            if task in {"class_object", "class_scene"}
            else "published-taskonomy-target"
        ),
        "target_transform": {
            "class_object": "official-final5k-mask+argmax-hard-label",
            "class_scene": "official-final5k-mask+argmax-hard-label",
            "room_layout": "official-precomputed-nine-vector-npy",
            "jigsaw": "official-1000-permutation-class-id",
            "segmentsemantic": f"zero-to-background+minus-one+nearest-resize-{input_size}",
            "normal": f"rgb-decode+bilinear-resize-{input_size}+tensor-0-1-unverified-scale",
            "autoencoder": f"rgb-decode+bilinear-resize-{input_size}+tensor-half-normalize",
        }[task],
        "transform": (
            "official-jigsaw-255-grid-random-crop64-per-tile-normalize"
            if task == "jigsaw"
            else f"deterministic-resize-{input_size}+tensor+half-normalize"
        ),
        "transform_fidelity": {
            "source_commit": TRANSNAS_UPSTREAM_COMMIT,
            "loader_semantics": "upstream-load-ops-port",
            "sampling": "zcp-test-deterministic-evaluation",
            "training_augmentation_match": False,
            "official_transnas_input_protocol_match": False,
        },
        "split_protocol": document["split_protocol"],
        "manifest_schema": TRANSNAS_MANIFEST_SCHEMA,
        "manifest_sha256": hashlib.sha256(_canonical_json(document)).hexdigest(),
        "license_requirement": document.get("license_requirement"),
    }
    return torch.stack(inputs), torch.stack(labels), protocol


__all__ = [
    "TRANSNAS_MANIFEST_NAMES",
    "TRANSNAS_MANIFEST_SCHEMA",
    "TRANSNAS_TASKS",
    "TRANSNAS_UPSTREAM_COMMIT",
    "build_transnas_input_manifest",
    "generate_transnas_input_manifest",
    "is_transnas_task",
    "load_transnas_input_batch",
]
