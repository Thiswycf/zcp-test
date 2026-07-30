from __future__ import annotations

import json
import shutil
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from zcp_test.cli import main
from zcp_test.data.transnas_inputs import (
    TRANSNAS_MANIFEST_SCHEMA,
    TRANSNAS_UPSTREAM_COMMIT,
    generate_transnas_input_manifest,
)
from zcp_test.inputs import make_input_batch


def _selection(name: str) -> np.ndarray:
    resource = files("zcp_test").joinpath("resources", "transnas", name)
    with resource.open("rb") as handle:
        return np.load(handle, allow_pickle=False).astype(bool)


def _write_sample(root: Path, sample_index: int) -> str:
    template = f"building/{{domain}}/point_{sample_index}_view_0_domain_{{domain}}.png"
    domains = {
        "rgb": root / template.replace("{domain}", "rgb"),
        "class_object": root
        / template.replace("{domain}", "class_object").replace(".png", ".npy"),
        "class_scene": root
        / template.replace("{domain}", "class_scene").replace(".png", ".npy"),
        "normal": root / template.replace("{domain}", "normal"),
        "room_layout": root
        / template.replace("{domain}", "room_layout").replace(".png", ".npy"),
        "segmentsemantic": root / template.replace("{domain}", "segmentsemantic"),
    }
    for path in domains.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    grid = np.zeros((32, 32, 3), dtype=np.uint8)
    grid[:, :, 0] = np.arange(32, dtype=np.uint8)[:, None] * 7
    grid[:, :, 1] = np.arange(32, dtype=np.uint8)[None, :] * 7
    grid[:, :, 2] = 40 + sample_index * 20
    Image.fromarray(grid).save(domains["rgb"])
    Image.fromarray(np.flip(grid, axis=1)).save(domains["normal"])
    segmentation = np.tile(np.array([0, 1, 2, 3], dtype=np.uint8), (32, 8))
    Image.fromarray(segmentation).save(domains["segmentsemantic"])

    object_logits = np.zeros(1000, dtype=np.float32)
    object_indices = np.flatnonzero(_selection("class_object_final5k.npy"))
    object_logits[object_indices[3 + sample_index]] = 10.0
    np.save(domains["class_object"], object_logits, allow_pickle=False)

    scene_logits = np.zeros(365, dtype=np.float32)
    scene_indices = np.flatnonzero(_selection("class_scene_final5k.npy"))
    scene_logits[scene_indices[4 + sample_index]] = 10.0
    np.save(domains["class_scene"], scene_logits, allow_pickle=False)
    np.save(
        domains["room_layout"],
        np.arange(9, dtype=np.float32) + sample_index,
        allow_pickle=False,
    )
    return template


@pytest.fixture
def taskonomy_root(tmp_path: Path) -> Path:
    root = tmp_path / "taskonomy"
    root.mkdir()
    templates = [_write_sample(root, sample_index) for sample_index in range(2)]
    (root / "building.json").write_text(json.dumps(templates), encoding="utf-8")
    split = tmp_path / "train_split.json"
    split.write_text(json.dumps({"filename_list": ["building.json"]}), encoding="utf-8")
    generate_transnas_input_manifest(
        split,
        root,
        root / "transnas-inputs.json",
        split="train",
        verify_files=True,
    )
    return root


@pytest.mark.parametrize(
    ("task", "input_size", "input_shape", "label_shape", "label_dtype"),
    [
        ("class_object", 256, (2, 3, 256, 256), (2,), torch.long),
        ("class_scene", 256, (2, 3, 256, 256), (2,), torch.long),
        ("room_layout", 256, (2, 3, 256, 256), (2, 9), torch.float32),
        ("segmentsemantic", 256, (2, 3, 256, 256), (2, 256, 256), torch.long),
        ("normal", 256, (2, 3, 256, 256), (2, 3, 256, 256), torch.float32),
        ("autoencoder", 256, (2, 3, 256, 256), (2, 3, 256, 256), torch.float32),
        ("jigsaw", 64, (2, 9, 3, 64, 64), (2,), torch.long),
    ],
)
def test_seven_transnas_tasks_load_real_inputs_and_targets(
    taskonomy_root: Path,
    task: str,
    input_size: int,
    input_shape: tuple[int, ...],
    label_shape: tuple[int, ...],
    label_dtype: torch.dtype,
):
    batch = make_input_batch(
        "dataset",
        task,
        2,
        input_size,
        1000,
        17,
        torch.device("cpu"),
        str(taskonomy_root),
    )

    assert tuple(batch.inputs.shape) == input_shape
    assert tuple(batch.labels.shape) == label_shape
    assert batch.labels.dtype == label_dtype
    assert batch.protocol["task"] == task
    assert batch.protocol["transform_fidelity"]["source_commit"] == TRANSNAS_UPSTREAM_COMMIT
    assert batch.protocol["transform_fidelity"]["training_augmentation_match"] is False
    assert batch.protocol["transform_fidelity"]["official_transnas_input_protocol_match"] is False
    assert batch.protocol["split_protocol"]["official_transnas_24_building_split"] is False
    assert batch.protocol["license_requirement"]
    assert batch.protocol["target_transform"]
    assert len(batch.protocol["sample_ids"]) == 2
    assert len(batch.fingerprint) == 64
    if task == "class_object":
        assert set(batch.labels.tolist()) == {3, 4}
    elif task == "class_scene":
        assert set(batch.labels.tolist()) == {4, 5}
    elif task == "segmentsemantic":
        assert set(batch.labels.unique().tolist()) == {0, 1, 2}
    elif task == "normal":
        assert batch.protocol["target_transform"].endswith("unverified-scale")


def test_prepare_transnas_input_cli_generates_default_manifest(
    taskonomy_root: Path, capsys
):
    manifest = taskonomy_root / "transnas-inputs.json"
    manifest.unlink()

    main(
        [
            "data",
            "prepare-transnas-input",
            "--data-root",
            str(taskonomy_root),
            "--split-json",
            str(taskonomy_root.parent / "train_split.json"),
            "--split",
            "train",
            "--verify-files",
        ]
    )

    summary = json.loads(capsys.readouterr().out)
    assert summary["records"] == 2
    assert summary["split"] == "train"
    assert Path(summary["output"]) == manifest
    assert manifest.is_file()


def test_jigsaw_is_deterministic_per_sample_and_seed(taskonomy_root: Path):
    arguments = (
        "dataset",
        "jigsaw",
        1,
        64,
        1000,
        91,
        torch.device("cpu"),
        str(taskonomy_root),
    )
    first = make_input_batch(*arguments)
    second = make_input_batch(*arguments)
    different = make_input_batch(*arguments[:5], 92, *arguments[6:])

    assert torch.equal(first.inputs, second.inputs)
    assert torch.equal(first.labels, second.labels)
    assert first.fingerprint == second.fingerprint
    assert first.fingerprint != different.fingerprint
    assert not torch.equal(first.inputs, different.inputs) or not torch.equal(
        first.labels, different.labels
    )


def test_manifest_generation_uses_official_split_and_relative_paths(taskonomy_root: Path):
    manifest = json.loads((taskonomy_root / "transnas-inputs.json").read_text(encoding="utf-8"))

    assert manifest["schema_version"] == TRANSNAS_MANIFEST_SCHEMA
    assert manifest["upstream"]["commit"] == TRANSNAS_UPSTREAM_COMMIT
    assert manifest["split"] == "train"
    assert len(manifest["records"]) == 2
    assert manifest["source"]["building_files"] == ["building.json"]
    for record in manifest["records"]:
        assert not Path(record["rgb"]).is_absolute()
        assert set(record["targets"]) == {
            "autoencoder",
            "class_object",
            "class_scene",
            "jigsaw",
            "normal",
            "room_layout",
            "segmentsemantic",
        }


def test_manifest_rejects_path_traversal(taskonomy_root: Path):
    manifest_path = taskonomy_root / "transnas-inputs.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["records"][0]["rgb"] = "../outside.png"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsafe relative path"):
        make_input_batch(
            "dataset",
            "class_object",
            1,
            256,
            75,
            1,
            torch.device("cpu"),
            str(taskonomy_root),
        )


def test_manifest_generator_rejects_building_path_traversal(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"filename_list": ["../outside.json"]}), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsafe relative path"):
        generate_transnas_input_manifest(split, root, root / "transnas-inputs.json")


def test_missing_manifest_fails_without_fallback(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="No TransNAS input manifest"):
        make_input_batch(
            "dataset",
            "normal",
            1,
            256,
            3,
            1,
            torch.device("cpu"),
            str(tmp_path),
        )


def test_packaged_final5k_and_jigsaw_resources_match_protocol():
    object_selection = _selection("class_object_final5k.npy")
    scene_selection = _selection("class_scene_final5k.npy")
    permutation_resource = files("zcp_test").joinpath(
        "resources", "transnas", "permutations_hamming_max_1000.npy"
    )
    with permutation_resource.open("rb") as handle:
        permutations = np.load(handle, allow_pickle=False)

    assert object_selection.shape == (1000,)
    assert object_selection.sum() == 75
    assert scene_selection.shape == (365,)
    assert scene_selection.sum() == 47
    assert permutations.shape == (1000, 9)
    assert all(np.array_equal(np.sort(row), np.arange(9)) for row in permutations)


@pytest.mark.parametrize(("task", "input_size"), [("class_object", 32), ("jigsaw", 256)])
def test_reference_task_input_size_is_strict(taskonomy_root: Path, task: str, input_size: int):
    with pytest.raises(ValueError, match="requires input_size"):
        make_input_batch(
            "dataset",
            task,
            1,
            input_size,
            1000,
            1,
            torch.device("cpu"),
            str(taskonomy_root),
        )


def test_missing_selected_task_file_fails_without_fallback(taskonomy_root: Path):
    manifest = json.loads((taskonomy_root / "transnas-inputs.json").read_text(encoding="utf-8"))
    missing = taskonomy_root / manifest["records"][0]["targets"]["room_layout"]
    missing.unlink()

    with pytest.raises(FileNotFoundError, match="room_layout"):
        make_input_batch(
            "dataset",
            "room_layout",
            2,
            256,
            9,
            3,
            torch.device("cpu"),
            str(taskonomy_root),
        )


def test_fingerprint_is_stable_when_dataset_root_moves(taskonomy_root: Path, tmp_path: Path):
    copy = tmp_path / "relocated"
    shutil.copytree(taskonomy_root, copy)
    first = make_input_batch(
        "dataset", "class_scene", 2, 256, 47, 12, torch.device("cpu"), str(taskonomy_root)
    )
    second = make_input_batch(
        "dataset", "class_scene", 2, 256, 47, 12, torch.device("cpu"), str(copy)
    )

    assert first.protocol == second.protocol
    assert first.fingerprint == second.fingerprint
