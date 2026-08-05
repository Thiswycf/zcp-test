import json
import subprocess
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

import zcp_test.cli as cli
import zcp_test.training.protocols as training_protocols
from zcp_test.artifacts import normalize_score_records
from zcp_test.cli import _load_architecture_spec, _stratified_subset, main
from zcp_test.inputs import make_input_batch
from zcp_test.spaces import SPACES, load_builtin_spaces
from zcp_test.training.protocols import (
    resolve_acceptance_protocol,
    validate_candidate_training_protocol,
    validate_formal_training_protocol,
)


def test_candidate_acceptance_launchers_lock_protocols_and_parse_as_shell():
    root = Path(__file__).resolve().parents[1]
    acceptance = root / "tools" / "acceptance"
    scripts = {
        "run-autoformer-imagenet-dual-one-percent.sh": ("autoformer", "500", "5"),
        "run-plainnet-mbv2-imagenet-dual-one-percent.sh": (
            "zennas_plainnet_mbv2",
            "150",
            "2",
        ),
        "run-proxyless-mbv2-imagenet-dual-one-percent.sh": (
            "ofa_proxyless_mbv2",
            "150",
            "2",
        ),
    }
    common = acceptance / "run-imagenet-candidate-dual-one-percent.sh"
    for path in [common, *(acceptance / name for name in scripts)]:
        subprocess.run(["bash", "-n", str(path)], check=True)

    common_source = common.read_text(encoding="utf-8")
    for candidate in ("zcp_selected.json", "candidates-manifest.json"):
        assert candidate in common_source
    assert "fixed_random.json" not in common_source
    assert "params_flops_matched.json" not in common_source
    assert "candidate manifest checksum/role mismatch" in common_source
    assert "parallel_single_gpu requires ZCP_PARALLEL_SINGLE_GPU_ACCEPTED=yes" in common_source
    assert "packed_single_gpu requires ZCP_PACKED_SINGLE_GPU_ACCEPTED=yes" in common_source
    assert "parallel_single_gpu requires exactly two GPU UUIDs" in common_source
    assert "packed_single_gpu requires exactly one GPU UUID" in common_source
    assert "three architecture JSON files" not in common_source
    assert '"acceptance_scope": "single_zcp_dual_one_percent"' in common_source
    assert "both zcp-selected $SPACE ImageNet dual-one-percent acceptance runs completed" in common_source
    assert "ZCP_CPU_AFFINITIES" in common_source
    assert "one ZCP candidate; two dual-one-percent protocols" in common_source
    assert 'CUDA_VISIBLE_DEVICES="$uuid"' in common_source
    assert "CUDA_DEVICE_ORDER=PCI_BUS_ID" in common_source
    assert "ZoneInfo(\"Asia/Shanghai\")" in common_source
    assert "acceptance_exec_immutable" in common_source
    assert 'flock -n "$descriptor"' in common_source
    assert "with_all_gpu_locks run_one 1" in common_source
    assert 'with_gpu_lock "${gpu_array[0]}" "imagenet-task:1:zcp-selected:full-data"' in common_source
    assert 'with_gpu_lock "${gpu_array[1]}" "imagenet-task:2:zcp-selected:one-percent-data"' in common_source
    assert 'with_gpu_lock "${gpu_array[0]}" "imagenet-packed-tasks:1,2:zcp-selected"' in common_source
    assert "GPU locks are acquired only while each scientific task is active" in common_source
    assert 'eval "exec ${descriptor}' not in common_source
    assert common_source.count("exec {descriptor}>&-") >= 4

    for name, (space, formal_epochs, full_epochs) in scripts.items():
        source = (acceptance / name).read_text(encoding="utf-8")
        assert f"ZCP_ACCEPTANCE_SPACE={space}" in source
        assert f"ZCP_FORMAL_EPOCHS={formal_epochs}" in source
        assert f"ZCP_FULL_DATA_EPOCHS={full_epochs}" in source


def test_plainnet_one_percent_source_search_launcher_is_scoped_and_resumable():
    root = Path(__file__).resolve().parents[1]
    script = root / "tools" / "acceptance" / "run-plainnet-source-search-one-percent.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    source = script.read_text(encoding="utf-8")

    assert "acceptance_exec_immutable" in source
    assert "ZCP_ACCEPTANCE_ROOT" in source
    assert "CUDA_DEVICE_ORDER=PCI_BUS_ID" in source
    assert "ZoneInfo(\"Asia/Shanghai\")" in source
    assert 'valid_candidates": 1000' in source
    assert '"search_budget_protocol": "one_percent_acceptance"' in source
    assert '"search_budget_fraction": 0.01' in source
    assert 'controller": "plainnet_source_aligned"' in source
    assert "configs/search/plainnet_mbv2_source_aligned.yaml" in source
    assert '--flops-target "$FLOPS_TARGET"' in source
    assert '--gpu "$GPU_UUID"' in source
    assert '--resume "$state"' in source
    assert '"formal_search_completed": False' in source
    assert '"one_percent_search_completed": sys.argv[2] == "completed"' in source


def test_darts_parallel_resume_preserves_global_batch_and_assigns_all_tasks():
    root = Path(__file__).resolve().parents[1]
    script = root / "tools" / "acceptance" / "resume-darts-imagenet-parallel-from-task2.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    source = script.read_text(encoding="utf-8")
    assert "longest_first_per_lane_locks_global_batch_128" in source
    assert "CUDA_VISIBLE_DEVICES=$uuid" in source
    assert "--device cuda:0" in source
    assert "torchrun" not in source
    assert "with_gpu_lock" in source
    assert "longest tasks 4-6 start first" in source
    assert 'with_gpu_lock "$uuid" "darts-task:2:fixed-random"' in source
    assert 'with_gpu_lock "$uuid" "darts-task:3:params-matched"' in source
    assert 'with_gpu_lock "${gpu_array[3]}" short_lane' not in source
    assert source.index('with_gpu_lock "${gpu_array[0]}" "darts-task:4:zcp-selected"') < source.index(
        'short_lane "${gpu_array[3]}"'
    )
    for task_index in range(2, 7):
        assert f"run_single {task_index} " in source


def test_legacy_darts_ddp_launcher_locks_only_active_task():
    root = Path(__file__).resolve().parents[1]
    script = root / "tools" / "acceptance" / "run-darts-imagenet-dual-one-percent.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    source = script.read_text(encoding="utf-8")
    assert "with_all_gpu_locks run_one 1" in source
    assert "GPU locks are held only while each DDP task is active" in source
    assert 'eval "exec ${descriptor}' not in source


def test_autoformer_aznas_acceptance_launcher_is_resumable_and_packed():
    root = Path(__file__).resolve().parents[1]
    script = root / "tools" / "acceptance" / "run-autoformer-aznas-random-8000.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    source = script.read_text(encoding="utf-8")
    assert "ZCP_AZNAS_POPULATION:-8000" in source
    assert "packed_2_plus_1_on_two_gpus" in source
    assert "--proxy az_nas_autoformer --aggregator az_nas_log_rank" in source
    assert '--generations 0' in source
    assert '--resume "$state"' in source
    assert 'acceptance_with_gpu_lock "$LOCK_DIR/$uuid.lock" "$LOCK_TIMEOUT"' in source
    assert 'CUDA_VISIBLE_DEVICES="$uuid"' in source
    assert "acceptance_exec_immutable" in source
    assert "lane_a_tasks" in source
    assert "lane_b_tasks" in source
    assert 'command=$2' in source
    assert "architecture-hash-v1" in source
    assert 'ZoneInfo("Asia/Shanghai")' in source


def test_acceptance_launcher_snapshot_survives_source_rewrite(tmp_path):
    root = tmp_path / "repo"
    library_dir = root / "tools" / "acceptance" / "lib"
    library_dir.mkdir(parents=True)
    source_library = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "acceptance"
        / "lib"
        / "launcher-runtime.sh"
    )
    (library_dir / "launcher-runtime.sh").write_text(
        source_library.read_text(encoding="utf-8"), encoding="utf-8"
    )
    launcher = root / "tools" / "acceptance" / "long-launcher.sh"
    launcher.write_text(
        """#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT_ROOT=${ZCP_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
OUTPUT_ROOT=${ZCP_ACCEPTANCE_ROOT:-$PROJECT_ROOT/output}
source "$PROJECT_ROOT/tools/acceptance/lib/launcher-runtime.sh"
acceptance_exec_immutable "$PROJECT_ROOT" "$OUTPUT_ROOT" "${BASH_SOURCE[0]}" "$@"
printf 'before\\n' >> "$OUTPUT_ROOT/result.txt"
sleep 1
printf 'after\\n' >> "$OUTPUT_ROOT/result.txt"
""",
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "zcp-test"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=root, check=True)

    process = subprocess.Popen(["bash", str(launcher)], cwd=root)
    snapshot_dir = root / "output" / "launcher-snapshots"
    for _ in range(100):
        if list(snapshot_dir.glob("long-launcher-*/source/tools/acceptance/long-launcher.sh")):
            break
        process.poll()
        assert process.returncode is None
        import time

        time.sleep(0.02)
    else:
        process.kill()
        raise AssertionError("launcher snapshot was not created")

    launcher.write_text("#!/usr/bin/env bash\nexit 127\n", encoding="utf-8")
    assert process.wait(timeout=5) == 0
    assert (root / "output" / "result.txt").read_text(encoding="utf-8") == "before\nafter\n"
    snapshots = list(snapshot_dir.glob("long-launcher-*/source/tools/acceptance/long-launcher.sh"))
    assert len(snapshots) == 1
    assert snapshots[0].stat().st_mode & 0o222 == 0
    assert snapshots[0].with_suffix(".sh.sha256").is_file()


def test_failed_lane_releases_flock_before_supervisor_finishes(tmp_path):
    lock_path = tmp_path / "gpu.lock"
    released = tmp_path / "released"
    supervisor = tmp_path / "supervisor.sh"
    supervisor.write_text(
        """#!/usr/bin/env bash
set -Eeuo pipefail
lock_path=$1
released=$2
exec {descriptor}>"$lock_path"
flock -n "$descriptor"
if (
    exec {descriptor}>&-
    sleep 0.1
    exit 7
  ); then
  exit_code=0
else
  exit_code=$?
fi
: > "$lock_path"
flock -u "$descriptor"
printf '%s\\n' "$exit_code" > "$released"
sleep 2
""",
        encoding="utf-8",
    )
    process = subprocess.Popen(["bash", str(supervisor), str(lock_path), str(released)])
    for _ in range(100):
        if released.is_file():
            break
        process.poll()
        assert process.returncode is None
        import time

        time.sleep(0.02)
    else:
        process.kill()
        raise AssertionError("failed lane did not report lock release")

    assert released.read_text(encoding="utf-8").strip() == "7"
    assert process.poll() is None
    subprocess.run(["flock", "-n", str(lock_path), "-c", "true"], check=True)
    process.terminate()
    process.wait(timeout=5)


def test_acceptance_freeze_candidates_cli_is_exposed():
    parser = cli.build_parser()
    arguments = parser.parse_args(
        [
            "acceptance",
            "freeze-candidates",
            "--search-run",
            "search-run",
            "--training-config",
            "training.yaml",
            "--output",
            "candidates",
        ]
    )
    assert arguments.action == "freeze-candidates"
    assert arguments.pool_size == 32
    assert arguments.classes == 1000
    assert arguments.supporting_search_run == []

    reconciliation = parser.parse_args(
        [
            "acceptance",
            "reconcile-search-cohort",
            "--cohort-root",
            "cohort",
            "--search-run",
            "run-a",
            "--expected-space",
            "autoformer",
            "--expected-population",
            "8000",
            "--expected-seed",
            "20260731",
            "--expected-components",
            "expressivity,trainability,complexity",
        ]
    )
    assert reconciliation.action == "reconcile-search-cohort"
    assert reconciliation.expected_seed == [20260731]


def test_evaluate_one_row_per_proxy_and_lazy_directories(tmp_path):
    output = tmp_path / "runs"
    main(
        [
            "evaluate",
            "--space",
            "nb201_topology",
            "--proxies",
            "er,naswot,synflow",
            "--count",
            "2",
            "--device",
            "cpu",
            "--input-source",
            "random",
            "--batch-size",
            "2",
            "--input-size",
            "8",
            "--output",
            str(output),
        ]
    )
    run = next(output.iterdir())
    rows = [json.loads(line) for line in (run / "scores.jsonl").read_text().splitlines()]
    assert len(rows) == 6
    assert all(row["schema_version"] == "2.1" for row in rows)
    assert all("target_epoch_budget" in row for row in rows)
    er_rows = [row for row in rows if row["proxy_id"] == "er"]
    assert all(row["primary_component"] == "score" for row in er_rows)
    assert all(set(row["components"]) == {"score"} for row in er_rows)
    assert not (run / "checkpoints").exists()
    assert not (run / "parts").exists()
    assert not (run / "reports").exists()


def test_evaluate_identity_can_come_from_yaml_config(tmp_path):
    output = tmp_path / "configured-runs"
    config = tmp_path / "evaluate.yaml"
    config.write_text(
        "evaluate:\n"
        "  space: nb201_topology\n"
        "  proxies: params\n"
        "  count: 1\n"
        "  device: cpu\n"
        "  input_source: random\n"
        "  batch_size: 1\n"
        "  input_size: 8\n"
        f"  output: {output}\n",
        encoding="utf-8",
    )

    main(["evaluate", "--config", str(config)])

    run = next(output.iterdir())
    rows = [json.loads(line) for line in (run / "scores.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["proxy_id"] == "params"
    assert rows[0]["proxy_version"] == "count-v2"
    assert rows[0]["direction"] == "maximize"
    assert rows[0]["resource_direction"] == "minimize"


def test_analyze_correlation_cli_retains_failed_rows_in_coverage(tmp_path):
    scores = tmp_path / "scores.jsonl"
    rows = [
        {
            "architecture_id": f"a{index}",
            "proxy_id": "synflow",
            "primary_component": "score",
            "components": {"score": float(index)},
            "score": float(index),
            "target_value": float(index),
            "seed": 1,
            "status": "ok",
        }
        for index in (1, 2, 3)
    ]
    rows.append(
        {
            "architecture_id": "a4",
            "proxy_id": "synflow",
            "primary_component": "score",
            "components": {},
            "score": None,
            "target_value": 4.0,
            "seed": 1,
            "status": "failed",
        }
    )
    scores.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "analysis"

    main(
        [
            "analyze",
            "correlation",
            "--scores",
            str(scores),
            "--bootstrap-samples",
            "0",
            "--output",
            str(output),
        ]
    )

    import pandas as pd

    record = pd.read_csv(output / "correlations.csv").iloc[0]
    assert record["total_count"] == 4
    assert record["failed_count"] == 1
    assert record["sample_count"] == 3
    assert record["coverage"] == pytest.approx(0.75)


def test_legacy_score_rows_fold_without_modifying_source():
    rows = [
        {"run_id": "r", "architecture_id": "a", "proxy_id": "er", "component": "mean", "score": 2.0},
        {"run_id": "r", "architecture_id": "a", "proxy_id": "er", "component": "sum", "score": 8.0},
    ]
    normalized = list(normalize_score_records(rows))
    assert len(normalized) == 1
    assert normalized[0]["score"] == 2.0
    assert normalized[0]["components"] == {"mean": 2.0, "sum": 8.0}


def test_dataset_input_does_not_fall_back_to_random(tmp_path):
    with pytest.raises(FileNotFoundError):
        make_input_batch(
            "dataset", "cifar10", 2, 32, 10, 1, torch.device("cpu"), str(tmp_path / "missing")
        )


def test_explicit_missing_benchmark_path_does_not_fall_back_to_catalog(tmp_path):
    with pytest.raises(FileNotFoundError, match="Explicit benchmark path"):
        main(
            [
                "evaluate",
                "--benchmark",
                "nasbench101",
                "--benchmark-path",
                str(tmp_path / "missing.jsonl"),
                "--device",
                "cpu",
                "--input-source",
                "random",
            ]
        )


@pytest.mark.parametrize(
    ("dataset_name", "split", "sample_count", "class_count", "expected_size", "counts"),
    [
        ("cifar10", "train", 50_000, 10, 500, {50}),
        ("cifar10", "valid", 10_000, 10, 100, {10}),
        ("cifar100", "train", 50_000, 100, 500, {5}),
        ("cifar100", "valid", 10_000, 100, 100, {1}),
        ("imagenet1k", "train", 1_281_167, 1000, 12_812, {12, 13}),
        ("imagenet1k", "valid", 50_000, 1000, 500, {1}),
    ],
)
def test_one_percent_subset_matches_canonical_split_sizes_and_distribution(
    dataset_name,
    split,
    sample_count,
    class_count,
    expected_size,
    counts,
):
    class Dataset(torch.utils.data.Dataset):
        targets = [index % class_count for index in range(sample_count)]

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, index):
            return index, self.targets[index]

    dataset = Dataset()
    first = _stratified_subset(dataset, 0.01, 2026)
    second = _stratified_subset(dataset, 0.01, 2026)
    selected_counts = Counter(dataset.targets[index] for index in first.indices)

    assert len(first) == expected_size, (dataset_name, split)
    assert first.indices == second.indices
    assert set(selected_counts.values()) == counts
    if expected_size < class_count:
        assert len(selected_counts) == expected_size


def test_stratified_subset_fails_closed_when_global_target_is_empty():
    class Dataset(torch.utils.data.Dataset):
        targets = [0] * 49

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, index):
            return index, self.targets[index]

    with pytest.raises(ValueError, match="empty subset"):
        _stratified_subset(Dataset(), 0.01, 7)


def test_config_cannot_enable_trusted_execution(tmp_path):
    config = tmp_path / "evaluate.yaml"
    config.write_text(
        "evaluate:\n  benchmark: nasbench201\n  trusted: true\n",
        encoding="utf-8",
    )

    with pytest.raises(PermissionError, match="explicitly"):
        main(["evaluate", "--config", str(config)])


def test_config_cannot_self_approve_formal_training_protocol(tmp_path):
    config = tmp_path / "train.yaml"
    config.write_text(
        "space: autoformer\n"
        "dataset: imagenet1k\n"
        "epochs: 1\n"
        "optimizer: adamw\n"
        "learning_rate: 0.001\n"
        "weight_decay: 0.01\n"
        "formal_training_ready: true\n",
        encoding="utf-8",
    )

    with pytest.raises(NotImplementedError, match="not approved"):
        main(["train", "--config", str(config), "--device", "cpu"])


def test_approved_formal_protocol_rejects_modified_recipe(tmp_path):
    config = tmp_path / "train.yaml"
    source = Path("configs/training/darts_cifar10.yaml").read_text(encoding="utf-8")
    config.write_text(source.replace("learning_rate: 0.025", "learning_rate: 0.1"), encoding="utf-8")

    with pytest.raises(ValueError, match="learning_rate"):
        main(["train", "--config", str(config), "--device", "cpu"])


def test_approved_formal_protocol_requires_real_data(monkeypatch, tmp_path):
    monkeypatch.setattr("zcp_test.cli._resolve_data_root", lambda args, dataset: None)
    config = Path("configs/training/darts_cifar10.yaml")
    load_builtin_spaces()
    architecture_path = tmp_path / "darts.json"
    architecture_path.write_text(
        json.dumps(SPACES.create("darts").sample(42).to_dict()), encoding="utf-8"
    )
    output = tmp_path / "runs"
    with pytest.raises(ValueError, match="data-root"):
        main(
            [
                "train",
                "--config",
                str(config),
                "--architecture",
                str(architecture_path),
                "--device",
                "cpu",
                "--output",
                str(output),
            ]
        )
    assert not output.exists()


def test_autoformer_candidate_training_protocol_is_locked():
    config = yaml.safe_load(
        Path("configs/training/autoformer_imagenet.yaml").read_text(encoding="utf-8")
    )
    assert validate_candidate_training_protocol(config) == "aznas-autoformer-scratch"
    assert config["epochs"] == 500
    assert config["warmup_epochs"] == 20
    assert config["warmup_learning_rate"] == pytest.approx(1e-6)
    assert config["minimum_learning_rate"] == pytest.approx(1e-5)
    assert config["validation_label_smoothing"] == 0.0
    assert config["exclude_bias_norm_from_weight_decay"] is True
    assert config["batch_size_semantics"] == "per_device"
    assert config["formal_training_ready"] is True
    assert config["formal_training_acceptance"].endswith(
        "autoformer_single_candidate_dual_one_percent_completion_20260804.json"
    )
    assert "formal_training_blockers" not in config
    config["mixup"] = 0.0
    with pytest.raises(ValueError, match="mixup"):
        validate_candidate_training_protocol(config)


def test_autoformer_formal_training_gate_is_released(monkeypatch, tmp_path):
    monkeypatch.setattr("zcp_test.cli._resolve_data_root", lambda args, dataset: None)
    load_builtin_spaces()
    architecture_path = tmp_path / "autoformer.json"
    architecture_path.write_text(
        json.dumps(SPACES.create("autoformer").sample(42).to_dict()), encoding="utf-8"
    )
    output = tmp_path / "runs"

    with pytest.raises(ValueError, match="data-root"):
        main(
            [
                "train",
                "--config",
                "configs/training/autoformer_imagenet.yaml",
                "--architecture",
                str(architecture_path),
                "--device",
                "cpu",
                "--output",
                str(output),
            ]
        )

    assert not output.exists()


def test_proxyless_candidate_training_protocol_is_locked():
    config = yaml.safe_load(
        Path("configs/training/ofa_proxyless_mbv2_imagenet.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert validate_candidate_training_protocol(config) == (
        "project-ofa-proxyless-mbv2-scratch-v1"
    )
    assert config["scheduler"] == "cosine_step"
    assert config["color_distortion"] == "tf"
    assert config["exclude_norm_from_weight_decay"] is True
    assert config["candidate_input_size_policy"] == "architecture_resolution"
    assert config["training_protocol_fidelity"] == (
        "project_candidate_resolution_adaptation"
    )
    assert config["formal_training_ready"] is True
    assert config["formal_training_acceptance"].endswith(
        "proxyless_single_candidate_dual_one_percent_completion_20260804.json"
    )
    assert "formal_training_blockers" not in config
    assert validate_formal_training_protocol(config) == (
        "project-ofa-proxyless-mbv2-scratch-v1"
    )


def test_formal_training_evidence_must_exist_and_match_checksum(monkeypatch, tmp_path):
    config = yaml.safe_load(
        Path("configs/training/ofa_proxyless_mbv2_imagenet.yaml").read_text(
            encoding="utf-8"
        )
    )
    monkeypatch.setattr(training_protocols, "PROJECT_ROOT", tmp_path)
    with pytest.raises(ValueError, match="evidence is missing"):
        validate_formal_training_protocol(config)


def test_proxyless_formal_training_gate_is_released(monkeypatch, tmp_path):
    monkeypatch.setattr("zcp_test.cli._resolve_data_root", lambda args, dataset: None)
    architecture_path = tmp_path / "proxyless.json"
    architecture_path.write_text(
        json.dumps(
            {
                "kernel_size": [3] * 21,
                "expand_ratio": [3] * 21,
                "depth": [2] * 5,
                "width_mult": 1.3,
                "resolution": 128,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "runs"

    with pytest.raises(ValueError, match="data-root"):
        main(
            [
                "train",
                "--config",
                "configs/training/ofa_proxyless_mbv2_imagenet.yaml",
                "--architecture",
                str(architecture_path),
                "--device",
                "cpu",
                "--output",
                str(output),
            ]
        )

    assert not output.exists()


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--epochs", "1"], "epoch schedule"),
        (["--data-fraction", "0.01"], "complete dataset"),
        (["--classes", "999"], "exactly 1000 classes"),
    ],
)
def test_proxyless_formal_training_rejects_protocol_overrides(
    tmp_path, extra, message
):
    architecture_path = tmp_path / "proxyless.json"
    architecture_path.write_text(
        json.dumps(
            {
                "kernel_size": [3] * 21,
                "expand_ratio": [3] * 21,
                "depth": [2] * 5,
                "width_mult": 1.3,
                "resolution": 128,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "runs"
    with pytest.raises(ValueError, match=message):
        main(
            [
                "train",
                "--config",
                "configs/training/ofa_proxyless_mbv2_imagenet.yaml",
                "--architecture",
                str(architecture_path),
                "--device",
                "cpu",
                "--output",
                str(output),
                *extra,
            ]
        )
    assert not output.exists()


def test_formal_training_requires_architecture_file(tmp_path):
    base = [
        "train",
        "--config",
        "configs/training/ofa_proxyless_mbv2_imagenet.yaml",
        "--device",
        "cpu",
        "--output",
        str(tmp_path / "runs"),
    ]
    with pytest.raises(ValueError, match="explicit frozen"):
        main(base)
    with pytest.raises(ValueError, match="existing JSON file"):
        main([*base, "--architecture", json.dumps({"resolution": 128})])


def test_proxyless_acceptance_training_uses_candidate_resolution(monkeypatch, tmp_path):
    captured = {}
    load_builtin_spaces()
    space = SPACES.create("ofa_proxyless_mbv2")
    architecture = space.canonicalize(
        {
            "kernel_size": [3] * 21,
            "expand_ratio": [3] * 21,
            "depth": [2] * 5,
            "width_mult": 1.3,
            "resolution": 128,
        }
    )

    @contextmanager
    def training_device(*args, **kwargs):
        yield torch.device("cpu"), {"selection_strategy": "test"}

    @contextmanager
    def training_run_context(output, argv, resolved, runtime, world_size, rank):
        captured["resolved"] = resolved
        yield SimpleNamespace(directory=tmp_path)

    def fake_train_model(*args, **kwargs):
        captured["run_identity"] = kwargs["run_identity"]
        return {"best_accuracy": 0.0}

    monkeypatch.setattr(cli, "_prepare_gpu", lambda args: None)
    monkeypatch.setattr(cli, "_training_device", training_device)
    monkeypatch.setattr(cli, "_training_run_context", training_run_context)
    monkeypatch.setattr(cli, "_build_training_model", lambda *args: torch.nn.Linear(1, 1))
    monkeypatch.setattr(cli, "_resolve_data_root", lambda args, dataset: tmp_path)
    monkeypatch.setattr(cli, "_real_loaders", lambda *args, **kwargs: ([], []))
    monkeypatch.setattr(cli, "train_model", fake_train_model)

    main(
        [
            "train",
            "--config",
            "configs/training/ofa_proxyless_mbv2_imagenet.yaml",
            "--architecture",
            json.dumps(architecture.to_dict()),
            "--acceptance-smoke",
            "--epochs",
            "2",
            "--data-fraction",
            "1.0",
            "--device",
            "cpu",
        ]
    )

    assert captured["resolved"]["configured_input_size"] == 224
    assert captured["resolved"]["input_size"] == 128
    assert captured["resolved"]["candidate_resolution"] == 128
    assert captured["resolved"]["candidate_input_size_policy"] == (
        "architecture_resolution"
    )
    assert captured["run_identity"]["input_size"] == 128

    with pytest.raises(ValueError, match="candidate input_size"):
        main(
            [
                "train",
                "--config",
                "configs/training/ofa_proxyless_mbv2_imagenet.yaml",
                "--architecture",
                json.dumps(architecture.to_dict()),
                "--acceptance-smoke",
                "--epochs",
                "2",
                "--data-fraction",
                "1.0",
                "--input-size",
                "224",
                "--device",
                "cpu",
            ]
        )


def test_proxyless_project_zcp_transfer_search_profile_is_explicitly_scoped():
    config = yaml.safe_load(
        Path("configs/search/proxyless_mbv2_zen_project_transfer.yaml").read_text(
            encoding="utf-8"
        )
    )["search"]

    assert config == {
        "controller": "generic",
        "space": "ofa_proxyless_mbv2",
        "proxy": "zen",
        "aggregator": "primary",
        "population": 1000,
        "generations": 0,
        "project_budget_protocol": "one_percent_planned_100k",
        "elite_ratio": 0.2,
        "seed": 20260731,
        "input_source": "random",
        "batch_size": 16,
        "classes": 1000,
        "dataset": "imagenet1k",
        "weight_mode": "independent_scratch",
        "bn_recalibration_batches": 0,
        "gpu": "auto",
        "output": "runs/search/proxyless-mbv2-zen-project-transfer",
    }
    assert "input_size" not in config
    assert "model_checkpoint" not in config


def test_proxyless_one_percent_project_budget_rejects_wrong_candidate_count():
    with pytest.raises(ValueError, match="exactly 1,000 candidate evaluations"):
        main(
            [
                "search",
                "--config",
                "configs/search/proxyless_mbv2_zen_project_transfer.yaml",
                "--population",
                "999",
                "--device",
                "cpu",
            ]
        )


def test_project_one_percent_budget_is_not_a_generic_space_fraction():
    with pytest.raises(ValueError, match="currently defined only"):
        main(
            [
                "search",
                "--space",
                "nb201_topology",
                "--project-budget-protocol",
                "one_percent_planned_100k",
                "--population",
                "1000",
                "--generations",
                "0",
                "--device",
                "cpu",
            ]
        )


def test_plainnet_candidate_training_protocol_is_locked():
    config = yaml.safe_load(
        Path("configs/training/zennas_plainnet_mbv2_imagenet.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert validate_candidate_training_protocol(config) == (
        "aznas-plainnet-mbv2-scratch-5e6683a2"
    )
    assert config["scheduler"] == "cosine_warmup_step"
    assert config["learning_rate_reference_batch_size"] == 256
    assert config["batch_size"] == 512
    assert config["bn_momentum"] == pytest.approx(0.01)
    assert config["use_se"] is True
    assert config["formal_training_ready"] is False


@pytest.mark.parametrize(
    "arguments",
    [
        ["--epochs", "4", "--data-fraction", "1.0"],
        ["--epochs", "500", "--data-fraction", "0.02"],
    ],
)
def test_autoformer_acceptance_training_rejects_non_one_percent_protocol(arguments):
    with pytest.raises(ValueError, match="1%"):
        main(
            [
                "train",
                "--config",
                "configs/training/autoformer_imagenet.yaml",
                "--acceptance-smoke",
                "--device",
                "cpu",
                *arguments,
            ]
        )


def test_autoformer_acceptance_training_rejects_less_than_one_percent_data():
    with pytest.raises(ValueError, match="exactly 1% data"):
        main(
            [
                "train",
                "--config",
                "configs/training/autoformer_imagenet.yaml",
                "--acceptance-smoke",
                "--device",
                "cpu",
                "--epochs",
                "500",
                "--data-fraction",
                "0.005",
            ]
        )


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [("--batch-size", "128", "batch_size"), ("--input-size", "192", "input_size")],
)
def test_autoformer_acceptance_training_rejects_shape_overrides(option, value, message):
    with pytest.raises(ValueError, match=message):
        main(
            [
                "train",
                "--config",
                "configs/training/autoformer_imagenet.yaml",
                "--acceptance-smoke",
                "--device",
                "cpu",
                "--epochs",
                "1",
                option,
                value,
            ]
        )


def test_autoformer_acceptance_training_requires_real_data(monkeypatch, tmp_path):
    monkeypatch.setattr("zcp_test.cli._resolve_data_root", lambda args, dataset: None)
    with pytest.raises(ValueError, match="data-root"):
        main(
            [
                "train",
                "--config",
                "configs/training/autoformer_imagenet.yaml",
                "--acceptance-smoke",
                "--device",
                "cpu",
                "--epochs",
                "5",
                "--output",
                str(tmp_path),
            ]
        )
    assert list(tmp_path.iterdir()) == []


def test_darts_formal_protocol_is_allowed_for_one_percent_acceptance(monkeypatch, tmp_path):
    monkeypatch.setattr("zcp_test.cli._resolve_data_root", lambda args, dataset: None)
    with pytest.raises(ValueError, match="data-root"):
        main(
            [
                "train",
                "--config",
                "configs/training/darts_cifar10.yaml",
                "--acceptance-smoke",
                "--device",
                "cpu",
                "--epochs",
                "6",
                "--data-fraction",
                "1.0",
                "--output",
                str(tmp_path),
            ]
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("config_path", "minimum_epochs"),
    [
        ("configs/training/darts_cifar10.yaml", 6),
        ("configs/training/darts_cifar100.yaml", 6),
        ("configs/training/darts_imagenet.yaml", 3),
    ],
)
def test_darts_acceptance_minimum_full_data_epochs(config_path, minimum_epochs):
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    assert (
        resolve_acceptance_protocol(config, minimum_epochs, 1.0)
        == "full_data_one_percent_epochs"
    )
    with pytest.raises(ValueError, match="at least 1% epochs"):
        resolve_acceptance_protocol(config, minimum_epochs - 1, 1.0)


@pytest.mark.parametrize("data_fraction", [0.0100001, 0.010001, 0.0099999])
def test_one_percent_acceptance_rejects_nearby_fractions(data_fraction):
    config = yaml.safe_load(
        Path("configs/training/darts_cifar10.yaml").read_text(encoding="utf-8")
    )
    with pytest.raises(ValueError, match="exactly 1% data"):
        resolve_acceptance_protocol(config, 600, data_fraction)


@pytest.mark.parametrize(
    ("mode", "epochs", "data_fraction", "expected_protocol", "expected_training_mode"),
    [
        (
            "--acceptance-smoke",
            6,
            1.0,
            "full_data_one_percent_epochs",
            "acceptance_smoke",
        ),
        (
            "--acceptance-smoke",
            600,
            0.01,
            "one_percent_data_protocol",
            "acceptance_smoke",
        ),
        ("--real-data-preflight", 1, 1.0, None, "real_data_preflight"),
    ],
)
def test_darts_real_data_modes_resolve_config_identity_and_ddp_batch(
    monkeypatch,
    tmp_path,
    mode,
    epochs,
    data_fraction,
    expected_protocol,
    expected_training_mode,
):
    captured = {}

    @contextmanager
    def training_device(*args, **kwargs):
        yield torch.device("cpu"), {"selection_strategy": "test"}

    @contextmanager
    def training_run_context(output, argv, resolved, runtime, world_size, rank):
        captured["resolved"] = resolved
        yield SimpleNamespace(directory=tmp_path)

    def fake_train_model(*args, **kwargs):
        captured["training_config"] = args[3]
        captured["run_identity"] = kwargs["run_identity"]
        return {"best_accuracy": 0.0}

    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(cli, "_prepare_gpu", lambda args: None)
    monkeypatch.setattr(cli, "_training_device", training_device)
    monkeypatch.setattr(cli, "_training_run_context", training_run_context)
    monkeypatch.setattr(cli, "_build_training_model", lambda *args: torch.nn.Linear(1, 1))
    monkeypatch.setattr(
        torch.nn.parallel,
        "DistributedDataParallel",
        lambda model, **kwargs: model,
    )
    monkeypatch.setattr(cli, "_resolve_data_root", lambda args, dataset: tmp_path)
    monkeypatch.setattr(cli, "_real_loaders", lambda *args, **kwargs: ([], []))
    monkeypatch.setattr(cli, "train_model", fake_train_model)

    main(
        [
            "train",
            "--config",
            "configs/training/darts_cifar10.yaml",
            mode,
            "--epochs",
            str(epochs),
            "--data-fraction",
            str(data_fraction),
        ]
    )

    assert captured["resolved"]["acceptance_protocol"] == expected_protocol
    assert captured["resolved"]["training_mode"] == expected_training_mode
    assert captured["resolved"]["configured_batch_size"] == 96
    assert captured["resolved"]["per_device_batch_size"] == 48
    assert captured["resolved"]["effective_global_batch_size"] == 96
    assert captured["resolved"]["schedule_epochs"] == 600
    assert captured["training_config"].schedule_epochs == 600
    assert captured["resolved"]["seed"] == 42
    assert captured["resolved"]["rank_seed"] == 42
    assert captured["resolved"]["deterministic"] is True
    assert captured["run_identity"]["acceptance_protocol"] == expected_protocol
    assert captured["run_identity"]["training_mode"] == expected_training_mode
    assert captured["run_identity"]["seed"] == 42
    assert captured["run_identity"]["data_fraction"] == data_fraction


def test_training_smoke_modes_are_mutually_exclusive():
    with pytest.raises(SystemExit) as error:
        main(
            [
                "train",
                "--config",
                "configs/training/autoformer_imagenet.yaml",
                "--smoke",
                "--acceptance-smoke",
            ]
        )
    assert error.value.code == 2


@pytest.mark.parametrize(
    "arguments",
    [
        ["--epochs", "2", "--data-fraction", "1.0"],
        ["--epochs", "1", "--data-fraction", "0.01"],
    ],
)
def test_real_data_preflight_requires_one_complete_data_epoch(arguments):
    with pytest.raises(ValueError, match="exactly one epoch over the complete dataset"):
        main(
            [
                "train",
                "--config",
                "configs/training/darts_cifar10.yaml",
                "--real-data-preflight",
                "--device",
                "cpu",
                *arguments,
            ]
        )


def test_real_data_preflight_requires_real_data(monkeypatch, tmp_path):
    monkeypatch.setattr("zcp_test.cli._resolve_data_root", lambda args, dataset: None)
    with pytest.raises(ValueError, match="data-root"):
        main(
            [
                "train",
                "--config",
                "configs/training/darts_cifar10.yaml",
                "--real-data-preflight",
                "--epochs",
                "1",
                "--data-fraction",
                "1.0",
                "--device",
                "cpu",
                "--output",
                str(tmp_path),
            ]
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("--batch-size", "8", "batch_size"),
        ("--input-size", "64", "input_size"),
    ],
)
def test_approved_formal_protocol_rejects_shape_overrides(option, value, message):
    config = Path("configs/training/darts_cifar10.yaml")
    with pytest.raises(ValueError, match=message):
        main(["train", "--config", str(config), option, value, "--device", "cpu"])


def test_formal_training_rejects_non_reference_space_before_protocol(tmp_path):
    config = tmp_path / "train.yaml"
    config.write_text(
        "space: darts_toy_legacy\n"
        "dataset: cifar10\n"
        "epochs: 1\n"
        "optimizer: sgd\n"
        "learning_rate: 0.1\n"
        "weight_decay: 0.0\n"
        "formal_training_ready: true\n",
        encoding="utf-8",
    )
    with pytest.raises(NotImplementedError, match="requires reference_model"):
        main(["train", "--config", str(config), "--device", "cpu"])


def test_incomplete_formal_training_protocol_is_rejected(tmp_path):
    config = tmp_path / "train.yaml"
    config.write_text(
        "space: autoformer\n"
        "dataset: imagenet1k\n"
        "epochs: 1\n"
        "optimizer: adamw\n"
        "learning_rate: 0.001\n"
        "weight_decay: 0.01\n"
        "formal_training_ready: false\n"
        "formal_training_blockers: [repeated augmentation]\n",
        encoding="utf-8",
    )

    with pytest.raises(NotImplementedError, match="repeated augmentation"):
        main(["train", "--config", str(config), "--device", "cpu"])


def test_train_rejects_manual_device_override_under_torchrun(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    with pytest.raises(ValueError, match="cuda:LOCAL_RANK"):
        main(
            [
                "train",
                "--config",
                "configs/training/darts_cifar10.yaml",
                "--smoke",
                "--device",
                "cpu",
            ]
        )


def test_train_architecture_accepts_inline_json_and_json_file(tmp_path):
    specification = {"normal": [], "reduce": []}
    assert _load_architecture_spec(json.dumps({"spec": specification})) == specification
    path = tmp_path / "architecture.json"
    path.write_text(json.dumps(specification), encoding="utf-8")
    assert _load_architecture_spec(str(path)) == specification
    with pytest.raises(ValueError, match="existing JSON file"):
        _load_architecture_spec("not-json")


def test_evaluate_rejects_approximation_without_explicit_opt_in(tmp_path):
    with pytest.raises(RuntimeError, match="allow-approximation"):
        main(
            [
                "evaluate",
                "--space",
                "darts_toy_legacy",
                "--proxies",
                "params",
                "--count",
                "1",
                "--device",
                "cpu",
                "--input-source",
                "random",
                "--output",
                str(tmp_path),
            ]
        )


def test_correlate_uses_selected_score_field_and_inner_join(tmp_path):
    scores = tmp_path / "scores.jsonl"
    targets = tmp_path / "targets.jsonl"
    output = tmp_path / "correlations.jsonl"
    score_rows = [
        {
            "architecture_id": architecture_id,
            "proxy_id": "p",
            "score": wrong,
            "alternate": correct,
            "status": status,
        }
        for architecture_id, wrong, correct, status in (
            ("a", 3.0, 1.0, "ok"),
            ("b", 2.0, 2.0, "ok"),
            ("c", 1.0, 3.0, "ok"),
            ("failed", 4.0, 4.0, "failed"),
        )
    ]
    scores.write_text("".join(json.dumps(row) + "\n" for row in score_rows))
    targets.write_text(
        "".join(
            json.dumps({"architecture_id": key, "accuracy": value}) + "\n"
            for key, value in (("a", 1.0), ("b", 2.0), ("c", 3.0), ("target-only", 9.0))
        )
    )
    main(
        [
            "correlate",
            "--scores",
            str(scores),
            "--targets",
            str(targets),
            "--output",
            str(output),
            "--score-field",
            "alternate",
            "--target-field",
            "accuracy",
        ]
    )
    row = json.loads(output.read_text())
    assert row["sample_count"] == 3
    assert row["spearman"] == pytest.approx(1.0)
    assert row["score_direction"] == "maximize"
    assert row["target_direction"] == "maximize"
    assert row["score_coverage"] == pytest.approx(1.0)
    assert row["target_coverage"] == pytest.approx(0.75)


def test_correlate_normalizes_score_and_target_directions(tmp_path):
    scores = tmp_path / "scores.jsonl"
    targets = tmp_path / "targets.jsonl"
    output = tmp_path / "correlations.jsonl"
    scores.write_text(
        "".join(
            json.dumps(
                {
                    "architecture_id": architecture_id,
                    "proxy_id": "latency",
                    "score": score,
                    "direction": "minimize",
                }
            )
            + "\n"
            for architecture_id, score in (("a", 3.0), ("b", 2.0), ("c", 1.0))
        ),
        encoding="utf-8",
    )
    targets.write_text(
        "".join(
            json.dumps({"architecture_id": architecture_id, "error": target}) + "\n"
            for architecture_id, target in (("a", 3.0), ("b", 2.0), ("c", 1.0))
        ),
        encoding="utf-8",
    )

    main(
        [
            "correlate",
            "--scores",
            str(scores),
            "--targets",
            str(targets),
            "--output",
            str(output),
            "--target-field",
            "error",
            "--target-direction",
            "minimize",
        ]
    )

    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["spearman"] == pytest.approx(1.0)
    assert row["score_direction"] == "minimize"
    assert row["target_direction"] == "minimize"
    assert row["direction_normalized_to_maximize"] is True


def test_correlate_never_merges_disjoint_score_protocols(tmp_path):
    scores = tmp_path / "scores.jsonl"
    targets = tmp_path / "targets.jsonl"
    output = tmp_path / "correlations.jsonl"
    score_rows = []
    target_rows = []
    for seed, pairs in (
        (1, (("a", 1.0), ("b", 2.0))),
        (2, (("c", 2.0), ("d", 1.0))),
    ):
        for architecture_id, score in pairs:
            score_rows.append(
                {
                    "architecture_id": architecture_id,
                    "proxy_id": "p",
                    "score": score,
                    "status": "ok",
                    "benchmark_id": "nasbench201",
                    "benchmark_version": "1.1",
                    "search_space_id": "nb201_topology",
                    "dataset": "cifar10-valid",
                    "target_metric": "accuracy",
                    "target_split": "valid",
                    "target_epoch_budget": 200,
                    "input_source": "dataset",
                    "input_fingerprint": f"batch-{seed}",
                    "model_fidelity": "reference_model",
                    "seed": seed,
                }
            )
            target_rows.append({"architecture_id": architecture_id, "accuracy": score})
    scores.write_text("".join(json.dumps(row) + "\n" for row in score_rows))
    targets.write_text("".join(json.dumps(row) + "\n" for row in target_rows))

    main(
        [
            "correlate",
            "--scores",
            str(scores),
            "--targets",
            str(targets),
            "--output",
            str(output),
            "--target-field",
            "accuracy",
        ]
    )

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 2
    assert {row["seed"] for row in rows} == {1, 2}
    assert {row["input_fingerprint"] for row in rows} == {"batch-1", "batch-2"}
    assert all(row["sample_count"] == 2 for row in rows)
    assert len({row["protocol_digest"] for row in rows}) == 2


@pytest.mark.parametrize("duplicate_source", ["scores", "targets"])
def test_correlate_rejects_duplicate_architecture_keys(tmp_path, duplicate_source):
    scores = tmp_path / "scores.jsonl"
    targets = tmp_path / "targets.jsonl"
    score_rows = [
        {
            "schema_version": "2.1",
            "architecture_id": "a",
            "proxy_id": "p",
            "score": 1.0,
            "components": {"score": 1.0},
        },
        {
            "schema_version": "2.1",
            "architecture_id": "b",
            "proxy_id": "p",
            "score": 2.0,
            "components": {"score": 2.0},
        },
    ]
    target_rows = [
        {"architecture_id": "a", "accuracy": 1.0},
        {"architecture_id": "b", "accuracy": 2.0},
    ]
    if duplicate_source == "scores":
        score_rows.append(dict(score_rows[0]))
    else:
        target_rows.append(dict(target_rows[0]))
    scores.write_text("".join(json.dumps(row) + "\n" for row in score_rows))
    targets.write_text("".join(json.dumps(row) + "\n" for row in target_rows))
    with pytest.raises(ValueError, match="Duplicate"):
        main(
            [
                "correlate",
                "--scores",
                str(scores),
                "--targets",
                str(targets),
                "--output",
                str(tmp_path / "out.jsonl"),
                "--target-field",
                "accuracy",
            ]
        )


def test_plainnet_throughput_preflight_is_not_formal_search():
    root = Path(__file__).resolve().parents[1]
    script = root / "tools" / "acceptance" / "run-plainnet-throughput-preflight.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    source = script.read_text(encoding="utf-8")
    assert "acceptance_exec_immutable" in source
    assert "ZCP_ACCEPTANCE_ROOT" in source
    assert "preflight-plainnet-source-aligned.py" in source
    assert "--accepted 3" in source
    assert "--flops-target 450m" in source
    assert "--lock-timeout" in source
    assert "--controller plainnet_source_aligned" not in source
    assert "/home/" not in source and "/public/" not in source

    runner = (
        root / "tools" / "acceptance" / "preflight-plainnet-source-aligned.py"
    ).read_text(encoding="utf-8")
    assert "valid_candidates=100_000" in runner
    assert "parent_pool=1_024" in runner
    assert "stop_after_accepted=args.accepted" in runner
    assert '"status": "waiting_for_gpu_lock"' in runner
    assert '"lock_acquired_at": timestamp()' in runner
    assert 'state.get("status") != "running"' in runner
    assert '"formal_search_completed": False' in runner
    assert "record_kind" in runner and "search summary" in runner


def test_plainnet_rerank_benchmark_is_planning_only():
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "tools" / "acceptance" / "benchmark-plainnet-rerank.py"
    ).read_text(encoding="utf-8")
    assert "cpu_only_planning_benchmark_not_formal_search" in source
    assert "formal_candidate_count" in source
    assert "conservative_cumulative_rerank_seconds" in source
