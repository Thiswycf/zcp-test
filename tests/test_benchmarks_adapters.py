import json

import pytest

from zcp_test.benchmarks import BENCHMARKS
from zcp_test.benchmarks.nasbench101 import NasBench101Adapter
from zcp_test.benchmarks.nasbench201 import NasBench201Adapter
from zcp_test.benchmarks.nasbench301 import NasBench301SurrogateAdapter
from zcp_test.benchmarks.nats import NatsSssAdapter, NatsTssAdapter
from zcp_test.benchmarks.transnasbench101 import TransNasBench101Adapter
from zcp_test.benchmarks.vitbench101 import VitBench101Adapter
from zcp_test.types import MetricSpec
from zcp_test.spaces.darts import DartsSpace


class FakeApi:
    def __init__(self, prefix="arch"):
        self.values = [f"{prefix}-{index}" for index in range(3)]
        self.calls = []

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]

    def arch(self, index):
        return self.values[index]

    def get_more_info(self, index, dataset, **kwargs):
        self.calls.append((index, dataset, kwargs))
        return {"valid-accuracy": 80.0 + index, "test-accuracy": 81.0 + index}


class FakeNatsMetadata:
    def __init__(self, seeds):
        self.seeds = seeds

    def get_dataset_seeds(self, dataset):
        assert dataset == "cifar10-valid"
        return self.seeds


class SeededFakeApi(FakeApi):
    def __init__(self):
        super().__init__()
        self.seed_values = {11: 79.0, 22: 83.0, 33: 81.0}
        self.metadata_calls = []

    def query_meta_info_by_index(self, index, *, hp):
        self.metadata_calls.append((index, hp))
        return FakeNatsMetadata(self.seed_values)

    def get_more_info(self, index, dataset, **kwargs):
        self.calls.append((index, dataset, kwargs))
        seed = kwargs["is_random"]
        value = 82.0 if seed is False else self.seed_values[seed]
        return {"valid-accuracy": value}


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_registry_contains_distinct_benchmarks():
    assert {"nasbench201", "nats_tss", "nats_sss", "nasbench101", "nasbench301_surrogate", "transnasbench101", "vitbench101"}.issubset(BENCHMARKS.names())


def test_nb201_and_nats_tss_share_space_not_identity(tmp_path):
    data = tmp_path / "benchmark"
    data.touch()
    nb_api, nats_api = FakeApi(), FakeApi()
    nb = NasBench201Adapter(str(data), version="1.1", api_factory=lambda _: nb_api)
    nats = NatsTssAdapter(str(data), version="1.0", api_factory=lambda _: nats_api)
    nb_arch = next(iter(nb.iter_architectures(end=1)))
    nats_arch = next(iter(nats.iter_architectures(end=1)))
    assert nb.search_space_id == nats.search_space_id == "nb201_topology"
    assert nb.benchmark_id != nats.benchmark_id
    assert nb_arch.benchmark_index == 0
    metric = MetricSpec("cifar10-valid", "valid", "accuracy", epoch_budget=12)
    assert nb.query_metrics(nb_arch, metric) == {"accuracy": 80.0}
    assert nats.query_metrics(nats_arch, metric) == {"accuracy": 80.0}
    assert nb_api.calls[0][2]["hp"] == "12"


def test_nats_sss_has_separate_space_and_budget(tmp_path):
    data = tmp_path / "sss"
    data.mkdir()
    adapter = NatsSssAdapter(str(data), version="1.0", api_factory=lambda _: FakeApi("channels"))
    assert adapter.search_space_id == "nats_size"
    with pytest.raises(ValueError, match="Epoch budget"):
        adapter.validate_metric(MetricSpec("cifar10", "test", "accuracy", epoch_budget=200))


def test_nats_seed_reduction_preserves_mean_and_explicit_seed(tmp_path):
    data = tmp_path / "tss"
    data.mkdir()
    api = SeededFakeApi()
    adapter = NatsTssAdapter(str(data), version="1.0", api_factory=lambda _: api)
    architecture = next(adapter.iter_architectures(end=1))

    mean = adapter.query_metrics(
        architecture,
        MetricSpec("cifar10-valid", "valid", "accuracy", 200, seed_reduction="mean"),
    )
    seeded = adapter.query_metrics(
        architecture,
        MetricSpec(
            "cifar10-valid",
            "valid",
            "accuracy",
            200,
            seed=22,
            seed_reduction="min",
        ),
    )

    assert mean == {"accuracy": 82.0}
    assert seeded == {"accuracy": 83.0}
    assert [call[2]["is_random"] for call in api.calls] == [False, 22]
    assert api.metadata_calls == []


@pytest.mark.parametrize(("reduction", "expected"), [("min", 79.0), ("max", 83.0)])
def test_nats_seed_reduction_enumerates_official_seeds(
    tmp_path, reduction, expected
):
    data = tmp_path / reduction
    data.mkdir()
    api = SeededFakeApi()
    adapter = NatsTssAdapter(str(data), version="1.0", api_factory=lambda _: api)
    architecture = next(adapter.iter_architectures(end=1))

    result = adapter.query_metrics(
        architecture,
        MetricSpec(
            "cifar10-valid",
            "valid",
            "accuracy",
            200,
            seed_reduction=reduction,
        ),
    )

    assert result == {"accuracy": expected}
    assert api.metadata_calls == [(0, "200")]
    assert [call[2]["is_random"] for call in api.calls] == [11, 22, 33]


def test_nats_seed_reduction_rejects_unknown_or_unavailable_enumeration(tmp_path):
    data = tmp_path / "tss"
    data.mkdir()
    api = FakeApi()
    adapter = NatsTssAdapter(str(data), version="1.0", api_factory=lambda _: api)
    architecture = next(adapter.iter_architectures(end=1))

    with pytest.raises(ValueError, match="Unsupported NATS seed reduction"):
        adapter.query_metrics(
            architecture,
            MetricSpec(
                "cifar10-valid",
                "valid",
                "accuracy",
                200,
                seed_reduction="median",
            ),
        )
    with pytest.raises(RuntimeError, match="requires API seed enumeration"):
        adapter.query_metrics(
            architecture,
            MetricSpec(
                "cifar10-valid",
                "valid",
                "accuracy",
                200,
                seed_reduction="min",
            ),
        )
    assert api.calls == []


def test_nb101_jsonl_validates_and_joins_by_architecture_id(tmp_path):
    path = tmp_path / "nb101.jsonl"
    spec = {"matrix": [[0, 1], [0, 0]], "operations": ["input", "output"]}
    write_jsonl(path, [{"record_kind": "benchmark_architecture", "benchmark_id": "nasbench101", "search_space_id": "nb101_dag", "benchmark_version": "only108", "benchmark_index": 0, "specification": spec, "metrics": [{"dataset": "cifar10", "split": "test", "metric_name": "accuracy", "epoch_budget": 108, "seed": 1, "value": 91.5}]}])
    adapter = NasBench101Adapter(str(path), version="only108")
    architecture = adapter.sample_architecture(seed=4)
    result = adapter.query_metrics(architecture, MetricSpec("cifar10", "test", "accuracy", 108, seed=1))
    assert result == {"accuracy": 91.5}


def test_vit_slice_protocol_is_not_silently_merged(tmp_path):
    path = tmp_path / "vit.jsonl"
    record = {"record_kind": "benchmark_architecture", "benchmark_id": "vitbench101", "search_space_id": "autoformer", "benchmark_version": "auto-prox-90ed458", "benchmark_index": 0, "protocol": "auto-prox-90ed458-autoformer-main", "specification": {"depth": 12, "hidden_dim": 384, "num_heads": [6] * 12, "mlp_ratio": [4.0] * 12}, "metrics": []}
    write_jsonl(path, [record])
    assert VitBench101Adapter(str(path), slice_id="autoformer_main").metadata()["slice_id"] == "autoformer_main"
    with pytest.raises(ValueError, match="protocol"):
        VitBench101Adapter(str(path), slice_id="autoformer_ext")


def test_transnas_micro_and_macro_are_explicit(tmp_path):
    path = tmp_path / "transnas.jsonl"
    write_jsonl(path, [{"record_kind": "benchmark_architecture", "benchmark_id": "transnasbench101", "search_space_id": "transnas_micro", "benchmark_version": "v10141024", "benchmark_index": 0, "specification": {"architecture": "64-41414-1_23_301"}, "metrics": [{"dataset": "class_object", "split": "test", "metric_name": "test_top1", "epoch_budget": 24, "value": 45.0}]}])
    adapter = TransNasBench101Adapter(str(path), space="micro")
    architecture = adapter.sample_architecture(0)
    assert adapter.query_metrics(architecture, MetricSpec("class_object", "test", "test_top1", 24)) == {"test_top1": 45.0}
    assert adapter.build_model(architecture, "class_object")(
        __import__("torch").randn(2, 3, 32, 32)
    ).shape == (2, 75)
    metadata = adapter.metadata()
    assert metadata["model_protocol"] == "official-encoder-and-task-head-pytorch-port"
    assert metadata["implementation_commit"] == "6d4231b1eb04e95750a5b2b6cf391db770bc25d6"
    with pytest.raises(ValueError, match="search_space_id"):
        TransNasBench101Adapter(str(path), space="macro")


def test_nb301_is_deterministic_unless_noise_requested(tmp_path):
    model_path = tmp_path / "ensemble"
    model_path.mkdir()
    architecture_path = tmp_path / "darts.jsonl"
    spec = DartsSpace().sample(0).spec
    write_jsonl(architecture_path, [{"record_kind": "benchmark_architecture", "benchmark_id": "nasbench301_surrogate", "search_space_id": "darts", "benchmark_version": "1.0", "benchmark_index": 0, "specification": spec, "metrics": []}])

    class Ensemble:
        def __init__(self):
            self.noise = None

        def predict(self, **kwargs):
            self.noise = kwargs["with_noise"]
            return 93.2

    ensemble = Ensemble()
    adapter = NasBench301SurrogateAdapter(str(model_path), architecture_path=str(architecture_path), ensemble_loader=lambda _: ensemble)
    result = adapter.query_metrics(adapter.sample_architecture(0), MetricSpec("cifar10", "test", "accuracy"))
    assert result == {"accuracy": 93.2}
    assert ensemble.noise is False

    generated = NasBench301SurrogateAdapter(
        str(model_path), ensemble_loader=lambda _: ensemble
    )
    generated_architecture = next(generated.iter_architectures(2, 3))
    assert generated_architecture.architecture_id == DartsSpace().sample(2).architecture_id
    assert generated.metadata()["architecture_source"] == "deterministic_darts_sampling"
    assert generated.metadata()["protocol"] == "nasbench301-surrogate-v1.0"
