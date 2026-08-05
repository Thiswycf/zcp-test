import copy
import json
import math
from pathlib import Path

import pytest
import torch

from zcp_test.artifacts import JsonlWriter, RunContext, merge_jsonl, read_jsonl
from zcp_test.artifacts.run import PROJECT_TIMEZONE_NAME, _package_versions, project_now_iso
from zcp_test.proxies.evaluator import evaluate_proxy
from zcp_test.proxies import PROXIES, load_builtin_proxies
from zcp_test.proxies.builtin import FunctionProxy, _meco, _meco_opt, _vkdnw
from zcp_test.reporting import correlation_summary
from zcp_test.search import EvolutionSearch, cache_key, load_search_state
from zcp_test.spaces import SPACES, load_builtin_spaces
from zcp_test.training import TrainingConfig, train_model
from zcp_test.training.checkpoint import atomic_torch_save, load_checkpoint
from zcp_test.training.trainer import (
    _collect_checkpoint_rng,
    _cosine_learning_rate,
    _normalized_checkpoint_config,
    _normalized_checkpoint_identity,
    _optimizer_parameter_groups,
    _restore_checkpoint_rng,
    _restore_training_log,
)
from zcp_test.types import ProxyCapability, ProxyOutput


def test_jsonl_merge_and_partial_recovery(tmp_path):
    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    JsonlWriter(first, 1).append({"id": "a", "value": 1})
    JsonlWriter(second, 1).append({"id": "a", "value": 2})
    JsonlWriter(second, 1).append({"id": "b", "value": 3})
    assert merge_jsonl([first, second], tmp_path / "merged.jsonl", ("id",)) == 2
    assert [row["value"] for row in read_jsonl(tmp_path / "merged.jsonl")] == [2, 3]


def test_runtime_package_versions_skip_uninstalled_packages(monkeypatch):
    from importlib import metadata

    def version(package):
        if package == "present":
            return "1.2.3"
        raise metadata.PackageNotFoundError(package)

    monkeypatch.setattr("zcp_test.artifacts.run.metadata.version", version)

    assert _package_versions(("present", "missing")) == {"present": "1.2.3"}


def test_project_timestamps_use_beijing_timezone():
    assert project_now_iso().endswith("+08:00")
    assert PROJECT_TIMEZONE_NAME == "Asia/Shanghai"


def test_run_context_logs_structured_events(tmp_path):
    with RunContext(tmp_path, ["zcp-test", "train"], {"dataset": "fixture"}) as run:
        run.event("training_batch_progress", epoch=0, batch=3, batch_count=10)

    assert "+0800_" in run.directory.name
    manifest = json.loads((run.directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["timezone"] == "Asia/Shanghai"
    assert manifest["started_at"].endswith("+08:00")
    assert manifest["ended_at"].endswith("+08:00")
    assert all(
        row["timestamp"].endswith("+08:00")
        for row in read_jsonl(run.directory / "events.jsonl")
    )
    log = (run.directory / "run.log").read_text(encoding="utf-8")
    assert "run_started {}" in log
    assert "training_batch_progress" in log
    assert '"batch": 3' in log
    assert '"batch_count": 10' in log
    assert "run_finished" in log
    assert '"status": "completed"' in log


def test_spaces_and_cache_keys():
    load_builtin_spaces()
    expected = {
        "nb201_topology",
        "nats_size",
        "nb101_dag",
        "nb101_toy_legacy",
        "darts",
        "darts_toy_legacy",
        "transnas_micro",
        "transnas_macro",
        "autoformer",
        "pit",
        "zennas_plainnet_mbv2",
        "ofa_proxyless_mbv2",
        "ofa_mbv3",
    }
    assert expected == set(SPACES.names())
    architecture = SPACES.create("darts").sample(1)
    assert cache_key(architecture, "er", "cifar10", 1, "x") != cache_key(
        architecture, "er", "cifar100", 1, "x"
    )


def test_proxy_state_isolation():
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(4, 3),
    )
    before = {name: value.clone() for name, value in model.state_dict().items()}
    result = evaluate_proxy(
        "gradnorm",
        model,
        torch.randn(2, 3, 8, 8),
        torch.tensor([0, 1]),
        torch.nn.CrossEntropyLoss(),
    )
    assert result.status.value == "ok"
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())


@pytest.mark.parametrize(
    ("proxy_id", "output", "message"),
    [
        (
            "contract_wrong_primary",
            ProxyOutput(1.0, primary_component="aux", components={"aux": 1.0}),
            "returned primary component 'aux', but declared 'score'",
        ),
        (
            "contract_undeclared_component",
            ProxyOutput(1.0, components={"score": 1.0, "hidden": 2.0}),
            "returned undeclared components: hidden",
        ),
        (
            "contract_inconsistent_score",
            ProxyOutput(1.0, components={"score": 2.0}),
            "primary component value inconsistent with score",
        ),
    ],
)
def test_proxy_output_must_match_declared_capability(monkeypatch, proxy_id, output, message):
    monkeypatch.setitem(
        PROXIES._entries,
        proxy_id,
        lambda: FunctionProxy(ProxyCapability(proxy_id), lambda *args: output),
    )

    result = evaluate_proxy(proxy_id, torch.nn.Linear(2, 2), torch.ones(1, 2))

    assert result.status.value == "failed"
    assert message in result.error_message


def test_params_and_flops_separate_accuracy_and_resource_directions():
    load_builtin_proxies()
    for proxy_id, version in (("params", "count-v2"), ("flops", "thop-v2")):
        capability = PROXIES.create(proxy_id).capability
        assert capability.version == version
        assert capability.direction.value == "maximize"
        assert capability.resource_direction.value == "minimize"


def test_proxy_state_isolation_removes_injected_buffers():
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(4, 3),
    )
    before = set(model.state_dict())
    inputs = torch.randn(2, 3, 8, 8)
    assert evaluate_proxy("naswot", model, inputs).status.value == "ok"
    assert evaluate_proxy("flops", model, inputs).status.value == "ok"
    assert set(model.state_dict()) == before


def test_synflow_uses_float64_for_deep_models_and_restores_dtype():
    layers = []
    for _ in range(50):
        layer = torch.nn.Linear(4, 4, bias=False)
        torch.nn.init.constant_(layer.weight, 2.0)
        layers.extend((layer, torch.nn.ReLU()))
    model = torch.nn.Sequential(*layers)
    before = {name: value.clone() for name, value in model.state_dict().items()}

    result = evaluate_proxy("synflow", model, torch.ones(2, 4))

    assert result.status.value == "ok"
    assert result.score is not None and math.isfinite(result.score)
    assert result.proxy_version == "zero-cost-nas-b5059bc"
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
    assert PROXIES.create("synflow").capability.requires_data is False
    assert PROXIES.create("synflow").capability.requires_inputs is True


def test_proxy_can_be_explicitly_marked_unsupported_by_input_contract():
    model = torch.nn.Linear(2, 2)

    result = evaluate_proxy(
        "params",
        model,
        unsupported_reason="task-specific labels are unavailable",
    )

    assert result.status.value == "unsupported"
    assert result.error_message == "task-specific labels are unavailable"
    assert result.primary_component == "score"


def test_unsupported_multicomponent_proxy_preserves_declared_primary_component():
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 2))

    result = evaluate_proxy(
        "az_nas",
        model,
        torch.randn(2, 1, 2, 2),
        labels=None,
        loss_fn=None,
    )

    assert result.status.value == "unsupported"
    assert "StaticAutoFormer and AZPlainNetMobileNetV2" in result.error_message
    assert result.primary_component == "expressivity"


@pytest.mark.parametrize(
    ("capability", "inputs", "labels", "loss_fn", "missing"),
    [
        (ProxyCapability("needs_inputs"), None, None, None, "inputs"),
        (
            ProxyCapability("needs_labels", requires_labels=True),
            torch.ones(1, 2),
            None,
            None,
            "labels",
        ),
        (
            ProxyCapability("needs_loss", requires_loss_fn=True),
            torch.ones(1, 2),
            None,
            None,
            "loss_fn",
        ),
    ],
)
def test_proxy_input_contract_fails_before_compute(
    monkeypatch, capability, inputs, labels, loss_fn, missing
):
    calls = 0

    def compute(*args):
        nonlocal calls
        calls += 1
        return 1.0

    monkeypatch.setitem(
        PROXIES._entries,
        capability.proxy_id,
        lambda: FunctionProxy(capability, compute),
    )
    result = evaluate_proxy(
        capability.proxy_id,
        torch.nn.Linear(2, 2),
        inputs,
        labels,
        loss_fn,
    )
    assert result.status.value == "unsupported"
    assert missing in result.error_message
    assert calls == 0


def test_input_independent_proxy_runs_without_tensor(monkeypatch):
    capability = ProxyCapability("no_inputs", requires_data=False, requires_inputs=False)
    monkeypatch.setitem(
        PROXIES._entries,
        capability.proxy_id,
        lambda: FunctionProxy(capability, lambda *args: 2.0),
    )
    result = evaluate_proxy("no_inputs", torch.nn.Linear(2, 2), inputs=None)
    assert result.status.value == "ok"
    assert result.score == 2.0


def test_builtin_loss_contracts_are_explicit():
    load_builtin_proxies()
    for proxy_id in ("gradnorm", "zico"):
        capability = PROXIES.create(proxy_id).capability
        assert capability.requires_inputs is True
        assert capability.requires_labels is True
        assert capability.requires_loss_fn is True
    for proxy_id in ("te_nas", "az_nas"):
        capability = PROXIES.create(proxy_id).capability
        assert capability.requires_inputs is True
        assert capability.requires_labels is False
        assert capability.requires_loss_fn is False
    assert PROXIES.create("params").capability.requires_inputs is False


def test_all_builtin_proxies_have_finite_cpu_contracts_and_provenance():
    load_builtin_spaces()
    load_builtin_proxies()
    architecture = SPACES.create("nb201_topology").sample(11)
    model = SPACES.create("nb201_topology").build_model(architecture, 3)
    inputs = torch.randn(4, 3, 8, 8)
    labels = torch.tensor([0, 1, 2, 0])
    for proxy_id in PROXIES.names():
        capability = PROXIES.create(proxy_id).capability
        context = None
        if proxy_id in {"er", "ter"}:
            from zcp_test.proxies.edge_adapters import capture_semantic_edge_activations
            from zcp_test.types import ProxyContext

            context = ProxyContext(
                inputs=inputs,
                labels=labels,
                loss_fn=torch.nn.CrossEntropyLoss(),
                model_family="cnn",
                edge_activations=capture_semantic_edge_activations(model, inputs),
            )
        result = evaluate_proxy(
            proxy_id,
            model,
            inputs,
            labels,
            torch.nn.CrossEntropyLoss(),
            context=context,
        )
        if "cnn" not in capability.model_families:
            assert result.status.value == "unsupported"
            continue
        if proxy_id in {"az_nas", "az_nas_plainnet"}:
            assert result.status.value == "unsupported"
            continue
        assert result.status.value == "ok", (proxy_id, result.error_message)
        assert result.score is not None and math.isfinite(result.score)
    assert result.implementation_fidelity != "unverified"


def test_meco_matches_pinned_upstream_feature_map_formula_and_cleans_hooks():
    class FeatureNetwork(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.convolution = torch.nn.Conv2d(3, 4, 3, padding=1, bias=False)
            self.activation = torch.nn.ReLU()

        def forward(self, inputs):
            return self.activation(self.convolution(inputs))

    torch.manual_seed(29)
    model = FeatureNetwork().eval()
    inputs = torch.randn(2, 3, 6, 6)
    with torch.no_grad():
        convolution = model.convolution(inputs)
        activation = model.activation(convolution)

    def minimum_channel_correlation_eigenvalue(output):
        feature_map = output[0].reshape(output.shape[1], -1)
        correlation = torch.nan_to_num(torch.corrcoef(feature_map))
        return torch.linalg.eigvals(correlation).real.min()

    expected = minimum_channel_correlation_eigenvalue(convolution)
    expected += 2 * minimum_channel_correlation_eigenvalue(activation)
    actual = _meco(model, inputs)

    assert actual == pytest.approx(float(expected), abs=1e-6)
    assert all(not module._forward_hooks for module in model.modules())


def test_meco_opt_is_an_official_variant_not_an_alias():
    load_builtin_proxies()
    meco = PROXIES.create("meco").capability
    optimized = PROXIES.create("meco_opt").capability

    assert meco.version == "hamstermimi-0d830dd-v2"
    assert optimized.version == "hamstermimi-0d830dd-v2"
    assert optimized.alias_of is None
    assert optimized.requires_data is False
    assert optimized.requires_inputs is True
    assert optimized.implementation_fidelity == "paper_formula_port_stabilized"
    assert "HamsterMimi/MeCo" in (optimized.source or "")


def test_meco_opt_matches_pinned_upstream_channel_sampling_formula():
    import random

    class FeatureNetwork(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.convolution = torch.nn.Conv2d(3, 12, 3, padding=1, bias=False)
            self.activation = torch.nn.ReLU()

        def forward(self, inputs):
            return self.activation(self.convolution(inputs))

    torch.manual_seed(31)
    model = FeatureNetwork().eval()
    inputs = torch.randn(2, 3, 6, 6)
    with torch.no_grad():
        convolution = model.convolution(inputs)
        activation = model.activation(convolution)

    random.seed(37)
    expected = 0.0
    for output in (convolution, activation, activation):
        feature_map = output[0].reshape(output.shape[1], -1)
        indices = random.sample(range(feature_map.shape[0]), 8)
        correlation = torch.nan_to_num(torch.corrcoef(feature_map[indices]))
        minimum = torch.linalg.eigvals(correlation).real.min()
        expected += float(minimum * feature_map.shape[0] / 8)

    random.seed(37)
    actual = _meco_opt(model, inputs)
    assert actual == pytest.approx(expected, abs=1e-6)
    assert all(not module._forward_hooks for module in model.modules())


def test_vkdnw_matches_pinned_fisher_decile_entropy_formula():
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(12, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 3),
    ).eval()
    reference = copy.deepcopy(model)
    inputs = torch.randn(3, 3, 2, 2)

    torch.manual_seed(43)
    actual = _vkdnw(model, inputs)

    torch.manual_seed(43)
    for module in reference.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            torch.nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    parameters = list(reference.parameters())[:128]
    fisher = torch.zeros(len(parameters), len(parameters))
    for sample in inputs:
        logits = reference(sample.unsqueeze(0)).squeeze(0)
        jacobian_rows = []
        for class_index in range(logits.numel()):
            gradients = torch.autograd.grad(
                logits[class_index], parameters, retain_graph=True
            )
            jacobian_rows.append(
                torch.stack([gradient.flatten()[0] for gradient in gradients])
            )
        jacobian = torch.stack(jacobian_rows)
        probability = torch.softmax(logits, dim=0) * 0.95 + 0.05 / logits.numel()
        covariance = torch.diag(probability) - probability[:, None] * probability[None, :]
        fisher += jacobian.T @ covariance @ jacobian
    fisher /= len(inputs)
    spectrum = torch.linalg.svdvals(fisher)
    quantiles = torch.quantile(spectrum, torch.arange(0.1, 1.0, 0.1))
    normalized = quantiles / quantiles.sum().clamp_min(1e-10)
    expected = -(normalized * torch.log(normalized + 1e-10)).sum()

    assert actual["entropy"] == pytest.approx(float(expected), abs=1e-5)
    assert actual["dimension"] == 4.0
    assert actual["single"] == pytest.approx(4.0 + float(expected), abs=1e-5)


def test_vkdnw_capability_uses_official_entropy_protocol():
    load_builtin_proxies()
    capability = PROXIES.create("vkdnw").capability

    assert capability.version == "ondratybl-d2ff276-v2"
    assert capability.model_families == ("cnn",)
    assert capability.requires_data is False
    assert capability.primary_component == "single"
    assert capability.components == ("single", "entropy", "dimension")
    assert capability.implementation_fidelity == "paper_formula_port_stabilized"
    assert "ondratybl/VKDNW" in (capability.source or "")


def test_aznas_autoformer_residual_features_components_and_rank_aggregation():
    from zcp_test.models.autoformer import AZNAS_SCRATCH_PROFILE, StaticAutoFormer
    from zcp_test.proxies.az_nas import autoformer_components, log_rank_aggregate

    torch.manual_seed(17)
    model = StaticAutoFormer(
        profile=AZNAS_SCRATCH_PROFILE,
        image_size=32,
        patch_size=16,
        num_classes=3,
        embed_dim=24,
        depth=2,
        num_heads=[2, 2],
        mlp_ratio=[2.0, 2.0],
        super_depth=14,
    ).eval()
    inputs = torch.randn(2, 3, 32, 32)

    residual_features = model.extract_res_features(inputs)
    torch.manual_seed(23)
    components = autoformer_components(model, inputs)
    torch.manual_seed(23)
    result = evaluate_proxy(
        "az_nas_autoformer",
        model,
        inputs,
        model_family="transformer",
    )

    assert len(residual_features) == 4
    assert all(feature.shape == (2, 5, 24) for feature in residual_features)
    assert torch.allclose(model(inputs), model.head(model.forward_features(inputs)))
    assert set(components) == {"expressivity", "trainability", "complexity"}
    assert components["complexity"] == model.official_complexity_ops()
    assert all(math.isfinite(value) for value in components.values())
    assert result.status.value == "ok"
    assert result.proxy_version == "aznas-5e6683-autoformer-stable-v1"
    assert result.components == pytest.approx(components)
    assert log_rank_aggregate(
        [{"left": 1.0, "right": 3.0}, {"left": 2.0, "right": 2.0}, {"left": 3.0, "right": 1.0}],
        ("left", "right"),
    ) == pytest.approx(
        [math.log(1 / 3), 2 * math.log(2 / 3), math.log(1 / 3)]
    )


def test_declared_transformer_proxies_execute_on_transformer_model():
    from zcp_test.models.autoformer import AZNAS_SCRATCH_PROFILE, StaticAutoFormer

    load_builtin_proxies()
    model = StaticAutoFormer(
        profile=AZNAS_SCRATCH_PROFILE,
        image_size=32,
        patch_size=16,
        num_classes=3,
        embed_dim=24,
        depth=2,
        num_heads=[2, 2],
        mlp_ratio=[2.0, 2.0],
        super_depth=14,
    )
    inputs = torch.randn(4, 3, 32, 32)
    labels = torch.tensor([0, 1, 2, 0])
    supported = [
        proxy_id
        for proxy_id in PROXIES.names()
        if "transformer" in PROXIES.create(proxy_id).capability.model_families
    ]
    for proxy_id in supported:
        result = evaluate_proxy(
            proxy_id,
            model,
            inputs,
            labels,
            torch.nn.CrossEntropyLoss(),
            model_family="transformer",
        )
        assert result.status.value == "ok", (proxy_id, result.error_message)
        assert result.score is not None and math.isfinite(result.score)


def test_checkpoint_loading_requires_trust(tmp_path):
    path = tmp_path / "checkpoint.pt"
    torch.save({"epoch": 0}, path)
    with pytest.raises(PermissionError):
        load_checkpoint(path)
    assert load_checkpoint(path, trusted=True)["epoch"] == 0


def test_statistics_and_search(tmp_path):
    assert correlation_summary([1, 2, 3], [2, 4, 6])["spearman"] == 1.0
    load_builtin_spaces()
    search = EvolutionSearch(
        SPACES.create("darts"),
        lambda architecture: float(int(architecture.architecture_id[:8], 16)),
        JsonlWriter(tmp_path / "search.jsonl", 1),
        4,
        seed=3,
        record_metadata={"weight_mode": "inherited_supernet"},
    )
    assert search.run(1).architecture.search_space_id == "darts"
    assert all(
        row["weight_mode"] == "inherited_supernet" for row in read_jsonl(tmp_path / "search.jsonl")
    )


def test_evolution_search_resolves_score_ties_by_architecture_id(tmp_path):
    load_builtin_spaces()
    path = tmp_path / "tied-search.jsonl"
    search = EvolutionSearch(
        SPACES.create("darts"),
        lambda _architecture: 1.0,
        JsonlWriter(path, 1),
        8,
        seed=37,
    )

    best = search.run(0)
    candidate_ids = {
        row["architecture_id"]
        for row in read_jsonl(path)
        if row["record_kind"] == "candidate"
    }

    assert best.architecture.architecture_id == min(candidate_ids)


def test_rank_aggregated_evolution_records_components_and_resumes(tmp_path):
    from zcp_test.proxies.az_nas import log_rank_aggregate

    load_builtin_spaces()
    space = SPACES.create("darts")

    def evaluator(architecture):
        value = int(architecture.architecture_id[:8], 16)
        return {
            "expressivity": float(value % 101 + 1),
            "trainability": float(value % 97 + 1),
            "complexity": float(value % 89 + 1),
        }

    def aggregator(rows):
        return log_rank_aggregate(
            rows, ("expressivity", "trainability", "complexity")
        )

    identity = {"proxy_id": "az_nas_fixture", "aggregator": "az_nas_log_rank"}
    state_path = tmp_path / "rank-state.json"
    first = EvolutionSearch(
        space,
        evaluator,
        JsonlWriter(tmp_path / "rank-first.jsonl", 1),
        4,
        seed=31,
        state_path=state_path,
        state_identity=identity,
        component_aggregator=aggregator,
    )
    first.run(1)
    rows = list(read_jsonl(tmp_path / "rank-first.jsonl"))
    candidate_rows = [row for row in rows if row["record_kind"] == "candidate"]
    summary_rows = [row for row in rows if row["record_kind"] == "generation_summary"]

    assert candidate_rows
    assert all(set(row["components"]) == {"expressivity", "trainability", "complexity"} for row in candidate_rows)
    assert all(math.isfinite(row["score"]) for row in candidate_rows)
    assert all(row["cohort_size"] == 4 for row in summary_rows)
    assert all(len(row["cohort_digest"]) == 64 for row in summary_rows)
    assert all(row["tie_method"] == "average" for row in summary_rows)
    assert all(row["rerank_scope"] == "current_generation_population" for row in summary_rows)

    resumed = EvolutionSearch(
        space,
        evaluator,
        JsonlWriter(tmp_path / "rank-resumed.jsonl", 1),
        4,
        seed=31,
        state_path=tmp_path / "rank-state-resumed.json",
        state_identity=identity,
        resume_state=load_search_state(state_path),
        component_aggregator=aggregator,
    )
    best = resumed.run(2)

    assert best.components is not None
    assert math.isfinite(best.score)
    assert list(read_jsonl(tmp_path / "rank-resumed.jsonl"))[: len(rows)] == rows


def test_evolution_search_resume_restores_population_rng_cache_and_history(tmp_path):
    load_builtin_spaces()
    space = SPACES.create("darts")

    def evaluator(architecture):
        return float(int(architecture.architecture_id[:8], 16))

    identity = {
        "search_space_id": "darts",
        "proxy_id": "fixture",
        "input_fingerprint": "batch-a",
        "seed": 19,
    }
    first_log = tmp_path / "first.jsonl"
    state_path = tmp_path / "search-state.json"
    EvolutionSearch(
        space,
        evaluator,
        JsonlWriter(first_log, 1),
        6,
        seed=19,
        state_path=state_path,
        state_identity=identity,
    ).run(1)

    resumed_log = tmp_path / "resumed.jsonl"
    resumed_log.touch()
    resumed = EvolutionSearch(
        space,
        evaluator,
        JsonlWriter(resumed_log, 1),
        6,
        seed=19,
        state_path=tmp_path / "resumed-state.json",
        resume_state=load_search_state(state_path),
        state_identity=identity,
    ).run(3)
    uninterrupted_log = tmp_path / "uninterrupted.jsonl"
    uninterrupted = EvolutionSearch(
        space,
        evaluator,
        JsonlWriter(uninterrupted_log, 1),
        6,
        seed=19,
    ).run(3)

    def scientific_trace(path):
        return [
            (
                row["record_kind"],
                row["generation"],
                row.get("architecture_id"),
                row.get("parents"),
                row.get("operation"),
                row.get("score"),
                row["cumulative_evaluations"],
                row["cumulative_cache_hits"],
            )
            for row in read_jsonl(path)
        ]

    assert resumed.architecture.architecture_id == uninterrupted.architecture.architecture_id
    assert scientific_trace(resumed_log) == scientific_trace(uninterrupted_log)
    summaries = [
        row["generation"]
        for row in read_jsonl(resumed_log)
        if row["record_kind"] == "generation_summary"
    ]
    assert summaries == [0, 1, 2, 3]
    assert not list(tmp_path.glob(".*search-state*.tmp"))

    mismatched = dict(identity, input_fingerprint="batch-b")
    with pytest.raises(ValueError, match="identity does not match"):
        EvolutionSearch(
            space,
            evaluator,
            JsonlWriter(tmp_path / "bad.jsonl", 1),
            6,
            seed=19,
            resume_state=load_search_state(state_path),
            state_identity=mismatched,
        )


def test_evolution_search_resumes_partial_initial_population(tmp_path):
    load_builtin_spaces()
    space = SPACES.create("darts")
    state_path = tmp_path / "partial-state.json"
    calls = 0

    def interrupted_evaluator(architecture):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise RuntimeError("simulated interruption")
        return float(int(architecture.architecture_id[:8], 16))

    with pytest.raises(RuntimeError, match="simulated interruption"):
        EvolutionSearch(
            space,
            interrupted_evaluator,
            JsonlWriter(tmp_path / "partial-first.jsonl", 1),
            6,
            seed=23,
            state_path=state_path,
            state_identity={"protocol": "partial-fixture"},
            initial_checkpoint_interval=1,
        ).run(0)

    partial = load_search_state(state_path)
    assert partial["completed_generation"] == -1
    assert len(partial["population"]) == 3
    assert partial["history"] == []

    def evaluator(architecture):
        return float(int(architecture.architecture_id[:8], 16))

    resumed_log = tmp_path / "partial-resumed.jsonl"
    resumed = EvolutionSearch(
        space,
        evaluator,
        JsonlWriter(resumed_log, 1),
        6,
        seed=23,
        state_path=tmp_path / "partial-complete-state.json",
        state_identity={"protocol": "partial-fixture"},
        resume_state=partial,
        initial_checkpoint_interval=1,
    ).run(1)
    uninterrupted_log = tmp_path / "partial-uninterrupted.jsonl"
    uninterrupted = EvolutionSearch(
        space,
        evaluator,
        JsonlWriter(uninterrupted_log, 1),
        6,
        seed=23,
    ).run(1)

    def trace(path):
        return [
            (
                row["record_kind"],
                row["generation"],
                row.get("architecture_id"),
                row.get("score"),
                row["cumulative_evaluations"],
                row["cumulative_cache_hits"],
            )
            for row in read_jsonl(path)
        ]

    assert resumed.architecture.architecture_id == uninterrupted.architecture.architecture_id
    assert trace(resumed_log) == trace(uninterrupted_log)


def test_training_artifacts(tmp_path):
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 2))
    data = torch.utils.data.TensorDataset(torch.randn(8, 3, 8, 8), torch.randint(2, (8,)))
    loader = torch.utils.data.DataLoader(data, batch_size=4)
    train_model(
        model, loader, loader, TrainingConfig(1, "sgd", 0.01, 0), tmp_path, torch.device("cpu")
    )
    assert (tmp_path / "checkpoints" / "last.pt").exists()


def test_autoformer_optimizer_exempts_bias_norm_and_tokens_from_weight_decay():
    from zcp_test.models.autoformer import AZNAS_SCRATCH_PROFILE, StaticAutoFormer

    model = StaticAutoFormer(
        profile=AZNAS_SCRATCH_PROFILE,
        image_size=32,
        patch_size=16,
        num_classes=3,
        embed_dim=24,
        depth=2,
        num_heads=[2, 2],
        mlp_ratio=[2.0, 2.0],
        super_depth=14,
    )
    groups = _optimizer_parameter_groups(model, 0.05, True)
    decay_ids = {id(parameter) for parameter in groups[0]["params"]}
    no_decay_ids = {id(parameter) for parameter in groups[1]["params"]}
    names = {name: id(parameter) for name, parameter in model.named_parameters()}

    assert groups[0]["weight_decay"] == 0.05
    assert groups[1]["weight_decay"] == 0.0
    assert names["class_token"] in no_decay_ids
    assert names["position_embedding"] in no_decay_ids
    assert names["blocks.0.attention.relative_key.vertical"] in no_decay_ids
    assert names["blocks.0.attention.relative_value.horizontal"] in no_decay_ids
    assert names["norm.weight"] in no_decay_ids
    assert names["head.bias"] in no_decay_ids
    assert names["head.weight"] in decay_ids
    assert not decay_ids & no_decay_ids


def test_proxyless_optimizer_exempts_only_normalization_from_weight_decay():
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3, bias=True),
        torch.nn.BatchNorm2d(4),
        torch.nn.Flatten(),
        torch.nn.Linear(16, 2, bias=True),
    )
    groups = _optimizer_parameter_groups(model, 4e-5, False, True)
    decay_ids = {id(parameter) for parameter in groups[0]["params"]}
    no_decay_ids = {id(parameter) for parameter in groups[1]["params"]}

    assert id(model[1].weight) in no_decay_ids
    assert id(model[1].bias) in no_decay_ids
    assert id(model[0].bias) in decay_ids
    assert id(model[3].bias) in decay_ids


def test_cosine_step_scheduler_matches_proxyless_batch_schedule(tmp_path):
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 4 * 4, 2))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.randn(4, 3, 4, 4), torch.randint(2, (4,))),
        batch_size=2,
        shuffle=False,
    )
    config = TrainingConfig(
        1,
        "sgd",
        0.1,
        0.0,
        scheduler="cosine_step",
        amp=False,
        nesterov=False,
    )

    train_model(model, loader, loader, config, tmp_path, torch.device("cpu"))

    record = next(read_jsonl(tmp_path / "training.jsonl"))
    checkpoint = load_checkpoint(tmp_path / "checkpoints" / "last.pt", trusted=True)
    assert record["learning_rate"] == pytest.approx(0.1)
    assert record["next_learning_rate"] == pytest.approx(0.0)
    assert checkpoint["scheduler"]["last_epoch"] == 2


def test_training_rejects_unknown_memory_format(tmp_path):
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 4 * 4, 2))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.randn(2, 3, 4, 4), torch.randint(2, (2,))),
        batch_size=2,
    )
    config = TrainingConfig(
        1,
        "sgd",
        0.1,
        0.0,
        amp=False,
        nesterov=False,
        memory_format="invalid",
    )

    with pytest.raises(ValueError, match="memory_format"):
        train_model(model, loader, loader, config, tmp_path, torch.device("cpu"))


def test_training_channels_last_smoke(tmp_path):
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3, padding=1),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(4, 2),
    )
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.randn(2, 3, 8, 8), torch.randint(2, (2,))),
        batch_size=2,
    )
    config = TrainingConfig(
        1,
        "sgd",
        0.1,
        0.0,
        amp=False,
        nesterov=False,
        memory_format="channels_last",
    )

    result = train_model(model, loader, loader, config, tmp_path, torch.device("cpu"))

    assert result["last_epoch"] == 0
    assert model[0].weight.is_contiguous(memory_format=torch.channels_last)


def test_cosine_warmup_step_scheduler_matches_aznas_sample_schedule(tmp_path):
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 4 * 4, 2))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.randn(4, 3, 4, 4), torch.randint(2, (4,))),
        batch_size=2,
        shuffle=False,
    )
    config = TrainingConfig(
        2,
        "sgd",
        0.1,
        0.0,
        scheduler="cosine_warmup_step",
        warmup_epochs=1,
        amp=False,
        nesterov=False,
    )

    train_model(model, loader, loader, config, tmp_path, torch.device("cpu"))

    records = list(read_jsonl(tmp_path / "training.jsonl"))
    checkpoint = load_checkpoint(tmp_path / "checkpoints" / "last.pt", trusted=True)
    assert records[0]["learning_rate"] == pytest.approx(0.05)
    assert records[0]["next_learning_rate"] == pytest.approx(0.05)
    assert records[1]["next_learning_rate"] == pytest.approx(0.0)
    assert checkpoint["scheduler"]["last_epoch"] == 4


def test_training_progress_callback_separates_batch_and_epoch_events(tmp_path):
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 4 * 4, 2))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.randn(4, 3, 4, 4), torch.randint(2, (4,))),
        batch_size=2,
        shuffle=False,
    )
    events = []

    train_model(
        model,
        loader,
        loader,
        TrainingConfig(1, "sgd", 0.1, 0.0, amp=False, nesterov=False),
        tmp_path,
        torch.device("cpu"),
        progress_callback=lambda kind, fields: events.append({"kind": kind, **fields}),
        progress_interval_seconds=3600,
    )

    batch_events = [event for event in events if event["kind"] == "training_batch_progress"]
    assert [event["split"] for event in batch_events] == ["train", "valid"]
    assert all(event["batch"] == event["batch_count"] == 2 for event in batch_events)
    assert all(event["eta_seconds"] == pytest.approx(0.0) for event in batch_events)
    assert events[-1]["kind"] == "training_epoch_completed"
    assert events[-1]["epoch"] == 0
    assert len(list(read_jsonl(tmp_path / "training.jsonl"))) == 1


def test_autoformer_cosine_schedule_matches_aznas_warmup_and_floor():
    config = TrainingConfig(
        500,
        "adamw",
        5e-4,
        0.05,
        warmup_epochs=20,
        warmup_learning_rate=1e-6,
        minimum_learning_rate=1e-5,
    )

    assert _cosine_learning_rate(config, 0, 500) == pytest.approx(1e-6)
    assert _cosine_learning_rate(config, 20, 500) == pytest.approx(5e-4)
    assert _cosine_learning_rate(config, 500, 500) == pytest.approx(1e-5)
    assert _cosine_learning_rate(config, 499, 500) > 1e-5


def test_validation_uses_plain_cross_entropy_when_training_uses_smoothing(tmp_path):
    torch.manual_seed(7)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 4 * 4, 3))
    inputs = torch.randn(4, 3, 4, 4)
    labels = torch.tensor([0, 1, 2, 1])
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(inputs, labels), batch_size=2, shuffle=False
    )
    config = TrainingConfig(
        1,
        "sgd",
        0.01,
        0.0,
        scheduler="none",
        label_smoothing=0.2,
        validation_label_smoothing=0.0,
        nesterov=False,
    )
    train_model(model, loader, loader, config, tmp_path, torch.device("cpu"))
    record = next(read_jsonl(tmp_path / "training.jsonl"))
    checkpoint = load_checkpoint(tmp_path / "checkpoints" / "last.pt", trusted=True)
    model.load_state_dict(checkpoint["model"])
    with torch.no_grad():
        expected = torch.nn.functional.cross_entropy(model(inputs), labels).item()
        smoothed = torch.nn.functional.cross_entropy(
            model(inputs), labels, label_smoothing=0.2
        ).item()

    assert record["valid_loss"] == pytest.approx(expected)
    assert record["valid_loss"] != pytest.approx(smoothed)


def test_training_keeps_label_smoothing_without_mixup(tmp_path, monkeypatch):
    observed: list[float] = []
    original = torch.nn.functional.cross_entropy

    def recording_cross_entropy(inputs, targets, *args, **kwargs):
        if torch.is_grad_enabled():
            observed.append(float(kwargs.get("label_smoothing", 0.0)))
        return original(inputs, targets, *args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "cross_entropy", recording_cross_entropy)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 4 * 4, 3))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.randn(4, 3, 4, 4), torch.tensor([0, 1, 2, 1])),
        batch_size=2,
        shuffle=False,
    )
    config = TrainingConfig(
        1,
        "sgd",
        0.01,
        0.0,
        scheduler="none",
        label_smoothing=0.2,
        validation_label_smoothing=0.0,
        nesterov=False,
    )

    train_model(model, loader, loader, config, tmp_path, torch.device("cpu"))

    assert observed
    assert set(observed) == {0.2}


def test_checkpoint_config_normalization_accepts_missing_default_fields():
    current = TrainingConfig(1, "sgd", 0.1, 0.0)
    legacy = dict(current.__dict__)
    for key in (
        "warmup_learning_rate",
        "minimum_learning_rate",
        "validation_label_smoothing",
        "exclude_bias_norm_from_weight_decay",
    ):
        legacy.pop(key)

    assert _normalized_checkpoint_config(legacy) == current.__dict__

    legacy["minimum_learning_rate"] = 1e-5
    assert _normalized_checkpoint_config(legacy) != current.__dict__


@pytest.mark.parametrize(
    ("training_mode", "acceptance_protocol", "expected_fraction"),
    [
        ("real_data_preflight", None, 1.0),
        ("acceptance_smoke", "one_percent_data_protocol", 0.01),
        ("acceptance_smoke", "one_percent_epochs_protocol", 1.0),
    ],
)
def test_checkpoint_identity_normalizes_only_known_legacy_fraction_protocols(
    training_mode, acceptance_protocol, expected_fraction
):
    identity = {
        "training_mode": training_mode,
        "acceptance_protocol": acceptance_protocol,
    }

    assert _normalized_checkpoint_identity(identity)["data_fraction"] == expected_fraction

    unknown = {"training_mode": "formal", "acceptance_protocol": None}
    assert "data_fraction" not in _normalized_checkpoint_identity(unknown)


def test_completed_checkpoint_does_not_require_rng_restore(tmp_path, monkeypatch):
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 4 * 4, 2))
    data = torch.utils.data.TensorDataset(torch.randn(4, 3, 4, 4), torch.randint(2, (4,)))
    loader = torch.utils.data.DataLoader(data, batch_size=2)
    config = TrainingConfig(1, "sgd", 0.01, 0.0, scheduler="none", nesterov=False)
    source = tmp_path / "source"
    train_model(model, loader, loader, config, source, torch.device("cpu"))
    checkpoint_path = source / "checkpoints" / "last.pt"
    checkpoint = load_checkpoint(checkpoint_path, trusted=True)
    checkpoint.pop("rng_by_rank", None)
    atomic_torch_save(checkpoint, checkpoint_path)
    monkeypatch.setattr(
        "zcp_test.training.trainer._restore_checkpoint_rng",
        lambda *args, **kwargs: pytest.fail("RNG restore should not run after the final epoch"),
    )

    result = train_model(
        model,
        loader,
        loader,
        config,
        tmp_path / "resumed",
        torch.device("cpu"),
        checkpoint_path,
        resume_trusted=True,
    )

    assert result["last_epoch"] == 0
    assert result["resumed_training_rows"] == 1


def test_training_scheduler_dispatch_and_resume_identity(tmp_path):
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 4 * 4, 2))
    data = torch.utils.data.TensorDataset(torch.randn(4, 3, 4, 4), torch.randint(2, (4,)))
    loader = torch.utils.data.DataLoader(data, batch_size=2)
    config = TrainingConfig(
        1,
        "sgd",
        0.01,
        0,
        scheduler="step",
        scheduler_gamma=0.5,
        nesterov=False,
    )
    identity = {
        "architecture_id": "a",
        "protocol": "test",
        "acceptance_protocol": "one_percent_data_protocol",
        "data_fraction": 0.01,
    }
    train_model(
        model,
        loader,
        loader,
        config,
        tmp_path,
        torch.device("cpu"),
        run_identity=identity,
    )
    record = next(read_jsonl(tmp_path / "training.jsonl"))
    assert record["next_learning_rate"] == pytest.approx(0.005)
    assert record["train_duration_seconds"] > 0
    assert record["valid_duration_seconds"] > 0
    assert record["train_samples_per_second"] > 0
    assert record["valid_samples_per_second"] > 0
    assert record["peak_memory_mb"] is None
    assert record["peak_reserved_memory_mb"] is None
    checkpoint = load_checkpoint(tmp_path / "checkpoints" / "last.pt", trusted=True)
    assert checkpoint["run_identity"] == identity
    resumed = tmp_path / "resumed"
    result = train_model(
        torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 4 * 4, 2)),
        loader,
        loader,
        config,
        resumed,
        torch.device("cpu"),
        tmp_path / "checkpoints" / "last.pt",
        resume_trusted=True,
        run_identity=identity,
    )
    assert result["resumed_training_rows"] == 1
    assert [row["epoch"] for row in read_jsonl(resumed / "training.jsonl")] == [0]
    with pytest.raises(ValueError, match="identity"):
        train_model(
            torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 4 * 4, 2)),
            loader,
            loader,
            config,
            tmp_path / "resume",
            torch.device("cpu"),
            tmp_path / "checkpoints" / "last.pt",
            resume_trusted=True,
            run_identity={**identity, "data_fraction": 0.0100001},
        )


def test_short_training_prefix_preserves_formal_cosine_schedule(tmp_path):
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 4 * 4, 2))
    data = torch.utils.data.TensorDataset(torch.randn(4, 3, 4, 4), torch.randint(2, (4,)))
    loader = torch.utils.data.DataLoader(data, batch_size=2)
    train_model(
        model,
        loader,
        loader,
        TrainingConfig(
            1,
            "sgd",
            0.025,
            0,
            nesterov=False,
            drop_path_prob=0.2,
            schedule_epochs=600,
        ),
        tmp_path,
        torch.device("cpu"),
    )
    record = next(read_jsonl(tmp_path / "training.jsonl"))
    assert record["learning_rate"] == pytest.approx(0.025)
    assert record["next_learning_rate"] == pytest.approx(
        0.025 * 0.5 * (1 + math.cos(math.pi / 600))
    )
    assert record["drop_path_prob"] == 0.0


def test_restore_training_log_handles_missing_source_duplicates_and_non_primary(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text(
        '{"epoch": 0, "valid_accuracy": 1.0}\n'
        '{"epoch": 1, "valid_accuracy": 2.0}\n'
        '{"epoch": 2, "valid_accuracy": 3.0}\n',
        encoding="utf-8",
    )
    destination = JsonlWriter(tmp_path / "destination.jsonl", fsync_every=1)
    destination.append({"epoch": 1, "valid_accuracy": 2.0})
    assert _restore_training_log(destination, source, checkpoint_epoch=1) == 1
    assert [row["epoch"] for row in read_jsonl(destination.path)] == [1, 0]
    assert _restore_training_log(destination, source, checkpoint_epoch=1) == 0
    assert _restore_training_log(destination, tmp_path / "missing.jsonl", 1) == 0
    assert _restore_training_log(None, source, 1) == 0

    portable = JsonlWriter(tmp_path / "portable.jsonl", fsync_every=1)
    history = [{"epoch": 0}, {"epoch": 1}, {"epoch": 2}]
    assert _restore_training_log(portable, tmp_path / "moved.jsonl", 1, history) == 2
    assert [row["epoch"] for row in read_jsonl(portable.path)] == [0, 1]


def test_atomic_checkpoint_removes_temporary_file_on_failure(monkeypatch, tmp_path):
    def fail_save(payload, path):
        del payload
        Path(path).write_bytes(b"partial")
        raise InterruptedError("injected")

    monkeypatch.setattr(torch, "save", fail_save)
    target = tmp_path / "checkpoint.pt"
    with pytest.raises(InterruptedError, match="injected"):
        atomic_torch_save({"epoch": 1}, target)
    assert not target.exists()
    assert not target.with_suffix(".pt.tmp").exists()


def test_training_sets_epoch_on_stateful_samplers(tmp_path):
    class EpochSampler(torch.utils.data.Sampler[int]):
        def __init__(self, dataset):
            self.dataset = dataset
            self.epochs = []

        def __iter__(self):
            return iter(range(len(self.dataset)))

        def __len__(self):
            return len(self.dataset)

        def set_epoch(self, epoch):
            self.epochs.append(epoch)

    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 4 * 4, 2))
    data = torch.utils.data.TensorDataset(torch.randn(4, 3, 4, 4), torch.randint(2, (4,)))
    train_sampler = EpochSampler(data)
    valid_sampler = EpochSampler(data)
    train_loader = torch.utils.data.DataLoader(data, batch_size=2, sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(data, batch_size=2, sampler=valid_sampler)
    train_model(
        model,
        train_loader,
        valid_loader,
        TrainingConfig(2, "sgd", 0.01, 0, nesterov=False),
        tmp_path,
        torch.device("cpu"),
    )
    assert train_sampler.epochs == [0, 1]
    assert valid_sampler.epochs == [0, 1]


def test_training_gradient_accumulation_steps_and_no_sync(tmp_path):
    from contextlib import contextmanager

    class AccumulationModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.network = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 4 * 4, 2))
            self.no_sync_calls = 0

        @contextmanager
        def no_sync(self):
            self.no_sync_calls += 1
            yield

        def forward(self, inputs):
            return self.network(inputs)

    model = AccumulationModel()
    data = torch.utils.data.TensorDataset(torch.randn(8, 3, 4, 4), torch.randint(2, (8,)))
    loader = torch.utils.data.DataLoader(data, batch_size=2)
    train_model(
        model,
        loader,
        loader,
        TrainingConfig(
            1,
            "sgd",
            0.01,
            0,
            nesterov=False,
            gradient_accumulation_steps=2,
        ),
        tmp_path,
        torch.device("cpu"),
    )
    record = next(read_jsonl(tmp_path / "training.jsonl"))
    assert record["optimizer_steps"] == 2
    assert model.no_sync_calls == 2


def test_non_primary_distributed_rank_does_not_write_artifacts(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda output, state: output.__setitem__(slice(None), [state, state]),
    )
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 4 * 4, 2))
    data = torch.utils.data.TensorDataset(torch.randn(4, 3, 4, 4), torch.randint(2, (4,)))
    loader = torch.utils.data.DataLoader(data, batch_size=2)
    train_model(
        model,
        loader,
        loader,
        TrainingConfig(1, "sgd", 0.01, 0, nesterov=False),
        tmp_path,
        torch.device("cpu"),
    )
    assert not (tmp_path / "training.jsonl").exists()
    assert not (tmp_path / "checkpoints").exists()


def test_distributed_checkpoint_collects_and_restores_rank_local_rng(monkeypatch):
    rank_one_state = {"rank": 1}
    monkeypatch.setattr("zcp_test.training.trainer.rng_state", lambda: rank_one_state)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    def gather(states, local_state):
        states[:] = [{"rank": 0}, local_state]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    checkpoint_rng = _collect_checkpoint_rng(True, 1)
    assert checkpoint_rng == {
        "rng": {"rank": 0},
        "rng_by_rank": [{"rank": 0}, rank_one_state],
    }

    restored = []
    monkeypatch.setattr("zcp_test.training.trainer.restore_rng", restored.append)
    _restore_checkpoint_rng(checkpoint_rng, True, 1)
    assert restored == [rank_one_state]


def test_distributed_resume_rejects_legacy_or_wrong_world_size_rng(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    with pytest.raises(ValueError, match="rank-local RNG"):
        _restore_checkpoint_rng({"rng": {"rank": 0}}, True, 1)
    with pytest.raises(ValueError, match="world size"):
        _restore_checkpoint_rng({"rng": {"rank": 0}, "rng_by_rank": [{"rank": 0}]}, True, 1)
