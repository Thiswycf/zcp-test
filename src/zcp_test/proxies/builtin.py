from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

from zcp_test.proxies import PROXIES
from zcp_test.proxies.base import ZeroCostProxy
from zcp_test.proxies.official import official_proxy_factories
from zcp_test.proxies.vision_transformer import dss_indicator
from zcp_test.types import ProxyCapability, ProxyContext, ScoreDirection


class FunctionProxy(ZeroCostProxy):
    def __init__(self, capability: ProxyCapability, function: Callable[..., float | dict[str, float]]) -> None:
        self.capability = capability
        self._function = function

    def compute(self, model: Any, inputs: Any, labels: Any | None = None, loss_fn: Any | None = None) -> float | dict[str, float]:
        return self._function(model, inputs, labels, loss_fn)


class ContextFunctionProxy(ZeroCostProxy):
    def __init__(self, capability: ProxyCapability, function: Callable[..., float]) -> None:
        self.capability = capability
        self._function = function

    def compute(
        self,
        model: Any,
        inputs: Any,
        labels: Any | None = None,
        loss_fn: Any | None = None,
    ) -> float:
        raise RuntimeError(f"{self.capability.proxy_id} requires ProxyContext")

    def compute_context(self, model: Any, context: ProxyContext) -> float:
        return self._function(model, context)


def _er_context(_model: Any, context: ProxyContext) -> float:
    from zcp_test.proxies.edge_rank import EdgeActivationBatch, compute_er_score

    batch = context.edge_activations
    if not isinstance(batch, EdgeActivationBatch):
        raise NotImplementedError("ER requires a semantic EdgeActivationBatch provider")
    return compute_er_score(edge.activation for edge in batch)


def _ter_context(_model: Any, context: ProxyContext) -> float:
    from zcp_test.proxies.edge_rank import EdgeActivationBatch, compute_ter_score

    batch = context.edge_activations
    if not isinstance(batch, EdgeActivationBatch):
        raise NotImplementedError("TER requires a semantic EdgeActivationBatch provider")
    return compute_ter_score(batch)


def _params(model: Any, *_: Any) -> float:
    return float(sum(parameter.numel() for parameter in model.parameters()))


def _flops(model: Any, inputs: Any, *_: Any) -> float:
    from thop import profile

    value, _ = profile(model, inputs=(inputs,), verbose=False)
    return float(value)


def _activation_matrix(model: Any, inputs: Any) -> Any:
    import torch

    activations = []
    handles = []

    def hook(_module: Any, _input: Any, output: Any) -> None:
        if isinstance(output, torch.Tensor) and output.ndim >= 2:
            activations.append(output.detach().flatten(1).float())

    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            handles.append(module.register_forward_hook(hook))
    model(inputs)
    for handle in handles:
        handle.remove()
    if not activations:
        raise ValueError("No eligible activations")
    width = min(value.shape[1] for value in activations)
    return torch.cat([value[:, :width] for value in activations], dim=1)


def _meco(model: Any, inputs: Any, *_: Any) -> float:
    import torch

    layer_scores: list[Any] = []
    handles: list[Any] = []

    def hook(_: Any, __: Any, output: Any) -> None:
        if not isinstance(output, torch.Tensor) or output.ndim < 3 or output.shape[0] == 0:
            return
        feature_map = output[0].detach().float().reshape(output.shape[1], -1)
        if feature_map.shape[0] < 2 or feature_map.shape[1] < 2:
            return
        correlation = torch.corrcoef(feature_map)
        correlation = torch.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
        eigenvalues = torch.linalg.eigvals(correlation).real
        score = eigenvalues.min()
        if torch.isfinite(score):
            layer_scores.append(score)

    try:
        handles = [module.register_forward_hook(hook) for module in model.modules()]
        model(inputs)
    finally:
        for handle in handles:
            handle.remove()
    if not layer_scores:
        raise ValueError("MeCo found no eligible feature maps")
    return float(torch.stack(layer_scores).sum().item())


def _meco_opt(model: Any, inputs: Any, *_: Any) -> float:
    import random

    import torch

    layer_scores: list[Any] = []
    handles: list[Any] = []

    def hook(_: Any, __: Any, output: Any) -> None:
        if not isinstance(output, torch.Tensor) or output.ndim < 3 or output.shape[0] == 0:
            return
        feature_map = output[0].detach().float().reshape(output.shape[1], -1)
        channel_count = feature_map.shape[0]
        if channel_count < 8 or feature_map.shape[1] < 2:
            return
        indices = random.sample(range(channel_count), 8)
        correlation = torch.corrcoef(feature_map[indices])
        correlation = torch.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
        eigenvalues = torch.linalg.eigvals(correlation).real
        score = eigenvalues.min() * channel_count / 8
        if torch.isfinite(score):
            layer_scores.append(score)

    try:
        handles = [module.register_forward_hook(hook) for module in model.modules()]
        model(inputs)
    finally:
        for handle in handles:
            handle.remove()
    if not layer_scores:
        raise ValueError("MeCo-opt found no eligible feature maps with at least eight channels")
    return float(torch.stack(layer_scores).sum().item())


def _vkdnw(model: Any, inputs: Any, *_: Any) -> float:
    import torch
    from torch.func import functional_call, jacrev, vmap

    def initialize(module: Any) -> None:
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            torch.nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)) and module.affine:
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def logits(output: Any) -> Any:
        if isinstance(output, (tuple, list)):
            tensors = [value for value in output if isinstance(value, torch.Tensor)]
            if not tensors:
                raise ValueError("VKDNW model output contains no tensor logits")
            output = tensors[-1]
        if not isinstance(output, torch.Tensor) or output.ndim != 2:
            raise ValueError("VKDNW requires two-dimensional classification logits")
        return output

    model.eval()
    model.apply(initialize)
    named_parameters = [(name, parameter) for name, parameter in model.named_parameters() if parameter.numel()]
    selected = dict(named_parameters[:128])
    if not selected:
        raise ValueError("VKDNW requires at least one trainable parameter tensor")
    selected_values = {
        name: parameter.detach().flatten()[:1] for name, parameter in selected.items()
    }
    base_parameters = {name: parameter.detach() for name, parameter in named_parameters}
    buffers = {name: buffer.detach() for name, buffer in model.named_buffers()}

    def predict(selected_parameters: dict[str, Any], sample: Any) -> Any:
        parameters = dict(base_parameters)
        for name, value in selected_parameters.items():
            base = base_parameters[name]
            parameters[name] = torch.cat((value, base.flatten()[1:])).reshape_as(base)
        return logits(
            functional_call(model, (parameters, buffers), (sample.unsqueeze(0),))
        ).squeeze(0)

    jacobian = vmap(jacrev(predict), in_dims=(None, 0))(selected_values, inputs)
    jacobian_matrix = torch.cat(
        [value.flatten(start_dim=2) for value in jacobian.values()], dim=2
    )
    prediction = logits(model(inputs))
    alpha = prediction.new_tensor(0.05)
    probability = torch.softmax(prediction, dim=1) * (1 - alpha) + alpha / prediction.shape[1]
    covariance = torch.diag_embed(probability) - probability.unsqueeze(2) * probability.unsqueeze(1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    covariance_sqrt = eigenvectors @ torch.diag_embed(eigenvalues.clamp_min(0).sqrt()) @ eigenvectors.transpose(1, 2)
    weighted_jacobian = covariance_sqrt @ jacobian_matrix
    fisher = torch.mean(weighted_jacobian.transpose(1, 2) @ weighted_jacobian, dim=0)
    spectrum = torch.linalg.svdvals(fisher)
    quantiles = torch.quantile(
        spectrum,
        torch.arange(0.1, 1.0, 0.1, device=spectrum.device, dtype=spectrum.dtype),
    )
    normalized = quantiles / quantiles.sum().clamp_min(1e-10)
    entropy = -(normalized * torch.log(normalized + 1e-10)).sum()
    dimension = float(len(named_parameters))
    return {
        "single": dimension + float(entropy.item()),
        "entropy": float(entropy.item()),
        "dimension": dimension,
    }


def _te_nas(model: Any, inputs: Any, *_: Any) -> float:
    from zcp_test.proxies.te_nas import te_nas_score

    return te_nas_score(model, inputs)


def _az_nas(model: Any, inputs: Any, *_: Any) -> dict[str, float]:
    from zcp_test.models.autoformer import StaticAutoFormer
    from zcp_test.models.plainnet import AZPlainNetMobileNetV2
    from zcp_test.proxies.az_nas import autoformer_components, plainnet_components

    if isinstance(model, StaticAutoFormer):
        return autoformer_components(model, inputs)
    if isinstance(model, AZPlainNetMobileNetV2):
        return plainnet_components(model, inputs)
    raise NotImplementedError(
        "AZ-NAS is defined only for StaticAutoFormer and AZPlainNetMobileNetV2"
    )


def _ac(model: Any, inputs: Any, *_: Any) -> dict[str, float]:
    from zcp_test.proxies.attention_metrics import attention_confidence

    return attention_confidence(model, inputs)


def _hi(model: Any, inputs: Any, *_: Any) -> dict[str, float]:
    from zcp_test.proxies.attention_metrics import head_importance

    return head_importance(model, inputs)


def _hc(model: Any, inputs: Any, *_: Any) -> dict[str, float]:
    from zcp_test.proxies.attention_metrics import head_confidence

    return head_confidence(model, inputs)


def _az_nas_autoformer(model: Any, inputs: Any, *_: Any) -> dict[str, float]:
    from zcp_test.proxies.az_nas import autoformer_components

    return autoformer_components(model, inputs)


def _az_nas_plainnet(model: Any, inputs: Any, *_: Any) -> dict[str, float]:
    from zcp_test.proxies.az_nas import plainnet_components

    return plainnet_components(model, inputs)


def _unsupported(name: str) -> Callable[..., float]:
    def compute(*_: Any) -> float:
        raise NotImplementedError(f"{name} requires its specialized compatibility implementation")

    return compute


_IMPLEMENTATIONS: dict[str, tuple[Callable[..., Any], ProxyCapability]] = {
    "params": (_params, ProxyCapability("params", version="count-v2", model_families=("cnn", "transformer"), requires_data=False, requires_inputs=False, resource_direction=ScoreDirection.MINIMIZE)),
    "flops": (_flops, ProxyCapability("flops", version="thop-v2", model_families=("cnn", "transformer"), dependencies=("thop",), resource_direction=ScoreDirection.MINIMIZE)),
    "meco": (_meco, ProxyCapability("meco", version="hamstermimi-0d830dd-v2", model_families=("cnn",), requires_data=False)),
    "meco_opt": (_meco_opt, ProxyCapability("meco_opt", version="hamstermimi-0d830dd-v2", model_families=("cnn",), requires_data=False)),
    "vkdnw": (_vkdnw, ProxyCapability("vkdnw", version="ondratybl-d2ff276-v2", model_families=("cnn",), requires_data=False, components=("single", "entropy", "dimension"), primary_component="single")),
    "te_nas": (_te_nas, ProxyCapability("te_nas", version="ter-score-a646c5a-rn-minus-ntk-v1", model_families=("cnn",), requires_data=False)),
    "az_nas": (_az_nas, ProxyCapability("az_nas", version="aznas-5e6683-dispatch-v1", model_families=("cnn", "transformer"), components=("expressivity", "progressivity", "trainability", "complexity"), primary_component="expressivity", formal_use="cohort_rank_aggregation")),
    "az_nas_autoformer": (_az_nas_autoformer, ProxyCapability("az_nas_autoformer", version="aznas-5e6683-autoformer-stable-v1", model_families=("transformer",), components=("expressivity", "trainability", "complexity"), primary_component="expressivity")),
    "az_nas_plainnet": (_az_nas_plainnet, ProxyCapability("az_nas_plainnet", version="aznas-5e6683-plainnet-stabilized-v1", model_families=("cnn",), components=("expressivity", "progressivity", "trainability", "complexity"), primary_component="expressivity")),
    "dss": (dss_indicator, ProxyCapability("dss", version="tf-tas-42616bc-code-protocol-port-v2", model_families=("transformer",), requires_data=False, components=("score", "attention_diversity", "mlp_saliency", "auxiliary_saliency"), primary_component="score")),
    "ac": (_ac, ProxyCapability("ac", version="acl23-2d76e01-vit-port-v1", model_families=("transformer",), components=("raw", "normalized"), primary_component="raw", protocol_domain="attention_transformer", formal_use="cross_domain_proxy_port")),
    "hi": (_hi, ProxyCapability("hi", version="acl23-2d76e01-vit-port-v1", model_families=("transformer",), components=("raw", "normalized"), primary_component="raw", protocol_domain="attention_transformer", formal_use="cross_domain_proxy_port")),
    "hc": (_hc, ProxyCapability("hc", version="acl23-2d76e01-vit-port-v1", model_families=("transformer",), components=("raw", "normalized"), primary_component="raw", protocol_domain="attention_transformer", formal_use="cross_domain_proxy_port")),
}

_SPECIALIZED: dict[str, tuple[str, ...]] = {}

_SOURCES = {
    "params": "https://github.com/Thiswycf/zcp-test",
    "flops": "https://github.com/Lyken17/pytorch-OpCounter",
    "gradnorm": "https://github.com/mohsaied/zero-cost-nas/blob/b5059bc42e2275534f584bc21a2d28ab8427cd8e/foresight/pruners/measures/grad_norm.py",
    "synflow": "https://github.com/mohsaied/zero-cost-nas/blob/b5059bc42e2275534f584bc21a2d28ab8427cd8e/foresight/pruners/measures/synflow.py",
    "naswot": "https://github.com/BayesWatch/nas-without-training/blob/b3a82a6642564df115f989ff940ec6b8ef9ca9d3/scores.py",
    "er": "https://github.com/Thiswycf/TER-Score/blob/a646c5a6e0b4633d06a153fe3cdc9b6ca3d9f06f/ZeroShotProxy/compute_ER_score.py",
    "ter": "https://github.com/Thiswycf/TER-Score/blob/a646c5a6e0b4633d06a153fe3cdc9b6ca3d9f06f/ZeroShotProxy/compute_TER_score.py",
    "meco": "https://github.com/HamsterMimi/MeCo/blob/0d830dd2f639f9d1ba3b5831a65df768d70fc93b/zero-cost-nas/foresight/pruners/measures/meco.py",
    "meco_opt": "https://github.com/HamsterMimi/MeCo/blob/0d830dd2f639f9d1ba3b5831a65df768d70fc93b/correlation/NAS_Bench_201.py",
    "jacob_cov": "https://github.com/mohsaied/zero-cost-nas/blob/b5059bc42e2275534f584bc21a2d28ab8427cd8e/foresight/pruners/measures/jacob_cov.py",
    "vkdnw": "https://github.com/ondratybl/VKDNW/blob/d2ff276d37d8ba2e9f8c04beb71499d0bd346146/NB201/ZeroShotProxy/compute_vkdnw_score.py",
    "near": "https://arxiv.org/abs/2408.08776",
    "swap": "https://github.com/pym1024/SWAP/blob/0853fc866051dca2b3b99d068502549de3686bd1/src/metrics/swap.py",
    "zen": "https://github.com/idstcv/ZenNAS/blob/d1d617e0352733d39890fb64ea758f9c85b28c1a/ZeroShotProxy/compute_zen_score.py",
    "zico": "https://github.com/SLDGroup/ZiCo/blob/b0fec65923a90e84501593f675b1e2f422d79e3d/ZeroShotProxy/compute_zico.py",
    "te_nas": "https://github.com/Thiswycf/TER-Score/blob/a646c5a6e0b4633d06a153fe3cdc9b6ca3d9f06f/ZeroShotProxy/compute_te_nas_score.py",
    "az_nas": "https://github.com/cvlab-yonsei/AZ-NAS/tree/5e6683a2cfa5c6d0dc34a1317a842497ba7eae47",
    "az_nas_autoformer": "https://github.com/cvlab-yonsei/AZ-NAS/tree/5e6683a2cfa5c6d0dc34a1317a842497ba7eae47/ImageNet_AutoFormer",
    "az_nas_plainnet": "https://github.com/cvlab-yonsei/AZ-NAS/blob/5e6683a2cfa5c6d0dc34a1317a842497ba7eae47/ImageNet_MBV2/ZeroShotProxy/compute_az_nas_score.py",
    "dss": "https://github.com/decemberzhou/TF_TAS/blob/42616bcf1b6bb643bf968a8342f8aaddc4f53f32/lib/training_free/indicators/dss.py",
    "ac": "https://github.com/aaronserianni/training-free-nas/blob/2d76e01b9586cad7340e8268dadba3056efd070b/BERT_metrics.ipynb",
    "hi": "https://github.com/aaronserianni/training-free-nas/blob/2d76e01b9586cad7340e8268dadba3056efd070b/BERT_metrics.ipynb",
    "hc": "https://github.com/aaronserianni/training-free-nas/blob/2d76e01b9586cad7340e8268dadba3056efd070b/BERT_metrics.ipynb",
}
_SOURCE_COMMITS = {
    "params": "zcp-test-project",
    "flops": "package:thop==0.1.1.post2209072238",
    "meco": "0d830dd2f639f9d1ba3b5831a65df768d70fc93b",
    "meco_opt": "0d830dd2f639f9d1ba3b5831a65df768d70fc93b",
    "gradnorm": "b5059bc42e2275534f584bc21a2d28ab8427cd8e",
    "synflow": "b5059bc42e2275534f584bc21a2d28ab8427cd8e",
    "naswot": "b3a82a6642564df115f989ff940ec6b8ef9ca9d3",
    "jacob_cov": "b5059bc42e2275534f584bc21a2d28ab8427cd8e",
    "near": "4d5d7f1bf005b67b352c078190c6810ca63fbadb",
    "swap": "0853fc866051dca2b3b99d068502549de3686bd1",
    "zen": "d1d617e0352733d39890fb64ea758f9c85b28c1a",
    "zico": "b0fec65923a90e84501593f675b1e2f422d79e3d",
    "te_nas": "a646c5a6e0b4633d06a153fe3cdc9b6ca3d9f06f",
    "az_nas": "5e6683a2cfa5c6d0dc34a1317a842497ba7eae47",
    "az_nas_autoformer": "5e6683a2cfa5c6d0dc34a1317a842497ba7eae47",
    "az_nas_plainnet": "5e6683a2cfa5c6d0dc34a1317a842497ba7eae47",
    "dss": "42616bcf1b6bb643bf968a8342f8aaddc4f53f32",
    "vkdnw": "d2ff276d37d8ba2e9f8c04beb71499d0bd346146",
    "ac": "2d76e01b9586cad7340e8268dadba3056efd070b",
    "hi": "2d76e01b9586cad7340e8268dadba3056efd070b",
    "hc": "2d76e01b9586cad7340e8268dadba3056efd070b",
}
_LICENSES = {
    "params": "MIT",
    "flops": "MIT",
    "meco": "NOASSERTION",
    "meco_opt": "NOASSERTION",
    "vkdnw": "GPL-3.0-only",
    "te_nas": "MIT",
    "az_nas": "GPL-3.0-only",
    "az_nas_autoformer": "GPL-3.0-only",
    "az_nas_plainnet": "GPL-3.0-only",
    "dss": "NOASSERTION",
    "near": "BSD-3-Clause",
    "ac": "Apache-2.0",
    "hi": "Apache-2.0",
    "hc": "Apache-2.0",
}
_PROTOCOL_DOMAINS = {
    "params": "generic_torch_model",
    "flops": "generic_torch_model_with_thop",
    "meco": "relu_cnn_feature_maps",
    "meco_opt": "relu_cnn_feature_maps",
    "vkdnw": "cnn_fisher_spectrum",
    "te_nas": "relu_cnn_rn_minus_ntk_condition",
    "az_nas": "autoformer_or_plainnet_dispatch",
    "az_nas_autoformer": "autoformer_transformer",
    "az_nas_plainnet": "zennas_plainnet_mbv2",
    "dss": "vision_transformer_attention_mlp",
}
for _proxy_name, (_proxy_function, _proxy_capability) in tuple(_IMPLEMENTATIONS.items()):
    if _proxy_name in {"params", "flops"}:
        _fidelity = "structural_measure"
    elif _proxy_name == "te_nas":
        _fidelity = "ter_score_first_party_adaptation"
    elif _proxy_name == "az_nas":
        _fidelity = "paper_formula_space_dispatch"
    elif _proxy_name in {"meco", "meco_opt", "vkdnw", "az_nas_autoformer", "az_nas_plainnet", "dss"}:
        _fidelity = "paper_formula_port_stabilized"
    elif _proxy_name in {"ac", "hi", "hc"}:
        _fidelity = "source_paper_official_port_to_vit"
    else:
        _fidelity = "paper_formula_port_unverified"
    _IMPLEMENTATIONS[_proxy_name] = (
        _proxy_function,
        replace(
            _proxy_capability,
            implementation_fidelity=_fidelity,
            source=_SOURCES.get(_proxy_name),
            source_commit=_SOURCE_COMMITS.get(_proxy_name),
            license=_LICENSES.get(_proxy_name),
            protocol_domain=(
                _PROTOCOL_DOMAINS.get(_proxy_name)
                or _proxy_capability.protocol_domain
            ),
            official_code_available=_proxy_name not in {"params", "flops"},
        ),
    )

for _name, (_function, _capability) in _IMPLEMENTATIONS.items():
    PROXIES.register(_name, lambda function=_function, capability=_capability: FunctionProxy(capability, function))

for _name, _factory in official_proxy_factories().items():
    PROXIES.register(_name, _factory)
for _name, _function, _requires_topology in (
    ("er", _er_context, False),
    ("ter", _ter_context, True),
):
    _capability = ProxyCapability(
        _name,
        version=f"ter-score-a646c5a-{_name}-v1",
        model_families=("cnn",),
        requires_data=False,
        requires_inputs=False,
        requires_edge_activations=True,
        requires_topology=_requires_topology,
        implementation_fidelity="ter_score_first_party_port",
        source=_SOURCES[_name],
        source_commit="a646c5a6e0b4633d06a153fe3cdc9b6ca3d9f06f",
        official_code_available=True,
        protocol_domain="semantic_4d_edge_activations",
        license="MIT",
    )
    PROXIES.register(
        _name,
        lambda function=_function, capability=_capability: ContextFunctionProxy(
            capability, function
        ),
    )
for _name, _families in _SPECIALIZED.items():
    _capability = ProxyCapability(_name, model_families=_families)
    PROXIES.register(_name, lambda name=_name, capability=_capability: FunctionProxy(capability, _unsupported(name)))
