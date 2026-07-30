from __future__ import annotations

from typing import Any

from zcp_test.types import Architecture


def num_classes(dataset: str) -> int:
    return {"cifar10": 10, "cifar10-valid": 10, "cifar100": 100, "ImageNet16-120": 120, "imagenet1k": 1000, "flowers": 102, "chaoyang": 4}.get(dataset, 10)


def nb201_model(architecture: Architecture, dataset: str) -> Any:
    from zcp_test.models.nb201 import build_nb201

    return build_nb201(str(architecture.spec["architecture"]), num_classes(dataset))


def nats_size_model(architecture: Architecture, dataset: str) -> Any:
    from zcp_test.models.nb201 import build_nats_sss

    value = architecture.spec.get("architecture")
    if isinstance(value, str):
        channels = [int(channel) for channel in value.split(":")]
    else:
        channels = [int(architecture.spec[f"stage_{index}"]) for index in range(5)]
    return build_nats_sss(channels, num_classes(dataset))


def nb101_model(architecture: Architecture, dataset: str) -> Any:
    from zcp_test.models.nb101 import Network

    return Network(architecture.spec, num_classes(dataset))


def registered_space_model(architecture: Architecture, dataset: str) -> Any:
    from zcp_test.spaces import SPACES, load_builtin_spaces

    load_builtin_spaces()
    if architecture.search_space_id in SPACES.names():
        space = SPACES.create(architecture.search_space_id)
        try:
            canonical = space.canonicalize(architecture.spec)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Cannot build {architecture.search_space_id} architecture "
                f"{architecture.architecture_id}: invalid specification"
            ) from error
        return space.build_model(canonical, num_classes(dataset))
    raise NotImplementedError(f"No model factory for {architecture.search_space_id}")


def vitbench_autoformer_model(architecture: Architecture, dataset: str) -> Any:
    from zcp_test.models.autoformer import StaticAutoFormer, VITBENCH_AUTOPROX_PROFILE
    from zcp_test.spaces import SPACES, load_builtin_spaces

    load_builtin_spaces()
    space = SPACES.create("autoformer")
    canonical = space.canonicalize(architecture.spec)
    return StaticAutoFormer(
        profile=VITBENCH_AUTOPROX_PROFILE,
        num_classes=num_classes(dataset),
        embed_dim=int(canonical.spec["hidden_dim"]),
        depth=int(canonical.spec["depth"]),
        num_heads=canonical.spec["num_heads"],
        mlp_ratio=canonical.spec["mlp_ratio"],
    )


def transnas_task_model(architecture: Architecture, dataset: str) -> Any:
    from zcp_test.models.transnas import TransNasTaskModel

    return TransNasTaskModel(str(architecture.spec["architecture"]), dataset)


def model_builder(architecture: Architecture, dataset: str) -> Any:
    if architecture.search_space_id == "nb201_topology":
        return nb201_model(architecture, dataset)
    if architecture.search_space_id == "nb101_dag":
        return nb101_model(architecture, dataset)
    if architecture.search_space_id == "nats_size":
        return nats_size_model(architecture, dataset)
    if architecture.search_space_id in {"transnas_micro", "transnas_macro"}:
        return transnas_task_model(architecture, dataset)
    if architecture.search_space_id == "autoformer":
        return vitbench_autoformer_model(architecture, dataset)
    return registered_space_model(architecture, dataset)
