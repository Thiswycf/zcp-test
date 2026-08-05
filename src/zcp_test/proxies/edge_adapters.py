from __future__ import annotations

from typing import Any

from zcp_test.proxies.edge_rank import EdgeActivation, EdgeActivationBatch


def capture_semantic_edge_activations(model: Any, inputs: Any) -> EdgeActivationBatch:
    from zcp_test.models.darts import Cell as DartsCell
    from zcp_test.models.nb101 import Module as Nb101Module
    from zcp_test.models.nb201 import InferCell as Nb201Cell
    from zcp_test.models.transnas import MicroCell as TransNasMicroCell

    captured: list[tuple[object, object, Any]] = []
    handles = []
    nb101_modules = [module for module in model.modules() if isinstance(module, Nb101Module)]
    for cell_index, cell in enumerate(nb101_modules):
        cell._edge_activation_capture = captured
        cell._edge_activation_prefix = f"nb101-cell-{cell_index}"

    def register(module: Any, source: object, target: object) -> None:
        def hook(_module: Any, _module_inputs: Any, output: Any) -> None:
            captured.append((source, target, output))

        handles.append(module.register_forward_hook(hook))

    for cell_index, cell in enumerate(
        module for module in model.modules() if isinstance(module, Nb201Cell)
    ):
        for target, (specifications, modules) in enumerate(
            zip(cell.nodes, cell.edges, strict=True), start=1
        ):
            for (_operation, source), edge_module in zip(
                specifications, modules, strict=True
            ):
                register(
                    edge_module,
                    f"nb201-cell-{cell_index}:{source}",
                    f"nb201-cell-{cell_index}:{target}",
                )

    for cell_index, cell in enumerate(
        module for module in model.modules() if isinstance(module, DartsCell)
    ):
        for edge_index, (source, edge_module) in enumerate(
            zip(cell._indices, cell._ops, strict=True)
        ):
            register(
                edge_module,
                f"darts-cell-{cell_index}:{source}",
                f"darts-cell-{cell_index}:{edge_index // 2 + 2}",
            )

    for cell_index, cell in enumerate(
        module for module in model.modules() if isinstance(module, TransNasMicroCell)
    ):
        edge_index = 0
        for target, node_code in enumerate(cell.code[1:], start=1):
            for source in range(len(node_code)):
                register(
                    cell.edges[edge_index],
                    f"transnas-cell-{cell_index}:{source}",
                    f"transnas-cell-{cell_index}:{target}",
                )
                edge_index += 1

    if not nb101_modules and not handles:
        raise NotImplementedError(
            "model has no registered NB101, NB201/NATS, DARTS, or TransNAS semantic edge provider"
        )
    try:
        model(inputs)
    finally:
        for handle in handles:
            handle.remove()
        for cell in nb101_modules:
            del cell._edge_activation_capture
            del cell._edge_activation_prefix
    return EdgeActivationBatch(
        EdgeActivation(source, target, activation)
        for source, target, activation in captured
    )


__all__ = ["capture_semantic_edge_activations"]
