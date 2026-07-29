# Operations and Research Guide

## GPU selection

The Conda environment persists `CUDA_DEVICE_ORDER=PCI_BUS_ID`. Run `zcp-test gpu list` to see the `nvidia-smi` index, PCI order, UUID, bus ID, model, free memory, utilization and visible logical index. GPU commands default to `--gpu auto`; selection ranks free memory descending, utilization ascending and bus ID ascending. If the best automatic candidate is locked by another `zcp-test` process, the command tries the next eligible GPU; an explicit INDEX, UUID or PCI bus ID never changes devices. Select a stable device with `--gpu GPU-...` or `--gpu 00000000:98:00.0`, filter with `--gpu-model`, and use `--device cpu` for CPU execution. The manifest records the physical identity and CUDA environment, and GPU exhaustion never silently falls back to CPU.

The lock coordinates only same-user processes that follow the `zcp-test` lock protocol; it is not a system-wide GPU reservation. CUDA visibility binding assumes the normal short-lived CLI process model, so embedded callers must not rebind multiple physical GPUs in one Python process.

## Evaluation

Dataset input is the default and requires `--data-root` or a `dataset_<name>` catalog asset. `--input-source random|noise` is an explicit ablation and is fingerprinted separately. Ten architectures and three proxies produce exactly 30 schema-2 JSONL records. A multi-component proxy stores one primary `score` plus its complete `components` mapping in the same record.

```bash
zcp-test evaluate --space nb201_topology --proxies er,naswot,synflow \
  --count 10 --data-root DATA/cifar10
```

Use `cifar10-valid` for NATS validation correlations; do not relabel CIFAR-10 test accuracy as validation. Run directories create `checkpoints`, `parts` and `reports` only when those artifacts are actually written.

## Adding a proxy

Run `zcp-test proxy scaffold my_proxy`, implement the generated trusted project-local module, declare a `ProxyCapability`, then run `zcp-test proxy validate my_proxy`. Multi-component formulas should return `ProxyOutput(score=..., primary_component=..., components=...)`. Never rely on mapping order to choose a score.

## Analysis and monitoring

`zcp-test report bundle RUN... --output REPORT` creates rebuildable CSV, PNG, SVG and static HTML. `zcp-test analyze correlation|compare|sensitivity` covers bootstrap correlations, top-k overlap, validation-only rank aggregation, transfer, cost/memory Pareto and sample-size convergence. `zcp-test analyze search|training` renders progression curves. `zcp-test monitor RUN --interval 5` tolerates an incomplete JSONL tail and atomically refreshes an auto-reloading HTML page.

## DARTS

`darts` is a standard normal/reduce genotype and cell implementation; the previous placeholder is named `darts_toy_legacy`. ZCP evaluation uses the lightweight `zcp` model profile. Complete configurations are `configs/training/darts_cifar10.yaml`, `darts_cifar100.yaml` and `darts_imagenet.yaml`. Use `--smoke --epochs 1` only for pipeline validation. Resume only a verified checkpoint with `--resume ... --trusted`.
