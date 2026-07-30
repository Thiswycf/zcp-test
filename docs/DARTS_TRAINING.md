# Standard DARTS Search and Training

`darts` denotes the standard normal/reduction genotype, not the legacy `{width, depth, op}` TinyConvNet. The read-only compatibility name for the old placeholder space is `darts_toy_legacy`.

## Search

```bash
zcp-test search \
  --space darts --proxy er \
  --population 20 --generations 5 \
  --input-source dataset --data-root /path/to/cifar10
```

ZCP search uses the lightweight `zcp` model profile. The architecture ID depends only on the canonical genotype; the model profile and input fingerprint remain part of run configuration and cache identity.

## Accepted Protocols

The static protocol audit is pinned to upstream DARTS commit `f276dd346a09ae3160f8e3aca5c7b193fda1da37`:

- Original CIFAR-10 evaluation recipe (`cnn/train.py`): 600 epochs, C=36, 20 cells, batch 96, SGD 0.025, momentum 0.9, weight decay `3e-4`, no Nesterov, cosine without warmup, auxiliary weight 0.4, drop-path target 0.2, cutout 16, and gradient clipping at 5.
- CIFAR-100 adaptation: the CIFAR-10 optimization and regularization recipe with CIFAR-100 statistics and a 100-class head. Upstream DARTS has no original CIFAR-100 training script, so this must not be reported as an “original CIFAR-100 reproduction.”
- Original ImageNet-1k evaluation recipe (`cnn/train_imagenet.py`): 250 epochs, C=48, 14 cells, batch 128, SGD 0.1, momentum 0.9, weight decay `3e-5`, no Nesterov, per-epoch `StepLR(gamma=0.97)` without warmup, label smoothing 0.1, auxiliary weight 0.4, drop-path target 0, and gradient clipping at 5.

Commands:

```bash
zcp-test train --config configs/training/darts_cifar10.yaml --data-root DATA/cifar10
zcp-test train --config configs/training/darts_cifar100.yaml --data-root DATA/cifar100
zcp-test train --config configs/training/darts_imagenet.yaml --data-root DATA/ImageNet1k
```

The formal protocol gate checks dataset, model scale, epochs, optimizer, momentum/Nesterov, scheduler, batch, auxiliary loss, drop path, gradient clipping, cutout/label smoothing, and the pinned implementation commit. Drop path follows the upstream `target * epoch / epochs` schedule; the final CIFAR training epoch is therefore `0.2 * 599 / 600`, not 0.2.

AMP and safe checkpoint/manifest handling in these configs are project runtime extensions, not features of the 2018 upstream scripts, and do not imply bitwise equivalence. All three formal DARTS profiles mark the published value as `batch_size_semantics: global`. DDP requires divisibility by `WORLD_SIZE` and derives the per-rank batch—for example, CIFAR batch 96 becomes 24 on four ranks—while preserving the published LR and effective global batch. Non-divisible launches fail instead of changing the protocol silently. DDP remains a modern runtime extension and is not evidence of bitwise reproduction.

## Dual One-Percent Acceptance

`--acceptance-smoke` uses real data without relabelling a shortened run as a full accuracy
reproduction. An approved DARTS profile accepts only:

- full data with at least 1% of the formal epochs: at least 6 epochs for CIFAR-10/100 and 3 for
  ImageNet-1k; or
- exactly 1% deterministic stratified data with the complete schedule: 600 CIFAR epochs or 250
  ImageNet-1k epochs.

```bash
zcp-test train --config configs/training/darts_cifar10.yaml \
  --acceptance-smoke --epochs 6 --data-fraction 1.0 \
  --architecture ARCH.json --data-root DATA/cifar10 --output RUNS

zcp-test train --config configs/training/darts_cifar10.yaml \
  --acceptance-smoke --epochs 600 --data-fraction 0.01 \
  --architecture ARCH.json --data-root DATA/cifar10 --output RUNS

zcp-test train --config configs/training/darts_imagenet.yaml \
  --acceptance-smoke --epochs 3 --data-fraction 1.0 \
  --architecture ARCH.json --data-root DATA/ImageNet1k --output RUNS
```

Runs record `acceptance_protocol=full_data_one_percent_epochs` or
`one_percent_data_protocol` in both resolved configuration and checkpoint identity. The 1% subset
uses an exact split-wide target; it does not inflate the fraction by forcing one item per class when
the target is smaller than the class count.

Before a formal launch, run one complete real-data epoch for throughput and pipeline validation:

```bash
zcp-test train --config configs/training/darts_cifar10.yaml \
  --real-data-preflight --epochs 1 --data-fraction 1.0 \
  --architecture ARCH.json --data-root DATA/cifar10 --output RUNS/preflight
```

Its `training_mode` is always `real_data_preflight`; it does not count toward either acceptance
protocol.

## TE-NAS Boundary

TE-NAS commit `9df78ffd98573035375b12e19b9007578cc4155d` delegates DARTS evaluation to `chenwydj/DARTS_evaluation`. At evaluation commit `f53b2b6975107885c44cf26e66620ff90a6dac4a`, the ImageNet defaults are 250 epochs, C=48, 14 cells, global batch 768 (documented for 8 GPUs), SGD 0.5, momentum 0.9, weight decay `3e-5`, no Nesterov, cosine, five warmup epochs, label smoothing 0.1, auxiliary weight 0.4, drop-path target 0, and gradient clipping at 5.

The current `configs/training/tenas_imagenet.yaml` uses batch 128, omits the five-epoch warmup, and cannot express the required global-batch-768/8-GPU semantics. That config is outside this task's permitted write scope. `tenas-retrain-imagenet` has therefore been removed from the formal protocol allowlist and now fails closed; it cannot be reported as a formal TE-NAS reproduction. It must also remain distinct from original DARTS ImageNet's SGD 0.1 + StepLR recipe.

## Resume Identity

Only explicitly trusted checkpoints may be resumed:

```bash
zcp-test train --config CONFIG \
  --architecture ARCH \
  --resume RUN/checkpoints/last.pt --trusted \
  --data-root DATA_ROOT
```

A checkpoint stores and restores the model, optimizer, scheduler, AMP scaler, RNG state, epoch, best metric, and portable training-log history. Resume strictly compares `TrainingConfig` and the run's `search_space_id`, canonical `architecture_id`, dataset, protocol, class count, input size, model fidelity, and training mode. Changes to architecture, protocol, scheduler/Nesterov configuration, or smoke/formal identity are rejected. In formal mode, protocol validation also prevents silent changes to auxiliary loss, drop path, or critical augmentation fields under an unchanged protocol name.
Before model construction, the training entry point seeds Python, NumPy, and PyTorch CPU/CUDA RNGs;
the base seed is part of checkpoint identity, while manifests record the per-rank seed and current
cuDNN/deterministic-algorithm state. This fixes model initialization that previously escaped
`--seed`, but it is not a claim of bitwise determinism for every CUDA kernel.

## Evidence Boundary

A short run must be described only as a smoke test:

```bash
zcp-test train --config configs/training/darts_cifar10.yaml --epochs 1 --smoke
```

It checks model forward/backward, optimizer/scheduler, auxiliary loss, AMP, JSONL, and checkpoint paths on synthetic data. It is not evidence of real-data accuracy, convergence, throughput, or a 250/600-epoch reproduction. Outstanding high-cost work includes multi-seed real-data 600-epoch CIFAR-10/CIFAR-100 runs, a 250-epoch ImageNet-1k run, and an 8-GPU 250-epoch TE-NAS run that preserves the upstream global-batch semantics.
