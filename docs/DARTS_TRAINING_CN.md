# 标准 DARTS 搜索与训练

`darts` 现在表示标准 normal/reduce genotype，而不是旧 `{width, depth, op}` TinyConvNet。历史占位空间只读兼容名称为 `darts_toy_legacy`。

## 搜索

```bash
zcp-test search \
  --space darts --proxy er \
  --population 20 --generations 5 \
  --input-source dataset --data-root /path/to/cifar10
```

ZCP 搜索使用轻量 `zcp` 模型 profile；架构 ID 只由 genotype 决定，模型 profile 和输入指纹进入运行配置与缓存边界。

## 完整训练 profile

正式训练当前只有 DARTS profiles 通过 `formal_training_ready` 门禁。AutoFormer 和两类 MBV2
已有独立静态参考模型，但其完整训练协议仍有明确 blocker；PiT 已有 Auto-Prox 发布编码对应的
三阶段静态参考拓扑但尚无正式训练配置；OFA-MBV3 已有官方结构静态子网和 BN recalibration，
但 inherited checkpoint 与正式训练配置尚未验收。

```bash
zcp-test train --config configs/training/darts_cifar10.yaml --data-root DATA/cifar10
zcp-test train --config configs/training/darts_cifar100.yaml --data-root DATA/cifar100
zcp-test train --config configs/training/darts_imagenet.yaml --data-root DATA/ImageNet1k
```

- CIFAR-10：600 epoch，C=36，20 cells，batch 96，SGD 0.025，cutout 16。
- CIFAR-100：相同标准适配协议，分类头为 100 类。
- ImageNet-1k 原始 DARTS：250 epoch，C=48，14 cells，batch 128，SGD 0.1，逐 epoch
  `StepLR(gamma=0.97)`，label smoothing 0.1。
- TE-NAS retrain：使用独立 `configs/training/tenas_imagenet.yaml`，SGD 0.5 与 cosine；不得与原始
  DARTS ImageNet profile 混称。

训练记录 auxiliary loss、drop-path 调度、梯度裁剪、top-1/top-5、实际/下一 epoch LR 和 optimizer step。恢复可信 checkpoint：

```bash
zcp-test train --config CONFIG --resume RUN/checkpoints/last.pt --trusted
```

短程验证使用 `--smoke --epochs 1`；它只验证模型、AMP、optimizer、JSONL 和 checkpoint，不代表正式精度。

指定固定架构时，`--architecture` 可以是含顶层 `spec` 的 JSON 文件，也可以是内联 JSON；
`spec` 必须符合配置中 space 的 canonical schema。不同 space 的 spec 不能互换，DARTS genotype
必须包含完整合法的 normal/reduce edge 与 concat。恢复时使用命令输出的准确 timestamp run，例如：

```bash
ARCH=/path/to/runs/search/YYYYMMDDTHHMMSSZ_runid/best_architecture.json
RUN=/path/to/runs/training/YYYYMMDDTHHMMSSZ_runid
zcp-test train --config configs/training/darts_cifar10.yaml \
  --architecture "$ARCH" \
  --resume "$RUN/checkpoints/last.pt" --trusted \
  --data-root /path/to/data/cifar10
```

架构文件和 checkpoint 必须来自兼容的 space/config。高成本 250/600 epoch profile 未完成验收时，
不得用 1-epoch smoke 结果推断正式精度或收敛。
