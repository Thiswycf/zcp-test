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

```bash
zcp-test train --config configs/training/darts_cifar10.yaml --data-root DATA/cifar10
zcp-test train --config configs/training/darts_cifar100.yaml --data-root DATA/cifar100
zcp-test train --config configs/training/darts_imagenet.yaml --data-root DATA/ImageNet1k
```

- CIFAR-10：600 epoch，C=36，20 cells，batch 96，SGD 0.025，cutout 16。
- CIFAR-100：相同标准适配协议，分类头为 100 类。
- ImageNet-1k：250 epoch，C=48，14 cells，batch 128，SGD 0.5，label smoothing 0.1。

训练记录 auxiliary loss、drop-path 调度、梯度裁剪、top-1/top-5、实际/下一 epoch LR 和 optimizer step。恢复可信 checkpoint：

```bash
zcp-test train --config CONFIG --resume RUN/checkpoints/last.pt --trusted
```

短程验证使用 `--smoke --epochs 1`；它只验证模型、AMP、optimizer、JSONL 和 checkpoint，不代表正式精度。
