# 标准 DARTS 搜索与训练

`darts` 表示标准 normal/reduce genotype，而不是旧 `{width, depth, op}` TinyConvNet。历史占位空间只读兼容名称为 `darts_toy_legacy`。

## 搜索

```bash
zcp-test search \
  --space darts --proxy er \
  --population 20 --generations 5 \
  --input-source dataset --data-root /path/to/cifar10
```

ZCP 搜索使用轻量 `zcp` 模型 profile；架构 ID 只由 canonical genotype 决定，模型 profile 和输入指纹进入运行配置与缓存边界。

## 已验收协议

本次静态协议验收固定到 DARTS 上游 commit `f276dd346a09ae3160f8e3aca5c7b193fda1da37`：

- CIFAR-10 原始评估配方：`cnn/train.py`，600 epoch，C=36，20 cells，batch 96，SGD 0.025、momentum 0.9、weight decay `3e-4`，不启用 Nesterov，cosine，无 warmup，auxiliary weight 0.4，drop-path target 0.2，cutout 16，梯度裁剪 5。
- CIFAR-100 适配配方：沿用上述 CIFAR-10 优化与正则化配方，仅替换数据统计和 100 类分类头。DARTS 上游没有对应的 original CIFAR-100 训练脚本，因此不得称为“原始 CIFAR-100 复现”。
- ImageNet-1k 原始评估配方：`cnn/train_imagenet.py`，250 epoch，C=48，14 cells，batch 128，SGD 0.1、momentum 0.9、weight decay `3e-5`，不启用 Nesterov，逐 epoch `StepLR(gamma=0.97)`，无 warmup，label smoothing 0.1，auxiliary weight 0.4，drop-path target 0，梯度裁剪 5。

对应命令：

```bash
zcp-test train --config configs/training/darts_cifar10.yaml --data-root DATA/cifar10
zcp-test train --config configs/training/darts_cifar100.yaml --data-root DATA/cifar100
zcp-test train --config configs/training/darts_imagenet.yaml --data-root DATA/ImageNet1k
```

正式协议门禁会核对数据集、模型规模、epoch、optimizer、momentum/Nesterov、scheduler、batch、auxiliary、drop-path、梯度裁剪、cutout/label smoothing 和固定实现 commit。当前实现按上游 `target * epoch / epochs` 调度 drop-path；600 epoch CIFAR 的最后一个训练 epoch 因而是 `0.2 * 599 / 600`，不是 0.2。

配置中的 AMP 与安全 checkpoint/manifest 是本项目运行时扩展，不来自 2018 年上游脚本；它们不能提供逐 bit 等价保证。三个正式 DARTS profile 将发布值明确标为 `batch_size_semantics: global`，DDP 会要求该值可被 `WORLD_SIZE` 整除并换算为每卡 batch，例如 CIFAR 的 96 在 4 rank 下为每卡 24；学习率和有效全局 batch 保持发布值。无法整除时直接失败，不静默改变协议。DDP 本身仍是现代运行时扩展，不能据此声称逐 bit 复现。

## 双重 1% 验收

`--acceptance-smoke` 使用真实数据，但不把短程结果冒充完整精度复现。正式 DARTS profile 只接受：

- 完整数据且不少于正式 epoch 的 1%：CIFAR-10/100 最少 6 epoch，ImageNet-1k 最少 3 epoch；
- 恰好 1% 确定性分层数据并跑完整 schedule：CIFAR-10/100 为 600 epoch，ImageNet-1k 为 250 epoch。

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

运行分别记录 `acceptance_protocol=full_data_one_percent_epochs` 或
`one_percent_data_protocol`，该字段同时进入解析配置和 checkpoint 恢复身份。1% 子集按 split 的
全局目标条数精确分配类别配额；目标条数少于类别数时不会用“每类至少一条”扩大实际比例。
6/3 epoch 模式保留正式 600/250-epoch 的 cosine/StepLR、warmup 与 drop-path 时间轴；
`epochs` 表示实际停止点，`schedule_epochs` 表示正式总周期。禁止把 schedule 压缩到短程运行长度。

### CIFAR-10/100 完成状态（2026-07-31）

三类冻结候选为 ER 搜索候选、固定随机候选和参数匹配随机池候选。CIFAR-10 与 CIFAR-100 均已
完成两套协议，因此全数据 × 6 epoch 共 6 runs，恰好 1% 数据 × 600 epoch 共 6 runs：

| 协议 | CIFAR-10 best valid top-1（ER / 固定随机 / 参数匹配随机） | CIFAR-100 best valid top-1（ER / 固定随机 / 参数匹配随机） |
|---|---|---|
| 全数据 × 6 epoch，保留 600-epoch schedule | `78.62 / 77.28 / 64.67` | `46.19 / 44.13 / 26.81` |
| 恰好 1% 数据 × 600 epoch | `42.0 / 45.0 / 46.0` | `15.0 / 12.0 / 11.0` |

确定性真实数据预检已通过；全数据短程协议和 1% 数据完整 schedule 各完成一次可信 checkpoint
恢复审计，分别恢复 6 条与 600 条训练历史；证据报告和机器可读摘要均已完成。详见
[`evidence/DARTS_CIFAR_DUAL_ONE_PERCENT_CN.md`](evidence/DARTS_CIFAR_DUAL_ONE_PERCENT_CN.md) 与
[`evidence/darts_cifar_dual_one_percent_summary.json`](evidence/darts_cifar_dual_one_percent_summary.json)。

该状态只表示 DARTS CIFAR 双重 1% 的工程与限定协议验收完成。它不是 600 epoch 全数据精度复现，
也不是多 seed 搜索收益证明：训练只使用 seed `20260731`，ER 候选只由一个固定 CIFAR-10 batch 和
一个初始化 seed 选出。两个协议给出的候选排序不同，分别测试早期全数据学习与小数据完整
schedule，禁止平均、合并或据此宣称 ER 稳定优于基线。ImageNet-1k 现已通过结构审计（1000 类、
1,281,167 张训练图、50,000 张验证图）和真实 loader 解码检查；DARTS ImageNet 双重 1% 已完成限定
验收，但首个 DDP 与其余单卡 run 的 BatchNorm 粒度不同。AutoFormer 单候选双重 1% 也已完成；
PlainNet-MBV2 和 Proxyless-MBV2 仍未完成。

正式启动前可用以下命令执行一个完整真实数据 epoch 的吞吐与流水线预检：

```bash
zcp-test train --config configs/training/darts_cifar10.yaml \
  --real-data-preflight --epochs 1 --data-fraction 1.0 \
  --architecture ARCH.json --data-root DATA/cifar10 --output RUNS/preflight
```

其 `training_mode` 固定为 `real_data_preflight`，不得计入上述任一 1% 验收。

## TE-NAS 边界

TE-NAS 主仓库 commit `9df78ffd98573035375b12e19b9007578cc4155d` 指向独立的 `chenwydj/DARTS_evaluation`；该评估仓库 commit `f53b2b6975107885c44cf26e66620ff90a6dac4a` 的 ImageNet 默认配方是 250 epoch、C=48、14 cells、全局 batch 768（README 明示面向 8 GPU）、SGD 0.5、momentum 0.9、weight decay `3e-5`、不启用 Nesterov、cosine、前 5 epoch warmup、label smoothing 0.1、auxiliary weight 0.4、drop-path target 0 和梯度裁剪 5。

现有 `configs/training/tenas_imagenet.yaml` 使用 batch 128 且没有 5 epoch warmup，也没有表达“全局 batch 768 / 8 GPU”的语义；它不在本次允许修改的配置范围内。`tenas-retrain-imagenet` 因此已从正式协议白名单移除，当前命令会 fail closed，而不能被当作 TE-NAS 正式复现。TE-NAS 也不得与 original DARTS ImageNet 的 SGD 0.1 + StepLR 配方混称。

## 恢复身份

恢复只接受显式可信的 checkpoint：

```bash
zcp-test train --config CONFIG \
  --architecture ARCH \
  --resume RUN/checkpoints/last.pt --trusted \
  --data-root DATA_ROOT
```

checkpoint 同时保存并恢复 model、optimizer、scheduler、AMP scaler、RNG、epoch、best metric 和训练日志历史。恢复前会严格比较 `TrainingConfig`，并比较 `search_space_id`、canonical `architecture_id`、dataset、protocol、classes、input size、model fidelity 和 training mode；架构、协议、scheduler/Nesterov 等训练配置或 smoke/formal 身份变化都会拒绝恢复。正式模式下协议门禁还防止在保留同一 protocol 名时静默修改 auxiliary、drop-path 或数据增强关键字段。
训练入口在模型构建前设置 Python、NumPy 与 PyTorch CPU/CUDA RNG，并将 base seed 纳入 checkpoint
identity；正式 profile 锁定 `deterministic: true`，manifest 同时记录各 rank 实际 seed 和
cuDNN/deterministic-algorithm 状态。不支持确定性实现的算子会失败，不回退非确定模式。该措施修复
过去 `--seed` 只控制 DataLoader、却未控制模型初始化的问题；跨软件/驱动版本仍不承诺逐 bit 一致。

## 证据边界

短程验证只能写成 smoke：

```bash
zcp-test train --config configs/training/darts_cifar10.yaml --epochs 1 --smoke
```

它只验证合成数据上的模型前后向、optimizer/scheduler、auxiliary、AMP、JSONL 和 checkpoint 通路，
不能作为上述真实数据双重 1% 证据。当前尚未完成的高成本项是 CIFAR-10/CIFAR-100 600 epoch
全数据精度复现和多 seed 搜索收益验证；DARTS ImageNet 数据资产已就绪，但双重 1% 与 250 epoch
正式训练尚未执行；保持上游全局 batch 语义的 8 GPU TE-NAS 250 epoch 复现也未完成。
