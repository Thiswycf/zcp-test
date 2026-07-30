# 搜索后完整训练操作手册

本手册用于没有 tabular“标准答案”的开放搜索空间。Benchmark 固定候选集应做 ZCP—真值相关性，
不重复把全部 benchmark 架构完整训练；DARTS、AutoFormer 和 MobileNet 搜索空间则先用 validation
协议搜索，再对选中、随机和参数/FLOPs 匹配架构从头训练。

## 1. 当前训练能力矩阵

| Space | 静态模型 fidelity | 正式训练 | 当前边界 |
|---|---|---|---|
| `darts` | `reference_model` | 已放行，尚未完成高成本精度验收 | CIFAR-10/100、原始 DARTS ImageNet、TE-NAS retrain 分 profile |
| `autoformer` | `reference_model` | **阻断** | repeated augmentation、分布式全局 batch/LR scaling、官方 fixture 未验收 |
| `ofa_proxyless_mbv2` | `reference_model` | **阻断** | scratch 与 inherited 权重协议已分离；正式训练仍缺颜色扰动、MAC 与分布式验收 |
| `zennas_plainnet_mbv2` | `reference_model` | **尚无正式配置** | ZenNAS/Zen-score 风格 PlainNet 与 Proxyless/OFA 空间必须分开 |
| `pit` | `reference_model` | **尚无正式配置** | Auto-Prox `90ed458` 三阶段静态拓扑；ViT-Bench vanilla/KD 真值只用于查询，不等同于本项目复训 |
| `ofa_mbv3` | `reference_model` | **尚无正式配置** | 官方五阶段/20-block 静态子网与 BN recalibration；尚未接入 inherited checkpoint/active weight export |

模型为 `reference_model` 不自动表示训练协议完备。非 smoke 训练同时要求配置
`formal_training_ready: true`，并且 `protocol` 及关键超参数必须匹配当前版本代码内置的已验收
profile；用户自写 YAML 不能通过设置该布尔值自行放行。未满足时 CLI 会列出 blocker 或不一致字段
并退出，不静默降级。正式运行不能覆盖已验收 profile 的 batch size 和 input size。

## 2. 从搜索结果选择架构

搜索 run 的架构必须来自 validation-only 选择，不得使用 benchmark test 指标调参。建议保留三组：

1. ZCP 搜索 best；
2. 固定 seed 随机候选；
3. 参数量或 FLOPs 匹配候选。

`--architecture` 接受文件或内联 JSON：

```bash
zcp-test train \
  --config configs/training/darts_cifar10.yaml \
  --architecture /path/to/best_architecture.json \
  --data-root /path/to/data/cifar10 \
  --output /path/to/runs/training

zcp-test train --config configs/training/darts_cifar10.yaml \
  --architecture '{"spec": {"normal": [...], "normal_concat": [2,3,4,5], "reduce": [...], "reduce_concat": [2,3,4,5]}}' \
  --smoke --epochs 1 --device cpu
```

内联示例中的 `...` 只是说明，不能直接执行。正式实验推荐使用 JSON 文件，便于保存 SHA-256 和
architecture ID。

## 3. DARTS 正式 profiles

```bash
zcp-test train --config configs/training/darts_cifar10.yaml \
  --architecture "$ARCH" --data-root /path/to/data/cifar10 --gpu auto
zcp-test train --config configs/training/darts_cifar100.yaml \
  --architecture "$ARCH" --data-root /path/to/data/cifar100 --gpu auto
zcp-test train --config configs/training/darts_imagenet.yaml \
  --architecture "$ARCH" --data-root /path/to/data/imagenet1k --gpu auto
```

原始 DARTS ImageNet 使用 LR 0.1 与逐 epoch StepLR；`tenas_imagenet.yaml` 使用 LR 0.5 与 cosine。
两者的 optimizer/scheduler 结果不可合并。短程 `--smoke` 使用合成输入，只验证构模、AMP、优化器、
JSONL 和 checkpoint。

## 4. AutoFormer 与 MobileNet 当前用法

下面命令只能 smoke；去掉 `--smoke` 会按设计失败并打印 blocker：

```bash
zcp-test train --config configs/training/autoformer_imagenet.yaml \
  --architecture "$AUTOFORMER_ARCH" --smoke --epochs 1 --gpu auto
zcp-test train --config configs/training/ofa_proxyless_mbv2_imagenet.yaml \
  --architecture "$MBV2_ARCH" --smoke --epochs 1 --gpu auto
```

不得把上述 smoke 写成 AZ-NAS 500 epoch 或 OFA/Proxyless 150 epoch 复现。Proxyless spec 使用
21 位固定位置 `kernel_size`/`expand_ratio`、五个可搜索 `depth`、固定 `width_mult=1.3` 和
128–224（步长 4）的 `resolution`；不能再传旧的“仅激活块”紧凑编码。官方 supernet checkpoint
和 active `ks/e/d` 权重导出已接入 `evaluate/search`，但每个 subnet 的独立真实数据 BN
recalibration 与 inherited accuracy 尚未验收，不能用未校准 ZCP smoke 替代。正式 scratch 训练
仍保持门禁阻断。

## 5. 恢复、监控与产物

```bash
zcp-test train --config "$CONFIG" --architecture "$ARCH" \
  --resume "$RUN/checkpoints/last.pt" --trusted \
  --data-root "$DATA_ROOT" --output /path/to/runs/training
zcp-test monitor "$RUN" --interval 5
zcp-test analyze training --source "$RUN/training.jsonl" --output "$RUN/reports/training"
```

恢复会比较 architecture ID、space、dataset、protocol、classes、input size 和训练配置，并恢复 model、
optimizer、scheduler、AMP scaler 与 RNG。`--trusted` 只确认操作者信任 checkpoint，不负责验证来源。

## 6. 高成本验收标签

- `full_data_one_percent_epochs`：完整数据，至少正式 epoch 的 1%；
- `one_percent_data_protocol`：确定性分层 1% 数据，跑完整 schedule；
- `full_reference_training`：完整数据与完整正式 schedule，且所有 protocol blocker 已关闭。

三类结果必须分开报告。OOM、NaN、恢复身份不一致和数据缺失均是失败，不得回退随机输入或缩小模型。
