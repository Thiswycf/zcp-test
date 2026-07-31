# 搜索后完整训练操作手册

本手册用于没有 tabular“标准答案”的开放搜索空间。Benchmark 固定候选集应做 ZCP—真值相关性，
不重复把全部 benchmark 架构完整训练；DARTS、AutoFormer 和 MobileNet 搜索空间则先用 validation
协议搜索，再对选中、随机和参数/FLOPs 匹配架构从头训练。

## 1. 当前训练能力矩阵

| Space | 静态模型 fidelity | 正式训练 | 当前边界 |
|---|---|---|---|
| `darts` | `reference_model` | 已放行，尚未完成高成本精度验收 | CIFAR-10/100、原始 DARTS ImageNet、TE-NAS retrain 分 profile |
| `autoformer` | `reference_model` | **阻断** | AZ-NAS Tiny/Small 500-epoch profile、no-decay、warmup/min-LR、plain validation CE 已锁定；双重 1% GPU 验收未完成 |
| `ofa_proxyless_mbv2` | `reference_model` | **阻断** | scratch 与 inherited 权重协议已分离；正式训练仍缺颜色扰动、MAC 与分布式验收 |
| `zennas_plainnet_mbv2` | `proxy_approximation` | **禁止正式训练** | 当前固定 stage MBConv 编码不是 ZenNAS/AZ-NAS structure-string 搜索空间；必须完成真实 PlainNet port 后才能升级 |
| `pit` | `reference_topology_pytorch_port` | **尚无正式配置** | Auto-Prox `90ed458` 三阶段拓扑、参数/MAC fixture 已核对；缺 checkpoint/逐层数值对照，ViT-Bench vanilla/KD 真值只用于查询 |
| `ofa_mbv3` | `reference_model` | **尚无正式配置** | 官方五阶段/20-block 静态子网与 BN recalibration；尚未接入 inherited checkpoint/active weight export |

详细上游对照、150/480 epoch 配方边界和升级门槛见
[`evidence/PLAINNET_MBV2_FIDELITY_AUDIT_CN.md`](evidence/PLAINNET_MBV2_FIDELITY_AUDIT_CN.md)；
AutoFormer optimizer、LR 与 validation 协议证据见
[`evidence/AUTOFORMER_TRAINING_PROTOCOL_CN.md`](evidence/AUTOFORMER_TRAINING_PROTOCOL_CN.md)。

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

AutoFormer 当前配置对应 AZ-NAS Tiny 搜索域：500 epoch、20 epoch warmup、AdamW、base LR
`5e-4`、`warmup_lr=1e-6`、`min_lr=1e-5`，并按有效 global batch 相对 512 线性缩放。bias、Norm、
class token 与 position embedding 不做 weight decay，validation 使用 plain cross entropy。
这些字段已经单元测试和 profile validator 锁定，但在双重 1% GPU 验收完成前仍不能去掉 `--smoke`。

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
每个 epoch 还记录 train/validation 耗时、样本吞吐、CUDA 峰值 allocated/reserved 显存；CPU 运行的
显存字段为 `null`。训练图将 accuracy、loss、LR/drop-path 与 epoch 耗时/峰值显存分面展示，避免
把量纲不同的优化指标混在同一纵轴。
模型构建前统一设置 Python、NumPy、PyTorch CPU/CUDA RNG；正式与候选 profile 锁定
`deterministic: true`，启用 PyTorch deterministic algorithms、cuDNN deterministic 和固定
CUBLAS workspace，manifest 记录 base seed、rank seed 与实际后端状态，checkpoint identity 记录
base seed。多 rank 使用 `rank_seed = base_seed + rank`；不支持确定性实现的算子会明确失败。即便
如此，跨 PyTorch/CUDA/驱动版本也不承诺逐 bit 一致，版本信息仍必须进入报告。

## 6. 高成本验收标签

- `full_data_one_percent_epochs`：完整数据，至少正式 epoch 的 1%；
- `one_percent_data_protocol`：确定性分层恰好 1% 数据，跑完整 schedule；
- `full_reference_training`：完整数据与完整正式 schedule，且所有 protocol blocker 已关闭。

三类结果必须分开报告。OOM、NaN、恢复身份不一致和数据缺失均是失败，不得回退随机输入或缩小模型。
第一类中的“6/3 epoch”是正式 600/250-epoch schedule 的前缀，而不是把 cosine、warmup 或
drop-path 总周期压缩成 6/3；resolved config 以 `schedule_epochs` 单独记录正式周期。
