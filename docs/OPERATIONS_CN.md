# CLI 运维与安全边界

## `--trusted` 信任边界

`--trusted` 只表示操作者已独立核验序列化输入，不会自动计算 checksum、隔离反序列化或让
pickle/PyTorch 文件变安全。原生 NAS-Bench-201、NATS-TSS/SSS、NAS-Bench-301 查询、checkpoint
恢复、ViT 转换和 legacy pickle 导入都必须在命令行显式确认：

```bash
zcp-test evaluate --config configs/benchmarks/nasbench201.yaml --trusted \
  --proxies params --count 1 --input-source random --device cpu
zcp-test train --config configs/training/darts_cifar10.yaml \
  --resume "$RUN/checkpoints/last.pt" --trusted
```

配置文件不能自行启用可信执行。先核验来源和摘要，再只对本次命令添加 `--trusted`。

## 配置优先级

`evaluate`、`correlate`、`search` 和旧版 `report` 接受 `--config`。配置可以直接包含参数，也可以
使用与命令同名的 section：

```yaml
evaluate:
  benchmark: nasbench101
  benchmark_version: full
  proxies: params,naswot
  count: 10
```

解析顺序为：CLI 默认值 → 匹配的配置值 → 命令行显式参数。`--count 20` 与标准 argparse 写法
`--count=20` 都会被识别为显式覆盖。所有命令都会拒绝未知键；`train` 额外允许版本化训练 profile
schema 中声明的模型、优化器、增强和协议字段，因此 `learnng_rate` 等拼写错误会在启动训练前
fail closed。训练配置还须通过 protocol validator，并检查 run 目录中的 resolved `config.yaml`。
YAML 中的 `trusted: true` 不能替代命令行 `--trusted`。

## GPU 锁

`--gpu-lock-timeout 0` 表示遇到锁立即失败；正数表示获取合格 GPU 锁的总等待秒数；负数非法。
`--gpu auto` 会在剩余时间内尝试下一张满足型号和显存条件的卡，显式 index、UUID 或 Bus ID 不会
换卡。锁只协调同一用户下遵循本协议的进程，不是系统级 GPU 预留；`--device` 会绕过物理卡选择
和锁。

## 数据输入与结果类型

`evaluate` 和 `search` 默认 `--input-source dataset`，必须提供 `--data-root` 或有效的
`dataset_<name>` catalog asset。真实数据缺失时直接失败，不会自动改用随机输入。

- **standard answer**：带 dataset/split/budget/seed 协议的 benchmark 发布记录；
- **surrogate**：例如 NAS-Bench-301 的模型预测，不是完整训练观测；
- **inherited**：使用 supernet 权重评价的 subnet 指标；
- **scratch**：架构独立从头训练得到的指标。

四类结果不得混合，NAS-Bench-201 真值也不能替代 NATS-TSS 真值。

## RUN 目录

`--output` 是父目录；实际运行目录为 `<output>/YYYYMMDDTHHMMSS+0800_<run-id>/`。后续报告、监控和
恢复必须使用命令输出 JSON 中的准确 `run` 值：

```bash
RUN=/path/to/runs/evaluate/YYYYMMDDTHHMMSS+0800_runid
zcp-test report bundle "$RUN" --output "$RUN/reports/bundle"
zcp-test monitor "$RUN" --interval 5
```

`report bundle` 会把没有直接 artifact 的父目录展开一层，并处理其中全部可识别 timestamp run；
`monitor` 仅在父目录恰好包含一个可识别 run 时自动进入。父目录有多个 run 时必须传入准确 `RUN`。
训练监控优先读取 `events.jsonl`：rank 0 默认约每 30 秒写入一次
`training_batch_progress`，每个 train/valid split 的最后一个 batch 也会写入；epoch 完成后写入
`training_epoch_completed`。`training.jsonl` 仍严格保持每个完成 epoch 一行，训练曲线只从该文件重建。
事件中的 `rank_local_samples` 是 rank 0 的本地计数，不是分布式全局精确样本数。旧 run 不会补写
heartbeat；若其 epoch 尚未结束，monitor 只能显示已有 artifact。
同一事件还会写入并即时 flush 到人类可读的 `run.log`；新 run 不应再出现“`events.jsonl` 有事件而
`run.log` 长期为 0 字节”。大型图像训练应把 `--data-root` 指向调用者已核验的本机高速盘副本，
不要根据目录名猜测介质速度；先用 `findmnt -T /path/to/imagenet1k` 确认挂载，再核对类别和文件数。
CLI 不会硬编码或自动改写数据根。

当前 `search` 尚无 `--resume`：`search.jsonl` 是可审计记录，不是 population/RNG/cache checkpoint。
中断后必须新建 run；不得把向旧文件追加记录称为恢复。训练恢复则使用同一架构、配置和协议身份，
并显式传入可信 `last.pt`。

## 范围切分与合并

`evaluate --start/--count` 可用于手工切分互不重叠的范围，但目前没有内置多进程 launcher 或
JSONL merge 子命令。优先保留每个分片 run 的 manifest，直接把多个 score 文件交给分析：

```bash
zcp-test analyze compare \
  --scores "$RUN_A/scores.jsonl" "$RUN_B/scores.jsonl" \
  --output /path/to/reports/partitions
```

若下游强制要求单文件，只能合并 resolved protocol 完全相同且范围不重叠的分片，并为
`zcp_test.artifacts.merge_jsonl` 明确唯一键后核对行数。不要用 `cat`：它无法发现重复评估、协议
混合或未写完的末行。合并文件是派生产物，不能替代各源 run manifest。

## `data fetch`

`data fetch` 只下载 catalog 中声明了 `source_url` 的单个 asset：

```bash
zcp-test data fetch ASSET_ID \
  --catalog /path/to/data/catalog.json \
  --destination /path/to/data/file
```

命令先写 `<destination>.part`，存在 catalog SHA-256 时进行核验，再原子替换目标。它不会展开
benchmark 组、解压、转换、注册新路径，也不提供 `data bootstrap` 的断点续传流程；无 checksum
时不能据此证明真实性。

## Legacy pickle 导入

```bash
zcp-test legacy import --source verified.pkl --output converted.jsonl --trusted
```

pickle 加载时可执行代码，只能在隔离环境中处理已核验来源。list 按元素输出，mapping 转成
`{"key": ..., "value": ...}`，其他对象转成单条 `{"value": ...}`。这只是形状迁移，不验证
score/target schema；使用前必须检查转换后的 JSONL，且不要覆盖源文件。

## NATS-SSS 跨数据集运行

先准备 NATS-SSS benchmark、确定性 1% manifest 和输入数据。NATS-SSS 有 32,768 个有限架构，
最低 1% 为 328 个；原生 NATS API 是序列化资产，因此 sample、inspect 和 evaluate 都需要
`--trusted`。ImageNet16 raw pickle 的转换步骤见[数据自举](DATA_BOOTSTRAP_CN.md)。

```bash
DATA=/path/to/data
CATALOG="$DATA/catalog.json"
AUDIT=/path/to/audit

zcp-test data bootstrap --root "$DATA" --benchmarks nats_sss \
  --catalog "$CATALOG" --yes
zcp-test benchmark sample nats_sss --catalog "$CATALOG" --trusted \
  --fraction 0.01 --seed 2026 --shards 4 \
  --output "$AUDIT/sampling/nats-sss-1pct-seed2026.json"
zcp-test benchmark inspect nats_sss --catalog "$CATALOG" --trusted \
  --dataset ImageNet16-120 --split valid --metric-name accuracy \
  --epoch-budget 90 --metric-seed-reduction mean
```

四个 shard 应分别启动，下面只展示 `--sample-shard 0`。正式 22 代理和核心 11 代理列表应沿用
锁定验收协议，不要把示例变量中的代理集合当作新的协议定义。

```bash
PROXIES=az_nas,er,er_conn,er_deg,er_dist,er_pr,flops,gradnorm,jacob_cov,meco,meco_opt,naswot,near,ntkt,params,swap,synflow,te_nas,ter,vkdnw,zen,zico
MANIFEST="$AUDIT/sampling/nats-sss-1pct-seed2026.json"

# CIFAR-100 dataset-specific ZCP：输入和 benchmark target 都是 CIFAR-100。
zcp-test evaluate --benchmark nats_sss --catalog "$CATALOG" --trusted \
  --sample-manifest "$MANIFEST" --sample-shard 0 \
  --dataset cifar100 --target-metric accuracy --target-split valid \
  --epoch-budget 90 --metric-seed-reduction mean --target-direction maximize \
  --input-source dataset --data-root /path/to/cifar100 \
  --input-size 32 --classes 100 --batch-size 16 \
  --proxies "$PROXIES" --seed 2026 --gpu auto \
  --output "$AUDIT/runs/nats-sss-cifar100-seed2026"

# ImageNet16-120 dataset-specific ZCP：不传 --data-root，按 catalog 解析安全 manifest。
zcp-test evaluate --benchmark nats_sss --catalog "$CATALOG" --trusted \
  --sample-manifest "$MANIFEST" --sample-shard 0 \
  --dataset ImageNet16-120 --target-metric accuracy --target-split valid \
  --epoch-budget 90 --metric-seed-reduction mean --target-direction maximize \
  --input-source dataset --input-size 16 --classes 120 --batch-size 16 \
  --proxies "$PROXIES" --seed 2026 --gpu auto \
  --output "$AUDIT/runs/nats-sss-imagenet16-seed2026"
```

这里的 `--dataset` 同时决定模型类别数语义、ZCP 输入协议和 NATS target dataset；因此两条命令
得到的是 **dataset-specific ZCP**。**Target-only transfer** 则要求保留源数据集 ZCP 分数及其
`input_fingerprint`，仅将同一 architecture ID 与另一个 dataset 的 NATS target 做一对一 join。
单次 `evaluate` 仍不使用独立 `--target-dataset`；正式 target-only 由分析阶段固定 source score
与 fingerprint，再按 architecture ID 连接其他数据集 target。三数据集各四个分片应一次性传给：

```bash
mapfile -t SCORES < <(find \
  /path/to/audit/h1-nats-sss-seed2026 \
  /path/to/audit/h1-nats-sss-cifar100-seed2026 \
  /path/to/audit/h1-nats-sss-imagenet16-seed2026 \
  -name scores.jsonl -type f | sort)
test "${#SCORES[@]}" -eq 12
zcp-test analyze benchmark --scores "${SCORES[@]}" \
  --benchmark nats_sss --view size \
  --output /path/to/audit/h1-nats-sss-cross-dataset-analysis
```

该命令现已生成 `dataset_proxy_target_matrix.csv`、`proxy_dataset_stability.csv`、
`target_dataset_transfer.csv` 和 `controlled_proxy_target_transfer.csv`。正式结果和 SHA 见
[跨数据集证据](evidence/NATS_SSS_CROSS_DATASET_CN.md)。

常见错误：

| 错误/现象 | 原因与处理 |
|---|---|
| `ImageNet16 conversion requires explicit --trusted` | raw 是 pickle；核验来源和 11 个 MD5 后显式添加 `--trusted`。 |
| `ImageNet16 MD5 mismatch` | 文件不是官方字节或下载损坏；不要 `--replace` 绕过，重新获取对应 batch。 |
| `Unsafe or corrupt ImageNet16 runtime` | manifest 或某个 `.npy` shard SHA 不匹配；重新复制完整安全目录或重新转换。 |
| `--input-source dataset requires --data-root or a configured dataset asset` | 未传 `--data-root`，且 catalog 没有 `dataset_imagenet16_120`。 |
| `nats_sss uses a native serialized format` | benchmark 查询仍需 `--trusted`；这与安全 `.npy` dataset 是否 trusted 无关。 |
| `Metric 'accuracy' for split 'valid' not in ...` | 使用精确 dataset `ImageNet16-120`、split `valid`、metric `accuracy`、budget `90`。 |
| CIFAR-100 找不到数据 | `--data-root` 必须是 torchvision CIFAR-100 已下载目录；命令不会隐式下载。 |
| 四个 shard 各自只得到 82 条 | 正常分片；最终分析必须合并四个互斥 shard，并按 evaluation seed 分组。 |

## Proxy scaffold

`zcp-test proxy scaffold NAME` 仅适用于可写源码 checkout 或 editable install。它会同时写入
`src/zcp_test/proxies/custom/NAME.py` 和 `tests/test_proxy_NAME.py`；普通只读 wheel/site-packages
不是支持目标。`proxy validate` 只在小型合成模型上检查有限值、权重隔离和 hook 清理，不代表完成
全 benchmark 科学验收。

## 训练架构文件与 fidelity

模型结构 fidelity 与正式训练协议是两个独立条件。`darts`、`autoformer`、
`ofa_proxyless_mbv2` 与 `zennas_plainnet_mbv2` 均拥有 `reference_model` 静态结构；后者使用
ZenNAS/AZ-NAS structure string、白名单 parser 和独立的 sample/mutate/crossover。只有配置中
`formal_training_ready: true` 的协议才能启动非 smoke 训练。当前正式放行的是 DARTS profiles；
AutoFormer、PlainNet-MBV2 与 Proxyless-MBV2 配置会列出尚未验收的 blocker并明确拒绝正式训练。`--smoke` 只验证
合成数据上的构模和训练流水线，不解除协议 blocker。

`--acceptance-smoke` 与 `--smoke` 互斥，使用真实数据且只接受两种代码锁定模式：

- 全数据、至少正式 epoch 的 1% 且不超过完整 schedule；AutoFormer 500 epoch profile 的最低值为 5 epoch；
- 恰好 1% 确定性分层数据、完整 500 epoch schedule。

第二种模式按整个 split 的 `round(N × 0.01)` 计算精确目标条数，再用最大余数法分配类别配额；
同余数类别使用固定 seed 决定顺序。若目标条数小于类别数（例如 ImageNet-1k 的 50,000 条
validation 数据只取 500 条），数学上不可能覆盖每一类，工具不会通过“每类至少一条”把 1% 偷换成 2%。

它允许在 `formal_training_ready: false` 时验证候选 recipe，但不会将候选协议升级为正式协议。
batch size 和 input size 不能通过 CLI 改写，缺少 `--data-root` 时直接失败。例如：

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=GPU-UUID-0,GPU-UUID-1
torchrun --standalone --nproc-per-node=2 -m zcp_test.cli train \
  --config configs/training/autoformer_imagenet.yaml \
  --acceptance-smoke --epochs 5 --data-fraction 1.0 \
  --architecture /path/to/autoformer-architecture.json \
  --data-root /path/to/imagenet1k --output /path/to/runs/acceptance
```

另一模式必须保留 `--epochs 500 --data-fraction 0.01`。短程真实图片夹具可验证 DDP、中断和恢复
机制，但不得记录为 `full_data_one_percent_epochs`。当前已通过的 2-rank 夹具验收生成一个
`interrupted` run 和一个新目录 completed run；恢复后的 `training.jsonl` 连续包含 epoch 0–4，
manifest 的 `runtime.resume` 保存 checkpoint SHA-256 与 source run ID，且无残留 `.tmp`。由于尚未
在完整 ImageNet-1k 上执行上述两种协议，AutoFormer 正式门禁继续关闭。
checkpoint 同时嵌入截至保存 epoch 的小型 `training_history`；原 run 日志路径不可用（例如复制到
另一台机器）时，新 run 仍可恢复连续曲线，原 JSONL 存在时则优先读取原始记录。

启动 6/3 epoch 或完整 schedule 前，先对每个 profile 和候选运行一个完整数据 epoch：

```bash
zcp-test train --config configs/training/darts_cifar10.yaml \
  --real-data-preflight --epochs 1 --data-fraction 1.0 \
  --architecture ARCH.json --data-root DATA/cifar10 --output RUNS/preflight
```

该模式使用真实数据、正式 batch 和 reference 模型，但只标记为 `real_data_preflight`；它不能替代
`full_data_one_percent_epochs` 或 `one_percent_data_protocol`，也不能用于宣称精度复现。参数必须
严格为 1 epoch 与完整数据，避免把任意缩小任务包装成预检。`training.jsonl` 的逐 epoch 资源字段
包括 `train_duration_seconds`、`valid_duration_seconds`、train/validation samples/s、
`peak_memory_mb` 和 `peak_reserved_memory_mb`；由此估算后续墙钟和显存，而不是只看进程启动负载。
`report bundle RUN...` 在多训练 run 时写出带 `source_run` 的 `training.csv`，并用 validation top-1、
validation loss、epoch 耗时和峰值显存四个分面比较各 run；返回值分别给出
`score_row_count` 与 `training_row_count`，不再把只有训练数据的 bundle 误报为“0 行结果”。
训练-only bundle 不创建空 `scores.csv`；搜索-only 或 score-only 产物也按同样的实际需要生成。

AutoFormer 配置固定 AZ-NAS commit `5e6683a2cfa5c6d0dc34a1317a842497ba7eae47`。真实数据 loader
使用三次 repeated augmentation；学习率按
`base_lr × per_device_batch × world_size × accumulation / 512` 缩放，因此官方 8×256 启动的
有效 LR 是 `0.002`，不是 YAML 中作为基准值的 `0.0005`。Cream T/S/B 与 AZ-NAS
Tiny/Small/Base 已有精确参数量和 `official_complexity_ops` golden。独立 THOP 对 AZ-NAS Tiny
给出 `1,100,420,352` MAC，而官方口径为 `1,380,128,376`，且 THOP 未计全 relative-position
参数；两列必须分开报告，官方自定义 `get_complexity` 不能称为通用 FLOPs。
多 GPU 使用 `torchrun`，且必须由启动器按 UUID 固定可见卡；不要同时传 `--device`：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-UUID-0,GPU-UUID-1,GPU-UUID-2,GPU-UUID-3 \
torchrun --standalone --nproc-per-node=4 -m zcp_test.cli train \
  --config configs/training/autoformer_imagenet.yaml \
  --smoke --epochs 1 --batch-size 2 --output /path/to/runs/training
```

每个进程内部使用 `cuda:LOCAL_RANK`。训练 loader 使用分布式 repeated-augmentation sampler，
指标跨 rank 求和，只由 rank 0 写 `manifest.json`、`training.jsonl` 和 checkpoint。AutoFormer 的
`gradient_accumulation_steps: auto` 将目标 global batch 固定为 2048：4 卡×每卡 256 时累积 2 次，
8 卡时累积 1 次。当前真实 2 卡 DARTS/AutoFormer smoke 与真实图片夹具的中断恢复已通过；完整
ImageNet-1k 的双重 1% 协议尚未验收，因此 AutoFormer `formal_training_ready` 仍为 false。
resolved config 分别保存 Cream 静态模型 commit `b799630a29995163f282b15e2f38701160272fd1`
和 AZ-NAS 训练 recipe commit，禁止用一个模糊 `implementation_commit` 覆盖两者。
上例是可直接执行的 DDP 流水线 smoke；移除 `--smoke` 的完整数据命令会按设计被协议门禁拒绝，
不能把未来正式命令伪装成当前可运行示例。

`ofa_proxyless_mbv2` 的 architecture spec 使用官方 supernet 位置语义：`kernel_size` 和
`expand_ratio` 均固定 21 项，五个 `depth` 决定每个最大深度 4 stage 激活多少前缀 block，最后一个
stage 固定深度 1。发布 supernet 的 `width_mult` 是 1.3，`resolution` 为 128–224、步长 4。
旧版按 `sum(depth)` 保存的紧凑数组不会被静默重解释，必须保留在旧结果读取路径或显式转换。

官方 inherited supernet 是模型资产，不是 benchmark 标准答案。首次使用先显式自举：

```bash
zcp-test data bootstrap --root /path/to/data \
  --benchmarks ofa_proxyless_supernet --catalog /path/to/data/catalog.json --yes
zcp-test evaluate --space ofa_proxyless_mbv2 --weight-mode ofa_inherited --trusted \
  --catalog /path/to/data/catalog.json --classes 1000 --proxies params,naswot \
  --count 2 --input-source dataset --dataset imagenet1k --data-root /path/to/imagenet1k \
  --bn-recalibration-batches 20 --bn-recalibration-batch-size 64 \
  --input-size 224 --gpu auto --output /path/to/runs/evaluate
```

`--trusted` 只应对已由内置 SHA-256 验证的官方 checkpoint 使用。checkpoint 在一次命令中只加载
一次，各架构按 21 位位置选择 active channel，并应用官方学习到的 7→5→3 kernel transform。
`scores.jsonl`/`search.jsonl` 会记录 `weight_mode=inherited_supernet`、checkpoint SHA-256、激活位置
和 BN 校准状态。省略校准参数时结果标记 `bn_recalibration_required=true`、
`bn_recalibrated_batches=0`。启用后，CLI 从真实 dataset root 确定性无放回采样独立批次，并记录
全部 sample ID、transform、batch 数和 SHA-256 指纹；数据不足或缺失会失败，不回退随机输入。
当前实现使用 `zcp-test-deterministic-v1` 的 resize/center-crop 协议，并明确标记
`official_protocol_match=false`，因此它可用于可重复 ZCP 对比，但在完成官方 OFA 数据 provider
数值对照前不能宣称发布 inherited accuracy。显式 random-input smoke 只能验证导出和 ZCP 流水线。

`--architecture` 接受现有 JSON 文件，或者内联 JSON 对象。两种形式都可使用带顶层 `spec` 的
artifact，也可直接给出 spec；spec 必须与配置中的 space 匹配：

```json
{
  "spec": {
    "normal": [],
    "normal_concat": [2, 3, 4, 5],
    "reduce": [],
    "reduce_concat": [2, 3, 4, 5]
  }
}
```

示例只展示外层格式；DARTS 实际 genotype 必须包含完整合法 edge。不同搜索空间的 spec 不能互换。
内联形式例如 `--architecture '{"spec": {...}}'`，适合调试；正式实验推荐保存文件以便 manifest
追溯。
正式训练必须提供真实 `--data-root` 或 dataset catalog asset；恢复 checkpoint 时必须使用兼容的
architecture/config，并在 CLI 显式传入 `--trusted`。

## TransNAS-Bench-101 七任务模型与输入契约

TransNAS 的 `dataset` 参数实际选择 Taskonomy 任务，不能再统一解释为分类数据集。当前官方
PyTorch port 对应 commit `6d4231b`：

| task | model output | 正式 validation target | budget | 方向 |
|---|---|---|---:|---|
| `class_scene` | `[B,47]` | `valid_top1` | 25 | maximize |
| `class_object` | `[B,75]` | `valid_top1` | 25 | maximize |
| `room_layout` | `[B,9]` | `valid_loss` | 25 | minimize |
| `jigsaw` | `[B,1000]` | `valid_top1` | 10 | maximize |
| `segmentsemantic` | `[B,17,256,256]` | `valid_mIoU` | 30 | maximize |
| `normal` | `[B,3,256,256]` | `valid_ssim` | 30 | maximize |
| `autoencoder` | `[B,3,256,256]` | `valid_ssim` | 30 | maximize |

### 标准答案与真实输入是两组不同资产

`data bootstrap --benchmarks transnasbench101` 下载的是约 105 MB 的 tabular 标准答案，可用于
query；它不包含 Taskonomy 图像和标签。Taskonomy 数据受独立 EULA 约束，新访问必须走
[官方获取方式](https://docs.omnidata.vision/starter_dataset_download.html#Examples)，不得由本项目静默
下载或再分发。许可文本见
[StanfordVL/taskonomy data LICENSE](https://github.com/StanfordVL/taskonomy/blob/master/data/LICENSE)。

更重要的是，论文正式实验使用随机选择的 24 栋建筑、120K 图像（80K/20K/20K），但公开仓库和
发布资产没有给出可验证的 24-building split、最终训练配置或逐任务完整 transform。因此，用户提供
Taskonomy split 后得到的是 **真实数据 contract protocol**，不是已证明的 TransNAS benchmark
reference input。除非作者 split/config 另行取得并校验，正式 H1 输入协议必须保持 blocked。

用户依法取得 TransNAS 使用的 Taskonomy/5k 子集后，数据根目录应保留上游模板结构，例如
`building/{domain}/point_0_view_0_domain_{domain}.png`。官方 split JSON 的 `filename_list` 指向
数据根目录内的逐 building JSON；逐 building JSON 列出包含 `{domain}` 的相对模板。生成安全索引：

```bash
zcp-test data prepare-transnas-input \
  --data-root /path/to/taskonomy-transnas5k \
  --split-json /path/to/taskonomy-train-split.json \
  --split train --verify-files
```

预期输出为 `/path/to/taskonomy-transnas5k/transnas-inputs.json`。生成器拒绝绝对路径、`..`、逃逸
symlink、重复 sample ID、缺失 domain 文件和错误上游 commit。运行期只读取该 manifest；不会回退
CIFAR 或随机输入。跨机器复制整个数据根目录后，输入 fingerprint 保持稳定。

将七任务共享根目录注册一次：

```bash
zcp-test data register dataset_transnas_taskonomy /path/to/taskonomy-transnas5k \
  --version taskonomy-contract-v1 \
  --protocol licensed-external-taskonomy-manifest-v1 --trusted --replace
```

分类任务使用上游 final5k mask 得到 75/47 类硬标签；Jigsaw 使用上游 1000 个 permutation，生成
确定性的 `[B,9,3,64,64]`；其余任务读取真实回归或 dense target。评估变换标记为
`zcp-test-deterministic-evaluation`，并记录 `training_augmentation_match=false`，因此不冒充官方训练增强。

### 1% 抽样与运行

micro 是 4,096 个架构的有限全集，最低 1% 为 41；macro 是 3,256 个架构，最低 1% 为 33：

```bash
CATALOG=~/.config/zcp-test/data.json
AUDIT=/path/to/audit

zcp-test benchmark sample transnasbench101 --catalog "$CATALOG" \
  --version v10141024 --transnas-space micro --fraction 0.01 \
  --seed 2026 --shards 4 --output "$AUDIT/transnas-micro-1pct-seed2026.json"

zcp-test benchmark sample transnasbench101 --catalog "$CATALOG" \
  --version v10141024 --transnas-space macro --fraction 0.01 \
  --seed 2026 --shards 4 --output "$AUDIT/transnas-macro-1pct-seed2026.json"
```

架构 1% manifest 保存 `search_space_id`、micro/macro variant、原始标准答案 SHA-256 和转换文件
SHA-256；它与输入 split fidelity 相互独立。下面的 class-object micro **contract-input** 示例产生
`41 × 22 = 902` 行，每个“架构 × 代理”一行，但在缺少作者 24-building split/config 时不得标为
正式 TransNAS H1：

```bash
zcp-test evaluate --benchmark transnasbench101 \
  --catalog "$CATALOG" --benchmark-version v10141024 --transnas-space micro \
  --sample-manifest "$AUDIT/transnas-micro-1pct-seed2026.json" \
  --dataset class_object --target-split valid --target-metric valid_top1 \
  --epoch-budget 25 --target-direction maximize --metric-seed-reduction mean \
  --input-source dataset --batch-size 2 --input-size 256 --classes 75 \
  --proxies az_nas,er,er_conn,er_deg,er_dist,er_pr,flops,gradnorm,jacob_cov,meco,meco_opt,naswot,near,ntkt,params,swap,synflow,te_nas,ter,vkdnw,zen,zico \
  --seed 2026 --gpu auto --output "$AUDIT/runs/transnas-micro-class-object"
```

Jigsaw 必须改为 `--input-size 64`。当前标签依赖 ZCP 仅对 `class_scene`、`class_object` 和
`jigsaw` 启用；`room_layout`、`segmentsemantic`、`normal`、`autoencoder` 尚缺经上游配置证明的统一
ZCP loss 契约，相关调用明确写为 `unsupported`。这不是失败伪装，也不能从 coverage 分母中删除。
专属报告的 `score_coverage.csv` 同时统计 `ok/failed/unsupported/skipped` 和 finite/paired coverage。

逐 task、逐 space 生成报告，禁止跨不同 metric 平均：

```bash
zcp-test analyze benchmark --scores /path/to/effective/transnas-micro.jsonl \
  --benchmark transnasbench101 --view transfer --benchmark-variant micro \
  --output /path/to/reports/transnas-micro
```

七个 head 的官方参数量与参数 shape multiset 已对照一致，但这不等于真实任务数值复现。显式
`--input-source random` 只能作为消融，不能与真实 Taskonomy 输入合并。

## ViT-Bench-101 发布切片研究

ViT-Bench 与开放 AutoFormer 搜索必须分开：前者查询发布 GT，不重新训练候选；后者没有完整 tabular
真值，使用 validation-only 搜索并对选中候选做 scratch training。公开 ViT-Bench 的 AutoFormer
main、来源说明不足的 extension 与 PiT 永不合并，vanilla、KD、ImageNet inherited 也永不合并。

开放 AutoFormer 的 AZ-NAS 搜索必须使用独立的论文组件端口和群体聚合器。下例用于验收项目自身的
探索性进化控制器，不是上游 8,000 随机候选协议：

```bash
zcp-test search --space autoformer \
  --proxy az_nas_autoformer --aggregator az_nas_log_rank \
  --population 32 --generations 20 --elite-ratio 0.25 \
  --dataset imagenet1k --input-source random --batch-size 2 --input-size 224 \
  --classes 1000 --seed 20260731 --gpu auto \
  --output /path/to/runs/search/autoformer-aznas
```

`az_nas_autoformer` 固定上游 AZ-NAS commit `5e6683a`：每个 block 保存 attention 残差后和 MLP
残差后的 `[B,N,C]` token，计算谱熵 expressivity、相邻残差 Jacobian trainability，以及 Cream
`official_complexity_ops`。协方差仅对浮点误差产生的负特征值执行 `clamp_min(0)`，因此版本为
`aznas-5e6683-autoformer-stable-v1`，fidelity 为 `paper_formula_port_stabilized`，不是逐位一致声明。
聚合器对三个组件分别执行 `rankdata/n`、取 log 后求和；不允许把 `expressivity` 单独冒充 AZ-NAS
最终分数。旧 `az_nas portable-v1` 是 NASWOT/GradNorm/参数量近似，正式 search 默认拒绝；仅显式
`--allow-approximation` 可做探索性消融。

本项目保留自己的 mutation/crossover/elite 控制器，并在每代按全部已评估组件缓存重新排名；这复现
AZ-NAS 组件与 log-rank 组合，但不是上游 AutoFormer 候选控制器的逐行复刻。`search.jsonl` 每个候选
同时保存 `components` 和聚合 `score`，`search-state.json` 保存原始组件缓存并支持恢复。generation 0
行数为 `population + 1 summary`；后续每代新增 `population - elite_count` 个候选和一条 summary。模型
初始化与代理随机向量使用 `architecture-hash-v1`，由 search seed 和 canonical architecture ID 派生；
同 seed 的两次独立 GPU smoke 在去除耗时字段后逐行一致。

来源对齐的验收 cohort 使用 generation 0、每个 seed 随机评估 8,000 个候选。launcher 会在一张卡
装箱两个已通过显存 smoke 的进程、第二张卡运行一个进程；等待逐卡锁不会阻塞调用者，并且每 100 次
评估原子保存一次未完成初始 population：

```bash
export ZCP_PYTHON=/path/to/envs/zcp-test/bin/python
export ZCP_GPU_UUIDS=GPU-UUID-A,GPU-UUID-B
export ZCP_GPU_LOCK_TIMEOUT_SECONDS=7200
tools/acceptance/run-autoformer-aznas-random-8000.sh
```

需要脱离终端时，可用用户级 `systemd-run --user --unit=zcp-test-autoformer-aznas-8000 --collect ...`
托管同一 launcher。检查 `runs/acceptance/autoformer-aznas-random-8000/status.json` 和各 seed 最新的
`search-state.json`；`completed_generation=-1` 表示 generation 0 尚未结束但可恢复。重新执行 launcher
会跳过 completed seed，并对最新 incomplete state 自动传入 `--resume`。不得绕过其他进程的 GPU 锁，
也不得把该 cohort 称为上游控制器逐行复刻：公式与 8,000 候选规模来源对齐，但采样器和 artifact
系统仍是显式版本化的项目实现。

首次机器初始化：

```bash
CATALOG=~/.config/zcp-test/data.json
DATA=/path/to/data
zcp-test data bootstrap --root "$DATA" --benchmarks vitbench101 \
  --catalog "$CATALOG" --yes
zcp-test data checklist --root "$DATA" --catalog "$CATALOG" --json
```

完整状态应为 `state=ready/raw_state=ready/runtime_state=ready/runtime_integrity=verified`。若只有安全
JSONL，则是 `state=partial` 但 `operational_ready=true`；查询可继续，离线重转换不可继续。

三个 index-0 smoke：

```bash
zcp-test benchmark inspect vitbench101 --catalog "$CATALOG" \
  --slice-id autoformer_main --start 0 \
  --dataset cifar100 --split test --metric-name accuracy_vanilla
zcp-test benchmark inspect vitbench101 --catalog "$CATALOG" \
  --slice-id autoformer_ext --start 0 \
  --dataset cifar100 --split test --metric-name accuracy_kd
zcp-test benchmark inspect vitbench101 --catalog "$CATALOG" \
  --slice-id pit --start 0 \
  --dataset cifar100 --split test --metric-name accuracy_vanilla
```

公开 commit 的三个文件各只有 100 条，而论文声明每数据集 500 GT 并使用无重叠 60%/40%
proxy-development/test。公开文件未给出该划分身份，因此下面只叫 minimum-5 发布切片预验收：

```bash
AUDIT=/path/to/audit
for SLICE in autoformer_main autoformer_ext pit; do
  zcp-test benchmark sample vitbench101 --catalog "$CATALOG" \
    --slice-id "$SLICE" --count 5 --seed 2026 \
    --output "$AUDIT/sampling/vitbench-${SLICE}-minimum5-seed2026.json"
done
```

真实 CIFAR-100 路径只写机器 catalog：

```bash
zcp-test data register dataset_cifar100 /path/to/cifar100 \
  --version torchvision-cifar100 \
  --protocol train-split-published-labels --trusted --replace
```

main 单 seed 示例：

```bash
PROXIES=az_nas,er,er_conn,er_deg,er_dist,er_pr,flops,gradnorm,jacob_cov,meco,meco_opt,naswot,near,ntkt,params,swap,synflow,te_nas,ter,vkdnw,zen,zico
zcp-test evaluate --benchmark vitbench101 --slice-id autoformer_main \
  --catalog "$CATALOG" \
  --sample-manifest "$AUDIT/sampling/vitbench-autoformer_main-minimum5-seed2026.json" \
  --sample-shard 0 --dataset cifar100 \
  --target-metric accuracy_vanilla --target-split test \
  --proxies "$PROXIES" --seed 2026 \
  --input-source dataset --data-root /path/to/cifar100 \
  --batch-size 2 --input-size 224 --classes 100 --gpu auto \
  --output "$AUDIT/runs/vitbench-autoformer-main-preacceptance"
```

预期核心 22 个迁移代理产生 5×22=`110` 行：`az_nas_autoformer` 是开放 AutoFormer 专用的第 23 个
显式代理，不纳入旧 22-proxy 兼容 sweep。当前支持矩阵为 80 `ok`、30 `unsupported`、0 `failed`。CLI 分别打印
`succeeded/failed/unsupported/skipped/non_ok`。切换 extension 时目标使用 `accuracy_kd`；PiT 可用
`accuracy_vanilla` 或 `accuracy_kd`，不能查询 ImageNet inherited。

```bash
RUN=/path/to/timestamped/run
zcp-test analyze correlation --scores "$RUN/scores.jsonl" \
  --output "$AUDIT/reports/vit/correlation" --bootstrap-samples 200 --top-k 1 3 5
zcp-test analyze benchmark --scores "$RUN/scores.jsonl" \
  --benchmark vitbench101 --view architecture \
  --dataset cifar100 --target-split test --benchmark-variant autoformer_main \
  --output "$AUDIT/reports/vit/architecture"
zcp-test report bundle "$RUN" --output "$AUDIT/reports/vit/bundle"
```

5 个候选的相关性置信区间很宽，只证明执行链路。正式升级条件、资产 SHA 与本机典型结果见
`docs/evidence/VITBENCH_PREFLIGHT_CN.md`。

PiT 构模当前标记为 `reference_topology_pytorch_port`，不是 `reference_model`。它适用于 ZCP 构模和
结构敏感性研究，但不构成官方训练数值复现；固定候选的 ground truth 仍来自切片 JSONL，而不是当前
PyTorch 模型的训练结果。`benchmark inspect`/`evaluate` 从 catalog 解析运行资产时会校验文件 SHA、
version 与 protocol；校验失败会停止。显式 `--benchmark-path` 是高级信任边界，不会借 catalog 替调用者
证明来源，正式运行应优先使用已校验 catalog。

## Artifact 行数与最小 schema

所有 `--output` 均先视为父目录；会创建 run 的命令在其下生成北京时间目录
`YYYYMMDDTHHMMSS+0800_<run-id>/`。manifest、events、status 与隔离文件名统一使用
`Asia/Shanghai` 和显式 `+08:00`/`+0800`；终端 JSON 的 `run` 才是后续命令应使用的路径。
旧 `...Z_...` run 只读兼容，不原地改写。

| 命令 | 规范 artifact | 预期行数 | 最小科学身份 |
|---|---|---:|---|
| `evaluate` | `scores.jsonl` | `架构数 × 代理数` | architecture/benchmark/space、proxy/version/component/direction、dataset/input fingerprint、fidelity、status |
| `search` | `search.jsonl` | candidate：`population + generations × (population-elite_count)`；summary：`generations+1` | generation、candidate/parent/mutation、proxy、资源约束、累计预算、模型/输入协议 |
| `train` | `training.jsonl`、`events.jsonl` | training：每个实际完成 epoch 一行；events：约每 30 秒及 split 末尾一行 | epoch 曲线指标；rank 0 本地 batch heartbeat、ETA 与 epoch 完成事件 |
| `correlate` | 用户指定 JSONL | 每个实际有 canonical-ID join 的 proxy 一行 | component、score/target direction、paired/coverage、统计量 |
| `report bundle` | CSV/HTML/可选图表 | 由可用 artifact 和字段决定 | 源 run、协议分组和派生产物类型 |

`search` 的 generation summary 与 candidate 是不同 `record_type`，不能把总行数误当候选数。
报告只在输入满足统计或曲线要求时生成 PNG/SVG，不创建没有数据依据的空图。

## AutoFormer 与两类 MobileNetV2 双重 1% 验收

三个空间使用独立配置、候选目录和结果目录，不能互换 architecture ID 或训练 recipe：

| 空间 | 启动器 | 全数据协议 | 1% 数据协议 |
|---|---|---:|---:|
| AutoFormer scratch | `run-autoformer-imagenet-dual-one-percent.sh` | 5/500 epoch | 500 epoch |
| ZenNAS PlainNet-MBV2 | `run-plainnet-mbv2-imagenet-dual-one-percent.sh` | 2/150 epoch | 150 epoch |
| Proxyless-MBV2 scratch | `run-proxyless-mbv2-imagenet-dual-one-percent.sh` | 2/150 epoch | 150 epoch |

每个候选目录必须先冻结三个结构化 JSON 文件：`zcp_selected.json`、`fixed_random.json` 和
`params_flops_matched.json`。第一个必须来自记录完整输入协议和代理版本的 ZCP 搜索；第二个使用固定
seed；第三个从独立随机池中同时匹配参数量与 FLOPs。不得把官方发布 architecture、手选 architecture
或只匹配参数量的候选标成 `zcp_selected`/`params_flops_matched`。

候选冻结使用已完成的搜索 run，而不是直接传入一个任意 architecture 文件：

```bash
SEARCH_RUN=/path/to/timestamped/search-run
zcp-test acceptance freeze-candidates \
  --search-run "$SEARCH_RUN" \
  --training-config configs/training/autoformer_imagenet.yaml \
  --seed 20260731 --pool-size 32 \
  --output /path/to/frozen-candidates/autoformer
```

命令要求 search manifest 为 `completed`，且 `search_identity` 完整包含 space、proxy/version、dataset、
input fingerprint 和 seed；`best_architecture.json` 还必须真实出现在 `search.jsonl` candidate 记录中。
输出严格为每种角色一份 JSON 加 `candidates-manifest.json`，其中保存 search/config/JSONL SHA-256、
训练配置 SHA-256、架构 ID、资源协议和匹配距离。训练 CLI 只读取其中的 `spec`，其余 provenance 保持
只读审计。

MobileNet 使用同一模型实现上的 THOP MAC 作为计算量约定。AutoFormer 使用 Cream/AZ-NAS 官方
`get_complexity` 口径，并明确写入 `generic_flops=false`；它不能被重命名为通用 FLOPs。匹配距离为参数量
和该空间计算量的 log-ratio L1，因此只表示资源相近，不表示精度、延迟或训练成本完全相同。

四卡后台启动示例：

```bash
export TZ=Asia/Shanghai
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export ZCP_IMAGENET_ROOT=/path/to/imagenet1k
export ZCP_TRAINING_CANDIDATES=/path/to/frozen-candidates/autoformer
export ZCP_GPU_UUIDS=GPU-...,GPU-...,GPU-...,GPU-...
setsid -f env ZCP_START_AT=1 \
  bash tools/acceptance/run-autoformer-imagenet-dual-one-percent.sh
```

PlainNet 和 Proxyless 只替换候选目录与启动器。启动器会验证 ImageNet 的 1000 类、1,281,167 个训练
文件和 50,000 个验证文件，校验 config 的 space/epoch，使用 GPU UUID 文件锁，并在工作树不干净时
拒绝启动。状态位于 `runs/acceptance/<space>-imagenet/status.json`，所有新时间使用北京时间。中断后先
审计最近 run 的 manifest/checkpoint，再用 `ZCP_START_AT=2..6` 从尚未完成的任务恢复；不要重复已完成
候选，也不要把 interrupted 记作 completed。

通用启动器默认采用 `sequential_ddp`。若单卡显存 smoke 已证明该 profile 的原始 batch 可放入一张卡，
可显式启用按候选并行：

```bash
export ZCP_EXECUTION_STRATEGY=parallel_single_gpu
export ZCP_PARALLEL_SINGLE_GPU_ACCEPTED=yes
```

此模式以四条单卡 lane 运行六项任务，但不会覆盖 config 的 batch、梯度累积或 LR。第二个变量是人为
验收闸门，不是自动显存证明；未做真实模型 forward/backward smoke 时不得设置。AutoFormer、PlainNet
和 Proxyless 必须分别验收，不能因为 DARTS 单卡可运行就直接放行。

如果“两进程同卡”的真实 forward/backward smoke 也已通过，可进一步同时启动六项任务：

```bash
export ZCP_EXECUTION_STRATEGY=packed_single_gpu
export ZCP_PACKED_SINGLE_GPU_ACCEPTED=yes
export ZCP_DATA_WORKERS=4
export ZCP_CPU_AFFINITIES='32-63,96-127;32-63,96-127;32-63,96-127;32-63,96-127'
```

`packed_single_gpu` 在两张卡上各放置两个独立 run，其余两张卡各一个；它提高的是项目总吞吐，不改变
单个 run 的 batch/LR。必须先确认两进程峰值显存总和有安全余量，并用较少 workers 防止 CPU 解码争用。
`ZCP_CPU_AFFINITIES` 可选，四段依次对应四个 GPU UUID；应按 `nvidia-smi topo -m` 选择 GPU 所属 NUMA
节点，不得照抄本机 CPU 编号到其他机器，也不要未经测量就把一个 NUMA 节点机械切成过小的互斥分组。
本机 16 逻辑核/任务的试验使吞吐下降约 6–8%；共享完整 NUMA1 的短时观察也没有证明优于基线，
因此现场已完全回退为系统默认 affinity。亲和性只保留为可选实验参数，不作为推荐默认。若没有 smoke
证据，继续使用 `parallel_single_gpu`。

训练 loader 使用 `--workers`，验证 loader 可独立使用 `--valid-workers`。1% ImageNet 验证集仅有 4 个
batch，因此验收脚本默认使用 2 个验证 worker；这不改变样本、顺序、transform、batch 或 LR。训练配置
还支持显式性能键：`prefetch_factor`、`valid_prefetch_factor`、`pin_memory`、`persistent_workers`、
`valid_persistent_workers`、
`non_blocking_transfer`、`memory_format: channels_last`、`cudnn_benchmark` 和 `allow_tf32`。默认值保持旧
协议；`channels_last` 只应在 CNN profile 单独 smoke 后启用。`cudnn_benchmark: true` 与
`deterministic: true` 冲突并会直接报错；TF32/非确定性设置可能改变数值轨迹，必须形成新的版本化训练
协议，不能用于续跑旧 checkpoint 或与旧候选结果无标记混合。

六项顺序固定为三个候选的全数据最少 1% epoch，再运行三个候选的 1% 数据完整 schedule。每个 run
必须有持续增长的 `run.log`/`events.jsonl`、每 epoch 的 `training.jsonl`、`last.pt`、`best.pt` 与最终
manifest。该验收用于放行训练实现，不等于论文完整数据完整 schedule 精度复现。

DARTS ImageNet 的正式 global batch 为 128。四卡 DDP 会把它拆成每卡 32，在 4090/4090D 上只占约
1.8 GiB 且同步开销明显；不能为追求利用率擅自扩大科学 batch。首项已完成后，可使用
`resume-darts-imagenet-parallel-from-task2.sh`：它将 task2–6 分成四条独立单卡 lane，每个 run 仍使用
global batch 128，但并行不同候选/协议。三个 250-epoch 任务优先在三条 lane 启动，task2/3 共用第四条；
每条 lane 只持有并在结束时释放自己的 GPU 锁，已完成 lane 可立即被后续任务复用，不再等待最慢任务。
这提高总吞吐而不改变单个实验的 batch/LR 协议；结果仍需逐 run 验证，不能把并行完成顺序当科学顺序。
