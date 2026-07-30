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

解析顺序为：CLI 默认值 → 匹配的配置值 → 命令行显式参数。当前版本应使用 `--count 20`，不要
写成 `--count=20`，以确保显式覆盖检测生效。解析器不存在的配置键会被忽略；正式使用前必须检查
run 目录中的 resolved `config.yaml`。YAML 中的 `trusted: true` 不能替代命令行 `--trusted`。

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

`--output` 是父目录；实际运行目录为 `<output>/YYYYMMDDTHHMMSSZ_<run-id>/`。后续报告、监控和
恢复必须使用命令输出 JSON 中的准确 `run` 值：

```bash
RUN=/path/to/runs/evaluate/YYYYMMDDTHHMMSSZ_runid
zcp-test report bundle "$RUN" --output "$RUN/reports/bundle"
zcp-test monitor "$RUN" --interval 5
```

`report bundle` 和 `monitor` 都不会递归搜索父目录下的 timestamp run。

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

## Proxy scaffold

`zcp-test proxy scaffold NAME` 仅适用于可写源码 checkout 或 editable install。它会同时写入
`src/zcp_test/proxies/custom/NAME.py` 和 `tests/test_proxy_NAME.py`；普通只读 wheel/site-packages
不是支持目标。`proxy validate` 只在小型合成模型上检查有限值、权重隔离和 hook 清理，不代表完成
全 benchmark 科学验收。

## 训练架构文件与 fidelity

模型结构 fidelity 与正式训练协议是两个独立条件。`darts`、`autoformer`、
`ofa_proxyless_mbv2` 和 `zennas_plainnet_mbv2` 可以拥有 `reference_model` 结构，但只有配置中
`formal_training_ready: true` 的协议才能启动非 smoke 训练。当前正式放行的是 DARTS profiles；
AutoFormer 与 Proxyless-MBV2 配置会列出尚未验收的 blocker 并明确拒绝正式训练。`--smoke` 只验证
合成数据上的构模和训练流水线，不解除协议 blocker。

`--acceptance-smoke` 与 `--smoke` 互斥，使用真实数据且只接受两种代码锁定模式：

- 全数据、最多正式 epoch 的 1%；AutoFormer 500 epoch profile 即最多 5 epoch；
- 最多 1% 确定性分层数据、完整 500 epoch schedule。

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
机制，但不得记录为 `full_data_one_percent_epoch_protocol`。当前已通过的 2-rank 夹具验收生成一个
`interrupted` run 和一个新目录 completed run；恢复后的 `training.jsonl` 连续包含 epoch 0–4，
manifest 的 `runtime.resume` 保存 checkpoint SHA-256 与 source run ID，且无残留 `.tmp`。由于尚未
在完整 ImageNet-1k 上执行上述两种协议，AutoFormer 正式门禁继续关闭。
checkpoint 同时嵌入截至保存 epoch 的小型 `training_history`；原 run 日志路径不可用（例如复制到
另一台机器）时，新 run 仍可恢复连续曲线，原 JSONL 存在时则优先读取原始记录。

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

预期 5×22=`110` 行：当前支持矩阵为 80 `ok`、30 `unsupported`、0 `failed`。CLI 分别打印
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
