# zcp-test

`zcp-test` 是独立的零成本代理评估与神经架构搜索项目，不修改 `TER-Score`、`Auto-Prox-AAAI24` 或其他参考仓库。

## 设计边界

- `benchmark_id` 与 `search_space_id` 分离。NAS-Bench-201 和 NATS-TSS 仅共享拓扑编码，不共享 API 或真值。
- NATS-TSS 与 NATS-SSS 使用不同数据目录、API 类型和完整训练预算。
- 运行期结果以 JSONL 为真源，CSV、HTML 和图表均可重新生成。
- 大型 benchmark 和训练数据不复制进仓库，由数据注册器记录路径、版本、协议和校验值。
- 任意 pickle、`.pth` benchmark 或 checkpoint 均不得隐式加载；旧数据转换必须显式声明 `--trusted`。
- 原生序列化 benchmark 查询和 checkpoint 恢复也必须显式添加 `--trusted`；该参数仅表示操作者已核验来源，不替代 SHA-256 校验。

## 安装

```bash
conda env create -f environment.yml
conda activate zcp-test
zcp-test doctor --catalog configs/data.example.json
```

环境只声明直接依赖，不使用冗余的 `pip freeze`。默认不安装 TensorFlow、torchaudio、PyG、Jupyter、ROCm Triton 等无关包。

## 使用

详细手册：

- [Benchmark 数据自举与离线迁移](docs/DATA_BOOTSTRAP_CN.md)
- [Benchmark 定制研究：预算、拓扑、size、任务迁移与 ViT 结构](docs/BENCHMARK_STUDIES_CN.md)
- [高引 ZCP 工作证据与本项目推广边界](docs/RESEARCH_EVIDENCE_CN.md)
- [保留的可复现研究实例](examples/studies/README_CN.md)
- [GPU 选择](docs/GPU_CN.md)
- [ZCP 评估与结果行数](docs/EVALUATE_CN.md)
- [新增代理](docs/ADD_PROXY_CN.md)
- [分析与监控](docs/ANALYSIS_CN.md)
- [训练协议、恢复与正式门禁](docs/TRAINING_CN.md)
- [配置优先级、RUN 目录与运维边界](docs/OPERATIONS_CN.md)
- [DARTS 搜索与训练](docs/DARTS_TRAINING_CN.md)
- [1% Benchmark 相关性验收](docs/ONE_PERCENT_ACCEPTANCE_CN.md)

```bash
zcp-test data list --catalog configs/data.example.json
zcp-test benchmark list
zcp-test benchmark inspect nasbench201 --trusted --catalog /path/to/data/catalog.json
zcp-test space inspect autoformer
zcp-test proxy inspect er

zcp-test gpu list
zcp-test evaluate --space nb201_topology --proxies er,naswot,synflow --count 10 --data-root /path/to/data/cifar10
zcp-test search --space darts --proxy er --population 20 --generations 5 --data-root /path/to/data/cifar10
ARCH=/path/to/darts-architecture.json
OUTPUT=/path/to/data/runs/training
zcp-test train --config configs/training/darts_cifar10.yaml --smoke \
  --architecture "$ARCH" --output "$OUTPUT"
# 从上一命令输出 JSON 复制准确的 timestamp run；仅恢复已核验的本项目 checkpoint。
RUN="$OUTPUT/YYYYMMDDTHHMMSS+0800_runid"
zcp-test train --config configs/training/darts_cifar10.yaml --smoke \
  --architecture "$ARCH" --output "$OUTPUT" \
  --resume "$RUN/checkpoints/last.pt" --trusted
```

ViT-Bench-101 的 AutoFormer 主切片、扩展切片与 PiT 分开转换和报告；vanilla、KD 与 inherited-supernet accuracy 不混合。NB301 默认使用 `with_noise=False`。

## 首次使用：显式准备 benchmark 数据

`evaluate` 不会隐式下载 benchmark 或训练数据。数据准备应作为独立、可审计的流程执行。
checklist 现在分别报告 `raw_state`、`runtime_state`、`runtime_integrity` 与
`operational_ready`。`state=ready` 要求所需原始资产和运行格式均就绪；若转换后的安全运行格式可用、
但原始文件缺失，则是 `state=partial, operational_ready=true`，表示可以查询但不能离线重建。
native benchmark 的外部 catalog 路径本身就是 raw/runtime 时会显示 `location=catalog_external`，
不表示文件已复制到 `--root`。只读 checklist 不会反序列化原生 pickle/PyTorch 文件，正式研究前
仍需执行 adapter smoke。
`runtime_integrity=verified` 对文件表示文件 SHA-256 匹配，对目录表示确定性的目录树摘要匹配；后者
是首次可信 bootstrap 在本机锁定的摘要，不应误称为上游公布 checksum。已有 catalog 条目为
`unpinned` 时，可重新运行同一 `data bootstrap --benchmarks ... --yes`：工具保留有效外部路径、不
重下载 ready 资产，只补写摘要，然后再次 checklist。

```bash
# 1. 表格查看来源、大小、目标路径和断点文件；磁盘余量字段见 --json。
zcp-test data checklist --root /path/to/data

# 2. 只下载并转换实际需要的 benchmark。
zcp-test data bootstrap \
  --root /path/to/data \
  --benchmarks nasbench101 \
  --catalog /path/to/data/catalog.json \
  --yes

# 3. 再次核验安装状态，并执行对应 benchmark smoke。
zcp-test data checklist --root /path/to/data --json \
  > /path/to/data/checklist-after.json
zcp-test benchmark inspect nasbench101 \
  --path /path/to/data/nasbench101/converted/full/manifest.json \
  --version full

# 4. 离线迁移前对运行期数据生成摘要 manifest。
zcp-test data export-manifest \
  --root /path/to/data \
  --benchmarks nasbench101 \
  --output /path/to/data/transfer/manifest.json

# 5. 在目标机器上验证按相同相对路径复制的数据。
zcp-test data import-manifest \
  --root /path/to/data/offline \
  --manifest /path/to/data/transfer/manifest.json
```

`export-manifest` 不复制数据；`import-manifest` 只验证迁移后的目录，不复制文件也不注册 catalog。
详细指南列出了各 benchmark 来源、规划大小、已固定和缺失的 checksum、协议边界、Google Drive
配额与断点、损坏和磁盘恢复、离线迁移以及 NAS-Bench-101 专有安全接口。本文不声称任何未在
用户环境中实际验证的下载已经成功。

### NATS-SSS 的 CIFAR-100 与 ImageNet16-120 输入

NATS-SSS benchmark 标准答案与 ZCP 输入数据是两类独立资产。NATS-SSS 原生 API 仍需
`--trusted`；ImageNet16 原始 batch 本身也是 pickle，不能直接交给运行期。先在可信机器上核对
官方 MD5，并显式转换为只读 `.npy` shards：

```bash
CATALOG=/path/to/data/catalog.json
zcp-test data convert-imagenet16 \
  --source /path/to/raw/ImageNet16 \
  --output /path/to/data/datasets/ImageNet16-120-safe \
  --trusted --register --catalog "$CATALOG"
zcp-test data verify dataset_imagenet16_120 --catalog "$CATALOG"
```

正式 `evaluate` 使用 `--dataset ImageNet16-120 --input-source dataset --catalog "$CATALOG"`；
不要再把 raw pickle 目录传给 `--data-root`。CIFAR-100 与 ImageNet16-120 的 dataset-specific ZCP
必须分别计算，因为输入、归一化、类别数和 `input_fingerprint` 都不同。只替换 benchmark target、
复用原 ZCP 的 target-only transfer 是另一协议。当前正式分析使用三数据集共 12 个 score 分片，
由 `analyze benchmark --benchmark nats_sss --view size` 分别生成 dataset-specific 矩阵和
target-only/controlled transfer 表；不得把两类数字混写。CIFAR-100 与 ImageNet16-120 各自
7,216/7,216 行成功、重复键 0。完整命令、安全边界和结果见[运维手册](docs/OPERATIONS_CN.md)、
[1% 验收](docs/ONE_PERCENT_ACCEPTANCE_CN.md)与
[跨数据集证据](docs/evidence/NATS_SSS_CROSS_DATASET_CN.md)。

## 正式训练

PlainNet-MBV2 的正式 AZ-NAS 搜索不能使用 generic `population × generations` 示例代替。固定入口为：

```bash
zcp-test search --config configs/search/plainnet_mbv2_source_aligned.yaml \
  --flops-target 450m --gpu auto \
  --output /path/to/runs/search/plainnet-aznas-450m
```

该入口强制 100,000 个有效候选、population 1024、batch 64/224、四组件全历史 log-rank 和无
crossover。正式运行前先按[操作手册](docs/OPERATIONS_CN.md#az-nas-plainnet-mbv2-搜索)执行 GPU
preflight 和 CPU rerank 估时；preflight 保留 100k 身份但在 3 个候选后保持 `running`，不得称为
完成搜索。本机 CPU rerank 保守估计约 4.21 小时，GPU 单候选耗时尚待排队预验收。

无标准答案搜索空间在正式训练前，先从一个 `completed` search run 冻结三类候选：

```bash
zcp-test acceptance freeze-candidates \
  --search-run /path/to/timestamped/search-run \
  --training-config configs/training/autoformer_imagenet.yaml \
  --output /path/to/frozen-candidates/autoformer
```

该命令拒绝缺少 proxy/version、输入指纹或 search JSONL 证据的手工候选。详细 provenance、资源
匹配语义、双重 1% 启动与恢复见[操作手册](docs/OPERATIONS_CN.md)。

- AutoFormer：AZ-NAS Tiny/Small 为 500 epoch、Base 为 300 epoch；基础 LR `5e-4` 按
  `per_device_batch × world_size / 512` 线性缩放（官方 8×256 时有效 LR `0.002`），AdamW、
  weight decay `0.05`、cosine、20 epoch warmup。当前已接入 repeated-augmentation sampler 和
  六个 Cream/AZ-NAS 官方子网参数量 golden。真实 2 卡混合 4090D/4090 DDP smoke 已验证共享 run、
  跨 rank 指标归约和仅 rank 0 写 artifact。六个官方 `get_complexity` golden 已按原命名保存，
  并与 THOP MAC 并列证明二者口径不同。真实 ImageNet 图片极小夹具上的 2-rank 中断/新 run 恢复
  已验证 `interrupted` manifest、checkpoint lineage、epoch 去重和 `.tmp` 清理；这只证明恢复机制，
  不等于全 ImageNet 验收。正式训练仍因两类 1% ImageNet 协议未完成而保持关闭。
- OFA/Proxyless MobileNetV2：ImageNet-1k、150 epoch、SGD/Nesterov、`0.05`、weight decay `4e-5`、label smoothing `0.1`。
- DARTS：提供 CIFAR-10 600 epoch profile。
- DARTS：同时提供 CIFAR-100 600 epoch 适配和 ImageNet-1k 250 epoch 官方评估 profile。

### DARTS 双重 1% 状态（2026-07-31）

DARTS CIFAR-10/100 已对 ER 搜索候选、固定随机候选和参数匹配随机池候选三类冻结架构完成两套
限定验收：全数据 × 6 epoch 共 6 runs，以及恰好 1% 数据 × 600 epoch 共 6 runs。确定性真实数据
预检、两次可信 checkpoint 恢复审计（每套协议各一次）和证据报告均已完成。完整结果、候选身份与
校验摘要见 [中文证据报告](docs/evidence/DARTS_CIFAR_DUAL_ONE_PERCENT_CN.md) 和
[机器可读摘要](docs/evidence/darts_cifar_dual_one_percent_summary.json)。

该结论只关闭 DARTS CIFAR 的双重 1% 工程与限定协议验收：它**不是** 600 epoch 全数据精度复现，
也**不是**多 seed 搜索收益证明。全部训练只使用 seed `20260731`，候选选择也只使用一个固定输入
batch 和一个初始化 seed。两套协议分别衡量早期全数据学习与小数据完整 schedule，候选排序不同，
不得对两套结果求平均后宣称 ER 稳定优于基线。本机 ImageNet-1k 已完成结构审计（1000 类、
1,281,167 张训练图、50,000 张验证图）和真实 loader 解码检查；DARTS ImageNet 六项双重 1%
已启动但尚未完成，因此状态是“运行中”，不是“已通过”。首轮发现低速盘小文件 I/O 与旧版空
`run.log` 问题；当前实现会把事件同时 flush 到 `events.jsonl` 和 `run.log`，重启时应通过
`--data-root` 显式选择经 `findmnt` 核验的高速本地副本。AutoFormer、PlainNet-MBV2 和
Proxyless-MBV2 的双重 1% 验收均未完成。

`--smoke` 只使用合成数据验证流水线。`--acceptance-smoke` 使用真实数据，并只允许“全数据且至少
1% epoch”或“恰好 1% 数据且完整 schedule”；它仍不解除 `formal_training_ready` 门禁，也不代表
论文精度复现。`--real-data-preflight --epochs 1 --data-fraction 1.0` 只用于在启动高成本任务前
测量一个完整数据 epoch 的吞吐、显存和流水线完整性，结果固定标记为 `real_data_preflight`，不得
计入双重 1% 验收。详见 [操作手册](docs/OPERATIONS_CN.md)。

catalog 中的 benchmark 路径在实际查询前会再次核对文件 SHA、version 和 protocol；错配会明确失败。
显式 `--benchmark-path` 不经过 catalog 完整性证明，只应在调用者已经独立核验来源时使用。

项目统一使用北京时间 `Asia/Shanghai`：新 run 目录为 `YYYYMMDDTHHMMSS+0800_<run-id>`，manifest、
events、status 和隔离文件名均携带 `+08:00`/`+0800`。历史 `...Z_...` run 保持只读兼容，不原地
改写；报告展示历史记录时应按其原始 offset 解析并转换为北京时间。

## 当前边界

- 评估命令支持 `--start/--count` 范围切分；多 GPU 目前采用每张卡独立启动一个范围并在结束后合并 JSONL，尚未内置多进程调度器。
- 已通过机器本地 catalog 完成 NAS-Bench-101、NAS-Bench-201、NATS-TSS/SSS、TransNAS
  micro/macro、NAS-Bench-301 performance surrogate、ViT-Bench AutoFormer/PiT 的真实 index-0
  查询；其他机器必须使用 `data bootstrap` 或本地 catalog 注册路径，仓库配置不保存本机路径。
- TransNAS 的 tabular 标准答案与 Taskonomy 输入是不同资产。原始/转换标准答案和 41/33 个架构的
  1% micro/macro manifest 已锁定；安全七任务 contract input provider 已实现。论文所用
  24-building/120K split 与最终 config 未公开，Taskonomy 数据还受独立 EULA 约束且本机尚未取得，
  因此正式真实输入 22-ZCP sweep 仍为 blocked，不能用任意 Taskonomy split、random/CIFAR fixture
  冒充。见 [TransNAS 预检证据](docs/evidence/TRANSNAS_PREFLIGHT_CN.md)。
- OFA MobileNetV3 保持可选 adapter；本次验收不把它的现代环境兼容性作为其他搜索空间的阻塞条件。
- DARTS 正式 profile 已放行，且 CIFAR-10/100 双重 1% 限定验收已完成；这不等于 600 epoch
  全数据精度复现。ImageNet-1k 已通过结构与 loader 预检，DARTS ImageNet 双重 1% 正在执行但
  尚未通过。大型图像训练应显式选择已核验的本机高速 `--data-root`，项目不硬编码机器路径。
  AutoFormer、PlainNet-MBV2 与
  Proxyless-MBV2 配置仍是可审计的候选 recipe，其双重 1% 均未完成，不得把
  `--smoke` 写成正式训练验收。`torchrun` 必须显式提供按 UUID 排列的 `CUDA_VISIBLE_DEVICES`；
  CLI 使用 `cuda:LOCAL_RANK`、DDP、分布式 sampler/指标归约和 rank-zero artifacts。AutoFormer
  可将梯度累积自动调整到 global batch 2048，因此 4 卡×256 使用 2 次累积。
