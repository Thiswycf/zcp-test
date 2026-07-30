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
zcp-test train --config configs/training/darts_cifar10.yaml --epochs 1 --smoke
# 仅恢复已核验的本项目 checkpoint
zcp-test train --config configs/training/darts_cifar10.yaml --resume /path/to/data/runs/training/checkpoints/last.pt --trusted
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

```bash
# 1. 查看来源、大小、目标路径、断点文件和磁盘余量。
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

## 正式训练

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

`--smoke` 只使用合成数据验证流水线。`--acceptance-smoke` 使用真实数据，并只允许“全数据且不超过
1% epoch”或“至多 1% 数据且完整 schedule”；它仍不解除 `formal_training_ready` 门禁，也不代表
论文精度复现。详见 [操作手册](docs/OPERATIONS_CN.md)。

catalog 中的 benchmark 路径在实际查询前会再次核对文件 SHA、version 和 protocol；错配会明确失败。
显式 `--benchmark-path` 不经过 catalog 完整性证明，只应在调用者已经独立核验来源时使用。

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
- DARTS 正式 profile 已放行；AutoFormer 与 MobileNet 配置仍是可审计的候选 recipe，不得把
  `--smoke` 写成正式训练验收。`torchrun` 必须显式提供按 UUID 排列的 `CUDA_VISIBLE_DEVICES`；
  CLI 使用 `cuda:LOCAL_RANK`、DDP、分布式 sampler/指标归约和 rank-zero artifacts。AutoFormer
  可将梯度累积自动调整到 global batch 2048，因此 4 卡×256 使用 2 次累积。
