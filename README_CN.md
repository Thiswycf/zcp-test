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
- [GPU 选择](docs/GPU_CN.md)
- [ZCP 评估与结果行数](docs/EVALUATE_CN.md)
- [新增代理](docs/ADD_PROXY_CN.md)
- [分析与监控](docs/ANALYSIS_CN.md)
- [DARTS 搜索与训练](docs/DARTS_TRAINING_CN.md)

```bash
zcp-test data list --catalog configs/data.example.json
zcp-test benchmark list
zcp-test benchmark inspect nasbench201
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
`ready` 的精确定义是：每个预期原始路径通过当前可用的 checksum/存在性检查，且每个声明的
运行期路径存在；`catalog_state` 单独报告。只读 checklist 不会反序列化原生 pickle/PyTorch
文件，因此正式研究前仍需执行文档中的 adapter smoke。

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

- AutoFormer：ImageNet-1k、224、300 epoch、AdamW、`5e-4`、weight decay `0.05`、cosine、20 epoch warmup。
- OFA/Proxyless MobileNetV2：ImageNet-1k、150 epoch、SGD/Nesterov、`0.05`、weight decay `4e-5`、label smoothing `0.1`。
- DARTS：提供 CIFAR-10 600 epoch profile。
- DARTS：同时提供 CIFAR-100 600 epoch 适配和 ImageNet-1k 250 epoch 官方评估 profile。

`--smoke` 只用于验证模型、优化器、AMP、JSONL 和 checkpoint 流程，不代表正式实验结果。

## 当前边界

- 评估命令支持 `--start/--count` 范围切分；多 GPU 目前采用每张卡独立启动一个范围并在结束后合并 JSONL，尚未内置多进程调度器。
- NAS-Bench-101 的下载、转换和 adapter 已提供，但本机没有官方 TFRecord，因此未执行真实 NB101 集成 smoke。
- OFA MobileNetV3 保持可选 adapter；本次验收不把它的现代环境兼容性作为其他搜索空间的阻塞条件。
- 150/300/600 epoch 正式训练配置已提供，本次仅执行短程 GPU smoke 与 checkpoint 恢复。
