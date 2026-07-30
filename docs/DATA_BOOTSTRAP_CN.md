# Benchmark 数据自举与离线迁移

本文只说明 benchmark 真值资产。CIFAR-10、CIFAR-100、ImageNet-1k 等训练数据是另一类
资产：请自行放到 `/path/to/data` 下，并显式传入对应的 `--data-root`。运行期数据加载器使用
`download=False`；`evaluate` 不会静默下载 benchmark，也不会静默下载训练集。

## 五步流程

一次部署应固定使用明确的数据根目录和 catalog。下列示例统一使用 `/path/to/data`；可以调整
其子目录，但 `checklist`、`bootstrap`、`export-manifest`、`import-manifest` 必须基于相同的
相对目录结构。

### 1. 下载前执行 checklist

```bash
zcp-test data checklist --root /path/to/data
zcp-test data checklist --root /path/to/data --json \
  > /path/to/data/checklist-before.json
```

表格是下载计划，不代表远端下载已经可用。JSON 还会给出原始路径、运行期路径、来源页面或
URL、规划大小、断点字节数、文件系统探测到的剩余空间和修复命令。

若使用非默认 catalog，应在 checklist、bootstrap、inspect 和 evaluate 中始终传入同一个
`--catalog /path/to/data/catalog.json`。`configs/benchmarks/*.yaml` 是可执行 evaluate 配置，
默认通过 `~/.config/zcp-test/data.json` 解析路径，不包含开发机绝对路径；例如：

```bash
zcp-test evaluate --config configs/benchmarks/nasbench201.yaml --trusted \
  --proxies params --count 1 --input-source random --device cpu
```

配置可以标识原生序列化 benchmark，但不能替操作者确认信任。原生 benchmark、checkpoint 和
pickle 加载都必须在独立核验来源与 checksum 后，于命令行显式传入 `--trusted`。

状态含义如下：

| 状态 | 精确定义 | 后续动作 |
|---|---|---|
| `missing` | 至少一个预期原始资产不存在，且未发现对应 `.part`。 | 自举该 benchmark，或把官方文件放到列出的原始路径。 |
| `partial` | 存在下载 `.part`，或转换型 benchmark 的安全 runtime 已可用但 raw 尚不完整。 | 检查 `operational_ready`；原样重跑同一 bootstrap 命令以续传或补齐 raw。 |
| `corrupt` | 带内置 SHA-256 的原始文件存在，但摘要不匹配。 | 隔离或删除损坏文件，再重新自举。 |
| `conversion_required` | 原始资产通过当前可用检查，但至少一个运行期路径不存在。 | 重跑 bootstrap，让它完成转换。 |
| `ready` | raw 与 runtime 均就绪；外部 native catalog 资产可同时充当 raw/runtime。 | 检查 `runtime_integrity` 与 `location`，正式评估前再做 benchmark smoke。 |

四个细分字段不能省略：`raw_state` 描述原始资产，`runtime_state` 描述 adapter 实际读取的格式，
`runtime_integrity=verified|unpinned|corrupt|missing` 描述 catalog 是否固定摘要，
`operational_ready` 只回答“当前能否初始化 adapter”。`operational_ready=true` 不会把总体
`partial` 自动升级成 `ready`。

### 2. 只自举实际需要的 benchmark

确认前先检查来源、预估大小、checksum 覆盖、上游条款和磁盘余量。交互运行会要求确认；
非交互自动化必须显式添加 `--yes`。

```bash
zcp-test data bootstrap \
  --root /path/to/data \
  --benchmarks nasbench101,nasbench201 \
  --catalog /path/to/data/catalog.json
```

```bash
zcp-test data bootstrap \
  --root /path/to/data \
  --benchmarks nasbench101,nasbench201 \
  --catalog /path/to/data/catalog.json \
  --yes
```

`--all` 会请求所有内置自举组；通常应优先逐个 benchmark 下载，以减少配额和磁盘风险：

```bash
zcp-test data bootstrap \
  --root /path/to/data \
  --all \
  --catalog /path/to/data/catalog.json \
  --yes
```

bootstrap 下载到 `.part`，通过重命名发布最终文件，按需转换格式，并把 ready 的运行期路径
注册到指定 catalog。JSON 返回 `"ok": true` 只表示该次调用通过已实现的检查；本文不声称
任何远端资产已在你的环境中成功下载。

### 3. 验证 ready 并执行 smoke

自举后重新检查：

```bash
zcp-test data checklist --root /path/to/data --json \
  > /path/to/data/checklist-after.json
```

`verify --all` 是严格的全量检查：只要任意一个内置 benchmark 组不是 `ready` 就会报错，
因此不要用它表达“只验证刚下载的一个 benchmark”。

```bash
zcp-test data verify --all --root /path/to/data
```

单个 catalog 条目的 `data verify` 检查路径存在性；若 catalog 中记录了 SHA-256，也会检查
摘要：

```bash
zcp-test data list --catalog /path/to/data/catalog.json
zcp-test data verify nasbench101 --catalog /path/to/data/catalog.json
```

catalog 验证不能替代 checklist 和查询 smoke。bootstrap 会为文件型 runtime（包括三个
ViT-Bench JSONL）登记 SHA-256；目录型 native runtime 仍可能显示 `runtime_integrity=unpinned`。
`benchmark inspect` 还会校验 JSONL 行内 benchmark/version/protocol/source SHA；原生序列化格式
仍必须显式传入 `--trusted`。index-0 最小查询 smoke 为：

```bash
CATALOG=/path/to/data/catalog.json
zcp-test benchmark inspect nasbench101 --catalog "$CATALOG" --dataset cifar10 --split valid --metric-name final_accuracy --epoch-budget 108
zcp-test benchmark inspect nasbench201 --trusted --catalog "$CATALOG" --dataset cifar10-valid --split valid --metric-name accuracy --epoch-budget 200
zcp-test benchmark inspect nats_tss --trusted --catalog "$CATALOG" --dataset cifar10-valid --split valid --metric-name accuracy --epoch-budget 200
zcp-test benchmark inspect nats_sss --trusted --catalog "$CATALOG" --dataset cifar10-valid --split valid --metric-name accuracy --epoch-budget 90
zcp-test benchmark inspect transnasbench101 --catalog "$CATALOG" --transnas-space micro --dataset class_object --split valid --metric-name valid_top1 --epoch-budget 25
zcp-test benchmark inspect transnasbench101 --catalog "$CATALOG" --transnas-space macro --dataset class_object --split valid --metric-name valid_top1 --epoch-budget 25
zcp-test benchmark inspect nasbench301_surrogate --trusted --catalog "$CATALOG" --dataset cifar10 --split test --metric-name accuracy
zcp-test benchmark inspect vitbench101 --catalog "$CATALOG" --slice-id autoformer_main --dataset cifar100 --split test --metric-name accuracy_vanilla
zcp-test benchmark inspect vitbench101 --catalog "$CATALOG" --slice-id pit --dataset cifar100 --split test --metric-name accuracy_vanilla

zcp-test evaluate \
  --benchmark nasbench101 \
  --catalog "$CATALOG" \
  --benchmark-version full \
  --proxies params \
  --count 1 \
  --input-source random \
  --device cpu \
  --output /path/to/data/smoke/nasbench101
```

只有每条命令退出码为零、adapter 身份/版本正确且 `query.value` 有限时，查询 smoke 才通过；
可选 evaluate 还必须产生非 failed 的 score 记录。这些步骤只验证接线，不代表完整 benchmark
的科研语义已被复核。

### 4. 导出离线迁移 manifest

`export-manifest` 不复制也不打包数据。它对选定 benchmark 的每个**运行期**文件或目录计算
摘要，记录相对于 root 的路径，并写入 manifest；原始下载文件和转换暂存状态不在其中。

```bash
zcp-test data export-manifest \
  --root /path/to/data/source \
  --benchmarks nasbench101,vitbench101 \
  --output /path/to/data/transfer/manifest.json
```

复制 manifest 和运行期数据，并保持相对于源 root 的目录结构。对于 NAS-Bench-101，必须复制
完整的 `converted/full` 目录，不能只复制 `manifest.json`：

```bash
mkdir -p /path/to/data/offline/nasbench101/converted
mkdir -p /path/to/data/offline/vitbench101
rsync -a \
  /path/to/data/source/nasbench101/converted/full \
  /path/to/data/offline/nasbench101/converted/
rsync -a \
  /path/to/data/source/vitbench101/converted \
  /path/to/data/offline/vitbench101/
```

通过获准的传输方式把 `/path/to/data/transfer/manifest.json` 和复制的数据移到离线机器。
manifest 只有摘要和相对路径，不包含 benchmark 数据本身。

`export-manifest` 会散列完整的 NB101 `converted/full` 目录，包括 `manifest.json`、
`hash-index.json`、`offsets.bin` 和全部 `architectures-*.jsonl` 分片；TransNAS-Bench-101 与
ViT-Bench-101 也按完整转换目录散列。迁移后仍需执行 adapter smoke，作为字节级目录摘要之外
的语义检查。

### 5. 通过验证完成 import

虽然命令名是 `import-manifest`，它既不复制文件，也不更新数据 catalog。它只验证 manifest
中的安全相对路径都位于目标 root 下，并比较导出时的摘要。

```bash
zcp-test data import-manifest \
  --root /path/to/data/offline \
  --manifest /path/to/data/transfer/manifest.json

zcp-test data checklist --root /path/to/data/offline
```

迁移后可显式传 `--benchmark-path`，也可以注册已验证的运行期路径：

```bash
zcp-test data register \
  nasbench101 \
  /path/to/data/offline/nasbench101/converted/full/manifest.json \
  --version full \
  --protocol official-tfrecord-converted \
  --catalog /path/to/data/offline/catalog.json
```

### 下载单个已注册 asset

`data fetch` 是底层单 asset 下载，不替代 bootstrap：

```bash
zcp-test data fetch ASSET_ID \
  --catalog /path/to/data/catalog.json \
  --destination /path/to/data/file
```

asset 必须声明 `source_url`。命令先写 `.part`，存在 catalog SHA-256 时核验，再原子发布目标。
它不会展开 benchmark 组、解压、转换原生数据、注册路径，也不提供 bootstrap 的断点续传流程；
没有 checksum 时，下载完成不能证明来源真实性。

## `ready` 保证什么、不保证什么

benchmark 组成为 `ready` 需要同时满足：

1. 每个内置原始资产的安装路径存在。如果该资产是文件且内置了 SHA-256，摘要必须匹配。
   没有固定摘要的原始资产只按“存在”通过；解压目录同样只按“存在”通过。
2. 该 benchmark 声明的每个运行期路径都存在且 catalog 校验通过。文件型 runtime 有登记摘要时
   必须匹配；目录型 runtime 可能只能标为 `unpinned`。

对于 NB201、NATS、NB301 等 native benchmark，机器本地 catalog 指向的外部 native 资产本身可
同时作为 raw/runtime；此时显示 `catalog_state=external_ready` 与 `location=catalog_external`，
不表示文件位于 `--root`。对于 NB101、TNB101、ViT-Bench 等转换型 benchmark，只有安全 runtime、
没有 raw 时总体是 `partial`，但 `operational_ready=true`，可继续只读查询。

checklist 不会为了声称 ready 而反序列化原生 `.pth`/pickle；那会破坏只读安全边界，也可能很
昂贵。因此 `ready` 后仍必须执行文档中的 adapter smoke。NB101 转换 manifest 与离线迁移
manifest 还会使用 SHA-256 绑定索引、offset 和全部分片。

状态优先级为 `corrupt`，然后是 `partial`/`missing`，再是 `conversion_required`，最后才是
`ready`。因此 `ready` 是精确的安装状态谓词，不是对上游真实性、许可合规、完整语义一致性
或论文结果可复现性的证明。必须保留来源，并执行符合目标协议的 smoke。

## 按 benchmark 下载

下表大小来自 checklist 的规划值，不是精确传输大小，也不是“磁盘够用”的保证。转换输出、
解压归档、保留的下载文件、`.part` 和临时数据库可能同时存在。

| Benchmark 组 | 版本与协议边界 | 规划大小 | 内置原始 SHA-256 | 来源与运行期结果 |
|---|---|---:|---|---|
| `nasbench101` | 官方 `full` TFRecord；4/12/36/108 epoch；转换为安全分片 JSONL | 2,085,986,016 B（约 1.94 GiB） | `3d64db8180fb1b0207212f9032205064312b6907a3bbc81eabea10db2f5c7e9c` | [Google NASBench](https://github.com/google-research/nasbench)；运行期 manifest 为 `/path/to/data/nasbench101/converted/full/manifest.json` |
| `nasbench201` | 原生 v1.1 `096897`；12/200 epoch 协议 | 4,700,000,000 B（约 4.38 GiB） | 未固定 | [NAS-Bench-201](https://github.com/D-X-Y/NAS-Bench-201)；原生 `.pth`，独立核验后查询时仍需显式 `--trusted` |
| `nats_tss` | 拓扑搜索空间，v1.0 `3ffb9`，12/200 epoch | 1,100,000,000 B（约 1.02 GiB） | 未固定 | [NATS-Bench](https://github.com/D-X-Y/NATS-Bench)；Google Drive tar，解压为原生 API 目录 |
| `nats_sss` | 尺寸搜索空间，v1.0 `50262`，12/90 epoch | 1,100,000,000 B（约 1.02 GiB） | 未固定 | [NATS-Bench](https://github.com/D-X-Y/NATS-Bench)；与 TSS 分离的 tar 和 API 目录 |
| `transnasbench101` | 官方 v10141024；micro 与 macro 是两个运行期表 | 105,000,000 B（约 100 MiB） | 未固定 | [上游 Drive 目录](https://drive.google.com/drive/folders/1HlLr2ihZX_ZuV3lJX_4i7q4w-ZBdhJ6o)；可信 `.pth` 转换为 `transnas_micro.jsonl` 和 `transnas_macro.jsonl` |
| `nasbench301_surrogate` | 官方 surrogate models v1.0；performance/runtime 分离 | 1,848,669,012 B（约 1.72 GiB） | `e807411d6a454841965d3157a977896683b716dc48743049bd6be0ce94210824` | [官方 Figshare](https://figshare.com/articles/software/nasbench301_models_v1_0_zip/13061510)；安全解压为 `xgb_v1.0` 与 `lgb_runtime_v1.0` |
| `vitbench101` | Auto-Prox commit `90ed458`；AutoFormer 主集、扩展集和 PiT 保持分离 | 62,925 B 规划值 | 三个固定摘要见下表 | [Auto-Prox 来源](https://github.com/lliai/Auto-Prox-AAAI24/tree/90ed458)；可信 `.pth` 转换为三个 JSONL 表 |

配额或磁盘紧张时逐个执行：

```bash
zcp-test data bootstrap --root /path/to/data --benchmarks nasbench101 --catalog /path/to/data/catalog.json --yes
zcp-test data bootstrap --root /path/to/data --benchmarks nasbench201 --catalog /path/to/data/catalog.json --yes
zcp-test data bootstrap --root /path/to/data --benchmarks nats_tss --catalog /path/to/data/catalog.json --yes
zcp-test data bootstrap --root /path/to/data --benchmarks nats_sss --catalog /path/to/data/catalog.json --yes
zcp-test data bootstrap --root /path/to/data --benchmarks transnasbench101 --catalog /path/to/data/catalog.json --yes
zcp-test data bootstrap --root /path/to/data --benchmarks nasbench301_surrogate --catalog /path/to/data/catalog.json --yes
zcp-test data bootstrap --root /path/to/data --benchmarks vitbench101 --catalog /path/to/data/catalog.json --yes
```

ViT-Bench-101 固定了以下源文件摘要：

| 切片 | SHA-256 | 指标协议规则 |
|---|---|---|
| `autoformer_main` | `712ad277546d9f7f565ce07885be7e0b98dcd8d0724fdd1120f595b517436eca` | AutoFormer 主切片，不与扩展切片合并。 |
| `autoformer_ext` | `05f5df6a41f338fb5f47eafebfc8758c75e451606856b278ccda1c60b26e7bca` | 扩展切片，保留独立身份。 |
| `pit` | `bdda89841d4105f99ab759e3243e7a2402929ba7a8430dac12a50256aa533bb2` | PiT 搜索空间。 |

vanilla、knowledge-distillation、inherited-supernet accuracy 是三种不同指标协议，不得合并为
同一个 target metric。

NAS-Bench-301 使用官方 Figshare v1.0 ensemble。performance 与 runtime 模型分别注册；默认
架构候选由标准 DARTS codec 按 seed 确定性采样，因此普通抽样 evaluate 不再要求伪造一个
“标准答案架构表”。如需分析固定候选集，仍可显式传入安全 JSONL `--architecture-path`。
`nb301` 依赖组固定使用兼容旧模型的 `ConfigSpace==0.4.21`，补齐官方包漏声明的推理依赖，且
不会安装 PyG；未使用的 GNN surrogate 训练模块不会被加载。

### 复用已有的外部数据目录

数据不必复制到 checklist 的 `--root`。可用 `data register` 把任意绝对路径写入机器本地 catalog；
当注册条目存在且有效时，checklist 显示 `state=ready`、`catalog_state=external_ready` 和
`location=catalog_external`。这表示运行时直接复用外部路径，不表示数据已经复制到 `--root`。
仓库配置只保留 `/path/to/data`，本机绝对路径只存在于 `~/.config/zcp-test/data.json`。

## 来源、checksum、信任与上游条款

- `checklist --json` 是当前安装版本内置 URL/来源页的权威视图；每次获取都应保存该输出。
- 缺少内置 checksum 表示“未固定”，不表示“校验通过”。添加 `--trusted` 前，应获取上游摘要
  或记录组织批准的本地摘要。
- `--trusted` 只允许加载 Python/PyTorch 原生序列化；它不核验来源、checksum、benchmark
  协议或安全性。
- bootstrap 不代替用户接受许可证，也不授予再分发权。下载或迁移前必须检查上游来源页、
  数据/仓库许可证、访问政策和引用要求；若条款缺失或含糊，应向上游确认，不能从“HTTP 可访问”
  推断权利。
- 实验来源记录至少应包含 URL、获取日期、精确字节数、SHA-256、benchmark 版本、split、
  epoch budget、seed reduction 和 metric protocol。

## Google Drive 配额与断点续传

NAS-Bench-201、两个 NATS 组和 TransNAS-Bench-101 使用 Google Drive URL。bootstrap 通过
`gdown` 开启 resume，并写入 `<destination>.part`；NATS tar 位于
`/path/to/data/.downloads`。

- 中断后原样重跑同一 bootstrap 命令，不要手工把 `.part` 改名为最终文件。
- Drive 报 quota exceeded 时，等待配额恢复，或通过来源页和获准的认证流程手工下载。把结果
  放到 checklist `raw_paths` 指定位置，再重跑 bootstrap 完成转换和注册。
- 工具不会绕过 Drive 配额、HTML 确认页或上游访问控制。
- 标准 HTTP 服务端若忽略 Range，下载器会重写 `.part`，不会把不兼容内容继续追加。
- 对没有固定 checksum 的资产，续传完成不能证明真实性；信任原生格式前仍需独立记录摘要。

## 损坏与磁盘空间恢复

1. 保存 `checklist --json`，检查 `state`、`raw_paths`、`runtime_paths`、`partial_bytes` 和
   `disk_probe`。
2. 对 `corrupt`，bootstrap 会先把坏文件重命名为 `.invalid-<timestamp>` 再重新下载；确认不再
   需要取证后可手工清理隔离文件。不要用 `--trusted` 掩盖损坏。
3. 对没有固定 checksum 的可疑资产，与上游或组织批准的摘要比对；不一致时删除最终文件和
   对应 `.part` 后重试。
4. 对 `partial`，若确属可信下载中断，应保留 `.part` 并续传；仅在来源错误或续传持续失败时
   删除它。
5. 对 `conversion_required`，保留原始源文件并重跑 bootstrap。NB101 转换会在摄取阶段使用
   临时 SQLite 状态，输出写完后才发布 manifest。

checklist 会报告剩余空间，但不会预留空间，也不会在开始前强制拒绝自举。应同时为原始文件、
`.part`、保留归档、解压目录、转换结果和临时状态预留空间。遇到 `ENOSPC` 时先停止其他写入，
释放空间但不要删除唯一已验证原始源，再重新 checklist 后续跑。

## NAS-Bench-101 专有安全接口

NAS-Bench-101 运行期不依赖 TensorFlow。其专用转换器会：

- 流式读取官方 TFRecord framing，默认验证 CRC-32C；
- 用项目内解析器读取 benchmark 的 `ModelMetrics` protobuf；
- 记录源文件 SHA-256 和大小；
- 摄取期间通过临时 SQLite 保存可重启 offset；
- 写出 `architectures-*.jsonl`、`hash-index.json`、定长 `offsets.bin`、每个分片摘要和原子
  manifest；
- 成功发布后删除转换状态。

通常应使用 bootstrap。受控转换或调试可调用 Python API：

```python
from zcp_test.data.nasbench101 import convert_nasbench101, read_indexed_record

manifest = convert_nasbench101(
    "/path/to/data/nasbench101/nasbench_full.tfrecord",
    "/path/to/data/nasbench101/converted/full",
    benchmark_version="full",
)
record = read_indexed_record(
    "/path/to/data/nasbench101/converted/full",
    "<official-module-hash>",
)
```

不要为了让损坏 TFRecord 转换成功而关闭 CRC。完整转换后仍应运行前文 adapter smoke。
2026-07-30 的发布验收已完成前文全部 adapter 的真实 index-0 查询；每台新机器仍必须用自己的
catalog 重跑，因为该结果不能替代对迁移文件或未来依赖栈的验证。

## 为什么 `evaluate` 不隐式自举

benchmark 评估先解析已经存在的显式 `--benchmark-path`，再查找匹配的 catalog 条目。两者都
不存在时会抛出 `FileNotFoundError`，并给出 checklist/bootstrap 提示；它不会调用 bootstrap。
`ZCP_DATA_ROOT` 只参与构造该修复提示，不是隐式 benchmark 路径解析器。

这样可以避免评估任务意外消耗网络配额、写满共享磁盘、替用户接受上游条款、加载刚下载的
可信格式，或在一次运行中改变数据版本。数据准备与验证必须是单独、可审计的步骤。

运行时通过 catalog 解析 benchmark 时，不只检查路径存在：文件型资产还会重新计算 SHA-256，并核对
该 benchmark/slice 的预期 `version` 与 `protocol`。任一不一致都会停止，而不是继续读取已篡改或错配的
JSONL。`data checklist` 的 `runtime_integrity=verified` 表示这三项当前一致；目录型上游 native 资产若
没有锁定目录摘要则仍显示 `unpinned`，不能解释成上游 checksum 已被证明。

## OFA-Proxyless supernet 模型资产

`ofa_proxyless_supernet` 是开放搜索空间的官方模型资产，**不是 benchmark 标准答案**。现有
`--benchmarks` 参数名为兼容早期 CLI 保留，实际也接受此类受版本约束的数据组：

```bash
zcp-test data bootstrap --root /path/to/data \
  --benchmarks ofa_proxyless_supernet --catalog /path/to/data/catalog.json --yes
zcp-test data checklist --root /path/to/data --catalog /path/to/data/catalog.json
```

资产来自 Once-for-All commit `f03b267` 的发布 URL，大小 `32,202,338` bytes，SHA-256 为
`10ce40eec63dd020b4fa0096b1ff3c1e81e5b740446ddef6a59651bb36e6b907`。运行期只在 CLI 显式
`--trusted --weight-mode ofa_inherited` 时以 PyTorch `weights_only=True` 读取；普通 evaluate 不会
隐式下载。跨机器复制后应重新 checklist/bootstrap 以注册本机绝对路径。
