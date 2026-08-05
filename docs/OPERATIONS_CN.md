# CLI 运维与安全边界

## CLI 命令索引

当前公开 CLI 共 40 个叶子/兼容入口。先用 `zcp-test --help` 查看一级命令，再用
`zcp-test COMMAND --help` 或 `zcp-test GROUP ACTION --help` 查看参数；以下名称可直接复制：

```text
doctor
gpu list
data list | register | verify | fetch | checklist | bootstrap
data export-manifest | import-manifest | convert-vit | convert-imagenet16 | prepare-transnas-input
benchmark list | inspect | sample
space list | inspect
proxy list | matrix | inspect | validate | scaffold
evaluate | correlate | search | train | report | monitor
report bundle
analyze correlation | compare | sensitivity | search | training | benchmark
acceptance freeze-candidates | reconcile-search-cohort | validate-plainnet-search
legacy import
```

推荐发现链：`benchmark list → benchmark inspect → benchmark sample/evaluate`；
`space list → space inspect → evaluate/search`；
`proxy list/matrix → proxy inspect/validate → evaluate`。`report` 是兼容单文件入口，
`report bundle` 才是多 run CSV/PNG/SVG/HTML 汇总入口。

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

锁文件存在不代表锁仍被持有，判定必须以操作系统 `flock` 为准，禁止仅凭文件名或旧 PID 文本删除
锁文件。Python 锁会在正常释放时清空 owner 文本；验收 launcher 只在实际 GPU 任务期间持锁：
四卡 DDP 每个任务单独获取并释放四锁，单卡并行也按科学任务获取并在该任务结束时立即释放。同一
lane 上的下一个串行任务必须重新竞争锁，不能由 lane/supervisor 跨任务续持。supervisor
做数据校验、候选复制、报告整理或等待其他 lane 时不得预占空闲 GPU。锁 holder 在启动训练子进程前
关闭任务侧继承的锁 FD；Python `fork` 子进程也会关闭继承副本，避免 `tee`、DataLoader worker 或
孤儿后代在 GPU 工作结束后继续持锁。

高成本验收 launcher 会在启动时要求工作树干净，并用 `git archive` 将启动 commit 的完整已跟踪源码
固化到该 run 的 `launcher-snapshots/`，写入 launcher SHA-256、整体设为只读后再 `exec`。长任务运行
期间可以继续修改主仓；已启动 supervisor 不再从被修改的原脚本继续读取，后启动的 lane 也只从固定
commit 导入 Python 与 config。结构化 `supervisor.log` 分别记录 task/packed-scope 持锁和释放时间、
child wait 状态及失败时的 `BASH_COMMAND`。这用于避免“GPU 已空闲但旧 supervisor 因控制流漂移仍
持锁”的低效情形；不得通过删除锁文件替代正常退出或 `flock` 探测。

### 诊断真实 `flock` owner

锁路径长期存在是正常现象，文件内容中的 PID 也只能作为提示。先用非阻塞 `flock` 探测内核锁；
探测成功表示该瞬间可获取，命令退出后会自动释放，不需要也不允许删除文件：

```bash
LOCK_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/zcp-test/gpu-locks"
for lock in "$LOCK_DIR"/*.lock; do
  [[ -e "$lock" ]] || continue
  if flock -n "$lock" -c true; then
    printf 'FREE  %s\n' "$lock"
  else
    printf 'HELD  %s\n' "$lock"
    lslocks -o PID,PPID,COMMAND,TYPE,MODE,PATH | grep -F -- "$lock" || true
    fuser -v "$lock" 2>/dev/null || true
  fi
done
```

`lslocks` 显示内核记录的 lock owner；`fuser`/`lsof` 只表示进程打开了文件，应作为定位候选而不是
单独的持锁结论。诊断到 PID 后核对 supervisor 和子进程树：

```bash
OWNER_PID=<pid-from-lslocks>
ps -o pid,ppid,stat,lstart,etime,cmd -p "$OWNER_PID"
pstree -aps "$OWNER_PID"
pgrep -a -P "$OWNER_PID" || true
```

只有实际 task（或同时运行多个任务的 packed scope）可以临时持锁。低 GPU 利用率本身不能证明任务
空闲，因为数据加载、验证、保存
checkpoint 时 GPU 可能短暂空闲；应同时检查 child、`run.log`、`events.jsonl`、manifest 和 GPU
进程。task/packed scope 结束后必须立即释放。若旧 supervisor 已无活跃训练 child，却仍持有内核锁，应先
通过对应 user service 或 PID 发送 `TERM`，等待正常清理，再重复非阻塞探测：

```bash
systemctl --user stop <unit>          # 有对应 user unit 时优先
# 或：kill -TERM "$OWNER_PID"
flock -n /path/to/GPU-UUID.lock -c 'echo lock-released'
```

禁止 `rm *.lock`“伪解锁”：删除 pathname 不会释放旧 inode 上的 kernel lock，反而可能让另一个
进程创建同名新文件并同时进入临界区。若仍有活跃 child，不得仅为释放 GPU 而终止 supervisor；
若没有活跃 child，则不应让 supervisor 继续预占空闲卡。所有判定都具有竞争窗口，执行终止操作前
应再次核对 PID、child 和 run identity。

### 单候选政策下的旧 supervisor 清理

单候选政策生效后，旧的 immutable supervisor 中尚处于 `queued`、且角色为 fixed-random 或
params/FLOPs-matched baseline 的任务不再获得启动授权，必须取消；已经启动的 `zcp-selected` 任务仍须
继续。现场处置证据见
`docs/evidence/autoformer_single_candidate_policy_intervention_20260804.json`。安全顺序是：

1. 先从候选 manifest、run manifest、launcher/status 和完整进程树识别每个 PID 的 candidate role，不能
   只凭命令行片段、GPU 编号或低利用率判断角色。
2. 仅终止“旧 supervisor 将自动启动、但新政策未授权”的 queued baseline 及其专属 lane holder；若它已
   启动，正常发送 `TERM`、等待训练子树退出，并把产物保留为 `interrupted`，不得伪写 `completed`。
3. 不得误杀 `zcp-selected` 训练树，也不得删除、替换 GPU 锁文件。若 selected 与旧 supervisor 共树，先
   隔离/等待 selected 进入 terminal manifest，再停止会继续派发 baseline 的旧 service。
4. 处置后同时验证 `nvidia-smi` 中显存/计算进程已符合预期，并用 `flock -n` 验证对应 kernel flock；只有
   未授权 baseline 使用的锁应释放，selected 正在执行时持锁是正确行为。

进程终止、manifest 状态、显存释放和 kernel flock 是四项独立证据，缺一项都不能宣布清理完成。

## 数据输入与结果类型

`evaluate` 和 `search` 默认 `--input-source dataset`，必须提供 `--data-root` 或有效的
`dataset_<name>` catalog asset。真实数据缺失时直接失败，不会自动改用随机输入。
这里的 `--data-root` 指向所选**训练数据集自身根目录**，不是 benchmark 资产总目录：CIFAR-10
应传 `/path/to/data/cifar10`，ImageNet 应传 `/path/to/data/ImageNet1k`；标准答案文件由
`--benchmark-path` 或 catalog 独立解析。

- **standard answer**：带 dataset/split/budget/seed 协议的 benchmark 发布记录；
- **surrogate**：例如 NAS-Bench-301 的模型预测，不是完整训练观测；
- **inherited**：使用 supernet 权重评价的 subnet 指标；
- **scratch**：架构独立从头训练得到的指标。

四类结果不得混合，NAS-Bench-201 真值也不能替代 NATS-TSS 真值。

代理协议身份还必须记录 `--proxy-batches`、`--proxy-repetitions` 和 `--score-selector`。
AZ-NAS 正式多组件排序使用 `aggregate:az_nas_log_rank`；`component:NAME` 只用于显式消融；
`te_nas` 只暴露 RN 减 NTK condition 的主标量。

正式 sweep 前先测代表性 pilot，再按 600 秒上限规划：

```bash
zcp-test acceptance plan-feasibility \
  --total-architectures TOTAL --pilot-architectures PILOT_N \
  --pilot-seconds PILOT_SECONDS --max-seconds 600
```

规划器依次尝试 1%、1‰、1‱，必要时降为 1 个架构，并始终返回 `coverage_claim=false`。
它只证明时间可行性，不证明覆盖。旧固定 1% 运行及 `docs/evidence/**` 保持只读，属于已撤销版本证据。

## RUN 目录

对于 `evaluate`、`search`、`train` 等创建 run 的命令，`--output` 是父目录；实际运行目录为
`<output>/YYYYMMDDTHHMMSS+0800_<run-id>/`。后续报告、监控和
恢复必须使用命令输出 JSON 中的准确 `run` 值：

```bash
RUN=/path/to/runs/evaluate/YYYYMMDDTHHMMSS+0800_runid
zcp-test report bundle "$RUN" --output "$RUN/reports/bundle"
zcp-test monitor "$RUN" --interval 5
```

`report bundle` 会把没有直接 artifact 的父目录展开一层，并处理其中全部可识别 timestamp run；
`monitor` 仅在父目录恰好包含一个可识别 run 时自动进入。父目录有多个 run 时必须传入准确 `RUN`。
也可直接传 `scores.jsonl`、`search.jsonl`、`events.jsonl` 或 `training.jsonl`；未指定 `--output` 时，
HTML 始终写入解析后 JSONL 所属 run 的 `reports/monitor.html`。自定义 `--output FILE.html` 表示精确
HTML 文件，其他值按目录处理并追加 `monitor.html`。`--once` 向 stdout 输出一个 JSON 对象；持续模式
每次刷新输出一个独立 JSON 对象流，整体不是单个 JSON 文档。稳定字段为 `source`、`output`、
`row_count`、`new_row_count`、`next_offset`、`ignored_partial_line`。

保留的单文件兼容入口不创建 bundle，且其 `--output` 是**精确文件路径**，不是父目录：

```bash
zcp-test report --source "$RUN/scores.jsonl" \
  --format csv --output "$RUN/reports/scores.csv"
zcp-test report --source "$RUN/training.jsonl" \
  --format plot --kind training --output "$RUN/reports/training.png"
```

`--format` 支持 `csv|html|plot`，默认 `csv`；`--kind training|search` 只对 `plot` 生效。成功 stdout
固定为 `{"rows": N, "output": "..."}`。CSV/HTML 可处理空 JSONL，plot 对空输入明确失败。该入口可
读取旧 run，但新研究汇总优先使用 `report bundle`。
训练监控优先读取 `events.jsonl`：rank 0 默认约每 30 秒写入一次
`training_batch_progress`，每个 train/valid split 的最后一个 batch 也会写入；epoch 完成后写入
`training_epoch_completed`。`training.jsonl` 仍严格保持每个完成 epoch 一行，训练曲线只从该文件重建。
事件中的 `rank_local_samples` 是 rank 0 的本地计数，不是分布式全局精确样本数。旧 run 不会补写
heartbeat；若其 epoch 尚未结束，monitor 只能显示已有 artifact。
同一事件还会写入并即时 flush 到人类可读的 `run.log`；新 run 不应再出现“`events.jsonl` 有事件而
`run.log` 长期为 0 字节”。大型图像训练应把 `--data-root` 指向调用者已核验的本机高速盘副本，
不要根据目录名猜测介质速度；先用 `findmnt -T /path/to/imagenet1k` 确认挂载，再核对类别和文件数。
CLI 不会硬编码或自动改写数据根。

## 查询类命令的 JSON 契约

`doctor`、`gpu list`、各 registry 的 `list|inspect` 和 `proxy matrix` 默认直接输出缩进 UTF-8 JSON，
不需要额外 `--json`。为兼容现有脚本，当前保持裸对象/数组，不增加外层 envelope：

| 命令 | 顶层类型与稳定字段 | 顺序/可选字段 |
|---|---|---|
| `doctor` | object：`python,python_supported,platform,packages,torch` | `--catalog` 增加 `data_catalog`；`--data-root` 增加 `benchmark_data` |
| `benchmark|space|proxy list` | 按 ID 排序的 string array | 新插件可增加元素 |
| `data list` | `DataAsset` object array | 按 `asset_id` 排序；字段为 `asset_id,path,version,sha256,source_url,protocol,trusted` |
| `gpu list` | GPU object array | PCI Bus ID 顺序；含物理 `index,uuid,bus_id,model`、显存/利用率、`pci_order,visible_logical_index,zcp_test_lock` |
| `benchmark inspect` | object：`metadata,capabilities` | 提供 `--metric-name` 时增加单架构 `query` |
| `space inspect` | object：`search_space_id,model_family,model_fidelity,sample` | `sample` 由 `--seed` 决定 |
| `proxy inspect` | capability object | 字段集合与 `proxy matrix` 每行相同 |
| `proxy matrix` | capability object array | 按 `proxy_id` 排序；只是静态声明，不是运行时 sweep |

`proxy` capability 的稳定字段详见 [新增代理](ADD_PROXY_CN.md)。新增稳定字段需要同步版本化文档和测试，
不能再直接把 dataclass `__dict__` 暴露为公共接口。命令失败时退出码非零；除 `proxy validate` 明确先输出
诊断 JSON 外，调用者不得假定失败 stdout 是完整 JSON。

当前 `search --resume /path/to/search-state.json` 支持从版本化状态恢复。状态包含 population、history、
组件缓存、累计 evaluations/cache hits、已完成 generation、RNG state 和完整 search identity；恢复时
space、proxy/version、aggregator、dataset、输入指纹、seed、population 等科学协议必须完全一致。
恢复会创建新的 run 并保存来源状态，不向旧 `search.jsonl` 直接追加；只有 manifest、state 和 JSONL
计数一致的 terminal run 才能称为完成。训练恢复则使用同一架构、配置和协议身份，并显式传入可信
`last.pt`。专用 PlainNet source-aligned controller 在完成独立状态 schema 验收前不得借用 generic
`search-state.json` 宣称上游搜索可恢复。

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
score/target schema；使用前必须检查转换后的 JSONL，且不要覆盖源文件。导入是一次性 fail-closed
转换：`--output` 不存在或是空文件时才可写；若目标已含任何内容必须失败，绝不能向非空 JSONL 追加，
也不能把旧行和本次从零编号的行混在一起。每条输出都有从 `0` 开始、连续且唯一的 `legacy_index`：
list 的预期行数为 `len(list)`，mapping 为 `len(mapping)`，其他对象固定为 `1`。命令返回的转换条数、
JSONL 实际行数和末行 `legacy_index + 1` 必须三者相等；失败或重试前先保留并审计已有目标，不得直接追加。

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

四个 shard 应分别启动，下面只展示 `--sample-shard 0`。旧 manifest 与旧代理集合只能用于只读复核；
新运行使用当前 23 ID，并重新生成协议身份，不能把旧行数解释为当前覆盖。

```bash
PROXIES=ac,az_nas,az_nas_autoformer,az_nas_plainnet,dss,er,flops,gradnorm,hc,hi,jacob_cov,meco,meco_opt,naswot,near,params,swap,synflow,te_nas,ter,vkdnw,zen,zico
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
`formal_training_ready: true` 的协议才能启动非 smoke 训练。当前正式放行的是 DARTS profiles、
AutoFormer AZ-NAS scratch profile 与版本化的 Proxyless-MBV2 candidate-resolution scratch profile；
PlainNet-MBV2 配置仍列出 blocker 并明确拒绝正式训练。`--smoke` 只验证
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
8 卡时累积 1 次。当前真实 2 卡 DARTS/AutoFormer smoke、真实图片夹具的中断恢复，以及 AutoFormer
单候选的全数据 5 epoch 与 1% 数据 500 epoch 验收均已通过；因此 AutoFormer
`formal_training_ready` 已显式置为 true。
resolved config 分别保存 Cream 静态模型 commit `b799630a29995163f282b15e2f38701160272fd1`
和 AZ-NAS 训练 recipe commit，禁止用一个模糊 `implementation_commit` 覆盖两者。
上例仍只是可直接执行的 DDP 流水线 smoke。要启动正式 500-epoch scratch 训练，必须移除 `--smoke`、
提供冻结的 AutoFormer 架构和真实 ImageNet 根目录，并保留 profile 的 batch/LR/增强字段：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-UUID-0,GPU-UUID-1,GPU-UUID-2,GPU-UUID-3 \
torchrun --standalone --nproc-per-node=4 -m zcp_test.cli train \
  --config configs/training/autoformer_imagenet.yaml \
  --architecture /path/to/zcp_selected.json \
  --data-root /path/to/imagenet1k \
  --output /path/to/runs/training/autoformer-formal
```

该命令现在可通过门禁，但仍是高成本启动命令；只有 terminal 500-epoch 全数据产物才能形成论文精度证据。

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
`41 × 23 = 943` 行，每个“架构 × 代理”一行，但在缺少作者 24-building split/config 时不得标为
正式 TransNAS H1：

```bash
zcp-test evaluate --benchmark transnasbench101 \
  --catalog "$CATALOG" --benchmark-version v10141024 --transnas-space micro \
  --sample-manifest "$AUDIT/transnas-micro-1pct-seed2026.json" \
  --dataset class_object --target-split valid --target-metric valid_top1 \
  --epoch-budget 25 --target-direction maximize --metric-seed-reduction mean \
  --input-source dataset --batch-size 2 --input-size 256 --classes 75 \
  --proxies ac,az_nas,az_nas_autoformer,az_nas_plainnet,dss,er,flops,gradnorm,hc,hi,jacob_cov,meco,meco_opt,naswot,near,params,swap,synflow,te_nas,ter,vkdnw,zen,zico \
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

通用进化控制器可在调用 ZCP 前施加显式资源上限，例如：

```bash
zcp-test search --space ofa_proxyless_mbv2 --proxy naswot \
  --population 100 --generations 500 --elite-ratio 0.25 \
  --max-parameters 10000000 --max-macs 600000000 \
  --constraint-max-attempts 1000 \
  --dataset imagenet1k --input-source dataset --data-root /path/to/imagenet1k \
  --gpu auto --output /path/to/runs/search/proxyless-resource-bounded
```

`--max-macs` 当前严格表示 THOP MAC（`compute_metric=thop_macs`），不是 FLOPs，也不是官方 lookup
table 或设备 latency；不匹配该口径时直接失败。被资源约束拒绝的候选不会运行 proxy、不会进入缓存，
恢复状态会保存 attempt/rejection 计数与 RNG，保证中断恢复和连续运行轨迹一致。`--max-parameters`
统计为构造出的完整模型上所有已注册 parameter 的 `numel()` 总和，不只统计可训练参数、backbone 或
当前激活路径；`--classes` 参与分类头构模，候选 `resolution`（没有时用 `--input-size`）参与资源模型的
输入协议，因此改动 classes/resolution 后必须重新测量，不能复用旧上限判定。

`--constraint-max-attempts N` 是“为生成下一个可接受候选而允许的连续拒绝上限”，不是整次 search 的
总采样预算；每接受一个候选后，下一候选重新获得至多 `N` 次机会，第 `N` 次连续拒绝后 fail-closed。
启用资源约束时，成功候选行和 generation summary 写入
`cumulative_constraint_attempts`/`cumulative_constraint_rejections`，`search-state.json` 写入
`constraint_attempts`/`constraint_rejections`，数值均为全 run 累计；拒绝项本身不产生 candidate 行。
中断会保留已经 fsync 的 `search.jsonl`、manifest 和最近一次原子 checkpoint state（初始 population
默认每 100 个已接受候选保存一次），因此最后一个 state 之后尚未 checkpoint 的拒绝计数可能需要重放，
不能据残留行数宣称 search 完成。该上限防止可行域为空时无限采样。PlainNet source-aligned controller 使用其固定
`--flops-target`，禁止与这些通用上限混用。硬件 latency 约束尚未实现，不能用 MAC 冒充。

### OFA Proxyless 协议边界

不要把 OFA tutorial 的 20-block、5-stage、`{160,176,192,208,224}` 域套到 Proxyless-MBV2。
固定 commit `f03b267` 的该 tutorial 实际加载 OFA-MobileNetV3；官方 Proxyless supernet 则包含 21 个
可配置 `ks/e` 位置和 6 个 block group，前 5 组 depth 为 `{2,3,4}`，最后一组 depth 固定为 1，
发布域为 width 1.3、resolution 128–224。项目因此保留 `ofa_proxyless_mbv2` 的 21-position 编码，
不注册伪 `ofa_proxyless_official_tutorial`。

模型/active-subnet 域有官方来源，不代表 ZCP 搜索协议也是官方的。使用项目通用进化控制器时，
`search_identity` 和每条 `search.jsonl` 会自动记录：ZCP 目标为
`experiment_kind=project_zcp_transfer`，`params`/`flops` 资源目标为
`project_resource_baseline`，两者均为 `controller_fidelity=project_controller_not_ofa_tutorial`、
`direct_search_protocol_evidence=false`。不得把这些结果写成官方 OFA ZCP search。完整来源见
[`evidence/ofa_proxyless_protocol_boundary_20260804.json`](evidence/ofa_proxyless_protocol_boundary_20260804.json)。

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

保留的 `3×8,000` cohort 是单候选/1% 政策生效前的历史搜索证据，不再定义当前工程验收预算。当前
搜索工程验收必须约为预声明参考预算的 1%，并记录参考预算、实际评估数、预算比例和截断控制器
fidelity。下面的旧 launcher 只用于只读审计或显式恢复历史产物，不得作为新验收工作负载启动。
该 launcher 会在一张卡
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

三个 seed 都完成后，先确认 cohort 根目录已有可解析的 `status.json`，其中预声明
`primary_selection_seed=20260731`、`supporting_robustness_seeds=[20260732,20260733]`；再为每个 seed
选择且只选择一个最新的 completed run 目录。每个 run 必须有 terminal-complete `manifest.json`、
`best_architecture.json`、`search.jsonl` 和 `search-state.json`，且 generation 0、population 8,000、
组件恰为 `expressivity,trainability,complexity`。可按实际时间戳替换三个 `RUN_*`：

```bash
COHORT=runs/acceptance/autoformer-aznas-random-8000
RUN_20260731="$COHORT/seed-20260731/<completed-run-directory>"
RUN_20260732="$COHORT/seed-20260732/<completed-run-directory>"
RUN_20260733="$COHORT/seed-20260733/<completed-run-directory>"

zcp-test acceptance reconcile-search-cohort \
  --cohort-root "$COHORT" \
  --search-run "$RUN_20260731" \
  --search-run "$RUN_20260732" \
  --search-run "$RUN_20260733" \
  --expected-space autoformer \
  --expected-population 8000 \
  --expected-seed 20260731 \
  --expected-seed 20260732 \
  --expected-seed 20260733 \
  --expected-components expressivity,trainability,complexity
```

`--search-run` 与 `--expected-seed` 都是可重复参数；run 的实际 seed 集必须与 expected seed 集完全相等，
seed 不得重复。预期 candidate 行总数是 `population × seeds = 8000 × 3 = 24000`，每个 run 另有且仅有
一条 generation summary。reconcile 还核验 search space、completed generation 0、population/state、
有限 score 与精确组件集合、重复 architecture ID 一致性、cache/evaluation/summary 计数、best selection
及跨 seed 搜索协议一致性。

成功后原子写 `$COHORT/cohort-validation.json`，其 schema 为：顶层
`schema_version="1.0"`、`validated_at`、`status="completed"`、`search_space_id`、
`expected_population_per_seed`、`expected_components[]`、`primary_selection_seed`、
`supporting_robustness_seeds[]`、`candidate_rows_total`、`unique_evaluations_total`、`cache_hits_total`、
`runs[]`、`supervisor_status_before_reconciliation`、`supervisor_detail_before_reconciliation`；每个 `runs[]`
项包含 `seed`、`run`、`run_id`、`ended_at`、`candidate_rows`、`unique_architectures`、
`unique_evaluations`、`cache_hits`、`summary_rows`、`best_architecture_id`、`best_selection`、
`search_manifest_sha256`、`search_config_sha256`、`search_jsonl_sha256`、`search_state_sha256` 和
`search_identity`。随后单独原子更新 `status.json` 为 `status="completed"`，写入
`detail`、`ended_at`、`updated_at`、`reconciled_at`、validation 路径/SHA-256 和总计数，同时把原 supervisor
终态保存在 `supervisor_terminal_status/detail`，即使 launcher 先前因 wait 误报 failed 也不丢失证据。

重复执行会重新完整校验并安全替换这两个 JSON，不追加 candidate、语义幂等；因 `validated_at` 更新，文件
字节和 SHA-256 不保证不变。任何输入校验失败时两个文件都不改。两个原子替换不是跨文件事务：若
`cohort-validation.json` 已替换后 `status.json` 写入失败，validation 可能是新的而 status 仍旧；此时先
核对 validation 内容与 SHA-256，再用完全相同参数重跑，不得手工把 status 改成 completed。

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
PROXIES=ac,az_nas,az_nas_autoformer,az_nas_plainnet,dss,er,flops,gradnorm,hc,hi,jacob_cov,meco,meco_opt,naswot,near,params,swap,synflow,te_nas,ter,vkdnw,zen,zico
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

这里保留的旧行数与支持矩阵只对应已撤销 proxy 版本，不能用于当前 23 ID。新运行必须按模型族能力
生成 `ok/failed/unsupported/skipped/non_ok`，不得预设覆盖数量。切换 extension 时目标使用 `accuracy_kd`；PiT 可用
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

创建 run 的命令才把 `--output` 视为父目录，并在其下生成北京时间目录
`YYYYMMDDTHHMMSS+0800_<run-id>/`。manifest、events、status 与隔离文件名统一使用
`Asia/Shanghai` 和显式 `+08:00`/`+0800`；终端 JSON 的 `run` 才是后续命令应使用的路径。
`report --source` 的输出是精确文件，`monitor --output FILE.html` 也是精确文件；派生工具的其余规则以
各命令章节为准。旧 `...Z_...` run 只读兼容，不原地改写。

| 命令 | 规范 artifact | 预期行数 | 最小科学身份 |
|---|---|---:|---|
| `evaluate` | `scores.jsonl` | `架构数 × 代理数` | architecture/benchmark/space、proxy/version/component/direction、dataset/input fingerprint、fidelity、status |
| `search` | `search.jsonl` | candidate：`population + generations × (population-elite_count)`；summary：`generations+1` | generation、candidate/parent/mutation、proxy、资源约束、累计预算、模型/输入协议 |
| `train` | `training.jsonl`、`events.jsonl` | training：每个实际完成 epoch 一行；events：约每 30 秒及 split 末尾一行 | epoch 曲线指标；rank 0 本地 batch heartbeat、ETA 与 epoch 完成事件 |
| `correlate` | 用户指定 JSONL | 每个“完整 score protocol × proxy”且有 canonical-ID join 的组合一行 | protocol digest、benchmark/version/dataset/budget/input/seed/fidelity、component、direction、paired/coverage、统计量 |
| `report bundle` | CSV/HTML/可选图表 | 由可用 artifact 和字段决定 | 源 run、协议分组和派生产物类型 |

`search` 的 generation summary 与 candidate 是不同 `record_type`，不能把总行数误当候选数。
报告只在输入满足统计或曲线要求时生成 PNG/SVG，不创建没有数据依据的空图。

## AZ-NAS PlainNet-MBV2 搜索

`az_nas_plainnet` 只定义于 `zennas_plainnet_mbv2`，不能用于 OFA/Proxyless。它移植固定上游 MBV2
公式的 expressivity、progressivity、trainability 和 `MasterNet.get_FLOPs()` complexity。上游完整
AZ-NAS 协议固定为 100,000 个有效候选、parent 参数 1024、前 11 个候选从初始结构替换一个
block、其后替换两个 block、无 crossover，并在每次插入后对全部历史候选四组件重新 log-rank：
固定 commit 的 Python 参数默认值虽为 512，但官方 450M/600M/1G 启动脚本显式覆盖为 1024；来源
脚本 SHA 与控制流证据见 `docs/evidence/plainnet_source_protocol_20260804.json`。

```bash
zcp-test search --config configs/search/plainnet_mbv2_source_aligned.yaml \
  --flops-target 450m --gpu auto \
  --output /path/to/runs/search/plainnet-aznas-450m
```

可将 `450m` 改为 `600m` 或 `1g`。已撤销旧版本的配置使用显式
`source_budget_protocol=one_percent_acceptance`：每档 1,000 个有效候选（上游 100k 的 1%）、batch
64、resolution 224、ImageNet-1k 随机输入、四组件 `az_nas_log_rank` 和独立 scratch 初始化。结果记录
`search_budget_fraction=0.01` 和截断 fidelity，不得称为 AZ-NAS 完整搜索复现。

```bash
python tools/acceptance/benchmark-plainnet-rerank.py \
  --output docs/evidence/plainnet_rerank_scaling.json
export ZCP_PLAINNET_PREFLIGHT_GPU_UUID=GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
bash tools/acceptance/run-plainnet-throughput-preflight.sh
```

历史 GPU 预验收保留 `valid_candidates=100000` 和完整 controller 身份，只在 3 个已接受候选后通过
API 暂停；state 必须保持 `running`、不得出现 `search_summary`，并写明
`formal_search_completed=false`。它固定 batch 64/224，用于估算显存和单候选耗时，不是缩小版正式
搜索。CPU benchmark
只测每次插入后的全历史 rank，不包含构模、proxy、mutation、拒绝候选和 JSONL I/O。本机 2026-08-04
测得完整 100k 的 CPU rerank 保守累计估计约 15,166 秒。该估计解释了工程验收改用显式 1% 预算；
它不是 1,000 候选任务的完成证据。

以下 1% 后台 launcher 属于已撤销固定比例协议，只用于只读复核历史 artifact，不是当前正式可行性
gate；每个目标声明一张 GPU UUID，三档可在三张卡上互不共享 state 地并行：

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export ZCP_PYTHON=/path/to/conda/envs/zcp-test/bin/python
export ZCP_PLAINNET_SEARCH_GPU_UUID=GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
export ZCP_PLAINNET_FLOPS_TARGET=450m  # 也可为 600m 或 1g
bash tools/acceptance/run-plainnet-source-search-one-percent.sh
```

输出根默认为 `runs/acceptance/plainnet-source-aligned-one-percent/<target>/`。`status.json` 记录
1,000 候选、`one_percent_acceptance` 和预算比例。launcher 会跳过已有 completed run；若存在未完成 run，
则从最新 `search-state.json` 创建新 run 恢复，并由 CLI 严格核对完整 identity 和源 journal SHA。
旧 100k 或预检 state 与 1% 协议 identity 不同，不能传给该 launcher 续跑；新协议从独立 run 开始。

完成后必须逐档运行公共验证器；它同时核验 launcher status、manifest、state、JSONL 行数与 SHA、连续
索引、缓存计数、有限组件和 best 一致性：

```bash
RUN=/path/to/runs/acceptance/plainnet-source-aligned-one-percent/450m/<timestamp_run>
zcp-test acceptance validate-plainnet-search \
  --run "$RUN" --expected-target 450m --expected-candidates 1000 \
  --expected-budget-protocol one_percent_acceptance
```

预期 `candidate_rows=1000`、`summary_rows=1`、`formal_search_completed=false` 且
`one_percent_search_completed=true`。600m/1g 必须分别替换 target 和 RUN，不能共用验证结果。

CLI 会在创建 run 前拒绝只用主组件排名或搜索空间不匹配。稳定化端口会裁剪协方差负特征值浮点噪声，
并令退化奇异值计算保持有限，因此版本为 `aznas-5e6683-plainnet-stabilized-v1`，不声明逐位一致。
`docs/evidence/aznas_plainnet_rank_smoke.json` 的两候选 GPU 证据只验证四组件、聚合与 artifact；其中
聚合分数并列，不能解释为搜索质量结论。

## AutoFormer 与两类 MobileNetV2 双重 1% 验收

三个空间使用独立配置、候选目录和结果目录，不能互换 architecture ID 或训练 recipe：

Proxyless-MBV2 的候选选择采用独立项目迁移配置，不冒充官方 OFA tutorial controller：

```bash
zcp-test search \
  --config configs/search/proxyless_mbv2_zen_project_transfer.yaml \
  --gpu auto --output /path/to/runs/search/proxyless-zen
```

该配置固定 1,000 个候选、generation 0、seed `20260731`、Zen、逐候选分辨率随机输入和 scratch
初始化，并声明 `project_budget_protocol=one_percent_planned_100k`。这里的 1% 是预先约定的
100,000 次工程评估预算的 1,000 次，不是巨大离散搜索空间基数的 1%，也不是 OFA 论文控制器预算。
CLI 会拒绝不等于 1,000 次的该协议。结果必须带 `search_budget_fraction=0.01`、
`search_budget_scope=engineering_acceptance_not_search_space_fraction`、`experiment_kind=project_zcp_transfer`、
`controller_fidelity=project_controller_not_ofa_tutorial`、`direct_search_protocol_evidence=false`。
ZenNAS 只提供 MobileNet-style 同族依据；本项目 OFA-Proxyless 域上的应用是推广实验，不是论文直接复现。
历史 3-candidate GPU smoke 仅证明兼容性；2026-08-04 起后续训练严格只使用重新冻结的唯一 winner。见
`docs/evidence/proxyless_mbv2_zen_project_transfer_smoke_20260804.json`。新预算 run
`a353e301f420` 已完成 1,000 candidate + 1 summary，winner 为
`968893a0cc5f0f687688`（resolution 128）；机器可读证据见
`docs/evidence/proxyless_one_percent_search_completion_20260804.json`。

| 空间 | 启动器 | 全数据协议 | 1% 数据协议 |
|---|---|---:|---:|
| AutoFormer scratch | `run-autoformer-imagenet-dual-one-percent.sh` | 5/500 epoch | 500 epoch |
| ZenNAS PlainNet-MBV2 | `run-plainnet-mbv2-imagenet-dual-one-percent.sh` | 2/150 epoch | 150 epoch |
| Proxyless-MBV2 scratch | `run-proxyless-mbv2-imagenet-dual-one-percent.sh` | 2/150 epoch | 150 epoch |

从 2026-08-04 起，工程验收只训练一个结构化 `zcp_selected.json`，并要求
`candidates-manifest.json` 锁定其搜索来源、架构 ID 和 checksum。该候选必须来自记录完整输入协议和
代理版本的 ZCP 搜索，不得把官方发布或手选架构改标为 `zcp_selected`。冻结工具仍可产生
`fixed_random.json` 和 `params_flops_matched.json`，但它们只供另行设计的充分训练研究实验使用，通用
双重 1% acceptance launcher 不再读取或训练它们。

原因是 1% 短训只能验证模型构建、真实数据、优化器/scheduler、checkpoint、恢复、日志和报告链路，
不能可靠证明 ZCP 候选优于随机或朴素资源代理。把三个候选都纳入工程验收会消耗约三倍 GPU 资源，
却不能形成有统计效力的搜索收益结论。若要比较候选优劣，必须另建预声明、多 seed、充分训练的研究
协议，不得复用 acceptance 结果作优越性声明。

候选冻结使用已完成的搜索 run，而不是直接传入一个任意 architecture 文件：

```bash
SEARCH_RUN=/path/to/timestamped/search-run
zcp-test acceptance freeze-candidates \
  --search-run "$SEARCH_RUN" \
  --training-config configs/training/autoformer_imagenet.yaml \
  --seed 20260731 --selected-only \
  --output /path/to/frozen-candidates/autoformer
```

多 seed 搜索不得直接平均不同候选集合中的原始 ZCP/rank 分数，也不得在搜索完成后从多个 winner 中
事后挑选。应在完成前预声明一个 primary run，其他 run 只作为稳健性证据：

```bash
PRIMARY=/path/to/seed-20260731/timestamped-run
SUPPORT_1=/path/to/seed-20260732/timestamped-run
SUPPORT_2=/path/to/seed-20260733/timestamped-run
zcp-test acceptance freeze-candidates \
  --search-run "$PRIMARY" \
  --supporting-search-run "$SUPPORT_1" \
  --supporting-search-run "$SUPPORT_2" \
  --training-config configs/training/autoformer_imagenet.yaml \
  --seed 20260731 --selected-only \
  --output /path/to/frozen-candidates/autoformer
```

命令要求三个 run 均为 completed、搜索空间/模型/代理/聚合器/种群和输入协议一致且 seed 唯一。
`zcp_selected.json` 只来自 primary run；supporting winner 仅写入 provenance。最终 population 的最高分
若并列，按 canonical architecture ID 升序稳定裁决，并同时记录原 `best_architecture.json` 的选择、
并列数以及 `search-state.json` checksum。固定随机和参数量/计算量匹配候选仍由冻结 seed 独立生成，
不得从 supporting winner 替代。

命令要求 search manifest 为 `completed`，且 `search_identity` 完整包含 space、proxy/version、dataset、
input fingerprint 和 seed；`best_architecture.json` 还必须真实出现在 `search.jsonl` candidate 记录中。
`--selected-only` 输出严格只有 `zcp_selected.json` 和 `candidates-manifest.json`，不会采样或测量两类
对照候选；其中保存 search/config/JSONL SHA-256、
训练配置 SHA-256、架构 ID、资源协议和匹配距离。训练 CLI 只读取其中的 `spec`，其余 provenance 保持
只读审计。

MobileNet 使用同一模型实现上的 THOP MAC 作为计算量约定。AutoFormer 使用 Cream/AZ-NAS 官方
`get_complexity` 口径，并明确写入 `generic_flops=false`；它不能被重命名为通用 FLOPs。匹配距离为参数量
和该空间计算量的 log-ratio L1，因此只表示资源相近，不表示精度、延迟或训练成本完全相同。

通过单卡显存 smoke 后的双卡后台启动示例：

```bash
export TZ=Asia/Shanghai
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export ZCP_IMAGENET_ROOT=/path/to/imagenet1k
export ZCP_TRAINING_CANDIDATES=/path/to/frozen-candidates/autoformer
export ZCP_GPU_UUIDS=GPU-...,GPU-...  # parallel_single_gpu 只需两张卡
export ZCP_EXECUTION_STRATEGY=parallel_single_gpu
export ZCP_PARALLEL_SINGLE_GPU_ACCEPTED=yes
setsid -f env ZCP_START_AT=1 \
  bash tools/acceptance/run-autoformer-imagenet-dual-one-percent.sh
```

PlainNet 和 Proxyless 只替换候选目录与启动器。启动器会验证 ImageNet 的 1000 类、1,281,167 个训练
文件和 50,000 个验证文件，校验 config 的 space/epoch，使用按任务/lane 持有的 GPU UUID 文件锁，
并在工作树不干净时拒绝启动。完成的 lane 会立即释放 GPU；supervisor 不会为了尚未开始或已经完成的
任务继续占锁。状态位于 `runs/acceptance/<space>-imagenet/status.json`，所有新时间使用北京时间。中断后先
审计最近 run 的 manifest/checkpoint，再用 `ZCP_START_AT=2` 从第二种协议恢复；不要重复已完成
协议，也不要把 interrupted 记作 completed。

在启动真实数据 1% 验收前，应先验证正式配置的单卡 micro-batch 显存。普通 `--smoke` 会主动缩小
ImageNet batch，因此不能作为显存证据；使用下列显式模式执行一个 synthetic epoch，保留配置中的
batch、模型、优化器和 AMP，但不读取真实数据，也不得报告为精度实验：

```bash
zcp-test train \
  --config configs/training/autoformer_imagenet.yaml \
  --smoke --full-batch-smoke --epochs 1 \
  --gpu GPU-... --output runs/smoke/autoformer-full-batch-memory
```

输出的 `training_mode` 必须为 `synthetic_full_batch_memory_smoke`，并核对
`per_device_batch_size`、峰值显存、是否 OOM 和 manifest。若失败，应先依据已发布 recipe 判断需要 DDP、
梯度累积或配置语义修正；不得静默减小 batch 后继续称原协议已通过。

通用启动器默认采用 `sequential_ddp`。若单卡显存 smoke 已证明该 profile 的原始 batch 可放入一张卡，
可显式启用按候选并行：

```bash
export ZCP_EXECUTION_STRATEGY=parallel_single_gpu
export ZCP_PARALLEL_SINGLE_GPU_ACCEPTED=yes
```

此模式在两张单卡上分别运行同一 ZCP 候选的两个协议，但不会覆盖 config 的 batch、梯度累积或 LR。
其余 GPU 不持锁，可供其他任务使用。第二个变量是人为
验收闸门，不是自动显存证明；未做真实模型 forward/backward smoke 时不得设置。AutoFormer、PlainNet
和 Proxyless 必须分别验收，不能因为 DARTS 单卡可运行就直接放行。

如果“两进程同卡”的真实 forward/backward smoke 也已通过，可让两个协议共享一张 GPU：

```bash
export ZCP_EXECUTION_STRATEGY=packed_single_gpu
export ZCP_PACKED_SINGLE_GPU_ACCEPTED=yes
export ZCP_DATA_WORKERS=4
export ZCP_GPU_UUIDS=GPU-...  # packed_single_gpu 必须且只能声明这一张卡
export ZCP_CPU_AFFINITIES='32-63,96-127'
```

`packed_single_gpu` 在第一张卡上放置两个独立 run；它减少占卡数，不改变单个 run 的 batch/LR。必须先
确认两进程峰值显存总和有安全余量，并用较少 workers 防止 CPU 解码争用。
`ZCP_CPU_AFFINITIES` 可选，此处唯一一段对应唯一 GPU UUID；应按 `nvidia-smi topo -m` 选择 GPU 所属 NUMA
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

两项固定为同一 `zcp-selected` 候选的“全数据 × 最少 1% epoch”和“1% 数据 × 完整 schedule”。每个 run
必须有持续增长的 `run.log`/`events.jsonl`、每 epoch 的 `training.jsonl`、`last.pt`、`best.pt` 与最终
manifest。该验收用于放行训练实现，不等于论文完整数据完整 schedule 精度复现。

DARTS ImageNet 的正式 global batch 为 128。四卡 DDP 会把它拆成每卡 32，在 4090/4090D 上只占约
1.8 GiB 且同步开销明显；不能为追求利用率擅自扩大科学 batch。首项已完成后，可使用
`resume-darts-imagenet-parallel-from-task2.sh`：它将 task2–6 分成四条独立单卡 lane，每个 run 仍使用
global batch 128，但并行不同候选/协议。三个 250-epoch 任务优先在三条 lane 启动，task2/3 共用第四条；
task2 与 task3 分别获取和释放第四张 GPU 的锁；两项串行任务之间不由 lane/supervisor 续持，因此可被
其他合格任务重新竞争。其余 task 也在各自结束时立即释放，不再等待最慢任务。
这提高总吞吐而不改变单个实验的 batch/LR 协议；结果仍需逐 run 验证，不能把并行完成顺序当科学顺序。
