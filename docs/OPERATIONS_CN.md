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
