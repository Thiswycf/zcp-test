# ViT-Bench-101 发布切片预验收

## 结论

当前实现已完成三份公开 GT 文件的安全自举、固定 SHA、JSONL 转换、严格切片隔离、真实模型构建、
真实 CIFAR-100 输入的 22-ZCP 预验收，以及通用/专有报告生成。但它的科学状态仍是
**partial release-slice preacceptance**，不能写成论文完整 ViT-Bench-101 复现或正式 H1 通过。

关键原因不是代码报错，而是上游公开证据不足：Auto-Prox 论文称每个数据集包含 500 个 ground truth，
并将完整 ViT-Bench-101 按 60% proxy search validation、40% proxy evaluation test 做无重叠划分；固定
仓库 commit `90ed458eff6948a6f0d23e440a8d21bbec50d091` 实际只发布三个各 100 条的 `.pth`，没有发布
500 条全集和 60/40 架构身份。三个文件由 commit
`69ccf82bfa8897ce3c955a700a5b0f046d8e5f87` 一次加入，提交说明也没有解释第二个 AutoFormer
切片的来源。因此不能自行把文件顺序解释成论文划分。

## 已验证资产

| 切片 | 条数 | 可用指标 | raw SHA-256 | runtime SHA-256 |
|---|---:|---|---|---|
| AutoFormer main | 100 | CIFAR-100/Flowers/Chaoyang vanilla+KD；ImageNet inherited | `712ad277...36eca` | `f862cb7f...640b5` |
| AutoFormer extension | 100 | CIFAR-100/Flowers/Chaoyang KD；ImageNet inherited | `05f5df6a...e7bca` | `439f6a25...2b8322` |
| PiT | 100 | CIFAR-100/Flowers/Chaoyang vanilla+KD | `bdda8984...3bb2` | `cd32dd4e...ce25` |

完整摘要见 `docs/evidence/vitbench_preflight_summary.json`。main、extension 与 PiT 必须分别报告；
extension 缺少 vanilla，PiT 缺少 ImageNet inherited，不能补零或用其他切片代替。

## 数据自举与查询

```bash
CATALOG=~/.config/zcp-test/data.json
DATA=/path/to/data

zcp-test data bootstrap --root "$DATA" \
  --benchmarks vitbench101 --catalog "$CATALOG" --yes
zcp-test data checklist --root "$DATA" --catalog "$CATALOG" --json

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

完整自举后应看到：

```text
state=ready
raw_state=ready
runtime_state=ready
runtime_integrity=verified
operational_ready=true
```

只有安全 JSONL 存在、raw 缺失时，应看到 `state=partial`、`operational_ready=true`；这表示查询可用，
但不能离线重建。普通 `evaluate` 不会自动下载。

## 发布切片预验收

以下命令只研究公开的 100 条切片，不是论文 500 条全集的 1% 复现。每切片采用最低 5 个候选，因而
实际是公开切片的 5%；文件名应使用 `minimum5`，不要标成 `one-percent-of-release`。

```bash
AUDIT=/path/to/audit
for SLICE in autoformer_main autoformer_ext pit; do
  zcp-test benchmark sample vitbench101 --catalog "$CATALOG" \
    --slice-id "$SLICE" --count 5 --seed 2026 \
    --output "$AUDIT/sampling/vitbench-${SLICE}-minimum5-seed2026.json"
done
```

机器 catalog 单独注册真实 CIFAR-100，不在仓库配置中写死路径：

```bash
zcp-test data register dataset_cifar100 /path/to/cifar100 \
  --version torchvision-cifar100 \
  --protocol train-split-published-labels --trusted --replace
```

以 main 为例：

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

当前能力矩阵下预期严格为 5×22=`110` 行：`80 ok`、`30 unsupported`、`0 failed`。
`er_conn/er_deg/er_dist/er_pr/gradnorm/ter` 不支持 Transformer，必须保留为 unsupported。CLI 现在分别
打印 `failed`、`unsupported`、`skipped` 与 `non_ok`，不再把 unsupported 伪称 failed。

## 分析与查看

```bash
RUN=/path/to/the/timestamped/run
zcp-test analyze correlation --scores "$RUN/scores.jsonl" \
  --output "$AUDIT/reports/vit-main/correlation" \
  --bootstrap-samples 200 --top-k 1 3 5
zcp-test analyze benchmark --scores "$RUN/scores.jsonl" \
  --benchmark vitbench101 --view architecture \
  --dataset cifar100 --target-split test \
  --benchmark-variant autoformer_main \
  --output "$AUDIT/reports/vit-main/architecture"
zcp-test report bundle "$RUN" --output "$AUDIT/reports/vit-main/bundle"
```

`analyze benchmark --view architecture` 分别分析 AutoFormer 的 depth/hidden/head/MLP 与 PiT 的
stage depth/base dimension/head/MLP。样本只有 5 时，相关系数置信区间极宽，图表只用于链路验收，
不能据此做稳定科学结论。

## 正式升级条件

只有满足以下全部条件，才能把状态升级为正式 ViT-Bench H1：

1. 获得论文所述 500 GT/数据集的可信资产或作者确认公开 100 条文件的精确角色；
2. 获得无重叠 60/40 架构 ID，proxy 开发只用 60%，最终相关性只用 40%；
3. 对 CIFAR-100、Flowers、Chaoyang 使用对应真实数据和明确的上游输入协议；
4. vanilla、KD、ImageNet inherited 分协议报告，extension 不并入 main；
5. 核心代理三 seed 使用同一 test architecture manifest；
6. 报告全部失败、unsupported、覆盖率、输入指纹和模型 fidelity。

## PiT 模型 fidelity 审计

公开 GT 的查询正确性与用于 ZCP 的模型实现 fidelity 是两个独立问题。当前 PiT PyTorch 实现已对齐
Auto-Prox 固定 commit 中的卷积 patch embedding、三阶段 transformer、depthwise pooling、QKV/projection、
`LayerNorm(eps=1e-6)` 和 `drop_path_rate * block_index / total_blocks`。固定架构
`base_dim=16, depth=[2,8,4], heads=[2,4,4], mlp_ratio=6` 的参数量为 `893,828`，当前锁定的
THOP fixture 为 `159,665,472` MACs。

但当前项目不加载上游训练 checkpoint，也没有完成逐层输出数值对照；模块创建顺序与 PyTorch 版本还会
影响同一 seed 下的具体初始化样本。因此 fidelity 必须写为
`reference_topology_pytorch_port`，不能写为 `reference_model`，也不能用它启动所谓 ViT-Bench 正式重训练。
ViT-Bench 固定候选只查询公开 GT；开放 PiT 研究若要训练，必须另行定义并验收训练协议。

## Fidelity 修正后的真实 GPU index-0 回归

在 catalog SHA/version/protocol 校验与 PiT fidelity 修正后，三个切片分别使用真实确定性 CIFAR-100
batch（2×3×224×224）执行 index-0 的 `params,naswot`：每片 2 行、2 `ok`、0 `failed`。AutoFormer
记录 `reference_model`，PiT 记录 `reference_topology_pytorch_port`；运行环境记录
`CUDA_DEVICE_ORDER=PCI_BUS_ID`。外部审计摘要 SHA-256 为
`55313b359380288d51967d7e6deaff57f3070d7eae1b50703efafa35ad03750b`。该 smoke 证明当前端到端
查询/构模/evaluator 链路，不提升公开切片的科学状态，也不替代 22-ZCP minimum-5 结果。
