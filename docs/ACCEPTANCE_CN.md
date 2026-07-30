# 验收报告

本文严格区分轻量软件验收与高成本科学验证。单元测试、代理可执行、adapter smoke 或 1 个合成
epoch 都不能证明论文数值复现或正式 benchmark 精度。

## 证据状态

| 范围 | 已记录证据 | 状态 | 能证明什么 |
|---|---|---|---|
| 单元/集成基线 | 2026-07-30 当前工作树：**216 tests passed** | 通过 | 小型 fixture、schema、adapter、报告、GPU、reference 构模和工作流契约；不替代真实数据或高成本科学验收 |
| 静态质量门禁 | Ruff、compileall、pip check、`git diff --check` 通过 | 通过 | 语法、依赖和基础仓库卫生；不代表科学正确性 |
| 覆盖率 | 第一方 source 总计 **86%**；CLI 80%、报告 96%/100%、converter 98%、doctor/legacy 100% | 通过 | 达到总计 85% 与列出的关键模块 80% 门槛；adapter 的真实数据契约仍需独立 smoke |
| Proxy sweep | 验收 sweep 纳入 **22 个注册代理** | 部分证据 | registry 覆盖和明确状态处理，不证明全部模型族上的论文数值 |
| DARTS smoke | `runs/training/20260729T055707Z_6737dcdb935c`：`completed`，1 个合成 epoch，写出 checkpoint | smoke 通过 | RTX 4090 上的 DARTS 构模、optimizer/AMP、training JSONL 和 checkpoint 写入 |
| Evaluate smoke | `runs/evaluate/20260729T055018Z_aa69ffaeb008`：`completed` | 仅历史 smoke | 10 架构、3 代理流水线完成；它不是 22-proxy sweep 产物 |
| Search smoke | `runs/search/` 保留一次失败和一次完成的 AutoFormer ER 搜索 | 部分证据 | 只证明当时的搜索流程；旧 manifest 不能独立重建当前模型 fidelity，失败记录不能隐藏 |

216 tests、Ruff、compileall、pip check、diff check 与 86% coverage 是当前可复核的低成本软件基线。
仓库保留 DARTS/evaluate/search manifest，但没有专门的 22-proxy sweep manifest，因此该 sweep 仍只能
部分独立重建；后续应保存命令、commit、环境和摘要证据。Conda 下 coverage 必须使用
`python -m coverage`，避免系统 `coverage` shebang 误用 base Python。

## 22 个代理的口径

22 个名称为 `az_nas`、`er`、`er_conn`、`er_deg`、`er_dist`、`er_pr`、`flops`、`gradnorm`、
`jacob_cov`、`meco`、`meco_opt`、`naswot`、`near`、`ntkt`、`params`、`swap`、`synflow`、
`te_nas`、`ter`、`vkdnw`、`zen`、`zico`。

sweep 表示每个名称经过统一 evaluator，并产生明确 `ok`、`unsupported` 或 `failed`；不表示每个
代理支持所有模型族，也不表示 `portable-v1` 与论文数值一致，更不表示已完成 standard-answer
相关性验收。

## DARTS smoke 边界

保留 run 执行：

```bash
zcp-test train --config configs/training/darts_cifar10.yaml --epochs 1 --smoke
```

它写出 `best.pt`、`last.pt`、`training.jsonl` 和 completed manifest，只能证明流水线连通；不能
证明 CIFAR-10 test accuracy、600 epoch 收敛、增强协议、多 GPU、任意 epoch 恢复或跨硬件复现。

## Fidelity 与结果协议

| Fidelity | 空间 | 验收后果 |
|---|---|---|
| `reference_topology_pytorch_port` | `nb101_dag`、`nb201_topology`、`nats_size` | 拓扑由 port 表示，ZCP 不自动等同 benchmark 原训练实现 |
| `reference_topology_pytorch_port` | `transnas_micro`、`transnas_macro` | 官方 encoder topology port；七个 Taskonomy task head 尚未完整实现 |
| `reference_model` | `darts`、`autoformer`、`pit`、`zennas_plainnet_mbv2`、`ofa_proxyless_mbv2`、`ofa_mbv3` | 静态模型结构已实现；正式训练仍须独立通过 `formal_training_ready` 门禁 |
| `proxy_approximation` | legacy toy | 只适合显式 opt-in 的方法学 smoke，不得参与正式训练或 reference 结论 |

AutoFormer 与 Proxyless-MBV2 的仓库配置当前明确 `formal_training_ready: false`：前者缺 repeated
augmentation sampler、分布式全局 batch/LR scaling 和官方参数/FLOPs fixture；后者缺已验证的
TensorFlow 风格颜色扰动、官方静态子网 fixture 和分布式全局 batch。CLI 会拒绝非 smoke 训练。
配置中的布尔值不能自行授权；正式训练还必须匹配代码内置的 DARTS 协议白名单及关键字段。

NAS-Bench-101/201、NATS 和转换后的 TransNAS 记录只有在明确 dataset/split/budget/seed 下才是
**standard answer**。NAS-Bench-301 是 **surrogate**，deterministic/noisy 属于不同协议。
ViT-Bench 可能是 **scratch**、蒸馏或 **inherited-supernet**，不得混合。

## 当前部分验收

- 保留 evaluate 只有 3 个代理，不是 22；其 40 行历史 score 是旧 component-long schema。
- 当前树只能独立复核 registry 数量，不能重建专门的 22-proxy sweep 产物。
- AutoFormer 成功 run 只证明搜索机制；同时保留一次 failed run。
- 多个上游原生资产没有固定 checksum，路径存在不等于真实性。
- bootstrap 和 index-0 smoke 不证明全记录、全部 budget/split 或跨机器覆盖。
- `--start/--count` 没有已验收 launcher/merge CLI；多文件分析可用，端到端多 GPU 尚未验收。
- `portable-v1` 与 topology port 在论文复现声明前仍需对照官方实现。
- PiT 已对发布切片的三阶段规格完成 `load → build → forward`；同一规格与 Auto-Prox
  `90ed458` 上游均为 893,828 参数且参数 shape multiset 一致。MAC golden、正式训练和 KD 复现
  仍未完成；vanilla/KD 标准答案只作为独立指标查询。
- OFA-MBV3 的全 3×3、expand 3、depth 2、width 1.0 子网与官方 commit `f03b267` 均为
  3,410,792 参数且参数 shape multiset 一致；BN recalibration 已实现。官方 inherited checkpoint、
  active-weight export 与正式训练仍未验收。

## 尚未完成的高成本验收

以下工作明确为 **未验收**，不得写成已完成：

1. DARTS CIFAR-10/CIFAR-100 600 epoch 与 ImageNet 250 epoch 正式训练。
2. AutoFormer 500 epoch 与 Proxyless-MBV2 150 epoch 正式训练协议尚未放行；静态 reference 模型
   不能替代未实现的训练 sampler、分布式语义和官方 fixture。
3. 在第二台干净机器完成 benchmark 下载、checksum 和来源核验。
4. 在全部支持的 dataset、split、budget、seed 上运行 22-proxy 全量评估。
5. NAS-Bench-101 全量评估或 NAS-Bench-301 理论 DARTS 空间穷举。
6. 多 GPU 启动、去重合并、重启和故障注入验收。
7. 论文数值复现、独立 seed 置信区间及与官方代码的成本/精度比较。

正式验收必须保留 manifest、resolved config、commit、环境、输入 hash、结果类型、失败行和准确命令。
在证据齐全前，只能声明轻量软件验收与 smoke 覆盖。
