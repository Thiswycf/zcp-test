# 验收报告

本文严格区分轻量软件验收与高成本科学验证。单元测试、代理可执行、adapter smoke 或 1 个合成
epoch 都不能证明论文数值复现或正式 benchmark 精度。

## 证据状态

| 范围 | 已记录证据 | 状态 | 能证明什么 |
|---|---|---|---|
| 单元/集成基线 | 2026-07-31 当前工作树：**365 tests passed** | 通过 | 小型 fixture、schema、adapter、报告、GPU、reference 构模和工作流契约；不替代真实数据或高成本科学验收 |
| 静态质量门禁 | Ruff、compileall、pip check、`git diff --check` 通过 | 通过 | 语法、依赖和基础仓库卫生；不代表科学正确性 |
| 覆盖率 | 第一方 source 总计 **87%**；CLI 80%、analysis/proxy studies 93%、benchmark report 96%、reports 100%、converter 98%、doctor/legacy 100% | 通过 | 达到总计 85% 与列出的关键模块 80% 门槛；adapter 的真实数据契约仍需独立 smoke |
| H1：1% proxy sweep | NB201、NATS-TSS、NATS-SSS/CIFAR-10-valid、NB101 与 NB301 deterministic surrogate 分别完成 22 代理 seed 2026 和核心 11 代理三 seed；ViT 三公开切片完成 minimum-5 单 seed 预验收 | **五个 benchmark 的当前既定协议完成，H1 整体进行中** | ViT 公开文件各 100 条，与论文 500 GT/数据集和无重叠 60/40 划分不闭合，因此不计入正式 H1；TNB101 仍受作者 split/config 与许可输入阻塞 |
| DARTS smoke | `runs/training/20260729T055707Z_6737dcdb935c`：`completed`，1 个合成 epoch，写出 checkpoint | smoke 通过 | RTX 4090 上的 DARTS 构模、optimizer/AMP、training JSONL 和 checkpoint 写入 |
| Evaluate smoke | `runs/evaluate/20260729T055018Z_aa69ffaeb008`：`completed` | 仅历史 smoke | 10 架构、3 代理流水线完成；它不是 22-proxy sweep 产物 |
| Search smoke | `runs/search/` 保留一次失败和一次完成的 AutoFormer ER 搜索 | 部分证据 | 只证明当时的搜索流程；旧 manifest 不能独立重建当前模型 fidelity，失败记录不能隐藏 |
| ViT/PiT 模型 fidelity | PiT 参数量、MAC、stage、QKV、pool、LN epsilon 与 drop-path fixture 已通过 | topology port 通过 | 仍缺官方 checkpoint/逐层数值对照，因此降为 `reference_topology_pytorch_port`，不称 `reference_model` |

365 tests、Ruff、compileall、pip check、diff check 与 87% coverage 是当前可复核的低成本软件基线。
NB201 已有专门的 22-proxy、1% 分层抽样单 seed 证据：sample manifest SHA、四个 run ID、四个
`scores.jsonl` SHA、失败键和相关性摘要见
[`evidence/NB201_ONE_PERCENT_22ZCP_CN.md`](evidence/NB201_ONE_PERCENT_22ZCP_CN.md)。原始 score 留在
外部审计目录，不进入 Git。Conda 下 coverage 必须使用
`python -m coverage`，避免系统 `coverage` shebang 误用 base Python。

核心 11 代理的 seed 2027/2028 也已完成；与 seed 2026 合并后为 5,181 行、5,172 成功、9 失败、
0 重复键。三 seed Spearman、跨 seed 排名稳定性、八个新增 run 和 score SHA 见
[`evidence/NB201_CORE_THREE_SEED_CN.md`](evidence/NB201_CORE_THREE_SEED_CN.md)。这只关闭 NB201
既定 seed 子项，H1 仍等待其余 benchmark。

## 22 个代理的口径

22 个名称为 `az_nas`、`er`、`er_conn`、`er_deg`、`er_dist`、`er_pr`、`flops`、`gradnorm`、
`jacob_cov`、`meco`、`meco_opt`、`naswot`、`near`、`ntkt`、`params`、`swap`、`synflow`、
`te_nas`、`ter`、`vkdnw`、`zen`、`zico`。

sweep 表示每个名称经过统一 evaluator，并产生明确 `ok`、`unsupported` 或 `failed`。NB201 seed
2026 的 22 个名称均有记录，但 `az_nas`、`naswot`、`te_nas` 各有一条非有限失败；`near` 和
`swap` 为常数，相关系数未定义。该结果不表示每个代理支持所有模型族，也不表示 `portable-v1`
与论文数值一致。出现数值完全相同的不同名称也不能据此认定算法独立，仍需 provenance/公式审计。

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
| `reference_topology_pytorch_port` | `transnas_micro`、`transnas_macro` | 官方 encoder 与七个 task head 的 PyTorch port；真实 Taskonomy contract provider 已实现，但正式 24-building split/config 未公开且本机无许可数据 |
| `reference_model` | `darts`、`autoformer`、`pit`、`zennas_plainnet_mbv2`、`ofa_proxyless_mbv2`、`ofa_mbv3` | 静态模型结构已实现；正式训练仍须独立通过 `formal_training_ready` 门禁 |
| `proxy_approximation` | legacy toy | 只适合显式 opt-in 的方法学 smoke，不得参与正式训练或 reference 结论 |

AutoFormer 与 Proxyless-MBV2 的仓库配置当前明确 `formal_training_ready: false`。AutoFormer 已有
AZ-NAS repeated-augmentation sampler、`base_lr × global_batch / 512` 缩放规则，以及 Cream T/S/B
和 AZ-NAS Tiny/Small/Base 六个精确参数量 golden。混合 4090D/4090 的 2-rank DARTS 与 AutoFormer
真实 GPU smoke 已验证 DDP、跨 rank 指标归约、单一共享 run 和仅 rank 0 写 artifact；失败 run
也会标为 failed。六个官方 `get_complexity` 数值已以 `official_complexity_ops` 独立字段复现；
AZ-NAS Tiny 的 THOP MAC 为 `1,100,420,352`，官方口径为 `1,380,128,376`，且 THOP 未计全
relative-position 参数，因此报告禁止将二者混称 FLOPs。真实 ImageNet 图片极小夹具上的 2-rank
中断/新 run 恢复已验证 `interrupted` manifest、checkpoint SHA-256/source run lineage、epoch 0–4
连续日志和 `.tmp` 清理；该证据只覆盖恢复机制，未覆盖完整 ImageNet 双重 1% 协议。Proxyless-MBV2 仍缺已验证的 TensorFlow
风格颜色扰动、官方 MAC fixture 和分布式全局 batch。CLI 会拒绝非 smoke 训练。
配置中的布尔值不能自行授权；正式训练还必须匹配代码内置的 DARTS 协议白名单及关键字段。

NAS-Bench-101/201、NATS 和转换后的 TransNAS 记录只有在明确 dataset/split/budget/seed 下才是
**standard answer**。NAS-Bench-301 是 **surrogate**，deterministic/noisy 属于不同协议。
ViT-Bench 可能是 **scratch**、蒸馏或 **inherited-supernet**，不得混合。

## 当前部分验收

- 保留 evaluate 只有 3 个代理，不是 22；其 40 行历史 score 是旧 component-long schema。
- Git 不保存 3,454 行原始 score；当前树可独立复核精简摘要、四个 score SHA、失败键和 registry
  provenance，但逐行重算仍需要外部审计目录或按手册重新运行。
- AutoFormer 成功 run 只证明搜索机制；同时保留一次 failed run。
- 多个上游原生资产没有固定 checksum，路径存在不等于真实性。
- bootstrap 和 index-0 smoke 不证明全记录、全部 budget/split 或跨机器覆盖。
- `--start/--count` 没有已验收 launcher/merge CLI；多文件分析可用，端到端多 GPU 尚未验收。
- 通用 correlation/compare/seed-sensitivity 与 NB201 topology 已用两组真实 CIFAR 输入、真实
  NB201 真值的 20 架构 run 完成工作流验收。每个 run 60 行；NASWOT 对 index 12 的非有限结果
  保留为 failed。该连续小样本不是 feature-stratified 1%，不得作为论文数值。证据见
  `docs/evidence/E2_E3_NB201_REAL_CN.md`。
- H1 已完成 NB201 seed 2026 的 feature-stratified 1% × 22 ZCP：157 架构、3,454 原始行、
  3,451 成功、3 失败且无重复键；修复 shard grouping 后，topology 报告含 157 architecture、
  942 edge、5 operation、6,720 correlation、840 operation effect 和 588 matched pair。核心 11 代理另两个 seed
  也已完成，三 seed 合并为 5,181 行、5,172 成功、9 失败。
  后续审计确认旧 `params`/`flops` 将资源约束方向错误用于 accuracy 相关性；现已拆分为
  `direction=maximize` 与 `resource_direction=minimize`，旧记录由 reader 显式迁移，相关证据按原始
  规模—精度方向重建。精简证据见
  `docs/evidence/NB201_ONE_PERCENT_22ZCP_CN.md`。
- H1 已独立完成 NATS-TSS 的同规模协议：22 代理 seed 2026 为 3,454 行、3,451 成功、3 失败；
  核心 11 代理三 seed 为 5,181 行、5,172 成功、9 失败。NATS-TSS 使用独立 adapter、版本、manifest
  和真值；与 NB201 共同的 157 个 topology 中有 31 个真值不同。当前状态为
  **“NB201 与 NATS-TSS 既定 seed 协议完成，H1 整体进行中”**。证据见
  `docs/evidence/NATS_TSS_ONE_PERCENT_CN.md`。
- H1 已完成 NATS-SSS 的 CIFAR-10-valid/90-epoch 协议：328 架构 × 22 代理为 7,216 行且全部成功；
  核心 11 代理三 seed 为 10,824 行且全部成功。修复了专属研究把 `run_id` 当协议、导致每 shard
  单独统计的错误；正式 size 表合并为 n=328，并按 evaluation seed 分离。CIFAR-100 与
  ImageNet16-120 rank transfer 仍待。证据见 `docs/evidence/NATS_SSS_ONE_PERCENT_CN.md`。
- H1 已完成 NB101 的正式 1% 既定协议：从 423,624 个架构中按 seed 2026 分层抽取 4,237 个；
  22 代理 seed 2026 为 93,214/93,214 成功，核心 11 代理 seed 2026/2027/2028 为
  139,821/139,821 成功，均无失败或重复任务键。4/12/36/108 epoch 的 repeat `mean/min/max`
  预算分析均已完成。TE-NAS `portable-v2` 仅为仓库可移植近似，不等同官方完整 TE-NAS；本结论
  只覆盖该固定样本和协议。证据见
  [`docs/evidence/NB101_ONE_PERCENT_CN.md`](evidence/NB101_ONE_PERCENT_CN.md) 与
  [`docs/evidence/nb101_one_percent_summary.json`](evidence/nb101_one_percent_summary.json)。
- `portable-v1` 与 topology port 在论文复现声明前仍需对照官方实现。
- TransNAS 七任务 head 已按上游 commit `6d4231b` 分离；同一 micro fixture 的官方/本项目参数量
  与完整 parameter-shape multiset 在七个任务均一致。已实现受控 Taskonomy manifest、七任务真实
  input/target loader、final5k 分类 mask 和确定性 Jigsaw。原始 105 MB 标准答案 SHA-256 为
  `1974b0ba…1364bc10`，micro/macro 转换表分别锁定为 `cc6c9fb2…3753ee2` 与
  `4818b9e6…6ae6bd4`；4,096/3,256 条及七任务 validation target 均完整有限。正式 1% manifest
  已冻结为 41/33 个架构。论文使用的 24-building/120K split 与最终 transform/config 未随公开发布；
  本机也尚无用户依法取得的 Taskonomy 数据。因此真实 task-input GPU ZCP 仍为 blocked，不能用
  fixture、Taskonomy 任意 split 或 random 输入冒充正式 H1；回归/dense 标签依赖 ZCP 也继续明确
  `unsupported`，直至 loss 契约有上游证据。
- PiT 已对发布切片的三阶段规格完成 `load → build → forward`；同一规格与 Auto-Prox
  `90ed458` 上游均为 893,828 参数且参数 shape multiset 一致。MAC golden、正式训练和 KD 复现
  仍未完成；vanilla/KD 标准答案只作为独立指标查询。
- OFA-MBV3 的全 3×3、expand 3、depth 2、width 1.0 子网与官方 commit `f03b267` 均为
  3,410,792 参数且参数 shape multiset 一致；BN recalibration 已实现。官方 inherited checkpoint、
  active-weight export 与正式训练仍未验收。
- OFA-Proxyless-MBV2 已改为官方 21 个 dynamic block 固定位置编码（五个最大深度 4 的可搜索
  stage，加一个固定末端 stage），正式空间固定使用发布 supernet 的 width 1.3，分辨率为
  128–224、步长 4。width 1.0 的全 3×3、expand 3、五个可搜索 stage depth 2 fixture 与官方
  commit `f03b267` 均为 2,500,632 参数，参数 shape multiset 一致；发布 width 1.3 对应 fixture
  均为 3,718,832 参数。官方 32,202,338-byte supernet checkpoint 已以固定 SHA-256 自举；混合
  `k/e/d` 子网经 active channel/kernel transform 导出后，与官方 `get_active_subnet` 参数量一致，
  同一输入最大绝对输出误差约 `1.9e-6`。真实 `evaluate` 与短程 `search` 已记录
  `inherited_supernet`、checkpoint 摘要、激活位置和 `bn_recalibration_required`。本机真实
  ImageNet-1k 已完成 1 个独立 batch 的确定性 BN 流水线 smoke，记录 sample ID、transform 和
  指纹；该项目协议明确 `official_protocol_match=false`。官方 data-provider 数值对照、inherited
  accuracy、MAC golden 和正式训练仍未验收。

## 尚未完成的高成本验收

以下工作明确为 **未验收**，不得写成已完成：

1. DARTS CIFAR-10/CIFAR-100 600 epoch 与 ImageNet 250 epoch 正式训练。
2. AutoFormer 500 epoch 与 Proxyless-MBV2 150 epoch 正式训练协议尚未放行；AutoFormer 的 sampler、
   LR、静态 fixture 和真实图片夹具恢复机制已验收，但“全 ImageNet × 至多 1% epoch”与“1% ImageNet
   × 完整 schedule”尚未执行。
3. 在第二台干净机器完成 benchmark 下载、checksum 和来源核验。
4. 在其余 benchmark 的目标 dataset、split、budget、task 上运行各自 1% 协议；NB201、NATS-TSS、
   NATS-SSS/CIFAR-10-valid、NB101 与 NB301 deterministic surrogate 已完成各自上述限定协议，
   NATS-SSS 跨数据集、TNB101 正式输入以及 ViT-Bench 500 条全集/60-40 身份仍待，因此完整项目
   H1 尚未完成。ViT 三个公开 100 条切片的 5×22 预验收不能替代该缺口。
5. NAS-Bench-101 全量评估或 NAS-Bench-301 理论 DARTS 空间穷举。
6. 多 GPU evaluate 的内置启动/去重合并；训练 DDP 启动与夹具级重启/故障注入已验收，但全数据级别未验收。
7. 论文数值复现、独立 seed 置信区间及与官方代码的成本/精度比较。

正式验收必须保留 manifest、resolved config、commit、环境、输入 hash、结果类型、失败行和准确命令。
在证据齐全前，只能声明轻量软件验收与 smoke 覆盖。

## 真实标准答案 index-0 验收（2026-07-30）

以下结果来自本机 catalog 与 `/path/to/data` 等价的数据根配置。表中的 `external catalog` 表示本机可用，
但文件不位于所检查的数据根目录；迁移到另一台机器时仍须执行 `data bootstrap` 或重新 `data register`，
不能复制仓库配置后假定可用。

| Benchmark / slice | 规模或协议 | index-0 查询 | 当前位置语义 |
|---|---|---:|---|
| NAS-Bench-101 full | 423,624 架构；4/12/36/108 epoch × 3 repeats | CIFAR-10 valid 108-epoch mean `0.9264155825` | data root |
| NAS-Bench-201 v1.1 | native API；200 epoch | CIFAR-10-valid valid accuracy `81.98266666` | external catalog |
| NATS-TSS v1.0 | `nats_bench.create(..., "tss")`；200 epoch | CIFAR-10-valid valid accuracy `81.98266666` | external catalog |
| NATS-SSS v1.0 | `nats_bench.create(..., "sss")`；90 epoch | CIFAR-10-valid valid accuracy `76.88799999` | external catalog |
| TNB101 micro | 4,096 架构 | class-scene valid top-1 `7.48407650` | data root / safe JSONL |
| TNB101 macro | 3,256 架构 | class-scene valid top-1 `52.97074127` | data root / safe JSONL |
| NB301 v1.0 | deterministic XGBoost surrogate | sampled DARTS accuracy `93.45854187` | external catalog |
| ViT AutoFormer main | 100 条 | CIFAR-100 vanilla `68.66` | data root / safe JSONL |
| ViT AutoFormer extension | 100 条；来源单独报告 | CIFAR-100 KD `78.07` | data root / safe JSONL |
| ViT PiT | 100 条 | CIFAR-100 vanilla `68.33` | data root / safe JSONL |

NB201 与 NATS-TSS 的 index-0 architecture ID 相同，证明 topology codec 可共享；二者的
`benchmark_id`、版本、加载 API 和指标来源仍保持独立，禁止合并成一个 benchmark。NB301 数值是
surrogate prediction，不是候选真实训练精度。AutoFormer extension 不含 vanilla 指标；对它查询
`accuracy_vanilla` 会失败，必须显式使用 `accuracy_kd` 或 `accuracy_inherited`。

本机已真实执行 ViT bootstrap：三份 raw 固定 SHA、三份 runtime JSONL SHA、严格 canonical
architecture ID、load→query→build→224 forward 均通过。三个切片各抽 5 个候选并执行 22 ZCP，
每切片严格得到 110 行、80 `ok`、30 `unsupported`、0 `failed`，同时生成 correlation、architecture
study 和 bundle。由于论文声明 500 GT/数据集及无重叠 60/40，而公开 commit 未给出全集与划分身份，
该证据状态保持 partial；详见 `docs/evidence/VITBENCH_PREFLIGHT_CN.md`。

可移植复验模板：

```bash
zcp-test data checklist --root /path/to/data --json
zcp-test benchmark inspect nasbench101 --data-root /path/to/data \
  --dataset cifar10 --split valid --metric-name final_accuracy \
  --epoch-budget 108 --metric-seed-reduction mean
zcp-test benchmark inspect nasbench201 --trusted --data-root /path/to/data \
  --dataset cifar10-valid --split valid --metric-name accuracy --epoch-budget 200
zcp-test benchmark inspect nats_tss --trusted --data-root /path/to/data \
  --dataset cifar10-valid --split valid --metric-name accuracy --epoch-budget 200
zcp-test benchmark inspect nats_sss --trusted --data-root /path/to/data \
  --dataset cifar10-valid --split valid --metric-name accuracy --epoch-budget 90
```

同一轮还对上述十个 benchmark/切片执行了真实 index-0 `build_model → params proxy`。所有调用均为
`succeeded=1, failed=0, score_rows=1`；参数量依次为 NB101 `8,555,530`、NB201/NATS-TSS
`129,306`、NATS-SSS `11,714`、TNB micro `24,618`、TNB macro `2,318,890`、NB301 DARTS
`239,802`、ViT main `5,710,180`、ViT extension `8,755,324`、PiT `893,828`。该 smoke 显式使用
`input_source=random` 且只运行数据无关的 `params`，因此仅证明真实架构能够构模并进入统一 evaluator，
不能作为真实输入消融或相关性结果。
