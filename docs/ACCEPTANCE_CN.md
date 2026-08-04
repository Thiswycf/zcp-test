# 验收报告

本文严格区分轻量软件验收与高成本科学验证。单元测试、代理可执行、adapter smoke 或 1 个合成
epoch 都不能证明论文数值复现或正式 benchmark 精度。

## 证据状态

| 范围 | 已记录证据 | 状态 | 能证明什么 |
|---|---|---|---|
| 单元/集成基线 | 2026-08-04 当前工作树：**563 tests passed**（38 个测试文件） | 通过 | 全量 pytest、Ruff、compileall、pip check、Bash 语法、JSON 与 diff 检查均通过；看板一致性在并行更新后单独复验；仅保留 4 条来自 THOP 上游 `distutils` 的非失败弃用警告 |
| 静态质量门禁 | Ruff、compileall、pip check、repository hygiene、panel check、`git diff --check` 均通过 | 通过 | 语法、依赖、面板检查和基础仓库卫生；不代表科学正确性 |
| 覆盖率 | 第一方 source 总计 **87%**；CLI **82%**、analysis 93%、proxy studies 94%、benchmark report 96%、reports 100%、ImageNet16 converter 83%、doctor/legacy 100% | 通过 | 达到总计 85% 与列出的关键模块 80% 门槛；adapter 的真实数据契约仍需独立 smoke |
| H1：1% proxy sweep | NB201、NATS-TSS、NATS-SSS 三数据集、NB101 与 NB301 deterministic surrogate 已完成限定协议；ViT 三公开切片完成 minimum-5 单 seed 预验收 | **五个 benchmark 的当前既定协议完成，H1 整体进行中** | NATS-SSS 跨数据集扩展为 1% 分层样本、单输入/初始化 seed，不是全空间结论；ViT 公开身份不完整，TNB101 仍受作者 split/config 与许可输入阻塞 |
| DARTS smoke | `runs/training/20260729T055707Z_6737dcdb935c`：`completed`，1 个合成 epoch，写出 checkpoint | smoke 通过 | RTX 4090 上的 DARTS 构模、optimizer/AMP、training JSONL 和 checkpoint 写入 |
| DARTS CIFAR 双重 1% | CIFAR-10/100 三候选：全数据 × 6 epoch 共 6 runs；1% 数据 × 600 epoch 共 6 runs；确定性预检、两次恢复审计和报告完成 | **限定协议通过** | 关闭工程与限定协议验收；不是 600 epoch 全数据精度复现或多 seed 搜索收益，两个协议不得平均 |
| Evaluate smoke | `runs/evaluate/20260729T055018Z_aa69ffaeb008`：`completed` | 仅历史 smoke | 10 架构、3 代理流水线完成；它不是 22-proxy sweep 产物 |
| AutoFormer 搜索与冻结 | AZ-NAS `3×8,000` cohort 已 reconcile：24,000 candidates、23,999 unique evaluations、1 cache hit；三个候选已冻结 | 搜索与冻结通过 | supervisor 在工作负载完成后失败的历史状态继续保留，但不得覆盖已验证的科学产物；supporting seed 仅作 provenance |
| AutoFormer 三候选 smoke | 三个冻结候选均完成 batch 256 synthetic memory smoke、原子 checkpoint 与可信 checkpoint-load/resume smoke | smoke 通过 | 证明构模、显存、optimizer/checkpoint 与恢复路径；随机输入精度无意义，不是 ImageNet 训练证据 |
| AutoFormer 单候选 real dual-1% V2 | `zcp-selected` 完成 full-data 5 epoch 与 one-percent-data 500 epoch，分别为 5/500 行，均有 terminal manifest、`last.pt` 与 `best.pt` | **限定协议通过** | task 5/6 基线因违反新政策而中止并排除；只证明实现、调度和恢复就绪，不是 500-epoch 全数据论文精度或搜索收益 |
| ViT/PiT 模型 fidelity | PiT 参数量、MAC、stage、QKV、pool、LN epsilon 与 drop-path fixture 已通过 | topology port 通过 | 仍缺官方 checkpoint/逐层数值对照，因此降为 `reference_topology_pytorch_port`，不称 `reference_model` |

当前完整 gate 实际执行并通过 563 tests（38 个测试文件）。第一方 source coverage 87%、CLI
coverage 82% 仍来自最近一次保留的 coverage gate；Ruff、compileall、pip check、Bash 语法、看板、
JSON 与 `git diff --check` 均通过，仅有 4 条来自 THOP 上游 `distutils` 的非失败弃用警告。
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

## DARTS CIFAR 双重 1% 与 smoke 边界

DARTS CIFAR-10/100 已冻结 ER 搜索候选、固定随机候选和参数匹配随机池候选，并完成：

| 协议 | CIFAR-10 best valid top-1（ER / 固定随机 / 参数匹配随机） | CIFAR-100 best valid top-1（ER / 固定随机 / 参数匹配随机） | 状态 |
|---|---|---|---|
| 全数据 × 6 epoch，保留 600-epoch schedule | `78.62 / 77.28 / 64.67` | `46.19 / 44.13 / 26.81` | 6/6 runs 完成 |
| 恰好 1% 数据 × 600 epoch | `42.0 / 45.0 / 46.0` | `15.0 / 12.0 / 11.0` | 6/6 runs 完成 |

确定性真实数据预检已通过；两套协议分别完成一次可信 checkpoint 恢复审计，恢复了 6 条和 600 条
历史记录；证据报告及脱敏 JSON 摘要也已完成。候选身份、run ID、校验摘要和详细边界见
[`evidence/DARTS_CIFAR_DUAL_ONE_PERCENT_CN.md`](evidence/DARTS_CIFAR_DUAL_ONE_PERCENT_CN.md) 与
[`evidence/darts_cifar_dual_one_percent_summary.json`](evidence/darts_cifar_dual_one_percent_summary.json)。

这是单一训练 seed `20260731` 下的工程与限定协议验收，不是 600 epoch 全数据精度复现，也不是
多 seed 搜索收益证明；ER 候选只由一个固定 CIFAR-10 batch 和一个初始化 seed 选出。两个协议的
候选排序不同，分别测试早期全数据学习和小数据完整 schedule，禁止求平均或合并成“ER 稳定优于
基线”的结论。ImageNet-1k 已通过 1000 类、1,281,167/50,000 张图和真实 loader 解码预检；
DARTS ImageNet 六项双重 1% 验收已于 2026-07-31 启动。锁定 `78d8118` 且读取 `/public` 机械盘的
首轮在 3 小时内未完成一个 epoch，`run.log` 也因 logger 未被调用而保持 0 字节，现已明确标为
`interrupted`，不构成完成证据。修复后的主仓 commit `c0c7815` 把事件同步 flush 到
`events.jsonl` 和 `run.log`，新增可移植四卡 launcher，并改用经 1000/1,281,167/50,000 文件数核验的
本机 NVMe 数据副本。新 run 的首分钟 rank-0 吞吐约 7.2 batches/s、首 epoch ETA 约 22 分钟；当前
六项 DARTS ImageNet 双重 1% 已于北京时间 2026-07-31 16:52 完成，共 759 个 epoch 记录，并生成
CSV、PNG、SVG 与 HTML bundle。三个 full-data × 3 epoch 候选的最终 valid top-1 为
`39.528/38.624/29.852`，三个 1%-data × 250 epoch 候选为 `9.6/10.6/5.0`（顺序均为
ZCP-selected/fixed-random/params-matched）。首个 full-data run 使用 4-GPU DDP、其余使用单 GPU，
普通 BatchNorm 的每设备统计粒度不同，因此六项通过的是实现/恢复验收，不是严格同拓扑搜索收益结论。
吞吐、装箱、断点与六项结果摘要见
[`evidence/gpu_throughput_optimization.json`](evidence/gpu_throughput_optimization.json)。
AutoFormer 单候选双重 1% 已完成；PlainNet-MBV2、Proxyless-MBV2 的双重 1% 仍未完成。

自 2026-08-04 起，未来“双重 1% 工程验收”只对一个 `zcp-selected` 架构执行两项协议：全数据 ×
至少 1% epoch，以及恰好 1% 数据 × 完整 schedule。历史已完成或已启动的三候选产物保持不变，不重跑、
不改写，但不再定义新验收门槛。短训无法可靠证明 ZCP 优于随机或参数量/FLOPs 基线，因此工程验收不再
为这两个对照额外消耗三倍资源；优越性结论必须来自另行预声明、充分训练且多 seed 的研究实验。

另有历史合成 smoke run 执行：

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
| `reference_model` | `darts`、`autoformer`、`zennas_plainnet_mbv2`、`ofa_proxyless_mbv2`、`ofa_mbv3` | 静态模型结构已实现；正式训练仍须独立通过 `formal_training_ready` 门禁 |
| `reference_topology_pytorch_port` | `pit` | 发布结构的 PyTorch port；不得冒充官方数值 reference 或正式训练实现 |
| `proxy_approximation` | legacy toy | 只适合显式 opt-in 的方法学 smoke，不得参与正式训练或 reference 结论 |

PlainNet-MBV2 的 structure-string port、golden 与训练门禁见
[`evidence/PLAINNET_MBV2_FIDELITY_AUDIT_CN.md`](evidence/PLAINNET_MBV2_FIDELITY_AUDIT_CN.md)。
正式搜索控制器已锁定上游 100k、population 1024、batch 64、1/2 block replacement、top-1023
parent、无 crossover 与逐插入全历史四组件 log-rank；1024 来自固定 commit 的官方启动脚本覆盖值，
不是 Python parser 的默认 512，来源 SHA 见
[`evidence/plainnet_source_protocol_20260804.json`](evidence/plainnet_source_protocol_20260804.json)。
CPU rerank 的本机保守累计估计为 15,165.56 秒（约 4.21 小时），但 GPU batch64/224 preflight 和
正式 100k 搜索尚未完成，因此不能称搜索验收通过。

AutoFormer 仓库配置已在单候选双重 1% 终态核验后置为 `formal_training_ready: true`；PlainNet-MBV2
与 Proxyless-MBV2 仍为 false。AutoFormer 已有
AZ-NAS repeated-augmentation sampler、`base_lr × global_batch / 512` 缩放规则，以及 Cream T/S/B
和 AZ-NAS Tiny/Small/Base 六个精确参数量 golden。混合 4090D/4090 的 2-rank DARTS 与 AutoFormer
真实 GPU smoke 已验证 DDP、跨 rank 指标归约、单一共享 run 和仅 rank 0 写 artifact；失败 run
也会标为 failed。六个官方 `get_complexity` 数值已以 `official_complexity_ops` 独立字段复现；
AZ-NAS Tiny 的 THOP MAC 为 `1,100,420,352`，官方口径为 `1,380,128,376`，且 THOP 未计全
relative-position 参数，因此报告禁止将二者混称 FLOPs。真实 ImageNet 图片极小夹具上的 2-rank
中断/新 run 恢复已验证 `interrupted` manifest、checkpoint SHA-256/source run lineage、epoch 0–4
连续日志和 `.tmp` 清理；该证据只覆盖恢复机制，未覆盖完整 ImageNet 双重 1% 协议。
Proxyless-MBV2 已完成 TensorFlow 风格颜色扰动、官方结构/参数 shape 对照、精确
`265,526,256` MAC golden、OFA float32 `265,526,240` profile、global-batch 语义和 BN-only
no-decay；PlainNet-MBV2 已完成真实 structure-string、参数/MAC golden 与 150-epoch candidate
profile。两者仍缺双重 1% GPU、distributed validation 和完整报告验收，CLI 因而继续拒绝非 smoke
训练。
配置中的布尔值不能自行授权；正式训练还必须匹配代码内置的 DARTS 协议白名单及关键字段。

### AutoFormer 搜索、冻结与 V2 训练状态

AutoFormer AZ-NAS 搜索 cohort 已完成并 reconcile：三个 seed 各处理 8,000 个候选，合计
24,000 candidate rows、23,999 次 unique evaluation 和 1 次 cache hit。历史 supervisor 在全部 seed
产物完成后以 failed 终止；该 post-completion orchestration 状态保留用于审计，但不能把三个已完成
manifest、generation summary 和 cohort 归并结果改写为科学失败。完整 launcher 防护不再只复制 Shell：
它使用 `git archive` 固化启动 commit 的全部已跟踪 Shell、Python 和 configs，写入只读
`launcher-snapshots` 并从该树执行；后续 lane 不导入主仓新代码。逐 lane 锁只在活跃 workload
期间持有，完成或中断后立即释放。实现边界和历史 exit 127 的证据见
[`evidence/gpu_launcher_snapshot_fix.json`](evidence/gpu_launcher_snapshot_fix.json)。

候选已按预声明 primary/supporting 协议冻结并通过审计：`zcp_selected` 为
`42e6457ccb580a092454`，`fixed_random` 为 `d904aacf51d2b0867df6`，
`params_flops_matched` 为 `41b5e6d4dc3279909487`。三者 canonical ID 唯一，supporting seeds
只作 provenance，不参与平均、替换 winner 或事后挑选。冻结 manifest SHA-256 为
`42dc72f29e141fa97c042c1979f390486962a97fa34cdbcd3394b556148bdb4a`；详见
[`evidence/autoformer_frozen_candidates.json`](evidence/autoformer_frozen_candidates.json)。

三个冻结候选均完成 configured micro-batch 256 的 synthetic full-batch memory smoke，写出原子
`last.pt`/`best.pt`，并分别通过 trusted checkpoint-load/resume smoke。该证据只验证显存容量、训练
控制路径和 checkpoint 身份恢复，随机输入 accuracy 无意义；详见
[`evidence/autoformer_frozen_candidate_smokes.json`](evidence/autoformer_frozen_candidate_smokes.json)。

real dual-1% V2 于 2026-08-04 11:07+08 以 commit/source snapshot `76a0fcd` 启动。单候选 gate 的
full-data 5 epoch 于 12:52 完成，1%-data 500 epoch 于 13:18 完成，分别保存 5/500 条连续 epoch
记录、有限指标以及 `last.pt`/`best.pt`。旧 immutable
supervisor 随后自动启动尚未开始的 task 5/6，这与 2026-08-04 生效的单候选政策冲突。两项基线任务已
定向中止并写为 `interrupted`，GPU0/1 显存降至约 89/15 MiB、利用率 0%，两把 kernel flock 已释放。
旧 main supervisor 已在单候选终态后由 watcher 清理。数据预检为 1,000 类、
1,281,167 train 和 50,000 validation 文件；micro-batch 256，global batch 2,048 由梯度累积实现，
学习率协议未改变。该限定验收允许把仓库 profile 的 `formal_training_ready` 置为 true，但不能被写成
500-epoch 全数据论文精度复现或 ZCP 优于基线。状态、checksum、锁审计和科学边界见
[`evidence/autoformer_dual_one_percent_launch.json`](evidence/autoformer_dual_one_percent_launch.json) 与
[`evidence/autoformer_single_candidate_policy_intervention_20260804.json`](evidence/autoformer_single_candidate_policy_intervention_20260804.json)、
[`evidence/autoformer_single_candidate_dual_one_percent_completion_20260804.json`](evidence/autoformer_single_candidate_dual_one_percent_completion_20260804.json)。

NAS-Bench-101/201、NATS 和转换后的 TransNAS 记录只有在明确 dataset/split/budget/seed 下才是
**standard answer**。NAS-Bench-301 是 **surrogate**，deterministic/noisy 属于不同协议。
ViT-Bench 可能是 **scratch**、蒸馏或 **inherited-supernet**，不得混合。

## 当前部分验收

- 保留 evaluate 只有 3 个代理，不是 22；其 40 行历史 score 是旧 component-long schema。
- Git 不保存 3,454 行原始 score；当前树可独立复核精简摘要、四个 score SHA、失败键和 registry
  provenance，但逐行重算仍需要外部审计目录或按手册重新运行。
- AutoFormer `3×8,000` 搜索 cohort 与候选冻结已经完成；post-completion supervisor failure 仍作为
  orchestration 证据保留。V2 dual-1% 仍为非终态 `running`，不得称训练验收完成。
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
  ImageNet16-120 dataset-specific 扩展也分别完成 328 × 22 = 7,216 行，全部成功且无重复稳定键；
  三数据集报告将 dataset-specific、固定源 score 的 target-only transfer、proxy 排名稳定性和
  size/stage 控制相关性分表保存。该扩展只覆盖同一 1% 分层样本和单 seed 2026。证据见
  `docs/evidence/NATS_SSS_ONE_PERCENT_CN.md` 与 `docs/evidence/NATS_SSS_CROSS_DATASET_CN.md`。
- H1 已完成 NB101 的正式 1% 既定协议：从 423,624 个架构中按 seed 2026 分层抽取 4,237 个；
  22 代理 seed 2026 为 93,214/93,214 成功，核心 11 代理 seed 2026/2027/2028 为
  139,821/139,821 成功，均无失败或重复任务键。4/12/36/108 epoch 的 repeat `mean/min/max`
  预算分析均已完成。TE-NAS `portable-v2` 仅为仓库可移植近似，不等同官方完整 TE-NAS；本结论
  只覆盖该固定样本和协议。证据见
  [`docs/evidence/NB101_ONE_PERCENT_CN.md`](evidence/NB101_ONE_PERCENT_CN.md) 与
  [`docs/evidence/nb101_one_percent_summary.json`](evidence/nb101_one_percent_summary.json)。
- `portable-v1` 与 topology port 在论文复现声明前仍需对照官方实现。
- AutoFormer 新增 `az_nas_autoformer`：固定 AZ-NAS commit `5e6683a`，保存每个 block 的 attention/MLP
  残差特征，计算 expressivity/trainability/官方 complexity，并按三个组件的 log-rank 聚合。两候选
  ImageNet-224 GPU smoke 生成 2 条候选和 1 条 summary，组件缓存可恢复；模型初始化按架构哈希固定，
  两次独立同 seed GPU run 去除耗时后逐行一致。精简证据见
  `docs/evidence/aznas_autoformer_rank_smoke.json`。数值稳定版对协方差负零误差执行 clamp，fidelity 为
  `paper_formula_port_stabilized`；项目 evolution controller 不是上游候选控制器的逐行复刻。项目
  `3×8,000` 搜索 cohort 与候选冻结已经完成，但必须标记为项目控制器结果，不能冒充上游控制器逐行复现。
- AutoFormer 新增显式 `--full-batch-smoke`。使用配置 micro-batch 256 的 sampled reference subnet
  已完成一个 synthetic epoch，峰值 allocated/reserved 为 `8920/10390 MiB`，见
  `docs/evidence/autoformer_full_batch_memory_smoke.json`。该结果只证明单进程显存和训练步链路，不是
  ImageNet 精度证据，也不能替代冻结后三候选的真实数据/恢复验收；运行期间同卡存在其他用户约 10 GiB
  进程，进一步证明项目锁不是系统级排他锁。
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

1. DARTS CIFAR-10/CIFAR-100 的 600 epoch **全数据精度复现**与多 seed 搜索收益验证仍未完成；
   已完成的是上述双重 1% 限定协议。DARTS ImageNet 六项双重 1% 已完成，但首项 DDP 与其余单卡
   存在 BatchNorm 粒度差异；250 epoch 全数据正式训练不在本次限定验收范围内且尚未执行。
2. PlainNet-MBV2 与 Proxyless-MBV2 的双重 1% 尚未完成。AutoFormer 的单候选双重 1% 已完成并解除
   profile 启动门禁，但 500 epoch 全数据精度复现与多 seed 搜索收益仍未完成。PlainNet 只完成 3 个
   accepted candidate 的 GPU preflight，明确 `formal_search_completed=false`；Proxyless-MBV2 150 epoch
   正式训练仍未放行。
3. 在第二台干净机器完成 benchmark 下载、checksum 和来源核验。
4. 在其余 benchmark 的目标 dataset、split、budget、task 上运行各自 1% 协议；NB201、NATS-TSS、
   NATS-SSS 三数据集、NB101 与 NB301 deterministic surrogate 已完成各自上述限定协议，
   TNB101 正式输入以及 ViT-Bench 500 条全集/60-40 身份仍待，因此完整项目
   H1 尚未完成。ViT 三个公开 100 条切片的 5×22 预验收不能替代该缺口。
5. NAS-Bench-101 全量评估或 NAS-Bench-301 理论 DARTS 空间穷举。
6. 多 GPU evaluate 的内置启动/去重合并；训练 DDP 启动与夹具级重启/故障注入已验收，但全数据级别未验收。
7. 论文数值复现、独立 seed 置信区间及与官方代码的成本/精度比较。

正式验收必须保留 manifest、resolved config、commit、环境、输入 hash、结果类型、失败行和准确命令。
对上述尚未关闭的项目，在证据齐全前只能声明已有的软件验收、限定协议或 smoke 覆盖。

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
architecture ID 与查询均通过。旧三个切片各 5 个候选 × 22 ZCP 的行数和工作流完整，但 AutoFormer
分数是在错误的跨实现静态模型上计算，现只保留为 legacy 工作流证据，必须用
`vitbench-autoprox-90ed458` profile 重算；详见
`docs/evidence/AUTOFORMER_FIDELITY_AUDIT_CN.md`。此外论文声明的 500 GT/数据集及无重叠 60/40
身份仍未公开，因此重算后的公开 100 条切片也只能保持 partial。

可移植复验模板：

```bash
CATALOG=/path/to/data/catalog.json
zcp-test data checklist --root /path/to/data --catalog "$CATALOG" --json
zcp-test benchmark inspect nasbench101 --catalog "$CATALOG" \
  --dataset cifar10 --split valid --metric-name final_accuracy \
  --epoch-budget 108 --metric-seed-reduction mean
zcp-test benchmark inspect nasbench201 --trusted --catalog "$CATALOG" \
  --dataset cifar10-valid --split valid --metric-name accuracy --epoch-budget 200
zcp-test benchmark inspect nats_tss --trusted --catalog "$CATALOG" \
  --dataset cifar10-valid --split valid --metric-name accuracy --epoch-budget 200
zcp-test benchmark inspect nats_sss --trusted --catalog "$CATALOG" \
  --dataset cifar10-valid --split valid --metric-name accuracy --epoch-budget 90
```

`benchmark inspect` 不会把 `--data-root` 当作 benchmark 自动发现路径；必须使用 bootstrap 写入的
catalog，或逐项传入准确 `--path`。

同一轮还对上述十个 benchmark/切片执行了真实 index-0 `build_model → params proxy`。所有调用均为
`succeeded=1, failed=0, score_rows=1`；参数量依次为 NB101 `8,555,530`、NB201/NATS-TSS
`129,306`、NATS-SSS `11,714`、TNB micro `24,618`、TNB macro `2,318,890`、NB301 DARTS
`239,802`、ViT main `5,710,180`、ViT extension `8,755,324`、PiT `893,828`。该 smoke 显式使用
`input_source=random` 且只运行数据无关的 `params`，因此仅证明真实架构能够构模并进入统一 evaluator，
不能作为真实输入消融或相关性结果。
