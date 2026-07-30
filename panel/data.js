window.ZCP_PANEL_DATA = {
  schemaVersion: 2,
  updatedAt: "2026-07-30 21:33 CST",
  project: {
    name: "zcp-test",
    purpose: "跟踪 reference 模型、Benchmark、ZCP、训练、文档与高成本验收，确保每项结论都能追溯到风险和可复现证据。"
  },
  statuses: {
    "待开始": { color: "#68766f", order: 4 },
    "进行中": { color: "#1769aa", order: 2 },
    "已完成": { color: "#08785e", order: 3 },
    "受阻": { color: "#b4233d", order: 0 },
    "预算内未完成": { color: "#a95a08", order: 1 }
  },
  priorities: {
    P0: { label: "P0 · 阻断", order: 0 },
    P1: { label: "P1 · 高", order: 1 },
    P2: { label: "P2 · 常规", order: 2 },
    P3: { label: "P3 · 可选", order: 3 }
  },
  tasks: [
    {
      id: "A1", phase: "审计", priority: "P0", title: "冻结基线与质量 gate",
      content: "记录仓库状态、测试基线、依赖环境和最终全量 gate，区分定向 smoke 与全仓结论。",
      purpose: "建立可复现验收起点，防止局部通过被误写为全量通过。",
      estimate: "1–2 小时", startedAt: "2026-07-30 11:30", finishedAt: "—", status: "进行中", progress: 90,
      detail: "最新完整 gate 为 287 tests passed、第一方 source coverage 87%、CLI coverage 80%；NATS-SSS 真实协议与 run_id shard grouping 修复回归均已纳入，Ruff、compileall、pip check 与 git diff check 通过。",
      acceptance: ["记录 Python/依赖环境", "全量 pytest 与 Ruff 结果可追溯", "报告并发未提交改动"],
      evidence: ["EV-BASELINE", "EV-LOW-COST-GATE", "EV-COVERAGE", "EV-GIT-CHECKPOINT", "EV-FULL-GATE-219", "EV-FULL-GATE-222", "EV-FULL-GATE-223", "EV-FULL-GATE-231", "EV-FULL-GATE-240", "EV-FULL-GATE-251", "EV-FULL-GATE-252", "EV-FULL-GATE-262", "EV-FULL-GATE-282", "EV-FULL-GATE-286", "EV-FULL-GATE-287", "EV-FINAL-2RANK-ACCEPTANCE", "EV-ACCEPTANCE-CLI", "EV-IMAGENET-DDP-RESUME", "EV-DDP-SMOKE", "EV-AUTOFORMER-COMPLEXITY", "EV-OFA-INHERITED", "EV-OFA-BN-REAL", "EV-TRANSNAS-HEADS", "EV-AUTOFORMER-PROTOCOL"], risks: ["R-CONCURRENCY"], updatedAt: "2026-07-30 21:06"
    },
    {
      id: "A2", phase: "审计", priority: "P0", title: "统一模型 fidelity 与协议",
      content: "统一 reference、surrogate、approximation、scratch 与 inherited 的元数据和阻断规则。",
      purpose: "防止不同模型保真度和训练协议在报告中被混用。",
      estimate: "2–4 小时", startedAt: "2026-07-30 11:40", finishedAt: "2026-07-30 13:18", status: "已完成", progress: 100,
      detail: "核心枚举、artifact 字段和 CLI 正式训练保护已接入。",
      acceptance: ["artifact 携带 fidelity", "unsupported 正式路径明确失败", "scratch 与 inherited 分离"],
      evidence: ["EV-FIDELITY"], risks: [], updatedAt: "2026-07-30 13:18"
    },
    {
      id: "A3", phase: "安全", priority: "P0", title: "trusted 与 correlate 安全修复",
      content: "限制 trusted 只能由显式 CLI 确认，并阻止相关性计算中的重复键和静默错配。",
      purpose: "消除不可信反序列化绕过和错误相关性结论。",
      estimate: "1–2 小时", startedAt: "2026-07-30 12:00", finishedAt: "2026-07-30 13:18", status: "已完成", progress: 100,
      detail: "配置不能开启 trusted；重复键报错；score-field 已生效。",
      acceptance: ["配置 trusted 被拒绝", "重复 join key 报错", "指定 score-field 有测试"],
      evidence: ["EV-SAFETY"], risks: [], updatedAt: "2026-07-30 13:18"
    },
    {
      id: "B1", phase: "Benchmark", priority: "P0", title: "NB201 与 NATS-SSS reference topology",
      content: "实现 topology、reduction stage 与 stage width 可感知的 PyTorch 模型构建。",
      purpose: "确保架构字段真实进入模型和 ZCP 计算。",
      estimate: "4–8 小时", startedAt: "2026-07-30 12:10", finishedAt: "2026-07-30 16:28", status: "已完成", progress: 100,
      detail: "NB201、NATS-TSS 和 NATS-SSS 的 native index-0 query→build→params proxy 均通过；NB201/TSS 共享 architecture ID 但 benchmark/API 身份保持独立。NATS 查询的 seed_reduction=min|max 已修复为枚举官方 seed 后显式归约，真实 index-0 得到 mean 81.982667、min 81.616000、max 82.240000；不能枚举时明确失败。",
      acceptance: ["字段敏感性测试", "native API 架构对照", "真实样本 query→build→proxy"],
      evidence: ["EV-NB201", "EV-REAL-BENCHMARKS", "EV-NATS-TSS-1PCT-REPORT", "EV-FULL-GATE-286"], risks: [], updatedAt: "2026-07-30 20:41"
    },
    {
      id: "B2", phase: "Benchmark", priority: "P1", title: "TransNAS 端到端构模契约",
      content: "打通真实 JSONL 的 load、query、build 和 proxy evaluate，并明确 task-head 边界。",
      purpose: "修复转换规格存在但无法构模的验收缺口。",
      estimate: "4–8 小时", startedAt: "2026-07-30 12:20", finishedAt: "2026-07-30 17:19", status: "已完成", progress: 100,
      detail: "七个 Taskonomy task head 已按官方 commit 6d4231b 分离；官方参数量与完整 parameter-shape multiset 对照一致，真实 micro index-0 七任务 build→params 全部成功。缺失 task input/label provider 的标签依赖代理明确为 unsupported。",
      acceptance: ["真实 JSONL 端到端通过", "七任务输出 shape 独立", "官方 params/shape fixture 对照", "缺输入协议时 unsupported", "不误称训练数值复现"],
      evidence: ["EV-TRANSNAS", "EV-TRANSNAS-HEADS", "EV-FULL-GATE-231"], risks: ["R-TRANSNAS"], updatedAt: "2026-07-30 17:19"
    },
    {
      id: "B3", phase: "Benchmark", priority: "P1", title: "NB101、NB301 与 ViT 指标语义复核",
      content: "核查标准答案、surrogate noise、ViT slice 与 epoch/seed 指标协议。",
      purpose: "保证离线标签和 surrogate 输出不会被误标为同一标准答案。",
      estimate: "3–6 小时", startedAt: "2026-07-30 13:00", finishedAt: "2026-07-30 16:28", status: "已完成", progress: 100,
      detail: "NB101、deterministic NB301、ViT main/extension/PiT 的真实 query 与构模 proxy 均通过；extension 仅含 KD/inherited，vanilla 查询按设计失败。NB101 full 后端与 4/12/36/108 budget 语义已核验，H1 的 4,237 架构分层 manifest 已生成；这不等于 22-ZCP 正式 sweep 已完成。",
      acceptance: ["slice 身份可追溯", "surrogate noise 明确记录", "metric seed/epoch 不静默降级"],
      evidence: ["EV-REAL-BENCHMARKS", "EV-NB101-H1-PREPARATION"], risks: [], updatedAt: "2026-07-30 21:18"
    },
    {
      id: "C1", phase: "Reference", priority: "P0", title: "AutoFormer 静态 scratch reference",
      content: "提供可独立构建的静态 subnet，使逐层 depth、head 与 MLP ratio 真实影响模型。",
      purpose: "支撑无 inherited supernet 条件下的 AZ-NAS scratch/static 研究。",
      estimate: "6–12 小时", startedAt: "2026-07-30 13:00", finishedAt: "—", status: "进行中", progress: 95,
      detail: "六个 Cream/AZ-NAS 官方逐层配置的参数量与 official_complexity_ops golden 均通过；THOP 交叉检查证明不能混称 FLOPs。候选协议已锁定完整增强与来源字段；最终 2-rank acceptance 1 epoch 验证配置分别记录 Cream 模型 commit b799630… 与 AZ-NAS 训练 commit 5e6683…，run completed、training.jsonl 仅 1 行且无 .tmp。恢复机制已验证，但双重 1% 正式协议尚未完成，formal_training_ready 仍为 false。",
      acceptance: ["逐层字段改变参数或算子", "分类 forward 通过", "非法编码拒绝", "metadata 明确不支持 inherited", "官方 fixture 对照"],
      evidence: ["EV-REFERENCE-MODELS", "EV-AUTOFORMER-PROTOCOL", "EV-DDP-SMOKE", "EV-AUTOFORMER-COMPLEXITY", "EV-FULL-GATE-252", "EV-FULL-GATE-262", "EV-FINAL-2RANK-ACCEPTANCE", "EV-ACCEPTANCE-CLI", "EV-IMAGENET-DDP-RESUME"], risks: ["R-AUTOFORMER"], updatedAt: "2026-07-30 18:35"
    },
    {
      id: "C2", phase: "Reference", priority: "P0", title: "OFA-Proxyless MBV2 官方 positional encoding/reference fixture",
      content: "对齐 OFA-Proxyless MBV2 官方 positional encoding，并用 reference fixture 核验静态 MBConv 的结构与参数量。",
      purpose: "在保留 PlainNet-MBV2 与 Proxyless/OFA-style MBConv 分离的同时，消除位置编码和公开架构对照缺口。",
      estimate: "8–16 小时", startedAt: "2026-07-30 13:10", finishedAt: "2026-07-30 16:38", status: "已完成", progress: 100,
      detail: "官方 21 dynamic-block positional encoding 已实现；registered space 固定 width 1.3、resolution 128..224 step 4。官方 commit f03b267 fixture 对齐 width1.0=2,500,632 params、width1.3=3,718,832 params，且 width1.0 参数 shape multiset 与官方完全一致。后续 inherited checkpoint 与 active-weight export 已进入 C3；MAC golden 与正式训练验收仍未完成。",
      acceptance: ["两种模型类型独立", "官方 21-block positional encoding 一致", "registered width/resolution 边界固定", "官方 width1.0/1.3 params fixture 对照", "width1.0 参数 shape multiset 完全一致"],
      evidence: ["EV-REFERENCE-MODELS", "EV-MBV2-FIXTURE-START", "EV-MBV2-REFERENCE", "EV-OFA-INHERITED"], risks: ["R-MBV2-FIXTURE", "R-MBV2-REMAINING", "R-OFA"], updatedAt: "2026-07-30 16:54"
    },
    {
      id: "C3", phase: "Reference", priority: "P1", title: "OFA inherited 与 BN calibration",
      content: "加载官方 supernet/checkpoint，执行 active subnet、继承权重和 BN 统计校准。",
      purpose: "支持真正的 OFA inherited accuracy，并与 scratch 结果严格分离。",
      estimate: "8–16 小时", startedAt: "2026-07-30 16:54", finishedAt: "—", status: "进行中", progress: 85,
      detail: "checkpoint 自举、active-weight export、evaluate/search provenance 已通过；本机真实 ImageNet-1k 上完成 1 个独立 batch 的确定性 BN smoke，记录 sample ID、transform 与 fingerprint，且无空目录。该项目协议明确 official_protocol_match=false；官方 data-provider 数值对照、inherited accuracy、MAC golden 与 formal training 仍待完成。",
      acceptance: ["官方 checkpoint 可校验和加载", "bootstrap ready 文件注册 catalog", "active channel/kernel transform 导出", "混合 k/e/d 子网与官方一致", "evaluate/search 记录 inherited provenance", "真实数据 BN recalibration accuracy"],
      evidence: ["EV-OFA-INHERITED", "EV-OFA-BN-REAL", "EV-FULL-GATE-223"], risks: ["R-OFA", "R-MBV2-REMAINING"], updatedAt: "2026-07-30 17:05"
    },
    {
      id: "C4", phase: "Reference", priority: "P3", title: "PiT 与可选 OFA-MBV3",
      content: "实现 ViT-Bench PiT reference，并评估含 SE/h-swish 的 OFA-MBV3 静态网络。",
      purpose: "补足条件性空间，同时控制非必要实现范围。",
      estimate: "8–16 小时", startedAt: "2026-07-30 15:42", finishedAt: "—", status: "进行中", progress: 85,
      detail: "PiT 已完成真实 GT load→build→224 forward，并与 Auto-Prox 90ed458 参数量 893,828 和参数 shape multiset 对齐。OFA-MBV3 已按官方五阶段/20-block 编码实现 SE、h-swish、静态子网和 BN recalibration，官方对照参数量 3,410,792 且 shape multiset 一致。PiT 固定 benchmark 候选无需重复完整训练；两者仍缺 MAC golden。OFA-Proxyless checkpoint 与 active-weight export 已在 C3 完成，OFA-MBV3 inherited 路径仍未单独验收。",
      acceptance: ["三阶段字段进入 PiT 模型", "真实 gt_pit load→build→224 forward", "PiT 官方参数 fixture", "OFA-MBV3 官方静态结构与 BN recalibration", "MAC fixture 与 inherited 边界"],
      evidence: ["EV-PIT-REFERENCE", "EV-MBV3-REFERENCE", "EV-OFA-INHERITED"], risks: ["R-PIT", "R-OFA"], updatedAt: "2026-07-30 16:54"
    },
    {
      id: "D1", phase: "训练", priority: "P0", title: "拆分 DARTS original 与 TE-NAS",
      content: "分离 optimizer、scheduler、drop-path 和 provenance 不同的训练 profile。",
      purpose: "修复原始 DARTS 与 TE-NAS retrain recipe 混用。",
      estimate: "2–4 小时", startedAt: "2026-07-30 12:40", finishedAt: "—", status: "进行中", progress: 80,
      detail: "配置和 scheduler 路径已拆分；GPU、恢复训练和完整 profile 复验待 gate。",
      acceptance: ["original/TE-NAS 名称分离", "step/cosine 行为测试", "恢复训练 LR 连续"],
      evidence: ["EV-TRAINING"], risks: [], updatedAt: "2026-07-30 14:10"
    },
    {
      id: "D2", phase: "训练", priority: "P1", title: "scheduler、恢复身份与 1% 数据",
      content: "实现 cosine/step/none、严格 checkpoint identity 和确定性分层子集。",
      purpose: "支持双重 1% 可恢复训练验收。",
      estimate: "3–6 小时", startedAt: "2026-07-30 12:45", finishedAt: "2026-07-30 13:37", status: "已完成", progress: 100,
      detail: "定向单测覆盖 scheduler、身份不匹配拒绝、分层子集、梯度累积与 non-primary artifact 禁写；acceptance CLI 锁定为全数据且不超过 1% epoch，或不超过 1% 数据且运行完整 schedule，候选配置同时锁定完整增强与来源字段。真实双卡中断/恢复验证新 run 的 epoch 0–4 连续、checkpoint SHA-256/source_run_id lineage 及仅 rank 0 写 artifacts；最终 2-rank acceptance 1 epoch run completed、仅 1 行训练记录且无 .tmp。",
      acceptance: ["三种 scheduler 有测试", "错误 checkpoint 身份拒绝", "全数据且不超过 1% epoch", "不超过 1% 数据且完整 schedule", "trusted checkpoint 恢复 lineage 可追溯", "仅 rank 0 写 artifacts"],
      evidence: ["EV-TRAINING", "EV-DDP-SMOKE", "EV-FULL-GATE-251", "EV-FULL-GATE-262", "EV-FINAL-2RANK-ACCEPTANCE", "EV-ACCEPTANCE-CLI", "EV-IMAGENET-DDP-RESUME"], risks: [], updatedAt: "2026-07-30 18:35"
    },
    {
      id: "E1", phase: "ZCP", priority: "P0", title: "22 ZCP 契约与算法 provenance",
      content: "逐项标记论文兼容、近似、别名和 unsupported，并建立 golden 验证。",
      purpose: "运行成功之外，验证代理公式和输入协议的可解释性。",
      estimate: "8–16 小时", startedAt: "2026-07-30 12:30", finishedAt: "—", status: "进行中", progress: 70,
      detail: "22/22 CPU sweep 与多个 benchmark 的真实协议已运行。NB101 正式旧 sweep 暴露 SynFlow v1/TE-NAS portable-v1 在部分深 DAG 上 float32 溢出；失败记录保留。现已实现 SynFlow double-v2 与 TE-NAS portable-v2，深模型 dtype/state 单测通过，已知失败 index 1566 的 CPU 回归 2/2 成功。版本升级不回写旧记录；正式旧 sweep 完成其余 20 代理后，再独立补跑两个 v2。alias、approximation 和论文公式 golden 审计仍未闭环。",
      acceptance: ["全部 proxy 可分类", "论文公式 golden fixture", "alias/approximation/常数与方向审计闭环", "approximation 在 artifact 可见"],
      evidence: ["EV-ZCP-SWEEP", "EV-NB201-1PCT-SUMMARY", "EV-NB101-SYNFLOW-V2-REGRESSION"], risks: ["R-PROXY", "R-NB101-SYNFLOW-OVERFLOW"], updatedAt: "2026-07-30 21:33"
    },
    {
      id: "E2", phase: "研究", priority: "P1", title: "通用 ZCP 分析",
      content: "验收互相关、top-k、稳定性、Pareto、transfer 与样本收敛分析。",
      purpose: "形成可复用且不会静默错配的研究报告。",
      estimate: "6–12 小时", startedAt: "2026-07-30 12:50", finishedAt: "—", status: "进行中", progress: 80,
      detail: "NB201、NATS-TSS 与 NATS-SSS/CIFAR10-valid 的真实 correlation、compare、三 seed stability 和 bundle 均已生成。专属研究曾把 run_id/source_run 纳入协议 grouping，导致互斥 shard 分别统计；现已移除 run 标识、按 evaluation seed 分组，并以测试锁定“同 seed 合并 shard、不同 seed 分离”。search/training、Pareto、跨数据集 transfer 和样本收敛仍待更多真实验收，因此不能标记完成。",
      acceptance: ["缺列与重复键明确报错", "coverage/ties/constant/direction 显式报告", "多 run 来源保留", "真实结果可生成全部表格", "search/training 分析真实验收"],
      evidence: ["EV-NB201-REAL-ANALYSIS", "EV-NB201-1PCT-SUMMARY", "EV-NB201-1PCT-REPORT", "EV-NB201-CORE-3SEED-SUMMARY", "EV-NB201-CORE-3SEED-REPORT", "EV-NATS-TSS-1PCT-SUMMARY", "EV-NATS-TSS-1PCT-REPORT", "EV-NATS-SSS-1PCT-SUMMARY", "EV-NATS-SSS-1PCT-REPORT", "EV-SHARD-GROUPING-FIX", "EV-FULL-GATE-282", "EV-FULL-GATE-286", "EV-FULL-GATE-287"], risks: [], updatedAt: "2026-07-30 21:06"
    },
    {
      id: "E3", phase: "研究", priority: "P2", title: "Benchmark 定制研究",
      content: "按预算、操作、size、任务和 ViT 参数分析结构偏置。",
      purpose: "避免不同 benchmark 只输出同一套泛化统计。",
      estimate: "8–16 小时", startedAt: "2026-07-30 13:00", finishedAt: "—", status: "进行中", progress: 70,
      detail: "修复 run_id shard grouping 后，NB201 与 NATS-TSS 当前 topology 表均为 157 architecture、942 edge、5 operations、6,720 correlations、840 operation effects、588 matched pairs、504 summaries；旧 26,880/3,360/168/168 是按四个 shard 分拆的错误口径，不再作为当前值。NATS-SSS/CIFAR10-valid size 表为 328 architecture、1,640 stage、12 summary、3,528 correlations、840 stage sensitivity、672 size-controlled correlations、112 strata。NATS-SSS 跨 CIFAR100/ImageNet16 及 NB101、NB301、TNB101、ViT 仍待。",
      acceptance: ["各 benchmark 有专属因子", "真实数据生成图表", "结论关联原始字段", "不同 benchmark 协议不混合或外推"],
      evidence: ["EV-NB201-REAL-ANALYSIS", "EV-NB201-1PCT-SUMMARY", "EV-NB201-1PCT-REPORT", "EV-NATS-TSS-1PCT-SUMMARY", "EV-NATS-TSS-1PCT-REPORT", "EV-NATS-SSS-1PCT-SUMMARY", "EV-NATS-SSS-1PCT-REPORT", "EV-SHARD-GROUPING-FIX"], risks: [], updatedAt: "2026-07-30 21:06"
    },
    {
      id: "F1", phase: "文档", priority: "P1", title: "中英文操作手册审计",
      content: "逐条验证跨机器数据自举、评估、搜索、训练和报告命令。",
      purpose: "保证文档示例可执行且不依赖作者机器路径。",
      estimate: "6–10 小时", startedAt: "2026-07-30 12:20", finishedAt: "—", status: "进行中", progress: 70,
      detail: "README 已加入中文主流程和有/无标准答案边界；新增统一训练手册，修正 DARTS ImageNet LR、内联 architecture 与 reference-model/正式训练协议混淆。完整命令矩阵仍需逐条复跑。",
      acceptance: ["无本机绝对路径", "中英文行为一致", "关键命令有 smoke 证据"],
      evidence: ["EV-DOC-AUDIT"], risks: ["R-CONCURRENCY"], updatedAt: "2026-07-30 15:02"
    },
    {
      id: "F2", phase: "报告", priority: "P1", title: "验收报告与复现实例",
      content: "整理完整、部分、受阻结论及对应命令、文件和运行产物。",
      purpose: "让所有验收判断都能被第三方复核。",
      estimate: "3–6 小时", startedAt: "2026-07-30 13:20", finishedAt: "—", status: "进行中", progress: 70,
      detail: "初稿和结构已具备；已链接 NB201、NATS-TSS、NATS-SSS/CIFAR10-valid 的单 seed/核心三 seed机器摘要、中文证据和 shard grouping 修复边界。NATS-SSS 跨数据集、NB101、NB301、TNB101、ViT 和高成本训练结果仍待回填。",
      acceptance: ["每个结论链接证据", "受阻原因明确", "不把 smoke 写成完整训练"],
      evidence: ["EV-NB201-1PCT-SUMMARY", "EV-NB201-1PCT-REPORT", "EV-NB201-CORE-3SEED-SUMMARY", "EV-NB201-CORE-3SEED-REPORT", "EV-NATS-TSS-1PCT-SUMMARY", "EV-NATS-TSS-1PCT-REPORT", "EV-NATS-SSS-1PCT-SUMMARY", "EV-NATS-SSS-1PCT-REPORT", "EV-SHARD-GROUPING-FIX"], risks: [], updatedAt: "2026-07-30 21:06"
    },
    {
      id: "F3", phase: "报告", priority: "P1", title: "report/monitor/run 发现",
      content: "支持单 run 自动发现，并对多 run 歧义和 legacy HTML 路径明确处理。",
      purpose: "避免把 runs 根目录误当成单个 run。",
      estimate: "2–3 小时", startedAt: "2026-07-30 12:50", finishedAt: "2026-07-30 13:39", status: "已完成", progress: 100,
      detail: "单 run、多 run 歧义和 legacy 路径均有测试。",
      acceptance: ["单 run 自动发现", "多 run 歧义报错", "legacy HTML 路径兼容"],
      evidence: ["EV-RUN-DISCOVERY"], risks: [], updatedAt: "2026-07-30 13:39"
    },
    {
      id: "F4", phase: "报告", priority: "P1", title: "HTML 实时任务看板",
      content: "维护无外部依赖的数据驱动看板，提供筛选、统计、风险、证据、详情和主题切换。",
      purpose: "让并发任务状态和验收边界可快速审阅、可持续更新。",
      estimate: "2–4 小时", startedAt: "2026-07-30 14:15", finishedAt: "2026-07-30 19:25", status: "已完成", progress: 100,
      detail: "看板以带缓存破坏参数的动态 data.js 脚本重载实现无需 F5 的更新，不依赖 fetch。提供立即刷新、自动刷新开关、15/30/60/300 秒间隔、上次成功刷新时间和倒计时；隐藏页暂停，恢复可见时立即检查。刷新后保留筛选状态并重绘，失败时恢复旧 data 对象并显示非阻塞错误。HTTP 静态服务审计已验证 index.html、app.js 与两个不同 cache-busting query 的 data.js 均返回 200；源码契约检查覆盖手动/自动触发、定时重排、visibility 暂停和失败回退。",
      acceptance: ["无 CDN 或服务端依赖", "任务必填字段齐全", "筛选/搜索/详情可用", "立即刷新按钮", "自动刷新开关与 15/30/60/300 秒间隔", "上次成功刷新与下次刷新倒计时", "visibility 隐藏暂停且恢复立即检查", "cache-busting data.js 且不依赖 fetch 或整页 reload", "保留筛选状态并避免重复静态监听器", "失败保留旧数据", "aria-live 可访问状态", "HTTP 静态资源与不同 query 实测 200", "Node 语法、数据契约与 diff 检查通过"],
      evidence: ["EV-PANEL", "EV-PANEL-REFRESH", "EV-PANEL-REFRESH-CONTROLS", "EV-PANEL-AUTO-REFRESH-NODE", "EV-PANEL-HTTP-REFRESH-AUDIT"], risks: [], updatedAt: "2026-07-30 20:17"
    },
    {
      id: "G1", phase: "验收", priority: "P0", title: "新增代码全量质量 gate",
      content: "执行全量测试、静态检查、覆盖率和关键模块回归。",
      purpose: "确认并发修复合并后无回归。",
      estimate: "2–4 小时", startedAt: "2026-07-30 14:50", finishedAt: "2026-07-30 15:42", status: "已完成", progress: 100,
      detail: "当前完整 gate 为 287 tests passed；第一方 source coverage 87%、CLI 80%，Ruff、compileall、pip check 与 git diff check 通过；NATS-SSS 协议和 shard grouping 回归均已纳入测试。",
      acceptance: ["全量 pytest", "Ruff 通过", "source coverage ≥85%", "关键模块 ≥80%"],
      evidence: ["EV-LOW-COST-GATE", "EV-COVERAGE", "EV-FULL-GATE-219", "EV-FULL-GATE-222", "EV-FULL-GATE-223", "EV-FULL-GATE-231", "EV-FULL-GATE-240", "EV-FULL-GATE-251", "EV-FULL-GATE-252", "EV-FULL-GATE-262", "EV-FULL-GATE-282", "EV-FULL-GATE-286", "EV-FULL-GATE-287", "EV-FINAL-2RANK-ACCEPTANCE", "EV-DDP-SMOKE", "EV-AUTOFORMER-COMPLEXITY"], risks: ["R-CONCURRENCY"], updatedAt: "2026-07-30 21:06"
    },
    {
      id: "G2", phase: "验收", priority: "P0", title: "全 Benchmark 真实 smoke",
      content: "在本机已注册真实资产上完成初始化、query、build 和最小 proxy。",
      purpose: "验证 adapter 与本地数据的实际可用性。",
      estimate: "3–8 小时", startedAt: "2026-07-30 16:18", finishedAt: "2026-07-30 16:28", status: "已完成", progress: 100,
      detail: "NB101、NB201、NATS-TSS/SSS、TNB micro/macro、NB301、ViT main/extension/PiT 共十个切片均完成真实 index-0 query→build→params proxy，全部 succeeded=1、failed=0。random input 仅作构模 smoke，不进入相关性结论。",
      acceptance: ["每个已注册 benchmark 有真实 smoke", "失败资产明确记录", "无 synthetic 替代"],
      evidence: ["EV-TRANSNAS", "EV-REAL-BENCHMARKS"], risks: [], updatedAt: "2026-07-30 16:28"
    },
    {
      id: "H1", phase: "高成本", priority: "P1", title: "至少 1% Benchmark 相关性",
      content: "在真实标准答案上执行 22 ZCP 分层相关性实验。",
      purpose: "验证代理排序而不仅是执行成功。",
      estimate: "12–24 小时", startedAt: "2026-07-30 18:55", finishedAt: "—", status: "进行中", progress: 52,
      detail: "NB201、NATS-TSS、NATS-SSS/CIFAR10-valid 当前既定协议完成，H1 整体仍进行中。NB101 正式旧 sweep 的四个固定 shard 均已在 GPU 上运行；auto GPU 启动延迟缺陷已修复，旧进程在约 120 秒后四卡均正常启动。旧 SynFlow v1/TE-NAS portable-v1 对部分深 NB101 架构发生 float32 溢出，失败继续保留；旧 sweep 继续收集其余 20 代理。SynFlow double-v2 与 TE-NAS portable-v2 已通过深模型单测及 index 1566 CPU 回归 2/2，待旧 sweep 结束后单独补跑，不把旧失败伪装为成功。核心 11 代理三 seed仍待；看板不固化瞬时行数。",
      acceptance: ["真实标签不少于 1%", "全部代理至少单 seed", "核心代理 3 seed", "预算记录完整"],
      evidence: ["EV-NB201-1PCT-SUMMARY", "EV-NB201-1PCT-REPORT", "EV-NB201-CORE-3SEED-SUMMARY", "EV-NB201-CORE-3SEED-REPORT", "EV-NATS-TSS-1PCT-SUMMARY", "EV-NATS-TSS-1PCT-REPORT", "EV-NATS-SSS-1PCT-SUMMARY", "EV-NATS-SSS-1PCT-REPORT", "EV-SHARD-GROUPING-FIX", "EV-NB101-H1-PREPARATION", "EV-NB101-FORMAL-SWEEP-START", "EV-GPU-LOCK-DELAY-FIX", "EV-NB101-SYNFLOW-V2-REGRESSION", "EV-FULL-GATE-282", "EV-FULL-GATE-286", "EV-FULL-GATE-287"], risks: ["R-BUDGET", "R-PROXY", "R-GPU-LOCK-DELAY", "R-NB101-SYNFLOW-OVERFLOW"], updatedAt: "2026-07-30 21:33"
    },
    {
      id: "H2", phase: "高成本", priority: "P1", title: "全数据 × 1% epoch",
      content: "在全训练数据上执行约 1% 正式 epoch，并验证曲线与恢复。",
      purpose: "验收 reference 模型、优化器、数据增强和 checkpoint。",
      estimate: "12–24 小时", startedAt: "—", finishedAt: "—", status: "待开始", progress: 0,
      detail: "acceptance CLI 已锁定全数据且不超过 1% epoch，并完成最终 2-rank acceptance 1 epoch 的配置 provenance 与 artifact 生命周期验证；本项完整真实 ImageNet 全数据协议仍未执行。",
      acceptance: ["真实全数据", "不超过 1% epoch", "恢复结果连续", "协议 metadata 完整"],
      evidence: ["EV-ACCEPTANCE-CLI", "EV-IMAGENET-DDP-RESUME", "EV-FINAL-2RANK-ACCEPTANCE"], risks: ["R-BUDGET", "R-AUTOFORMER", "R-OFA"], updatedAt: "2026-07-30 18:35"
    },
    {
      id: "H3", phase: "高成本", priority: "P2", title: "1% 数据 × 完整 schedule",
      content: "使用确定性 1% 数据跑完整 scheduler 和 checkpoint 生命周期。",
      purpose: "验证长期调度和监控，不作为正式精度结论。",
      estimate: "24–48 小时", startedAt: "—", finishedAt: "—", status: "待开始", progress: 0,
      detail: "acceptance CLI 已锁定不超过 1% 数据且运行完整 schedule；当前仅完成真实 ImageNet 极小夹具的恢复机制验证，本项长期 schedule 尚未执行。",
      acceptance: ["不超过 1% 数据", "完整 schedule", "监控产物持续写入", "48 小时内停止", "报告不声称正式精度"],
      evidence: ["EV-ACCEPTANCE-CLI", "EV-IMAGENET-DDP-RESUME"], risks: ["R-BUDGET", "R-AUTOFORMER"], updatedAt: "2026-07-30 18:29"
    },
    {
      id: "I1", phase: "发布", priority: "P1", title: "清理、提交与发布",
      content: "审查敏感信息、大文件、数据、缓存和本机路径，分阶段整理提交。",
      purpose: "保证发布仓库可复现且不泄露本地资产。",
      estimate: "1–3 小时", startedAt: "—", finishedAt: "—", status: "待开始", progress: 0,
      detail: "误保存的 Google 搜索结果 .html 已删除并复查；其余发布清理仍需等待全部验收任务完成或形成明确受阻结论。",
      acceptance: ["无凭据和本机路径", "无数据/大模型误提交", "状态与验收报告一致"],
      evidence: ["EV-REPO-HYGIENE"], risks: ["R-CONCURRENCY"], updatedAt: "2026-07-30 14:50"
    }
  ],
  risks: [
    { id: "R-CONCURRENCY", severity: "高", status: "开放", title: "并发工作区冲突", description: "多个工作者正在修改仓库，状态和测试基线可能快速变化。", mitigation: "只修改授权路径；最终 gate 前重新读取 git status，不回退他人改动。", taskIds: ["A1", "F1", "G1", "I1"] },
    { id: "R-NATIVE", severity: "高", status: "关闭", title: "真实 Benchmark index-0 smoke 已完成", description: "十个 benchmark/切片的真实 query、构模与 params proxy 均通过；external catalog 资产仍不等于 data root 自包含。", mitigation: "跨机器继续执行 bootstrap/checklist；后续 1% 相关性必须使用真实 dataset input。", taskIds: ["B1", "B3", "G2"] },
    { id: "R-TRANSNAS", severity: "中", status: "开放", title: "TransNAS Taskonomy input/label provider 未接入", description: "七任务 encoder/head 结构已对照官方，但真实 task input、dense/regression label、训练数值、latency/FLOPs 尚未复现。", mitigation: "后续接入受许可的 Taskonomy 数据与 task-specific loss；在此之前标签依赖代理返回 unsupported，random-input 结果只作消融。", taskIds: ["B2", "H1"] },
    { id: "R-AUTOFORMER", severity: "高", status: "开放", title: "AutoFormer 双重 1% 正式协议尚未验收", description: "恢复机制与最终 2-rank acceptance 1 epoch 的配置 provenance、completed manifest、单行训练记录和无 .tmp 已通过；模型 commit b799630… 与训练 commit 5e6683… 已分离记录。上述仍不是完整 ImageNet 的全数据≤1% epoch 或≤1%数据+完整 schedule。", mitigation: "formal_training_ready 继续为 false；分别执行并验收两种真实 ImageNet 1% 锁定协议后再放行。", taskIds: ["C1", "H2", "H3"] },
    { id: "R-MBV2-FIXTURE", severity: "高", status: "关闭", title: "OFA-Proxyless MBV2 positional/params fixture 已验收", description: "官方 commit f03b267 的 21 dynamic-block positional encoding、width1.0/1.3 参数量及 width1.0 参数 shape multiset 已完成对照。", mitigation: "保留固定 commit、registered space 边界和回归测试；MAC 与训练边界由独立风险继续跟踪。", taskIds: ["C2"] },
    { id: "R-MBV2-REMAINING", severity: "高", status: "开放", title: "OFA-Proxyless MBV2 MAC 与正式训练未验收", description: "params/shape fixture 已通过，但官方 MAC golden 尚缺，正式训练协议也未完成验收；不得由静态 reference 结论外推训练精度或成本。", mitigation: "补充同一官方 commit 的 MAC fixture；关闭训练 blocker 并完成正式 profile 验收前，仅报告 static scratch reference。", taskIds: ["C2", "H2"] },
    { id: "R-OFA", severity: "高", status: "开放", title: "OFA inherited accuracy 与完整协议未验收", description: "官方 checkpoint、catalog bootstrap、active-weight export、子网数值一致性、evaluate/search 和真实 ImageNet 确定性 BN smoke 已完成；当前项目 BN 协议不等同官方 data provider，accuracy、MAC golden 与 formal training 仍未完成。", mitigation: "对照官方 OFA data provider 的抽样、transform 与 BN 统计并执行 inherited accuracy；补齐 MAC golden。正式训练 blocker 关闭前不得外推 inherited 或 scratch 训练结论。", taskIds: ["C2", "C3", "C4", "H2"] },
    { id: "R-PROXY", severity: "高", status: "开放", title: "代理可运行不等于论文一致", description: "22/22 sweep 不能替代公式、聚合方向和输入协议的 golden 验证。", mitigation: "为核心代理增加论文级数值 fixture 和 provenance。", taskIds: ["E1", "H1"] },
    { id: "R-COVERAGE", severity: "中", status: "关闭", title: "报告模块覆盖率已达标", description: "当前第一方 coverage 87%、CLI 80%，总计与关键模块门槛均已达到。", mitigation: "维持现有覆盖率 gate，后续改动继续执行全量回归。", taskIds: ["A1", "F3", "G1"] },
    { id: "R-PIT", severity: "中", status: "开放", title: "PiT MAC 对照尚未完成", description: "真实 GT 的 224 forward、官方参数量与参数 shape multiset 已对齐；MAC golden 尚缺。PiT 是固定 benchmark 候选，不要求重复完整训练。", mitigation: "补充同一官方 commit 的 MAC fixture 后关闭结构验收；vanilla/KD 指标继续分协议查询。", taskIds: ["C4"] },
    { id: "R-BUDGET", severity: "中", status: "监控", title: "高成本任务预算", description: "1% benchmark 与双重 1% 训练可能超出 GPU 或 48 小时上限。", mitigation: "最多 4 GPU；24 小时复核；48 小时硬停止并保留部分结论。", taskIds: ["H1", "H2", "H3"] },
    { id: "R-GPU-LOCK-DELAY", severity: "中", status: "关闭", title: "auto GPU 非零锁超时启动延迟已修复", description: "旧实现先等待最佳卡再探测其他卡，导致约 120 秒启动延迟；旧四个 NB101 进程随后均正常占用四卡，未形成数据失败。", mitigation: "auto 选择现先以零超时探测全部候选，再在一个全局 timeout 内轮询；tests/test_gpu.py 15 passed 且 Ruff 通过，保留回归测试。", taskIds: ["H1"] },
    { id: "R-NB101-SYNFLOW-OVERFLOW", severity: "中", status: "监控", title: "NB101 旧 SynFlow/TE-NAS float32 溢出", description: "正式旧 sweep 中 SynFlow v1 与 TE-NAS portable-v1 对部分深 NB101 DAG 返回非有限值；这些失败必须作为旧版本结果保留。", mitigation: "使用 SynFlow double-v2 与 TE-NAS portable-v2；深模型单测和 index 1566 CPU 回归 2/2 已通过。旧 sweep 先完成其余 20 代理，再单独补跑两个新版本并按版本合并，不覆盖旧失败。", taskIds: ["E1", "H1"] }
  ],
  evidence: [
    { id: "EV-NB101-SYNFLOW-V2-REGRESSION", time: "2026-07-30 21:33", title: "NB101 深模型 SynFlow/TE-NAS v2 回归", result: "旧 SynFlow v1/TE-NAS portable-v1 的 float32 非有限失败保留。SynFlow double-v2 使用 float64 并恢复模型 dtype/state，TE-NAS portable-v2 采用该组件；深模型单测通过，已知失败 benchmark index 1566 的 CPU 真数据回归为 synflow/te_nas 2/2 成功。正式旧 sweep 结束后仅补跑这两个新版本。", command: "pytest -q tests/test_core.py -k synflow; zcp-test evaluate --benchmark nasbench101 --sample-manifest <INDEX_1566_MANIFEST> --proxies synflow,te_nas --device cpu", taskIds: ["E1", "H1"] },
    { id: "EV-GPU-LOCK-DELAY-FIX", time: "2026-07-30 21:33", title: "auto GPU 候选锁探测修复", result: "auto GPU 在非零 lock timeout 下现先零超时探测全部候选，再使用一个全局 timeout 轮询；tests/test_gpu.py 15 passed、Ruff 通过。修复前启动的四个 NB101 shard 在约 120 秒后均已正常运行，启动延迟未记为数据失败。", command: "pytest -q tests/test_gpu.py; ruff check src/zcp_test/cli.py src/zcp_test/gpu.py tests/test_gpu.py", taskIds: ["H1"] },
    { id: "EV-NB101-FORMAL-SWEEP-START", time: "2026-07-30 21:23", title: "NB101 正式 22-ZCP 四分片启动", result: "4 架构 × 22 ZCP GPU smoke 已完成 88/88。正式 4,237 架构 × 22 ZCP seed 2026 在四张 GPU 上启动：shard 0 为 1,060 架构/23,320 调用，shard 1–3 各为 1,059 架构/23,298 调用；当前只判定运行中，不记录瞬时完成行数。启动时观察到的 auto GPU 锁等待延迟后续已修复，未被记为数据失败。", command: "zcp-test evaluate --benchmark nasbench101 --sample-manifest <NB101_MANIFEST> --sample-shard 0..3 --proxies <22_ZCP> --seed 2026 --gpu <FIXED_GPU>", taskIds: ["H1"] },
    { id: "EV-NB101-H1-PREPARATION", time: "2026-07-30 21:18", title: "NB101 H1 后端、预算与抽样准备", result: "NB101 full 后端及 4/12/36/108 budget 已核验；population 423,624 的 proportional feature-stratified 1% manifest 含 4,237 架构，SHA-256 e54cba029c74197037f1268f1689ec2b198261641133fccd6acda4a89c67c347。初始 GPU 4 架构 × 22 ZCP smoke 为 88/88 成功，但风险验证仍进行中；正式 22-ZCP 与核心三 seed未执行。", command: "zcp-test benchmark sample nasbench101 --fraction 0.01 --seed 2026 --shards 4; sha256sum <AUDIT_ROOT>/sampling/nb101-1pct-seed2026.json; GPU 4-architecture × 22-ZCP smoke", taskIds: ["B3", "H1"] },
    { id: "EV-FULL-GATE-287", time: "2026-07-30 21:06", title: "NATS-SSS 与 shard grouping 修复后完整 gate", result: "全量 287 tests passed；第一方 source coverage 87%、CLI coverage 80%，Ruff、compileall、pip check 与 git diff check 通过。回归覆盖同 seed 跨 shard 合并、不同 evaluation seed 分离及 NATS-SSS 协议。", command: "conda run -n zcp-test python -m coverage run -m pytest -q && conda run -n zcp-test python -m coverage report -m && conda run -n zcp-test ruff check . && conda run -n zcp-test python -m compileall -q src tests && conda run -n zcp-test python -m pip check && git diff --check", taskIds: ["A1", "E2", "E3", "G1", "H1"] },
    { id: "EV-NATS-SSS-1PCT-REPORT", time: "2026-07-30 21:02", title: "NATS-SSS CIFAR10-valid 1% 中文证据", result: "NATS-SSS v1.0 的 CIFAR10-valid/90-epoch 协议完成：328×22=7,216 行全部成功，核心 11 代理三 seed=10,824 行全部成功，均无重复键；size 专属表与跨数据集待办边界均已记录。", command: "docs/evidence/NATS_SSS_ONE_PERCENT_CN.md", taskIds: ["E2", "E3", "F2", "H1"] },
    { id: "EV-NATS-SSS-1PCT-SUMMARY", time: "2026-07-30 21:02", title: "NATS-SSS CIFAR10-valid 机器摘要", result: "manifest SHA-256 07767985afbad7d498acf062620aea5ef7b66b2bfc8b3db3fea3a0fc768e1992；单 seed 7,216/7,216/0/0，核心三 seed 10,824/10,824/0/0；合并 SHA-256 81622da200341d5d025086e5a8da849ae1e12b5b66d9cf78f8a0e12badf47d4d。", command: "docs/evidence/nats_sss_one_percent_summary.json", taskIds: ["E2", "E3", "F2", "H1"] },
    { id: "EV-SHARD-GROUPING-FIX", time: "2026-07-30 21:01", title: "专属研究 shard grouping 修复", result: "run_id/source_run 不再作为科学协议分组键；同一 evaluation seed 的互斥 shard 合并，不同 seed 保持分离。重建后 NB201/NATS-TSS topology 当前值均为 157/942/5/6,720/840/588/504；NATS-SSS size 当前值为 328/1,640/12/3,528/840/672/112。旧 26,880/3,360/168/168 是按 shard 拆分的错误口径。", command: "docs/evidence/NB201_ONE_PERCENT_22ZCP_CN.md; docs/evidence/NATS_TSS_ONE_PERCENT_CN.md; docs/evidence/NATS_SSS_ONE_PERCENT_CN.md", taskIds: ["E2", "E3", "F2", "H1"] },
    { id: "EV-FULL-GATE-286", time: "2026-07-30 20:41", title: "NATS-TSS 协议与 seed reduction 修复后完整 gate", result: "全量 286 tests passed；第一方 source coverage 87%、CLI coverage 80%，Ruff、compileall、pip check 与 git diff check 通过。回归覆盖 NATS mean/min/max seed reduction、CLI/bundle failed coverage 与既有协议。", command: "conda run -n zcp-test python -m coverage run -m pytest -q && conda run -n zcp-test python -m coverage report -m && conda run -n zcp-test ruff check . && conda run -n zcp-test python -m compileall -q src tests && conda run -n zcp-test python -m pip check && git diff --check", taskIds: ["A1", "B1", "E2", "G1", "H1"] },
    { id: "EV-NATS-TSS-1PCT-REPORT", time: "2026-07-30 20:38", title: "NATS-TSS 1% 与核心三 seed 中文证据", result: "独立 NATS-TSS adapter/真值协议完成 157 架构 × 22 代理 seed 2026 和核心 11 代理三 seed；报告记录失败、相关性、跨 seed 稳定性、topology 表、NB201 真值差异及 min/max seed reduction 修复。真实 index-0 为 mean 81.982667、min 81.616000、max 82.240000。", command: "docs/evidence/NATS_TSS_ONE_PERCENT_CN.md", taskIds: ["B1", "E2", "E3", "F2", "H1"] },
    { id: "EV-NATS-TSS-1PCT-SUMMARY", time: "2026-07-30 20:38", title: "NATS-TSS 1% 机器可读摘要", result: "manifest SHA-256 c8280222f5d51a534124f2ed58f104ecb0d5593797481e7c3acc4a6338d18a5c；22 代理单 seed 为 3,454/3,451/3/0，核心三 seed 为 5,181/5,172/9/0，合并 SHA-256 9efbe925701b34490b0904ef01ee6f0d50625a489044de78f73fbac2cf6101e9。157 个共享 topology 中 31 个 target 不同。", command: "docs/evidence/nats_tss_one_percent_summary.json", taskIds: ["E2", "E3", "F2", "H1"] },
    { id: "EV-FULL-GATE-282", time: "2026-07-30 20:17", title: "NB201 三 seed与 coverage 修复后完整 gate", result: "全量 282 tests passed；第一方 source coverage 87%、CLI coverage 80%，Ruff、compileall、pip check 与 git diff check 通过。新增回归覆盖 CLI 与 report bundle 对 failed invocation 的 total/failed/invalid/coverage 分母保留。", command: "conda run -n zcp-test python -m coverage run -m pytest -q && conda run -n zcp-test python -m coverage report -m && conda run -n zcp-test ruff check . && conda run -n zcp-test python -m compileall -q src tests && conda run -n zcp-test python -m pip check && git diff --check", taskIds: ["A1", "E2", "G1", "H1"] },
    { id: "EV-NB201-CORE-3SEED-REPORT", time: "2026-07-30 20:09", title: "NB201 核心 11 代理三 seed 中文证据", result: "seed 2026/2027/2028 在同一 157 架构 manifest 上共 5,181 行、5,172 成功、9 失败、0 重复键；报告保留失败覆盖率、三 seed Spearman、跨 seed 排名稳定性、八个新增 run 与边界说明。H1 仍等待其他 benchmark。", command: "docs/evidence/NB201_CORE_THREE_SEED_CN.md", taskIds: ["E2", "F2", "H1"] },
    { id: "EV-NB201-CORE-3SEED-SUMMARY", time: "2026-07-30 20:09", title: "NB201 核心 11 代理三 seed 机器摘要", result: "JSON 摘要锁定 157 架构、11 核心代理、三个 seed、5,181 行、5,172 成功、9 失败、0 重复键，sample manifest SHA-256 与合并 scores SHA-256 均可复核。", command: "docs/evidence/nb201_core_three_seed_summary.json", taskIds: ["E2", "F2", "H1"] },
    { id: "EV-PANEL-HTTP-REFRESH-AUDIT", time: "2026-07-30 20:17", title: "HTTP 无 F5 刷新机制复核", result: "刷新源码契约覆盖 cache-busting 动态 data.js、立即刷新、自动开关、可选间隔、隐藏页暂停/恢复检查和失败旧数据回退；HTTP 静态服务下 index.html、app.js 及两个不同 refresh query 的 data.js 均返回 200 且 payload 一致。", command: "python -m http.server 18768 --bind 127.0.0.1 --directory panel; curl index.html app.js 'data.js?refresh=audit-manual-1' 'data.js?refresh=audit-auto-2'; node panel/check-data.js", taskIds: ["F4"] },
    { id: "EV-NB201-1PCT-REPORT", time: "2026-07-30 19:38", title: "NB201 1% × 22 ZCP 中文证据", result: "中文证据逐项记录 157 个分层样本、3,454 行、3,451 成功、3 失败、四个 shard/run/checksum、22 个主组件相关性、正式 topology 表规模与结论边界；明确 H1 仅为 NB201 单 seed 完成，核心 11 代理另两个 seed 和其他 benchmark 仍待执行。", command: "docs/evidence/NB201_ONE_PERCENT_22ZCP_CN.md", taskIds: ["E1", "E2", "E3", "F2", "H1"] },
    { id: "EV-NB201-1PCT-SUMMARY", time: "2026-07-30 19:38", title: "NB201 1% × 22 ZCP 机器可读摘要", result: "JSON 摘要锁定 sample manifest SHA-256 9b9e7b0e8b7e59b76cee386cf6221bdac3f9b463a9a4729f68faffcd671391bc；157 架构 × 22 代理共 3,454 行，3,451 成功、3 失败、0 重复键，并保留 alias、portable approximation、常数输出与方向转换审计字段。", command: "docs/evidence/nb201_one_percent_22zcp_summary.json", taskIds: ["E1", "E2", "E3", "F2", "H1"] },
    { id: "EV-NB201-REAL-ANALYSIS", time: "2026-07-30 18:55", title: "真实 NB201 通用分析工作流", result: "两个真实 CIFAR-10 dataset-input seed、20 个 NB201 架构和 params/naswot/synflow 已生成 correlation、compare、seed sensitivity 及 topology 的 CSV、PNG、SVG、HTML；非有限 NASWOT 明确保留为 failed。该证据只验收真实工作流，不代表 1% 科学结论。", command: "docs/evidence/E2_E3_NB201_REAL_CN.md; docs/evidence/nb201_real_analysis_summary.json", taskIds: ["E2", "E3"] },
    { id: "EV-PANEL-AUTO-REFRESH-NODE", time: "2026-07-30 19:25", title: "看板刷新机制静态与 Node 验收", result: "自动刷新、立即刷新、可选轮询间隔、缓存破坏、旧数据回退、页面可见性和 ARIA DOM 契约检查通过；data.js、app.js、check-data.js 语法及 panel diff 格式检查通过。该证据不包含 file:// 或 HTTP 浏览器交互实测。", command: "node --check panel/data.js && node --check panel/app.js && node --check panel/check-data.js && node panel/check-data.js && git diff --check -- panel", taskIds: ["F4"] },
    { id: "EV-FULL-GATE-262", time: "2026-07-30 18:35", title: "最终完整质量 gate", result: "全量 262 项测试通过；第一方 source coverage 87%、CLI coverage 80%，Ruff、compileall、pip check 与 git diff check 均通过。", command: "conda run -n zcp-test python -m coverage run -m pytest -q && conda run -n zcp-test python -m coverage report -m && conda run -n zcp-test ruff check . && conda run -n zcp-test python -m compileall -q src tests && conda run -n zcp-test python -m pip check && git diff --check", taskIds: ["A1", "C1", "D2", "G1"] },
    { id: "EV-FINAL-2RANK-ACCEPTANCE", time: "2026-07-30 18:35", title: "最终 2-rank acceptance 1 epoch", result: "候选协议已锁定完整增强与来源字段。最终 2-rank acceptance 1 epoch 的配置将模型来源固定为 Cream commit b799630a29995163f282b15e2f38701160272fd1，将训练来源固定为 AZ-NAS commit 5e6683a2cfa5c6d0dc34a1317a842497ba7eae47；run status=completed，training.jsonl 仅 1 行，且无 .tmp 文件。", command: "2-rank acceptance 1-epoch run；审计 config provenance、manifest status、training.jsonl 行数与 .tmp", taskIds: ["A1", "C1", "D2", "G1", "H2"] },
    { id: "EV-ACCEPTANCE-CLI", time: "2026-07-30 18:29", title: "Acceptance CLI 双模式锁定", result: "acceptance CLI 仅接受两种真实数据模式：全数据且训练不超过 1% epoch，或不超过 1% 数据且运行完整 schedule；相关 acceptance/resume 定向测试共 69 项通过。", command: "69 项 acceptance/resume 定向 pytest", taskIds: ["A1", "C1", "D2", "H2", "H3"] },
    { id: "EV-IMAGENET-DDP-RESUME", time: "2026-07-30 18:29", title: "真实 ImageNet 双卡中断与可信恢复", result: "混合 4090D/4090 双卡使用真实 ImageNet 图片极小夹具，在 epoch 0 后 SIGTERM；源 run manifest=interrupted 且无 .tmp。新 run 从 trusted checkpoint 恢复，training.jsonl 连续记录 epoch 0–4，runtime.resume 包含 checkpoint SHA-256 与 source_run_id，且仅 rank 0 产生 artifacts。该结果只证明恢复机制，不代表完整双重 1% 协议。", command: "混合 4090D/4090 torchrun + epoch 0 后 SIGTERM；trusted checkpoint 新 run resume；manifest/training.jsonl/runtime.resume/rank artifacts 审计", taskIds: ["A1", "C1", "D2", "H2", "H3"] },
    { id: "EV-FULL-GATE-252", time: "2026-07-30 18:12", title: "AutoFormer complexity 分口径合入后完整 gate", result: "全量 252 项测试通过；第一方 source coverage 86%、CLI 80%，Ruff、compileall、pip check 与 git diff check 均通过。", command: "conda run -n zcp-test python -m coverage run -m pytest -q && conda run -n zcp-test python -m coverage report -m && conda run -n zcp-test ruff check . && conda run -n zcp-test python -m compileall -q src tests && conda run -n zcp-test python -m pip check && git diff --check", taskIds: ["A1", "C1", "G1"] },
    { id: "EV-AUTOFORMER-COMPLEXITY", time: "2026-07-30 18:11", title: "六个官方 complexity_ops golden 与独立 THOP 对照", result: "Cream T/S/B 和 AZ-NAS Tiny/Small/Base 的官方 get_complexity 数值逐项复现，并以 official_complexity_ops 命名且 metadata 明确 generic_flops=false。AZ-NAS Tiny 的 THOP MAC=1,100,420,352，官方口径=1,380,128,376；THOP 参数量也少于真实参数量，证明 relative-position 等算子/参数未被同口径覆盖，报告不得混列。", command: "pytest -q tests/test_reference_models.py -k autoformer; independent THOP profile on AZ-NAS Tiny", taskIds: ["C1", "G1"] },
    { id: "EV-FULL-GATE-251", time: "2026-07-30 18:04", title: "DDP 与自动累积合入后完整质量 gate", result: "全量 251 项测试通过；第一方 source coverage 86%、CLI 80%，Ruff、compileall、pip check 与 git diff check 均通过。", command: "conda run -n zcp-test python -m coverage run -m pytest -q && conda run -n zcp-test python -m coverage report -m && conda run -n zcp-test ruff check . && conda run -n zcp-test python -m compileall -q src tests && conda run -n zcp-test python -m pip check && git diff --check", taskIds: ["A1", "C1", "D2", "G1"] },
    { id: "EV-DDP-SMOKE", time: "2026-07-30 17:54", title: "混合 4090D/4090 两卡 DDP 真实 smoke", result: "DARTS 与 AutoFormer 分别以 GPU UUID 固定两张混合型号显卡完成 torchrun 2-rank/1-epoch smoke；每次仅创建一个共享 run、一行 training JSONL 和一套 last/best checkpoint，manifest 记录两卡型号、UUID 顺序、NCCL 与 world size。AutoFormer batch=1 因 Mixup 偶数约束明确失败并留下 failed manifest，batch=2 通过。", command: "CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-UUID-0,GPU-UUID-1 torchrun --standalone --nproc-per-node=2 -m zcp_test.cli train --config configs/training/{darts_cifar10,autoformer_imagenet}.yaml --smoke --epochs 1", taskIds: ["C1", "D2", "G1"] },
    { id: "EV-FULL-GATE-240", time: "2026-07-30 17:45", title: "AutoFormer 协议补强后完整质量 gate", result: "全量 240 项测试通过；第一方 source coverage 86%，Ruff、compileall、pip check 与 git diff check 均通过。", command: "conda run -n zcp-test python -m coverage run -m pytest -q && conda run -n zcp-test python -m coverage report -m && conda run -n zcp-test ruff check . && conda run -n zcp-test python -m compileall -q src tests && conda run -n zcp-test python -m pip check && git diff --check", taskIds: ["A1", "C1", "G1"] },
    { id: "EV-AUTOFORMER-PROTOCOL", time: "2026-07-30 17:44", title: "AutoFormer 官方参数与训练语义阶段验收", result: "Cream T/S/B 与 AZ-NAS Tiny/Small/Base 六个真实逐层 spec 的参数量逐项精确一致；真实 loader 接入三次 repeated augmentation，trainer 每 epoch 调用 sampler.set_epoch；AZ-NAS 8×256 线性缩放得到有效 LR 0.002。检测 WORLD_SIZE>1 时明确拒绝伪分布式独立训练，formal_training_ready 继续保持 false。", command: "pytest -q tests/test_reference_models.py tests/test_cli_commands.py tests/test_core.py tests/test_config_inputs.py tests/test_workflow.py", taskIds: ["C1", "D2", "H2"] },
    { id: "EV-PANEL-REFRESH-CONTROLS", time: "2026-07-30 17:29", title: "无需 F5 的自动刷新控制增强", result: "保留动态 data.js 加载与旧数据回退，新增显眼的立即刷新按钮、自动刷新开关、30 秒倒计时和隐藏页提示；visibility 恢复立即检查，cache-busting 与并发去重继续生效。HTTP server 下 index.html 与两条不同 refresh URL 均返回 200，两份 data.js 内容一致。", command: "node --check panel/data.js && node --check panel/app.js && node panel/check-data.js && python -m http.server 8768 --bind 127.0.0.1 --directory panel; curl index.html 'data.js?refresh=panel-test-1' 'data.js?refresh=panel-test-2'; git diff --check -- panel", taskIds: ["F4"] },
    { id: "EV-TRANSNAS-HEADS", time: "2026-07-30 17:18", title: "TransNAS 七任务 head 官方结构对照", result: "class_scene/object、room_layout、jigsaw、segmentsemantic、normal、autoencoder 的独立输出契约已实现；同一 micro fixture 与官方 commit 6d4231b 的参数量和完整 parameter-shape multiset 七任务全部一致。真实 micro index-0 七任务 build→params 均为 ok；normal+gradnorm 在缺 label provider 时返回 unsupported。", command: "pytest tests/test_reference_topologies.py tests/test_benchmarks_adapters.py; official fixture comparison; seven-task real adapter params sweep", taskIds: ["B2", "G2"] },
    { id: "EV-FULL-GATE-231", time: "2026-07-30 17:19", title: "TransNAS 七任务合入后完整质量 gate", result: "全量 231 项测试通过；第一方 source coverage 86%，Ruff、compileall、pip check 与 git diff check 均通过。", command: "conda run -n zcp-test python -m coverage run -m pytest -q && conda run -n zcp-test python -m coverage report -m && conda run -n zcp-test ruff check . && conda run -n zcp-test python -m compileall -q src tests && conda run -n zcp-test python -m pip check && git diff --check", taskIds: ["A1", "B2", "G1"] },
    { id: "EV-OFA-BN-REAL", time: "2026-07-30 17:03", title: "真实 ImageNet-1k 确定性 BN smoke", result: "OFA inherited 子网在本机真实 ImageNet-1k 上完成 1 batch×2 samples 的独立 BN recalibration；score row 记录 required=false、batches=1、sample IDs、resize/center-crop transform 与 SHA-256 fingerprint，run 未生成空 checkpoints/parts/reports。协议明确 official_protocol_match=false，不作为 inherited accuracy。", command: "zcp-test evaluate --space ofa_proxyless_mbv2 --weight-mode ofa_inherited --trusted --input-source dataset --dataset imagenet1k --bn-recalibration-batches 1 --bn-recalibration-batch-size 2 --proxies params --count 1 --device cpu", taskIds: ["C3"] },
    { id: "EV-FULL-GATE-223", time: "2026-07-30 17:05", title: "真实 BN 流水线合入后完整质量 gate", result: "全量 223 项测试通过；第一方 source coverage 86%，Ruff、compileall、pip check 与 git diff check 均通过。", command: "conda run -n zcp-test python -m coverage run -m pytest -q && conda run -n zcp-test python -m coverage report -m && conda run -n zcp-test ruff check . && conda run -n zcp-test python -m compileall -q src tests && conda run -n zcp-test python -m pip check && git diff --check", taskIds: ["A1", "C3", "G1"] },
    { id: "EV-OFA-INHERITED", time: "2026-07-30 16:54", title: "OFA-Proxyless inherited checkpoint 与 active subnet 阶段通过", result: "官方 checkpoint 32,202,338 bytes 已下载到外部 data root，SHA256=10ce40...6b907；新增 ofa_proxyless_supernet bootstrap 组并修复 ready 文件不注册 catalog。active channel 与 7→5→3 learned kernel transform 导出完成；混合 k/e/d 子网参数与官方 get_active_subnet 一致，同输入 max abs≈1.9e-6。真实 CPU evaluate 1架构×params 与 population2/generation1 短 search 均成功并记录 inherited provenance；随后完整 222-test gate 通过。", command: "专项 pytest（core/reference/data 组合）; OFA inherited CPU evaluate 1×params; short search population=2 generations=1", taskIds: ["A1", "C3"] },
    { id: "EV-FULL-GATE-222", time: "2026-07-30 16:58", title: "OFA inherited 合入后完整质量 gate", result: "全量 222 项测试通过；第一方 source coverage 86%，Ruff、compileall、pip check 与 git diff check 均通过；短 search 的 5 行 candidate/summary 记录全部携带 inherited weight mode 与 checkpoint SHA-256。", command: "conda run -n zcp-test python -m coverage run -m pytest -q && conda run -n zcp-test python -m coverage report -m && conda run -n zcp-test ruff check . && conda run -n zcp-test python -m compileall -q src tests && conda run -n zcp-test python -m pip check && git diff --check", taskIds: ["A1", "C3", "G1"] },
    { id: "EV-FULL-GATE-219", time: "2026-07-30 16:40", title: "当前完整质量 gate 通过", result: "全量 219 项测试通过；第一方 source coverage 86%，Ruff、compileall、pip check 与 git diff check 均通过。历史 216-test gate 继续保留为当时事件。", command: "conda run -n zcp-test pytest && conda run -n zcp-test python -m coverage run -m pytest -q && conda run -n zcp-test python -m coverage report -m && conda run -n zcp-test ruff check . && conda run -n zcp-test python -m compileall -q src tests && conda run -n zcp-test python -m pip check && git diff --check", taskIds: ["A1", "G1"] },
    { id: "EV-MBV2-REFERENCE", time: "2026-07-30 16:38", title: "OFA-Proxyless MBV2 官方 positional/params fixture 通过", result: "实现官方 21 dynamic-block positional encoding；registered space 固定 width 1.3、resolution 128..224 step 4。官方 commit f03b267 fixture 中 width1.0=2,500,632 params、width1.3=3,718,832 params，width1.0 参数 shape multiset 与官方完全一致；test_reference_models 39 passed，Ruff 与 diff check 通过。", command: "pytest -q tests/test_reference_models.py && ruff check . && git diff --check", taskIds: ["C2"] },
    { id: "EV-MBV2-FIXTURE-START", time: "2026-07-30 16:32", title: "OFA-Proxyless MBV2 官方 fixture 主任务启动", result: "当前主任务已切换为官方 positional encoding/reference fixture 对齐并标记进行中；现阶段仅记录任务边界，尚无 params/MAC fixture 通过结论。", command: "任务交接：OFA-Proxyless MBV2 official positional encoding/reference fixture", taskIds: ["C2"] },
    { id: "EV-REAL-BENCHMARKS", time: "2026-07-30 16:28", title: "全 Benchmark 真实 index-0 smoke", result: "NB101、NB201、NATS-TSS/SSS、TNB micro/macro、deterministic NB301、ViT main/extension/PiT 的 query→build→params proxy 全部 succeeded=1、failed=0；未用 synthetic benchmark 或伪造真值。", command: "zcp-test benchmark inspect ...; zcp-test evaluate --benchmark ... --proxies params --count 1 --input-source random --device cpu", taskIds: ["B1", "B2", "B3", "G2"] },
    { id: "EV-GIT-CHECKPOINT", time: "2026-07-30 16:15", title: "低成本审计阶段提交", result: "216-test/86%-coverage、PiT/OFA-MBV3 reference、训练协议白名单、文档和动态看板已提交为 9e939b8；数据、runs 和 checkpoint 未进入 Git。远端推送等待 GitHub 登录。", command: "git commit -m 'Complete low-cost audit and reference model pass'", taskIds: ["A1", "A2", "C4", "F1", "G1", "I1"] },
    { id: "EV-PIT-REFERENCE", time: "2026-07-30 16:13", title: "PiT reference 官方结构对照", result: "真实 gt_pit 规格完成 load→build→224 forward；与 Auto-Prox 90ed458 同为 893,828 参数且参数 shape multiset 一致。MAC golden 尚未完成。", command: "PYTHONPATH=/tmp/Auto-Prox-AAAI24 conda run -n zcp-test python /tmp/check_upstream_pit_isolated.py", taskIds: ["C4"] },
    { id: "EV-MBV3-REFERENCE", time: "2026-07-30 16:13", title: "OFA-MBV3 静态 reference 对照", result: "按官方五阶段/20-block 编码实现 SE、h-swish、width rounding 与 BN recalibration；全 3×3/e3/d2/w1.0 子网双方均为 3,410,792 参数且 shape multiset 一致。", command: "PYTHONPATH=/tmp/once-for-all conda run -n zcp-test python /tmp/check_ofa_mbv3.py", taskIds: ["C4"] },
    { id: "EV-PROTOCOL-GATE", time: "2026-07-30 16:13", title: "正式训练协议不可由 YAML 自授权", result: "新增代码内 DARTS 协议白名单与关键字段校验；拒绝未知协议、篡改 recipe、batch/input override，并在缺数据时构模前失败。", command: "pytest -q tests/test_workflow.py -k 'approved_formal or non_reference_space'", taskIds: ["A2", "D1", "D2"] },
    { id: "EV-COVERAGE-85", time: "2026-07-30 16:13", title: "第一方与关键模块覆盖率达标", result: "全量 216 项测试通过；第一方 coverage 86%，CLI 80%，benchmark_report 96%，reports 100%。", command: "python -m coverage run -m pytest -q && python -m coverage report -m", taskIds: ["A1", "G1"] },
    { id: "EV-HYGIENE-REGRESSION", time: "2026-07-30 15:06", title: "README 本机路径回归已修复", result: "全量 coverage 运行中的 repository hygiene 测试发现 README 新增本机路径；已改为通用 TER-Score 描述，定向卫生测试通过。", command: "pytest -q tests/test_repository_hygiene.py && scan-machine-specific-paths README.md docs configs src tests panel", taskIds: ["F1", "G1", "I1"] },
    { id: "EV-COVERAGE-FIRST-PARTY", time: "2026-07-30 15:06", title: "第一方源码覆盖率复核", result: "明确排除 vendored NAS-Bench API 和生成 protobuf 后，第一方 source coverage 为 82%；仍低于 85%，CLI 为 62%。", command: "coverage run -m pytest -q && coverage report -m", taskIds: ["G1"] },
    { id: "EV-SAFETY-COVERAGE", time: "2026-07-30 15:04", title: "安全工具覆盖率补强", result: "新增 doctor、trusted legacy pickle 和 benchmark converter 契约测试共 8 项；定向覆盖率 converter 98%、doctor 100%、legacy 100%。", command: "coverage run --source=zcp_test.doctor,zcp_test.legacy,zcp_test.data.converters -m pytest -q tests/test_safety_tools.py", taskIds: ["G1", "A3"] },
    { id: "EV-DOC-AUDIT", time: "2026-07-30 15:02", title: "中文主指引与训练语义修订", result: "README 增加中文优先研究流程；新增 TRAINING_CN；修正 DARTS 原始/TE-NAS profile、内联 architecture 和正式训练门禁说明；本地 Markdown 链接检查无断链。", command: "python local-markdown-link-audit && git diff --check -- README.md docs", taskIds: ["F1", "F2", "D1", "D2"] },
    { id: "EV-PANEL-REFRESH", time: "2026-07-30 15:26", title: "看板无整页自动刷新复核", result: "手动刷新、30 秒自动拉取、页面恢复可见时刷新、并发去重、失败保留旧数据及原子 live status 已复核；index.html 与两条不同 refresh 查询 URL 均返回 HTTP 200，两次 data.js 内容一致且请求 URL 唯一，不复用缓存 URL。", command: "node --check panel/data.js && node --check panel/app.js; python -m http.server 8768 --bind 127.0.0.1 --directory panel >/tmp/zcp-panel-http.log 2>&1 & server_pid=$!; curl --noproxy '*' http://127.0.0.1:8768/index.html; curl --noproxy '*' 'http://127.0.0.1:8768/data.js?refresh=1785396332891103153-1'; curl --noproxy '*' 'http://127.0.0.1:8768/data.js?refresh=1785396332894871583-2'; kill \"$server_pid\"", taskIds: ["F4"] },
    { id: "EV-LOW-COST-GATE", time: "2026-07-30 16:13", title: "最新低成本质量 gate", result: "全量测试 216 passed；Ruff、compileall、pip check 与 diff check 通过。", command: "conda run -n zcp-test pytest && conda run -n zcp-test ruff check . && conda run -n zcp-test python -m compileall -q src tests && conda run -n zcp-test python -m pip check && git diff --check", taskIds: ["A1", "G1"] },
    { id: "EV-COVERAGE", time: "2026-07-30 16:13", title: "覆盖率 gate 通过", result: "第一方 coverage 86%、CLI 80%、benchmark_report 96%、reports 100%，总计与关键模块覆盖率门槛均通过。", command: "conda run -n zcp-test python -m coverage run -m pytest -q && conda run -n zcp-test python -m coverage report -m", taskIds: ["A1", "G1"] },
    { id: "EV-REPO-HYGIENE", time: "2026-07-30 14:50", title: "误保存搜索页清理", result: "误保存的 Google 搜索结果 .html 已删除；复查未发现 panel/index.html 之外的 .html/.htm 文件。", command: "find . -path './panel/index.html' -prune -o -type f \\( -name '*.html' -o -name '*.htm' \\) -print", taskIds: ["I1"] },
    { id: "EV-PANEL", time: "2026-07-30 14:25", title: "HTML 看板静态验收", result: "数据驱动筛选、统计、风险、证据、详情、主题、响应式和无障碍交互完成；脚本及数据契约检查通过。", command: "node --check panel/data.js && node --check panel/app.js && git diff --check -- panel", taskIds: ["F4"] },
    { id: "EV-REFERENCE-MODELS", time: "2026-07-30 14:15 前", title: "Reference 模型定向测试", result: "StaticAutoFormer、PlainNetMobileNetV2 与 StaticMobileNetV2 共 22 项测试通过，Ruff 通过。", command: "conda run -n zcp-test pytest -q tests/test_reference_models.py", taskIds: ["C1", "C2"] },
    { id: "EV-ZCP-SWEEP", time: "2026-07-30 13:51", title: "22 ZCP CPU sweep", result: "NB201 FX 可追踪修正后，22/22 proxy 均可执行；论文 golden 验证未包含。", command: "CPU proxy sweep", taskIds: ["E1"] },
    { id: "EV-RUN-DISCOVERY", time: "2026-07-30 13:39", title: "Run discovery 测试", result: "单 run 自动发现、多 run 歧义和 legacy HTML 路径测试通过。", command: "定向 pytest", taskIds: ["F3"] },
    { id: "EV-TRAINING", time: "2026-07-30 13:37", title: "训练控制路径测试", result: "scheduler、checkpoint identity 和确定性 1% 分层子集测试通过。", command: "定向 pytest", taskIds: ["D1", "D2"] },
    { id: "EV-TRANSNAS", time: "2026-07-30 13:34", title: "TransNAS 真实 JSONL smoke", result: "load→query→build→evaluate 单架构 CPU 路径通过。", command: "真实数据 CPU smoke", taskIds: ["B2", "G2"] },
    { id: "EV-NB201", time: "2026-07-30 13:28", title: "NB201/NATS topology 定向测试", result: "topology 与 width 字段进入模型，定向模型测试通过。", command: "定向 pytest", taskIds: ["B1"] },
    { id: "EV-SAFETY", time: "2026-07-30 13:18", title: "安全边界测试", result: "trusted 配置绕过被拒绝，重复键和 score-field 行为有测试。", command: "定向 pytest", taskIds: ["A3"] },
    { id: "EV-FIDELITY", time: "2026-07-30 13:18", title: "Model fidelity 契约", result: "正式训练保护与 artifact fidelity 字段接入。", command: "定向 pytest", taskIds: ["A2"] },
    { id: "EV-BASELINE", time: "2026-07-30 12:30", title: "首轮测试基线", result: "首轮既有测试通过；该证据早于后续并发改动，需由最终 gate 替代。", command: "pytest", taskIds: ["A1"] }
  ]
};
