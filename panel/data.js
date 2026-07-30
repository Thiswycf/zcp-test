window.ZCP_PANEL_DATA = {
  schemaVersion: 2,
  updatedAt: "2026-07-30 17:19 CST",
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
      detail: "TransNAS 七任务 head 合入后的最新完整 gate 为 231 passed、第一方 source coverage 86%，Ruff、compileall、pip check 与 git diff check 通过。",
      acceptance: ["记录 Python/依赖环境", "全量 pytest 与 Ruff 结果可追溯", "报告并发未提交改动"],
      evidence: ["EV-BASELINE", "EV-LOW-COST-GATE", "EV-COVERAGE", "EV-GIT-CHECKPOINT", "EV-FULL-GATE-219", "EV-FULL-GATE-222", "EV-FULL-GATE-223", "EV-FULL-GATE-231", "EV-OFA-INHERITED", "EV-OFA-BN-REAL", "EV-TRANSNAS-HEADS"], risks: ["R-CONCURRENCY"], updatedAt: "2026-07-30 17:19"
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
      detail: "NB201、NATS-TSS 和 NATS-SSS 的 native index-0 query→build→params proxy 均通过；NB201/TSS 共享 architecture ID 但 benchmark/API 身份保持独立。",
      acceptance: ["字段敏感性测试", "native API 架构对照", "真实样本 query→build→proxy"],
      evidence: ["EV-NB201", "EV-REAL-BENCHMARKS"], risks: [], updatedAt: "2026-07-30 16:28"
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
      detail: "NB101、deterministic NB301、ViT main/extension/PiT 的真实 query 与构模 proxy 均通过；extension 仅含 KD/inherited，vanilla 查询按设计失败。",
      acceptance: ["slice 身份可追溯", "surrogate noise 明确记录", "metric seed/epoch 不静默降级"],
      evidence: ["EV-REAL-BENCHMARKS"], risks: [], updatedAt: "2026-07-30 16:28"
    },
    {
      id: "C1", phase: "Reference", priority: "P0", title: "AutoFormer 静态 scratch reference",
      content: "提供可独立构建的静态 subnet，使逐层 depth、head 与 MLP ratio 真实影响模型。",
      purpose: "支撑无 inherited supernet 条件下的 AZ-NAS scratch/static 研究。",
      estimate: "6–12 小时", startedAt: "2026-07-30 13:00", finishedAt: "—", status: "进行中", progress: 70,
      detail: "静态分类模型、输入/编码校验、metadata 与字段敏感性测试完成；官方发布配置参数/FLOPs fixture 尚待对照。",
      acceptance: ["逐层字段改变参数或算子", "分类 forward 通过", "非法编码拒绝", "metadata 明确不支持 inherited", "官方 fixture 对照"],
      evidence: ["EV-REFERENCE-MODELS"], risks: ["R-AUTOFORMER"], updatedAt: "2026-07-30 14:15"
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
      detail: "定向单测覆盖 scheduler、身份不匹配拒绝和分层子集复现。",
      acceptance: ["三种 scheduler 有测试", "错误 checkpoint 身份拒绝", "1% subset 可复现"],
      evidence: ["EV-TRAINING"], risks: [], updatedAt: "2026-07-30 13:37"
    },
    {
      id: "E1", phase: "ZCP", priority: "P0", title: "22 ZCP 契约与算法 provenance",
      content: "逐项标记论文兼容、近似、别名和 unsupported，并建立 golden 验证。",
      purpose: "运行成功之外，验证代理公式和输入协议的可解释性。",
      estimate: "8–16 小时", startedAt: "2026-07-30 12:30", finishedAt: "—", status: "进行中", progress: 60,
      detail: "22/22 CPU sweep 与既有 GPU sweep 可运行；论文级 golden 数值验证尚未完成。",
      acceptance: ["全部 proxy 可分类", "论文公式 golden fixture", "approximation 在 artifact 可见"],
      evidence: ["EV-ZCP-SWEEP"], risks: ["R-PROXY"], updatedAt: "2026-07-30 14:15"
    },
    {
      id: "E2", phase: "研究", priority: "P1", title: "通用 ZCP 分析",
      content: "验收互相关、top-k、稳定性、Pareto、transfer 与样本收敛分析。",
      purpose: "形成可复用且不会静默错配的研究报告。",
      estimate: "6–12 小时", startedAt: "2026-07-30 12:50", finishedAt: "—", status: "进行中", progress: 55,
      detail: "主要分析函数已存在；CLI 边界和真实 run bundle 待验收。",
      acceptance: ["缺列明确报错", "多 run 来源保留", "真实结果可生成全部表格"],
      evidence: [], risks: [], updatedAt: "2026-07-30 14:15"
    },
    {
      id: "E3", phase: "研究", priority: "P2", title: "Benchmark 定制研究",
      content: "按预算、操作、size、任务和 ViT 参数分析结构偏置。",
      purpose: "避免不同 benchmark 只输出同一套泛化统计。",
      estimate: "8–16 小时", startedAt: "2026-07-30 13:00", finishedAt: "—", status: "进行中", progress: 45,
      detail: "study 模块已有实现；真实数据表、图和论文解释待复核。",
      acceptance: ["各 benchmark 有专属因子", "真实数据生成图表", "结论关联原始字段"],
      evidence: [], risks: [], updatedAt: "2026-07-30 14:15"
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
      estimate: "3–6 小时", startedAt: "2026-07-30 13:20", finishedAt: "—", status: "进行中", progress: 40,
      detail: "初稿和结构已具备，等待最终实验结果回填。",
      acceptance: ["每个结论链接证据", "受阻原因明确", "不把 smoke 写成完整训练"],
      evidence: [], risks: [], updatedAt: "2026-07-30 14:15"
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
      estimate: "2–4 小时", startedAt: "2026-07-30 14:15", finishedAt: "2026-07-30 14:59", status: "已完成", progress: 100,
      detail: "页面每 30 秒无整页重载地拉取 data.js，切回页面立即检查，并支持手动刷新；并发触发会合并，失败时恢复旧数据，刷新状态通过原子 live region 提示。",
      acceptance: ["无 CDN", "任务必填字段齐全", "筛选/搜索/详情可用", "自动与手动刷新可用", "cache-busting 数据请求", "失败保留旧数据", "可访问状态提示", "node --check 通过"],
      evidence: ["EV-PANEL", "EV-PANEL-REFRESH"], risks: [], updatedAt: "2026-07-30 15:26"
    },
    {
      id: "G1", phase: "验收", priority: "P0", title: "新增代码全量质量 gate",
      content: "执行全量测试、静态检查、覆盖率和关键模块回归。",
      purpose: "确认并发修复合并后无回归。",
      estimate: "2–4 小时", startedAt: "2026-07-30 14:50", finishedAt: "2026-07-30 15:42", status: "已完成", progress: 100,
      detail: "当前完整 gate 为 231 项测试通过；第一方 source coverage 86%，Ruff、compileall、pip check 与 git diff check 通过。",
      acceptance: ["全量 pytest", "Ruff 通过", "source coverage ≥85%", "关键模块 ≥80%"],
      evidence: ["EV-LOW-COST-GATE", "EV-COVERAGE", "EV-FULL-GATE-219", "EV-FULL-GATE-222", "EV-FULL-GATE-223", "EV-FULL-GATE-231"], risks: ["R-CONCURRENCY"], updatedAt: "2026-07-30 17:19"
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
      estimate: "12–24 小时", startedAt: "—", finishedAt: "—", status: "待开始", progress: 0,
      detail: "计划最多使用 4 GPU；全部代理单 seed，核心代理 3 seed。",
      acceptance: ["真实标签不少于 1%", "全部代理至少单 seed", "核心代理 3 seed", "预算记录完整"],
      evidence: [], risks: ["R-BUDGET", "R-PROXY"], updatedAt: "2026-07-30 13:50"
    },
    {
      id: "H2", phase: "高成本", priority: "P1", title: "全数据 × 1% epoch",
      content: "在全训练数据上执行约 1% 正式 epoch，并验证曲线与恢复。",
      purpose: "验收 reference 模型、优化器、数据增强和 checkpoint。",
      estimate: "12–24 小时", startedAt: "—", finishedAt: "—", status: "待开始", progress: 0,
      detail: "reference fixture 和 G1 gate 通过后执行。",
      acceptance: ["真实全数据", "约 1% epoch", "恢复结果连续", "协议 metadata 完整"],
      evidence: [], risks: ["R-BUDGET", "R-AUTOFORMER", "R-OFA"], updatedAt: "2026-07-30 13:50"
    },
    {
      id: "H3", phase: "高成本", priority: "P2", title: "1% 数据 × 完整 schedule",
      content: "使用确定性 1% 数据跑完整 scheduler 和 checkpoint 生命周期。",
      purpose: "验证长期调度和监控，不作为正式精度结论。",
      estimate: "24–48 小时", startedAt: "—", finishedAt: "—", status: "待开始", progress: 0,
      detail: "设置 24 小时检查点和 48 小时硬停止。",
      acceptance: ["完整 schedule", "监控产物持续写入", "48 小时内停止", "报告不声称正式精度"],
      evidence: [], risks: ["R-BUDGET"], updatedAt: "2026-07-30 13:50"
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
    { id: "R-AUTOFORMER", severity: "高", status: "开放", title: "AutoFormer 尚无官方 fixture 验收", description: "静态字段敏感性已通过，但还未与发布配置参数量、FLOPs 和精度对照。", mitigation: "加入官方 Tiny/Small/Base fixture；inherited 路径保持 false。", taskIds: ["C1", "H2"] },
    { id: "R-MBV2-FIXTURE", severity: "高", status: "关闭", title: "OFA-Proxyless MBV2 positional/params fixture 已验收", description: "官方 commit f03b267 的 21 dynamic-block positional encoding、width1.0/1.3 参数量及 width1.0 参数 shape multiset 已完成对照。", mitigation: "保留固定 commit、registered space 边界和回归测试；MAC 与训练边界由独立风险继续跟踪。", taskIds: ["C2"] },
    { id: "R-MBV2-REMAINING", severity: "高", status: "开放", title: "OFA-Proxyless MBV2 MAC 与正式训练未验收", description: "params/shape fixture 已通过，但官方 MAC golden 尚缺，正式训练协议也未完成验收；不得由静态 reference 结论外推训练精度或成本。", mitigation: "补充同一官方 commit 的 MAC fixture；关闭训练 blocker 并完成正式 profile 验收前，仅报告 static scratch reference。", taskIds: ["C2", "H2"] },
    { id: "R-OFA", severity: "高", status: "开放", title: "OFA inherited accuracy 与完整协议未验收", description: "官方 checkpoint、catalog bootstrap、active-weight export、子网数值一致性、evaluate/search 和真实 ImageNet 确定性 BN smoke 已完成；当前项目 BN 协议不等同官方 data provider，accuracy、MAC golden 与 formal training 仍未完成。", mitigation: "对照官方 OFA data provider 的抽样、transform 与 BN 统计并执行 inherited accuracy；补齐 MAC golden。正式训练 blocker 关闭前不得外推 inherited 或 scratch 训练结论。", taskIds: ["C2", "C3", "C4", "H2"] },
    { id: "R-PROXY", severity: "高", status: "开放", title: "代理可运行不等于论文一致", description: "22/22 sweep 不能替代公式、聚合方向和输入协议的 golden 验证。", mitigation: "为核心代理增加论文级数值 fixture 和 provenance。", taskIds: ["E1", "H1"] },
    { id: "R-COVERAGE", severity: "中", status: "关闭", title: "报告模块覆盖率已达标", description: "第一方 coverage 86%、CLI 80%、benchmark_report 96%、reports 100%，总计与关键模块门槛均已达到。", mitigation: "维持现有覆盖率 gate，后续改动继续执行全量回归。", taskIds: ["A1", "F3", "G1"] },
    { id: "R-PIT", severity: "中", status: "开放", title: "PiT MAC 对照尚未完成", description: "真实 GT 的 224 forward、官方参数量与参数 shape multiset 已对齐；MAC golden 尚缺。PiT 是固定 benchmark 候选，不要求重复完整训练。", mitigation: "补充同一官方 commit 的 MAC fixture 后关闭结构验收；vanilla/KD 指标继续分协议查询。", taskIds: ["C4"] },
    { id: "R-BUDGET", severity: "中", status: "监控", title: "高成本任务预算", description: "1% benchmark 与双重 1% 训练可能超出 GPU 或 48 小时上限。", mitigation: "最多 4 GPU；24 小时复核；48 小时硬停止并保留部分结论。", taskIds: ["H1", "H2", "H3"] }
  ],
  evidence: [
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
