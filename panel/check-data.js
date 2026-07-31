"use strict";

const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const dataPath = path.join(__dirname, "data.js");
const appPath = path.join(__dirname, "app.js");
const indexPath = path.join(__dirname, "index.html");
const stylesPath = path.join(__dirname, "styles.css");
const readmePath = path.join(__dirname, "README.md");
const context = { window: {} };
vm.runInNewContext(fs.readFileSync(dataPath, "utf8"), context, { filename: dataPath });
const data = context.window.ZCP_PANEL_DATA;
const appSource = fs.readFileSync(appPath, "utf8");
const indexSource = fs.readFileSync(indexPath, "utf8");
const stylesSource = fs.readFileSync(stylesPath, "utf8");
const readmeSource = fs.readFileSync(readmePath, "utf8");
const errors = [];

function assert(condition, message) {
  if (!condition) errors.push(message);
}

function assertUnique(items, kind) {
  const ids = new Set();
  for (const item of items) {
    assert(item && typeof item.id === "string" && item.id.length > 0, `${kind} 存在空 ID`);
    assert(!ids.has(item.id), `${kind} ID 重复：${item.id}`);
    ids.add(item.id);
  }
  return ids;
}

function parsePanelTime(value) {
  if (value === "—" || value == null || value === "") return null;
  const match = String(value).match(/^(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2})/);
  if (!match) return Number.NaN;
  return Date.parse(`${match[1]}-${match[2]}-${match[3]}T${match[4]}:${match[5]}:00+08:00`);
}

function checkTime(value, label, now) {
  const parsed = parsePanelTime(value);
  assert(parsed === null || Number.isFinite(parsed), `${label} 时间格式无效：${value}`);
  if (Number.isFinite(parsed)) assert(parsed <= now, `${label} 晚于当前时间：${value}`);
  return parsed;
}

assert(data && typeof data === "object", "缺少 ZCP_PANEL_DATA");
assert(data?.project?.name, "缺少 project.name");
assert(Array.isArray(data?.tasks), "tasks 必须是数组");
assert(Array.isArray(data?.risks), "risks 必须是数组");
assert(Array.isArray(data?.evidence), "evidence 必须是数组");

const refreshDomIds = [
  "refresh-data", "auto-refresh-toggle", "refresh-interval", "refresh-status",
  "refresh-success", "refresh-checked", "refresh-countdown", "refresh-file-hint"
];
for (const id of refreshDomIds) {
  assert(indexSource.includes(`id="${id}"`), `刷新控件缺少 #${id}`);
}
const reloadStateDomIds = [
  "search", "status-filter", "phase-filter", "priority-filter", "sort-order"
];
for (const id of reloadStateDomIds) {
  assert(indexSource.includes(`id="${id}"`), `页面重载状态恢复缺少 #${id}`);
  assert(appSource.includes(`$("#${id}").value = state.`), `页面重载状态未恢复到 #${id}`);
}
assert(!appSource.includes('$("#search-input")'), "状态恢复仍引用不存在的 #search-input");
assert(!appSource.includes('$("#sort-select")'), "状态恢复仍引用不存在的 #sort-select");
assert(indexSource.includes('aria-live="polite"'), "刷新状态缺少 aria-live");
assert(indexSource.includes('aria-atomic="true"'), "刷新状态缺少 aria-atomic");
assert(indexSource.includes('aria-pressed="true"'), "自动刷新按钮缺少 aria-pressed");
assert(indexSource.includes('http-equiv="Cache-Control"'), "页面缺少静态托管 no-store 提示");
assert(indexSource.includes('no-cache, no-store, must-revalidate'), "页面 Cache-Control 未声明 no-store");
assert(indexSource.includes("最后成功刷新"), "刷新栏未显示最后成功刷新时间");
assert(indexSource.includes("上次检查"), "刷新栏未显示上次检查时间");
assert(indexSource.includes('<script src="data.js"></script>'), "file:// 回退缺少直接 data.js 脚本");
assert(indexSource.includes('id="project-status"'), "页面缺少项目状态展示");
assert(appSource.includes('`项目状态：${data.project.status}`'), "项目状态未由数据驱动渲染");
assert(indexSource.indexOf('<script src="data.js"></script>') < indexSource.indexOf('<script src="app.js"></script>'), "data.js 必须先于 app.js 加载");
assert(stylesSource.includes(".refresh-primary"), "立即刷新按钮缺少可见样式");
assert(!/\.refresh-primary\s*\{[^}]*display\s*:\s*none/is.test(stylesSource), "立即刷新按钮被样式隐藏");
assert(stylesSource.includes('#refresh-status[data-state="error"]'), "刷新失败缺少非阻断错误样式");
for (const seconds of [5, 15, 30, 60]) {
  assert(indexSource.includes(`<option value="${seconds}"`), `刷新间隔缺少 ${seconds} 秒选项`);
}
assert(appSource.includes('new URL("data.js", window.location.href)'), "刷新未使用兼容 file/HTTP 的 data.js URL");
assert(appSource.includes('url.searchParams.set("refresh"'), "刷新未对 data.js 使用缓存破坏参数");
assert(appSource.includes('document.head.append(script)'), "刷新未动态挂载最新 data.js 脚本");
assert(appSource.includes('script.onload = () =>'), "刷新未处理 data.js 成功加载");
assert(appSource.includes('script.onerror = () =>'), "刷新未处理 data.js 加载失败");
assert(appSource.includes("const dataScriptTimeoutMs = 10_000"), "data.js 动态加载缺少明确超时");
assert(appSource.includes("timeoutId = window.setTimeout(() =>"), "data.js 超时未启动计时器");
assert(appSource.includes("window.clearTimeout(timeoutId)"), "data.js 加载结束后未清理超时计时器");
assert(appSource.includes("if (settled) return"), "data.js 加载缺少成功/失败/超时竞态保护");
assert(appSource.includes("script.onload = null"), "data.js 加载 cleanup 未释放 onload");
assert(appSource.includes("script.onerror = null"), "data.js 加载 cleanup 未释放 onerror");
assert(appSource.includes('new Error("加载 data.js 超时（10 秒）")'), "data.js 超时缺少明确失败信息");
assert(appSource.includes('$("#refresh-data").addEventListener("click"'), "立即刷新按钮未绑定点击处理");
assert(appSource.includes('refreshData(true).finally(scheduleAutoRefresh)'), "立即刷新未重新安排自动刷新");
assert(appSource.includes('setAutoRefresh(!autoRefreshEnabled, true)'), "自动刷新开关未切换状态");
assert(appSource.includes('setRefreshInterval(event.target.value, true)'), "刷新间隔选择未生效或未持久化");
assert(appSource.includes("[5, 15, 30, 60].includes(value)"), "持久化刷新间隔与页面选项不一致");
assert(appSource.includes('refreshData(false).finally(scheduleAutoRefresh)'), "自动刷新定时器未重新载入 data.js");
assert(appSource.includes('document.addEventListener("visibilitychange"'), "刷新未处理页面可见性变化");
assert(appSource.includes('document.visibilityState !== "visible"'), "隐藏页面未暂停自动刷新调度");
assert(appSource.includes('cancelScheduledRefresh()'), "缺少自动刷新定时器取消逻辑");
assert(appSource.includes('if (refreshPromise)'), "刷新请求缺少并发去重判断");
assert(appSource.includes('return refreshPromise'), "并发刷新未复用当前请求");
assert(appSource.includes('if (refreshInitialized) return'), "刷新初始化缺少重复事件保护");
assert(appSource.includes('if (refreshCountdownTimer === null)'), "刷新倒计时缺少重复计时器保护");
assert(appSource.includes("window.ZCP_PANEL_DATA = previousData"), "刷新失败时未保留旧数据");
assert(appSource.includes('`最后成功刷新：${formatClock()}`'), "刷新成功后未更新时间戳");
assert(appSource.includes('`上次检查：${formatClock()}`'), "刷新完成后未更新检查时间");
assert(appSource.includes('`数据更新时间（北京时间）：${formatPanelTime(data.updatedAt)} · schema v${data.schemaVersion}`'), "未明确显示北京时间数据更新时间");
assert(appSource.includes("window.scrollTo(viewport.left, viewport.top)"), "刷新后未恢复滚动位置");
assert(appSource.includes('window.location.protocol === "file:"'), "缺少 file:// 兼容提示逻辑");
assert(appSource.includes("reloadPageWithCacheBusting()"), "file:// 模式缺少页面级刷新回退");
assert(appSource.includes('url.searchParams.set("refresh"'), "页面级刷新回退缺少缓存破坏参数");
assert(appSource.includes("saveReloadState()"), "页面级刷新回退未保存筛选状态");
assert(appSource.includes("restoreReloadState()"), "页面级刷新回退未恢复筛选状态");
assert(indexSource.includes("若浏览器阻止动态脚本，则保留筛选状态并自动重载页面"), "file:// 回退提示不完整");
assert(appSource.includes("可点击“立即刷新”重试"), "刷新失败未提示重试");
assert(readmeSource.includes("python -m http.server 8768 --directory panel"), "README 缺少静态服务器命令");
assert(readmeSource.includes("file://"), "README 缺少 file:// 限制说明");
assert(readmeSource.includes("无需手动按 F5"), "README 未说明 file:// 页面重载回退");
assert(appSource.includes('$("#status-filter").value = state.status'), "刷新后未恢复状态筛选");
assert(appSource.includes('$("#phase-filter").value = state.phase'), "刷新后未恢复阶段筛选");
assert(appSource.includes('$("#priority-filter").value = state.priority'), "刷新后未恢复优先级筛选");
assert(appSource.includes('status.setAttribute("aria-busy", "true")'), "刷新开始时未设置 aria-busy");
assert(appSource.includes('status.removeAttribute("aria-busy")'), "刷新结束时未清除 aria-busy");
assert(appSource.includes("setRefreshInterval"), "缺少可选自动刷新间隔");
assert(!/\bfetch\s*\(/.test(appSource), "看板刷新不应依赖 fetch（file:// 不兼容）");
assert(!/window\.location\.reload\s*\(/.test(appSource), "页面回退应使用 cache-busting URL，而不是普通 reload");

if (data && Array.isArray(data.tasks) && Array.isArray(data.risks) && Array.isArray(data.evidence)) {
  const now = Date.now();
  const taskIds = assertUnique(data.tasks, "任务");
  const riskIds = assertUnique(data.risks, "风险");
  const evidenceIds = assertUnique(data.evidence, "证据");
  const requiredTaskFields = [
    "id", "phase", "priority", "title", "content", "purpose", "estimate", "startedAt",
    "finishedAt", "status", "progress", "detail", "acceptance", "evidence", "risks", "updatedAt"
  ];

  assert(data.project?.status === "active", "项目总体状态必须保持 active");
  const dartsEvidence = data.evidence.find((entry) => entry.id === "EV-DARTS-CIFAR-12-RUNS");
  assert(dartsEvidence?.result.includes("合计 12 runs"), "DARTS CIFAR 完成数未记录为 12 runs");
  assert(dartsEvidence?.result.includes("确定性预检重复一致"), "DARTS 确定性预检结果缺失");
  assert(dartsEvidence?.result.includes("两个恢复审计与三组报告完成"), "DARTS 恢复审计或报告状态缺失");
  const dartsImageNetFailure = data.evidence.find((entry) => entry.id === "EV-DARTS-IMAGENET-PREFLIGHT-FAILED");
  assert(dartsImageNetFailure?.result.includes("2026-07-31T01:45:39Z 状态 failed"), "DARTS ImageNet preflight 失败时间或状态缺失");
  assert(dartsImageNetFailure?.result.includes("[Errno 32] Broken pipe"), "DARTS ImageNet preflight 错误缺失");
  assert(dartsImageNetFailure?.result.includes("1 行 training.jsonl") && dartsImageNetFailure?.result.includes("last.pt/best.pt"), "DARTS ImageNet preflight 部分产物缺失");
  assert(dartsImageNetFailure?.result.includes("不构成完成证据") && dartsImageNetFailure?.result.includes("旧 manifest 不回写 completed"), "DARTS ImageNet preflight 失败边界不明确");
  const dartsTask = data.tasks.find((entry) => entry.id === "D1");
  const highCostTask = data.tasks.find((entry) => entry.id === "H2");
  const longScheduleTask = data.tasks.find((entry) => entry.id === "H3");
  const dartsLiveTask = data.tasks.find((entry) => entry.id === "J4");
  for (const task of [dartsTask, dartsLiveTask]) {
    assert(task?.status === "已完成" && task?.progress === 100 && task?.finishedAt === "2026-07-31 16:52", `DARTS ImageNet 任务 ${task?.id || "?"} 未同步完成状态`);
  }
  for (const task of [highCostTask, longScheduleTask]) {
    assert(task?.status === "进行中", `高成本任务 ${task?.id || "?"} 不应因 DARTS 子范围完成而整体完成`);
  }
  assert(dartsTask?.evidence.includes("EV-DARTS-IMAGENET-PREFLIGHT-FAILED"), "DARTS 任务未引用失败证据");
  assert(highCostTask?.evidence.includes("EV-DARTS-IMAGENET-PREFLIGHT-FAILED"), "高成本任务未引用失败证据");
  const dartsStdoutFix = data.evidence.find((entry) => entry.id === "EV-DARTS-CLI-STDOUT-FIX");
  assert(dartsStdoutFix?.result.includes("d0ccc6f"), "DARTS CLI stdout 修复 commit 缺失");
  assert(dartsStdoutFix?.result.includes("fb72a87"), "DARTS checkpoint 兼容修复 commit 缺失");
  assert(dartsStdoutFix?.result.includes("旧 preflight manifest 继续保持 failed") && dartsStdoutFix?.result.includes("不回写 completed"), "DARTS 原始失败 manifest 边界不明确");
  const dartsResumeEvidence = data.evidence.find((entry) => entry.id === "EV-DARTS-IMAGENET-ZERO-INCREMENT-RESUME");
  assert(dartsResumeEvidence?.result.includes("<audit-root>/training/darts-imagenet-preflight-resume-audit/20260731T020100Z_735cd5d3c551"), "DARTS 脱敏恢复审计路径缺失");
  assert(dartsResumeEvidence?.result.includes("manifest=completed") && dartsResumeEvidence?.result.includes("resumed_training_rows=1"), "DARTS 零增量恢复结果缺失");
  assert(dartsResumeEvidence?.result.includes("training JSONL SHA-256 与旧 failed run 完全一致"), "DARTS 恢复 SHA 一致性缺失");
  assert(dartsResumeEvidence?.result.includes("旧原始 run manifest 仍保持 failed"), "DARTS 旧 manifest 未保持 failed");
  const dartsNextSix = data.evidence.find((entry) => entry.id === "EV-DARTS-IMAGENET-NEXT-SIX");
  assert(dartsNextSix?.result.includes("历史启动前计划") && dartsNextSix?.result.includes("旧 commit 78d8118 run 已 interrupted"), "DARTS 六项历史计划边界缺失");
  assert(dartsNextSix?.result.includes("EV-DARTS-C0C7815-RUNNING"), "DARTS 当前运行证据指向缺失");
  const dartsSixLaunched = data.evidence.find((entry) => entry.id === "EV-DARTS-IMAGENET-SIX-LAUNCHED");
  assert(dartsSixLaunched?.result.includes("2026-07-31 10:12:06 +08 启动"), "DARTS 六项验收启动时间缺失");
  assert(dartsSixLaunched?.result.includes("commit 78d8118") && dartsSixLaunched?.result.includes("运行 3 小时无完成 epoch"), "旧 DARTS commit 或停滞时长缺失");
  assert(dartsSixLaunched?.result.includes("run.log 为 0 字节") && dartsSixLaunched?.result.includes("收到 SIGTERM"), "旧 DARTS 日志或 SIGTERM 结果缺失");
  assert(dartsSixLaunched?.result.includes("supervisor status 与 manifest 均为 interrupted"), "旧 DARTS interrupted 状态缺失");
  assert(dartsSixLaunched?.result.includes("不得标 completed"), "旧 DARTS 未完成边界缺失");
  assert(!/supervisor PID/i.test(dartsSixLaunched?.result || ""), "DARTS 看板不应跟踪本机监督器 PID");
  const oldDartsRun = data.evidence.find((entry) => entry.id === "EV-DARTS-OLD-78D8118-INTERRUPTED");
  assert(oldDartsRun?.result.includes("旧 commit 78d8118") && oldDartsRun?.result.includes("旧 HDD 数据"), "旧 DARTS run 身份缺失");
  assert(oldDartsRun?.result.includes("运行 3 小时仍无完成 epoch") && oldDartsRun?.result.includes("run.log 为 0 字节"), "旧 DARTS 停滞证据缺失");
  assert(oldDartsRun?.result.includes("收到 SIGTERM") && oldDartsRun?.result.includes("supervisor status 与 manifest 均为 interrupted"), "旧 DARTS 中断证据缺失");
  const flushEvidence = data.evidence.find((entry) => entry.id === "EV-RUNCONTEXT-FLUSH-ACCEPTANCE");
  assert(flushEvidence?.result.includes("commit c0c7815"), "RunContext flush commit 缺失");
  assert(flushEvidence?.result.includes("RunContext.event") && flushEvidence?.result.includes("同步 flush") && flushEvidence?.result.includes("events.jsonl/run.log"), "RunContext 同步 flush 语义缺失");
  assert(flushEvidence?.result.includes("tools/acceptance 四卡脚本"), "四卡 acceptance 脚本证据缺失");
  const newDartsRun = data.evidence.find((entry) => entry.id === "EV-DARTS-C0C7815-RUNNING");
  assert(newDartsRun?.result.includes("<project-runs>/acceptance") && newDartsRun?.result.includes("<fast-imagenet-root>"), "新 DARTS run 脱敏路径缺失");
  assert(newDartsRun?.result.includes("完整核验") && newDartsRun?.result.includes("commit c0c7815"), "新 DARTS 数据或 commit 身份缺失");
  assert(newDartsRun?.result.includes("约 7.2 batches/s") && newDartsRun?.result.includes("首 epoch ETA 约 22 分钟"), "新 DARTS 吞吐或 ETA 缺失");
  assert(newDartsRun?.result.includes("events.jsonl 与 run.log 均持续增长"), "新 DARTS 实时日志增长证据缺失");
  assert(newDartsRun?.result.includes("仍为 running") && newDartsRun?.result.includes("不得标 completed"), "新 DARTS 运行状态边界缺失");
  const heartbeatEvidence = data.evidence.find((entry) => entry.id === "EV-LIVE-TRAINING-HEARTBEAT");
  assert(heartbeatEvidence?.result.includes("commit 0e3ad8e") && heartbeatEvidence?.result.includes("commit c0c7815"), "训练 heartbeat commit 链缺失");
  assert(heartbeatEvidence?.result.includes("training_batch_progress") && heartbeatEvidence?.result.includes("training_epoch_completed"), "训练 heartbeat 事件类型缺失");
  assert(heartbeatEvidence?.result.includes("同步 flush events.jsonl/run.log"), "训练 heartbeat 同步 flush 缺失");
  assert(heartbeatEvidence?.result.includes("78d8118 run") && heartbeatEvidence?.result.includes("interrupted"), "旧 heartbeat run 中断边界缺失");
  assert(heartbeatEvidence?.result.includes("新 c0c7815 run 的 events.jsonl 与 run.log 均增长"), "新 heartbeat run 增长状态缺失");
  const legacyHeartbeatRisk = data.risks.find((entry) => entry.id === "R-LEGACY-TRAINING-NO-HEARTBEAT");
  assert(legacyHeartbeatRisk?.status === "关闭" && legacyHeartbeatRisk?.description.includes("78d8118"), "旧训练无 heartbeat 风险未关闭隔离");
  assert(legacyHeartbeatRisk?.description.includes("supervisor status 与 manifest 均为 interrupted"), "旧训练风险缺少 interrupted 结果");
  assert(legacyHeartbeatRisk?.mitigation.includes("commit c0c7815") && legacyHeartbeatRisk?.mitigation.includes("不回写 completed"), "旧训练隔离缓解措施缺失");
  for (const task of [dartsTask, highCostTask, longScheduleTask, dartsLiveTask]) {
    assert(task?.evidence.includes("EV-LIVE-TRAINING-HEARTBEAT"), `任务 ${task?.id || "?"} 未引用 heartbeat 证据`);
    assert(task?.evidence.includes("EV-DARTS-OLD-78D8118-INTERRUPTED"), `任务 ${task?.id || "?"} 未引用旧 run 中断证据`);
    assert(task?.evidence.includes("EV-RUNCONTEXT-FLUSH-ACCEPTANCE"), `任务 ${task?.id || "?"} 未引用 flush 修复证据`);
    assert(task?.evidence.includes("EV-DARTS-C0C7815-RUNNING"), `任务 ${task?.id || "?"} 未引用新 run 证据`);
    assert(task?.risks.includes("R-LEGACY-TRAINING-NO-HEARTBEAT"), `任务 ${task?.id || "?"} 未引用旧运行 heartbeat 风险`);
  }
  const oldDdpDiagnosis = data.evidence.find((entry) => entry.id === "EV-DARTS-OLD-DDP-DIAGNOSIS");
  assert(oldDdpDiagnosis?.result.includes("global batch 128") && oldDdpDiagnosis?.result.includes("每卡 32") && oldDdpDiagnosis?.result.includes("约 1.8 GiB"), "旧 DDP batch 或显存诊断缺失");
  assert(oldDdpDiagnosis?.result.includes("supervisor 已退出") && oldDdpDiagnosis?.result.includes("孤立 ranks") && oldDdpDiagnosis?.result.includes("task2 已明确 interrupted"), "旧 DDP 衔接或中断结论缺失");
  const parallelRun = data.evidence.find((entry) => entry.id === "EV-DARTS-PARALLEL-1449");
  assert(parallelRun?.result.includes("task6 params-matched 1%-data 已完成 250/250") && parallelRun?.result.includes("manifest=completed"), "DARTS task6 完成状态缺失");
  assert(parallelRun?.result.includes("2026-07-31 16:52+08:00 全部完成") && parallelRun?.result.includes("759 training rows"), "DARTS 六项完成时间或 training rows 缺失");
  assert(parallelRun?.result.includes("CSV/PNG/SVG/HTML"), "DARTS report bundle 格式缺失");
  assert(parallelRun?.result.includes("full-data zcp/fixed/params=39.528/38.624/29.852") && parallelRun?.result.includes("1%-data=9.6/10.6/5.0"), "DARTS 六项 final valid top1 缺失");
  assert(!/\/public\/|\/home\/|\bPID\b/.test(`${oldDdpDiagnosis?.result || ""} ${parallelRun?.result || ""}`), "DARTS 并行证据不得泄露绝对路径或 PID");
  assert(dartsLiveTask?.detail.includes("task6 已完成 250/250") && dartsLiveTask?.detail.includes("manifest=completed"), "DARTS 实时任务未同步 task6 完成");
  assert(dartsLiveTask?.detail.includes("2026-07-31 16:52+08:00 全部完成") && dartsLiveTask?.detail.includes("759 training rows") && dartsLiveTask?.detail.includes("CSV/PNG/SVG/HTML"), "DARTS 实时任务未同步六项完成产物");
  assert(dartsLiveTask?.detail.includes("39.528/38.624/29.852") && dartsLiveTask?.detail.includes("9.6/10.6/5.0"), "DARTS 实时任务未同步 final valid top1");
  assert(dartsLiveTask?.detail.includes("4-GPU DDP") && dartsLiveTask?.detail.includes("单 GPU") && dartsLiveTask?.detail.includes("BatchNorm 统计粒度风险"), "DARTS 实时任务未保留 BatchNorm 粒度风险");
  const dartsRisk = data.risks.find((entry) => entry.id === "R-DARTS-IMAGENET-DATA");
  assert(dartsRisk?.status === "监控" && dartsRisk?.description.includes("六项已全部完成") && dartsRisk?.description.includes("4-GPU DDP") && dartsRisk?.description.includes("单 GPU") && dartsRisk?.description.includes("BatchNorm 统计粒度不同"), "DARTS BatchNorm 粒度风险状态不准确");
  assert(dartsRisk?.mitigation.includes("执行拓扑") && dartsRisk?.mitigation.includes("相同 GPU 拓扑复跑"), "DARTS BatchNorm 风险缓解措施缺失");
  const launcherEvidence = data.evidence.find((entry) => entry.id === "EV-ACCEPTANCE-FREEZE-PARALLEL-LAUNCHERS");
  assert(launcherEvidence?.result.includes("commit 11dcc88") && launcherEvidence?.result.includes("freeze-candidates") && launcherEvidence?.result.includes("completed/versioned search identity") && launcherEvidence?.result.includes("search JSONL") && launcherEvidence?.result.includes("checksum") && launcherEvidence?.result.includes("三候选和 manifest"), "候选冻结协议证据缺失");
  assert(launcherEvidence?.result.includes("commit ebc9799") && launcherEvidence?.result.includes("parallel_single_gpu") && launcherEvidence?.result.includes("AutoFormer、PlainNet、Proxyless") && launcherEvidence?.result.includes("全仓门禁通过"), "通用单卡并行启动器或门禁证据缺失");
  assert(launcherEvidence?.result.includes("不代表 DARTS 六项训练完成"), "启动器能力与训练完成边界缺失");
  const numaEvidence = data.evidence.find((entry) => entry.id === "EV-DARTS-NUMA1-AFFINITY-OBSERVATION");
  assert(numaEvidence?.result.includes("物理 GPU4–7") && numaEvidence?.result.includes("均位于 NUMA1"), "DARTS GPU 物理位置或 NUMA 证据缺失");
  assert(numaEvidence?.result.includes("20 秒采样") && numaEvidence?.result.includes("SM 多数约 40–60%") && numaEvidence?.result.includes("周期性为 0%") && numaEvidence?.result.includes("4.8–5.5 GiB"), "DARTS GPU 采样证据缺失");
  assert(numaEvidence?.result.includes("互斥 16 逻辑核分片实验") && numaEvidence?.result.includes("吞吐下降约 6–8%") && numaEvidence?.result.includes("1% 任务 epoch 变长") && numaEvidence?.result.includes("结论 rejected"), "DARTS 分片 affinity rejected 证据缺失");
  assert(numaEvidence?.result.includes("完整 NUMA1 共享列表 32–63,96–127") && numaEvidence?.result.includes("约 7.4–7.6 batch/s") && numaEvidence?.result.includes("无正向证据") && numaEvidence?.result.includes("背景负载也可能变化") && numaEvidence?.result.includes("结论 inconclusive"), "DARTS 完整 NUMA affinity inconclusive 证据缺失");
  assert(numaEvidence?.result.includes("无收益即回退") && numaEvidence?.result.includes("全部恢复 affinity=0–127") && numaEvidence?.result.includes("不再处于 NUMA 优化观察"), "DARTS affinity 完全回退状态缺失");
  assert(numaEvidence?.result.includes("显式可选 ZCP_CPU_AFFINITIES") && numaEvidence?.result.includes("不设默认") && numaEvidence?.result.includes("不代表训练完成"), "DARTS affinity 代码默认与完成边界缺失");
  assert(numaEvidence?.command.includes("只读审计") && numaEvidence?.command.includes("training.jsonl 行数"), "DARTS training.jsonl 只读审计指引缺失");
  assert(!/\/public\/|\/home\/|\bPID\b/.test(`${numaEvidence?.result || ""} ${numaEvidence?.command || ""}`), "DARTS NUMA 证据不得泄露绝对路径或 PID");
  assert(![numaEvidence?.result, dartsTask?.detail, dartsLiveTask?.detail, dartsRisk?.description].some((value) => String(value || "").includes("完整NUMA亲和性观察中") || String(value || "").includes("已提速") || String(value || "").includes("已优化")), "DARTS NUMA 文案不得保留仍在观察或正向收益表述");
  assert(highCostTask?.detail.includes("39.528/38.624/29.852") && highCostTask?.detail.includes("DARTS 六项 report bundle 已生成"), "H2 未同步 DARTS full-data 完成指标");
  assert(longScheduleTask?.detail.includes("三个 1%-data 任务均已完成 250/250") && longScheduleTask?.detail.includes("9.6/10.6/5.0") && longScheduleTask?.detail.includes("DARTS 六项已完成"), "H3 未同步 DARTS 1%-data 完成状态");
  const laneLockEvidence = data.evidence.find((entry) => entry.id === "EV-DARTS-LANE-LOCK-4FC2C3D");
  assert(laneLockEvidence?.result.includes("task6 完成后继承锁自然释放") && laneLockEvidence?.result.includes("AutoFormer systemd 服务随即自动启动"), "DARTS 锁释放与自动衔接证据缺失");
  assert(laneLockEvidence?.result.includes("commit 4fc2c3d") && laneLockEvidence?.result.includes("longest-first + per-lane lock release"), "DARTS lane 锁修复证据缺失");
  assert(laneLockEvidence?.result.includes("旧 run 未注入强制解锁"), "DARTS 旧 run 不强制解锁边界缺失");
  const httpRecoveryEvidence = data.evidence.find((entry) => entry.id === "EV-PANEL-HTTP-8768-8769-RECOVERY");
  assert(httpRecoveryEvidence?.result.includes("8768/8769") && httpRecoveryEvidence?.result.includes("监听状态但 curl 超时") && httpRecoveryEvidence?.result.includes("服务重启"), "看板 HTTP 服务故障与恢复证据缺失");
  assert(httpRecoveryEvidence?.result.includes("index、data.js 与 monitor 根目录") && httpRecoveryEvidence?.result.includes("均可通过 HTTP 访问"), "看板 HTTP 可访问性验证缺失");
  assert(!/\/public\/|\/home\/|\bPID\b/.test(`${httpRecoveryEvidence?.result || ""} ${httpRecoveryEvidence?.command || ""}`), "看板 HTTP 恢复证据不得泄露绝对路径或 PID");
  const panelTask = data.tasks.find((entry) => entry.id === "F4");
  assert(panelTask?.detail.includes("8768/8769") && panelTask?.detail.includes("curl 超时") && panelTask?.detail.includes("已重启服务") && panelTask?.detail.includes("monitor 根目录"), "F4 未同步 HTTP 服务恢复状态");
  const fullGate468 = data.evidence.find((entry) => entry.id === "EV-FULL-GATE-468");
  assert(fullGate468?.result.includes("commit c0c7815") && fullGate468?.result.includes("468 tests 全部通过"), "当前主仓 468 pytest 结果缺失");
  assert(fullGate468?.result.includes("Ruff") && fullGate468?.result.includes("compileall") && fullGate468?.result.includes("pip check"), "当前主仓 468 静态门禁缺失");
  const fullGate467 = data.evidence.find((entry) => entry.id === "EV-FULL-GATE-467");
  assert(fullGate467?.result.includes("collected 467 tests 且全部通过"), "历史主仓 467 coverage pytest 结果缺失");
  assert(fullGate467?.result.includes("source coverage 87%") && fullGate467?.result.includes("CLI coverage 82%"), "历史主仓 467 coverage 缺失");
  assert(fullGate467?.result.includes("已由 EV-FULL-GATE-468 取代"), "历史 467 门禁未标记为被 468 取代");
  const baselineTask = data.tasks.find((entry) => entry.id === "A1");
  const qualityGateTask = data.tasks.find((entry) => entry.id === "G1");
  for (const task of [baselineTask, qualityGateTask]) {
    assert(task?.detail.includes("commit c0c7815") && task?.detail.includes("468 tests 全部通过"), `任务 ${task?.id || "?"} 缺少当前 468 gate`);
    assert(task?.detail.includes("Ruff") && task?.detail.includes("compileall") && task?.detail.includes("pip check"), `任务 ${task?.id || "?"} 缺少当前静态门禁`);
  }
  const releaseTask = data.tasks.find((entry) => entry.id === "I1");
  const workspaceEvidence = data.evidence.find((entry) => entry.id === "EV-WORKSPACE-CONSOLIDATION");
  assert(releaseTask?.detail.includes("旧 integration worktree 已删除") && releaseTask?.detail.includes("<project-runs>/acceptance/audit-archive"), "发布任务未同步工作区收敛");
  assert(workspaceEvidence?.result.includes("旧 integration worktree 已删除") && workspaceEvidence?.result.includes("旧 audit 工作区已移入 <project-runs>/acceptance/audit-archive"), "工作区收敛证据缺失");
  assert(workspaceEvidence?.result.includes("父目录当前仅保留主仓"), "工作区唯一主仓状态缺失");
  assert(!/\/public\/|\/home\//.test(JSON.stringify(data)), "看板数据不得泄露本机绝对路径");
  const fullGate465 = data.evidence.find((entry) => entry.id === "EV-FULL-GATE-465");
  assert(fullGate465?.result.includes("465 tests passed") && fullGate465?.result.includes("pytest 退出码 0"), "历史全仓 465 tests 或 pytest 结果缺失");
  assert(fullGate465?.result.includes("source coverage 87%") && fullGate465?.result.includes("CLI coverage 82%"), "历史 465 coverage 缺失");
  assert(fullGate465?.result.includes("Ruff") && fullGate465?.result.includes("compileall") && fullGate465?.result.includes("pip check"), "历史 465 静态门禁缺失");
  assert(fullGate465?.result.includes("已由主仓 EV-FULL-GATE-467 取代"), "历史 465 门禁未标记为被 467 取代");
  const fullGate456 = data.evidence.find((entry) => entry.id === "EV-FULL-GATE-456-COLLECT");
  assert(fullGate456?.result.includes("已由 EV-FULL-GATE-465 取代"), "历史 456 门禁未标记为被 465 取代");
  const plainnetTask = data.tasks.find((entry) => entry.id === "C2");
  const ofaTask = data.tasks.find((entry) => entry.id === "C3");
  const plainnetEvidence = data.evidence.find((entry) => entry.id === "EV-PLAINNET-FIDELITY-AUDIT");
  const plainnetRisk = data.risks.find((entry) => entry.id === "R-PLAINNET-FIDELITY");
  const ofaEvidence = data.evidence.find((entry) => entry.id === "EV-PROXYLESS-MBV2-TRAINING-PROTOCOL-FIDELITY");
  assert(plainnetTask?.status === "已完成" && plainnetTask?.progress === 100, "PlainNet 结构与候选协议任务未标记完成");
  assert(plainnetTask?.title.includes("PlainNet 真实 structure-string 与候选协议"), "PlainNet 任务标题未更新");
  for (const phrase of ["structure-string parser", "12 类 SuperResIDWE block", "SE", "sample/mutate/crossover", "参数/MAC golden", "150 epoch candidate profile"]) {
    assert(plainnetTask?.detail.includes(phrase), `PlainNet 任务详情缺少 ${phrase}`);
    assert(plainnetEvidence?.result.includes(phrase), `PlainNet 证据缺少 ${phrase}`);
  }
  assert(plainnetTask?.detail.includes("相关模块 199 项通过") && plainnetEvidence?.result.includes("相关模块 199 项通过"), "PlainNet 199 项专项测试结果缺失");
  assert(plainnetTask?.detail.includes("全仓 465 tests passed"), "PlainNet 任务缺少全仓 465 tests 结果");
  assert(plainnetTask?.detail.includes("source coverage 87%") && plainnetTask?.detail.includes("CLI coverage 82%"), "PlainNet coverage 结果缺失");
  assert(plainnetTask?.detail.includes("Ruff") && plainnetTask?.detail.includes("compileall") && plainnetTask?.detail.includes("pip check"), "PlainNet 静态门禁结果缺失");
  const plainnetTrainingBlocker = "双重 1% GPU / distributed validation / checkpoint resume / reporting 仍开放";
  assert(plainnetTask?.detail.includes("结构与候选协议完成") && plainnetTask?.detail.includes(plainnetTrainingBlocker), "PlainNet 完成范围或正式训练 blocker 缺失");
  assert(plainnetEvidence?.result.includes(plainnetTrainingBlocker) && plainnetEvidence?.result.includes("不得称正式训练完成"), "PlainNet 证据未保留正式训练边界");
  assert(plainnetRisk?.status === "开放" && plainnetRisk?.mitigation.includes("distributed validation") && plainnetRisk?.mitigation.includes("checkpoint resume") && plainnetRisk?.mitigation.includes("reporting"), "PlainNet 正式训练风险未保持开放");
  assert(ofaTask?.status === "进行中", "OFA 任务应保持进行中");
  assert(ofaTask?.detail.includes("c5234b8"), "OFA MAC golden commit 缺失");
  assert(ofaTask?.detail.includes("265,526,256"), "OFA MAC golden 数值缺失");
  assert(ofaTask?.detail.includes("265,526,240"), "OFA float32 profile 数值缺失");
  assert(ofaTask?.detail.includes("双重 1% / distributed validation / reporting"), "OFA 训练阻断边界缺失");
  assert(ofaEvidence?.result.includes("c5234b8"), "OFA 证据缺少 MAC golden commit");
  assert(ofaEvidence?.result.includes("265,526,256") && ofaEvidence?.result.includes("265,526,240"), "OFA 证据缺少 MAC 数值");
  assert(ofaEvidence?.result.includes("双重 1% / distributed validation / reporting"), "OFA 证据缺少训练阻断边界");
  const pitTask = data.tasks.find((entry) => entry.id === "C4");
  const pitEvidence = data.evidence.find((entry) => entry.id === "EV-PIT-REFERENCE");
  const pitRisk = data.risks.find((entry) => entry.id === "R-PIT");
  for (const phrase of ["base_dim=16", "depth=[2,8,4]", "heads=[2,4,4]", "mlp=6", "classes=100", "input=224"]) {
    assert(pitEvidence?.result.includes(phrase), `PiT 证据缺少规格 ${phrase}`);
  }
  assert(pitEvidence?.result.includes("90ed458eff6948a6f0d23e440a8d21bbec50d091"), "PiT 锁定上游 commit 缺失");
  assert(pitEvidence?.result.includes("分别运行 THOP") && pitEvidence?.result.includes("159,665,472 MAC"), "PiT THOP MAC golden 缺失");
  assert(pitEvidence?.result.includes("sum parameters 均为 893,828") && pitEvidence?.result.includes("shape multiset 一致"), "PiT 参数量或 shape multiset 对照缺失");
  assert(pitEvidence?.result.includes("MAC golden 已完成"), "PiT MAC golden 完成状态缺失");
  assert(pitEvidence?.command.includes("https://github.com/lliai/Auto-Prox-AAAI24/tree/90ed458eff6948a6f0d23e440a8d21bbec50d091"), "PiT 可信 GitHub commit URL 缺失");
  assert(pitEvidence?.command.includes("docs/evidence/VITBENCH_PREFLIGHT_CN.md"), "PiT 仓库相对证据文档缺失");
  assert(!pitEvidence?.command.includes("/tmp") && !pitEvidence?.command.includes("PYTHONPATH="), "PiT 公共操作指引不得保留本机临时路径");
  assert(pitTask?.detail.includes("159,665,472 MAC") && pitTask?.detail.includes("893,828") && pitTask?.detail.includes("参数/MAC golden 已完成"), "PiT 任务详情未同步 MAC golden");
  assert(pitEvidence?.result.includes("reference_topology_pytorch_port") && pitEvidence?.result.includes("缺官方 checkpoint 与逐层数值对照"), "PiT fidelity 边界缺失");
  assert(pitRisk?.description.includes("缺少官方 checkpoint 和逐层数值对照") && pitRisk?.mitigation.includes("reference_topology_pytorch_port"), "PiT 风险未保持 topology port 边界");
  const autoFormerTask = data.tasks.find((entry) => entry.id === "C1");
  const autoFormerEvidence = data.evidence.find((entry) => entry.id === "EV-AUTOFORMER-TRAINING-PROTOCOL-FIDELITY");
  assert(autoFormerTask?.detail.includes("source-pinned commit 5e6683a") && autoFormerTask?.detail.includes("aznas-5e6683-autoformer-stable-v1") && autoFormerTask?.detail.includes("paper_formula_port_stabilized"), "AutoFormer AZ-NAS 代理身份缺失");
  assert(autoFormerTask?.detail.includes("status=running") && autoFormerTask?.detail.includes("不再 queued") && autoFormerTask?.detail.includes("100/100/200"), "AutoFormer AZ-NAS 任务运行状态缺失");
  assert(autoFormerTask?.detail.includes("尚未完成") && autoFormerTask?.detail.includes("候选未冻结") && autoFormerTask?.detail.includes("不是上游控制器逐行复现"), "AutoFormer AZ-NAS 任务边界缺失");
  assert(autoFormerEvidence?.result.includes("133 项专项相关测试通过"), "AutoFormer 专项测试计数缺失");
  assert(autoFormerEvidence?.result.includes("不是全仓测试总数"), "AutoFormer 专项测试范围未明确");
  const azNasEvidence = data.evidence.find((entry) => entry.id === "EV-AUTOFORMER-AZNAS-STABLE-SMOKE");
  assert(azNasEvidence?.result.includes("source-pinned commit 5e6683a") && azNasEvidence?.result.includes("az_nas_autoformer") && azNasEvidence?.result.includes("aznas-5e6683-autoformer-stable-v1") && azNasEvidence?.result.includes("paper_formula_port_stabilized"), "AutoFormer AZ-NAS source/version/fidelity 证据缺失");
  assert(azNasEvidence?.result.includes("attention/MLP residual features") && azNasEvidence?.result.includes("expressivity、trainability、official complexity") && azNasEvidence?.result.includes("az_nas_log_rank"), "AutoFormer AZ-NAS feature 或 score 证据缺失");
  assert(azNasEvidence?.result.includes("旧 az_nas portable") && azNasEvidence?.result.includes("正式 search 默认拒绝"), "AutoFormer 旧 portable 正式搜索拒绝边界缺失");
  assert(azNasEvidence?.result.includes("两个独立同 seed GPU smoke") && azNasEvidence?.result.includes("2 candidates + 1 summary") && azNasEvidence?.result.includes("architecture-hash-v1") && azNasEvidence?.result.includes("去除耗时字段后逐行一致"), "AutoFormer AZ-NAS 确定性 smoke 证据缺失");
  assert(azNasEvidence?.result.includes("AutoFormer 尚未完成") && azNasEvidence?.result.includes("候选未冻结") && azNasEvidence?.result.includes("不是上游控制器逐行复现"), "AutoFormer AZ-NAS 验收边界缺失");
  assert(azNasEvidence?.result.includes("3 seeds × 8,000 随机候选") && azNasEvidence?.result.includes("zcp-test-autoformer-aznas-8000.service") && azNasEvidence?.result.includes("status=running") && azNasEvidence?.result.includes("不再 queued"), "AutoFormer AZ-NAS 运行身份缺失");
  assert(azNasEvidence?.result.includes("每 100 candidates 原子保存 partial state"), "AutoFormer AZ-NAS partial state 协议缺失");
  assert(azNasEvidence?.result.includes("双进程峰值显存 2,105 MiB") && azNasEvidence?.result.includes("峰值利用率 99%"), "AutoFormer packed smoke 资源证据缺失");
  assert(azNasEvidence?.result.includes("GPU 累计、epoch 末同步"), "AutoFormer trainer 指标累计方式缺失");
  assert(azNasEvidence?.result.includes("首批 partial state 为 100/100/200"), "AutoFormer 首批 partial state 进度缺失");
  assert(azNasEvidence?.result.includes("GPU5 双进程平均 SM 81.07%") && azNasEvidence?.result.includes("p50 88%") && azNasEvidence?.result.includes("max 99%") && azNasEvidence?.result.includes("GPU6 单进程平均 SM 44.25%"), "AutoFormer 30 秒 GPU 吞吐实测缺失");
  assert(azNasEvidence?.command.includes("docs/evidence/aznas_autoformer_rank_smoke.json"), "AutoFormer AZ-NAS 仓库相对证据文档缺失");
  assert(azNasEvidence?.command.includes("docs/evidence/gpu_throughput_optimization.json"), "AutoFormer GPU 吞吐证据文档缺失");
  const autoFormerRisk = data.risks.find((entry) => entry.id === "R-AUTOFORMER");
  assert(autoFormerRisk?.status === "开放" && autoFormerRisk?.description.includes("status=running") && autoFormerRisk?.description.includes("100/100/200") && autoFormerRisk?.description.includes("尚未完成") && autoFormerRisk?.description.includes("候选仍未冻结"), "AutoFormer 风险未保持运行与候选冻结边界");
  const budgetRisk = data.risks.find((entry) => entry.id === "R-BUDGET");
  assert(budgetRisk?.description.includes("允许增加高成本验收时间预算"), "高成本验收追加时间预算授权缺失");
  for (const riskId of ["R-DDP-RANK-RNG"]) {
    const risk = data.risks.find((entry) => entry.id === riskId);
    assert(risk && ["开放", "受阻"].includes(risk.status), `关键风险 ${riskId} 未保持开放或受阻`);
  }
  assert(data.risks.find((entry) => entry.id === "R-DARTS-IMAGENET-DATA")?.status === "监控", "DARTS 完成后 BatchNorm 粒度风险应保持监控");
  assert(data.risks.find((entry) => entry.id === "R-LAUNCHER-EXIT-CODE")?.status === "关闭", "旧 supervisor 继承锁风险未关闭");
  for (const taskId of ["H2", "H3"]) {
    const task = data.tasks.find((entry) => entry.id === taskId);
    assert(task?.status !== "已完成", `任务 ${taskId} 不应标记为全部完成`);
  }

  checkTime(data.updatedAt, "面板 updatedAt", now);
  for (const task of data.tasks) {
    for (const field of requiredTaskFields) assert(task[field] !== undefined, `任务 ${task.id || "?"} 缺少 ${field}`);
    assert(Boolean(data.statuses?.[task.status]), `任务 ${task.id} 状态无效：${task.status}`);
    assert(Boolean(data.priorities?.[task.priority]), `任务 ${task.id} 优先级无效：${task.priority}`);
    assert(Number.isFinite(task.progress) && task.progress >= 0 && task.progress <= 100, `任务 ${task.id} progress 越界`);
    assert(Array.isArray(task.acceptance) && Array.isArray(task.evidence) && Array.isArray(task.risks), `任务 ${task.id} 的列表字段无效`);
    const startedAt = checkTime(task.startedAt, `任务 ${task.id} startedAt`, now);
    const finishedAt = checkTime(task.finishedAt, `任务 ${task.id} finishedAt`, now);
    const updatedAt = checkTime(task.updatedAt, `任务 ${task.id} updatedAt`, now);
    assert(task.status === "已完成" ? Number.isFinite(finishedAt) : finishedAt === null, `任务 ${task.id} 状态与 finishedAt 不一致`);
    if (Number.isFinite(startedAt) && Number.isFinite(updatedAt)) assert(startedAt <= updatedAt, `任务 ${task.id} updatedAt 早于 startedAt`);
    if (Number.isFinite(finishedAt) && Number.isFinite(updatedAt)) assert(finishedAt <= updatedAt, `任务 ${task.id} updatedAt 早于 finishedAt`);
    for (const id of task.evidence) assert(evidenceIds.has(id), `任务 ${task.id} 引用未知证据 ${id}`);
    for (const id of task.risks) assert(riskIds.has(id), `任务 ${task.id} 引用未知风险 ${id}`);
  }

  for (const risk of data.risks) {
    assert(["高", "中", "低"].includes(risk.severity), `风险 ${risk.id} severity 无效：${risk.severity}`);
    assert(Array.isArray(risk.taskIds), `风险 ${risk.id} taskIds 必须是数组`);
    for (const id of risk.taskIds || []) assert(taskIds.has(id), `风险 ${risk.id} 引用未知任务 ${id}`);
  }

  for (const entry of data.evidence) {
    checkTime(entry.time, `证据 ${entry.id} time`, now);
    assert(entry.title && entry.result && entry.command, `证据 ${entry.id} 缺少标题、结果或命令`);
    assert(Array.isArray(entry.taskIds), `证据 ${entry.id} taskIds 必须是数组`);
    for (const id of entry.taskIds || []) assert(taskIds.has(id), `证据 ${entry.id} 引用未知任务 ${id}`);
  }
}

if (errors.length) {
  console.error(`数据完整性检查失败（${errors.length} 项）：`);
  for (const error of errors) console.error(`- ${error}`);
  process.exitCode = 1;
} else {
  console.log(`数据完整性检查通过：${data.tasks.length} tasks / ${data.risks.length} risks / ${data.evidence.length} evidence`);
}
