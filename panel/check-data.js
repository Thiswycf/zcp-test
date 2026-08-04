"use strict";

const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const dataPath = path.join(__dirname, "data.js");
const appPath = path.join(__dirname, "app.js");
const indexPath = path.join(__dirname, "index.html");
const stylesPath = path.join(__dirname, "styles.css");
const readmePath = path.join(__dirname, "README.md");
const liveStatusPath = path.join(__dirname, "live-status.py");
const gitignorePath = path.join(__dirname, "..", ".gitignore");
const operationsCnPath = path.join(__dirname, "..", "docs", "OPERATIONS_CN.md");
const operationsPath = path.join(__dirname, "..", "docs", "OPERATIONS.md");
const context = { window: {} };
vm.runInNewContext(fs.readFileSync(dataPath, "utf8"), context, { filename: dataPath });
const data = context.window.ZCP_PANEL_DATA;
const appSource = fs.readFileSync(appPath, "utf8");
const indexSource = fs.readFileSync(indexPath, "utf8");
const stylesSource = fs.readFileSync(stylesPath, "utf8");
const readmeSource = fs.readFileSync(readmePath, "utf8");
const liveStatusSource = fs.readFileSync(liveStatusPath, "utf8");
const gitignoreSource = fs.readFileSync(gitignorePath, "utf8");
const operationsCnSource = fs.readFileSync(operationsCnPath, "utf8");
const operationsSource = fs.readFileSync(operationsPath, "utf8");
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
for (const id of ["live-status-title", "live-freshness", "live-warning", "live-status-content"]) {
  assert(indexSource.includes(`id="${id}"`), `实时运行卡缺少 #${id}`);
}
assert(indexSource.includes("实时运行"), "页面缺少实时运行卡标题");
assert(stylesSource.includes(".live-status-grid") && stylesSource.includes(".live-warning.is-stale"), "实时运行卡或陈旧警告样式缺失");
assert(appSource.includes('new URL("live.json", window.location.href)'), "实时刷新未请求 live.json");
assert(appSource.includes('fetch(url.href, { cache: "no-store"'), "live.json 未使用 no-store fetch");
assert(appSource.includes('url.searchParams.set("refresh"'), "live.json 刷新缺少 cache-busting");
assert(appSource.includes("Promise.allSettled([loadFreshDataScript(), loadFreshLiveStatus()])"), "静态与实时状态未在同一刷新周期获取");
assert(appSource.includes("静态看板不受影响") && appSource.includes("继续显示静态任务、风险与证据"), "live.json 缺失降级文案不完整");
assert(appSource.includes("stale_after_seconds") && appSource.includes("live.json 已"), "live.json 陈旧判断或警告缺失");
assert(appSource.includes("rate_per_second") && appSource.includes("eta_seconds"), "实时卡缺少速率或 ETA 展示");
assert(appSource.includes("utilization_percent") && appSource.includes("memory_used_mib"), "实时卡缺少 GPU 利用率或显存展示");
assert(appSource.includes("candidateTotal") && appSource.includes("candidateTarget"), "AutoFormer workload 进度未使用 candidate rows");
assert(appSource.includes("uniqueEvaluations") && appSource.includes("cacheHits"), "AutoFormer 实时卡缺少 unique evaluations 或 cache hits");
assert(appSource.includes("workloadStatus") && appSource.includes("supervisorStatus"), "AutoFormer workload 与 supervisor 状态未独立归并");
assert(appSource.includes("workload ${escapeHtml(workloadStatus)}") && appSource.includes("supervisor ${escapeHtml(supervisorStatus)}"), "AutoFormer 双状态 badge 未渲染");
assert(stylesSource.includes(".live-badges") && stylesSource.includes(".live-status.status-failed"), "AutoFormer 双 badge 或 supervisor failed 样式缺失");
assert(appSource.includes("独立 orchestration warning"), "Supervisor failure 未明确保持为独立 orchestration warning");
assert(readmeSource.includes("python -m http.server 8768 --directory panel"), "README 缺少静态服务器命令");
assert(readmeSource.includes("python panel/live-status.py --once"), "README 缺少 live-status --once 命令");
assert(readmeSource.includes("python panel/live-status.py --watch --interval 15"), "README 缺少 live-status watch 命令");
assert(readmeSource.includes("systemd-run --user --unit=zcp-test-panel-live"), "README 缺少 live-status systemd-run 示例");
assert(readmeSource.includes("file://"), "README 缺少 file:// 限制说明");
assert(readmeSource.includes("无需手动按 F5"), "README 未说明 file:// 页面重载回退");
assert(appSource.includes('$("#status-filter").value = state.status'), "刷新后未恢复状态筛选");
assert(appSource.includes('$("#phase-filter").value = state.phase'), "刷新后未恢复阶段筛选");
assert(appSource.includes('$("#priority-filter").value = state.priority'), "刷新后未恢复优先级筛选");
assert(appSource.includes('status.setAttribute("aria-busy", "true")'), "刷新开始时未设置 aria-busy");
assert(appSource.includes('status.removeAttribute("aria-busy")'), "刷新结束时未清除 aria-busy");
assert(appSource.includes("setRefreshInterval"), "缺少可选自动刷新间隔");
assert((appSource.match(/\bfetch\s*\(/g) || []).length === 1, "看板只应使用一次 fetch 获取可降级的 live.json");
assert(!/window\.location\.reload\s*\(/.test(appSource), "页面回退应使用 cache-busting URL，而不是普通 reload");

for (const phrase of [
  "runs/acceptance/darts-imagenet-parallel/status.json",
  "runs/acceptance/autoformer-aznas-random-8000",
  "search-state.json",
  "Asia/Shanghai",
  "nvidia-smi",
  "os.replace",
  "--once",
  "--watch",
  "--interval"
]) {
  assert(liveStatusSource.includes(phrase), `live-status.py 缺少结构：${phrase}`);
}
assert(liveStatusSource.includes("with temporary.open") && liveStatusSource.includes("os.fsync"), "live.json 缺少临时文件落盘流程");
assert(liveStatusSource.includes("except (OSError, json.JSONDecodeError)"), "live-status.py 未容忍暂缺或不完整 JSON");
assert(gitignoreSource.split(/\r?\n/).includes("/panel/live.json"), ".gitignore 未忽略 panel/live.json");

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
  const fullGate563 = data.evidence.find((entry) => entry.id === "EV-FULL-GATE-563");
  const fullGate545 = data.evidence.find((entry) => entry.id === "EV-FULL-GATE-545");
  assert(fullGate563?.result.includes("38 个测试文件") && fullGate563?.result.includes("563 tests 全部通过"), "当前工作树 563 pytest 结果缺失");
  assert(fullGate563?.result.includes("Ruff") && fullGate563?.result.includes("compileall") && fullGate563?.result.includes("pip check") && fullGate563?.result.includes("Bash") && fullGate563?.result.includes("JSON") && fullGate563?.result.includes("diff"), "当前工作树 563 静态门禁缺失");
  assert(fullGate545?.result.includes("37 个测试文件") && fullGate545?.result.includes("545 tests 全部通过"), "历史工作树 545 pytest 结果缺失");
  const fullGate467 = data.evidence.find((entry) => entry.id === "EV-FULL-GATE-467");
  assert(fullGate467?.result.includes("collected 467 tests 且全部通过"), "历史主仓 467 coverage pytest 结果缺失");
  assert(fullGate467?.result.includes("source coverage 87%") && fullGate467?.result.includes("CLI coverage 82%"), "历史主仓 467 coverage 缺失");
  assert(fullGate467?.result.includes("已由 EV-FULL-GATE-545 取代"), "历史 467 门禁链缺失");
  const baselineTask = data.tasks.find((entry) => entry.id === "A1");
  const qualityGateTask = data.tasks.find((entry) => entry.id === "G1");
  for (const task of [baselineTask, qualityGateTask]) {
    assert(task?.detail.includes("38 files") && task?.detail.includes("563 tests"), `任务 ${task?.id || "?"} 缺少当前 563 gate`);
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
  const ofaProtocolCorrection = data.evidence.find((entry) => entry.id === "EV-OFA-PROXYLESS-PROTOCOL-CORRECTION-20260804");
  const ofaDirectSearchRisk = data.risks.find((entry) => entry.id === "R-OFA-DIRECT-SEARCH-FIDELITY");
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
  assert(ofaTask?.detail.includes("现有 ofa_proxyless_mbv2 模型域正确"), "Proxyless-MBV2 模型域结论缺失");
  for (const marker of ["project_zcp_transfer", "project_resource_baseline", "controller_fidelity=project_controller_not_ofa_tutorial", "direct_search_protocol_evidence=false"]) {
    assert(ofaTask?.detail.includes(marker), `OFA 任务缺少协议标记 ${marker}`);
    assert(ofaProtocolCorrection?.result.includes(marker), `OFA 纠错证据缺少协议标记 ${marker}`);
  }
  assert(ofaEvidence?.result.includes("c5234b8"), "OFA 证据缺少 MAC golden commit");
  assert(ofaEvidence?.result.includes("265,526,256") && ofaEvidence?.result.includes("265,526,240"), "OFA 证据缺少 MAC 数值");
  assert(ofaEvidence?.result.includes("双重 1% / distributed validation / reporting"), "OFA 证据缺少训练阻断边界");
  assert(ofaProtocolCorrection?.result.includes("commit f03b267") && ofaProtocolCorrection?.result.includes("20-block/5-stage") && ofaProtocolCorrection?.result.includes("resolution={160,176,192,208,224}") && ofaProtocolCorrection?.result.includes("OFA-MBV3"), "OFA-MBV3 tutorial 归属纠错缺失");
  assert(ofaProtocolCorrection?.result.includes("21 个动态 ks/e 位置") && ofaProtocolCorrection?.result.includes("6 个 block groups") && ofaProtocolCorrection?.result.includes("前 5 组 depth 可变") && ofaProtocolCorrection?.result.includes("最后一组 depth 固定为 1"), "Proxyless 动态位置或 block-group depth 边界缺失");
  assert(ofaProtocolCorrection?.result.includes("width=1.3") && ofaProtocolCorrection?.result.includes("resolution=128..224"), "Proxyless width 或 resolution 范围缺失");
  assert(ofaDirectSearchRisk?.status === "监控" && ofaDirectSearchRisk?.description.includes("模型域") && ofaDirectSearchRisk?.description.includes("不是官方 OFA tutorial 的直接搜索协议证据"), "Proxyless direct-search fidelity 风险缺失");
  assert(ofaDirectSearchRisk?.mitigation.includes("只有新增独立官方直接搜索协议证据后才能调整 fidelity"), "Proxyless fidelity 升级门槛缺失");
  assert(!JSON.stringify(data).includes("ofa_proxyless_official_tutorial"), "不得新增或宣称虚构的 Proxyless official tutorial 协议名");
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
  assert(autoFormerTask?.detail.includes("24,000/24,000 candidate rows") && autoFormerTask?.detail.includes("unique evaluations=23,999") && autoFormerTask?.detail.includes("cache hits=1"), "AutoFormer cohort 完成度或缓存语义缺失");
  assert(autoFormerTask?.detail.includes("workload completed") && autoFormerTask?.detail.includes("failed/exit 127") && autoFormerTask?.detail.includes("orchestration warning"), "AutoFormer workload/supervisor 双状态边界缺失");
  assert(autoFormerTask?.detail.includes("zcp_selected=42e6457ccb580a092454") && autoFormerTask?.detail.includes("fixed_random=d904aacf51d2b0867df6") && autoFormerTask?.detail.includes("params_flops_matched=41b5e6d4dc3279909487"), "AutoFormer 三个冻结候选身份缺失");
  assert(autoFormerTask?.detail.includes("42dc72f29e141fa97c042c1979f390486962a97fa34cdbcd3394b556148bdb4a") && autoFormerTask?.detail.includes("supporting seed 只作 provenance"), "AutoFormer 冻结 manifest 或 supporting seed 边界缺失");
  assert(autoFormerTask?.detail.includes("batch=256 synthetic full-batch memory smoke") && autoFormerTask?.detail.includes("GPU0/1/3") && autoFormerTask?.detail.includes("completed/exit 0"), "AutoFormer 三候选 memory smoke 状态缺失");
  assert(autoFormerTask?.detail.includes("不是实际 ImageNet 精度") && autoFormerTask?.detail.includes("不是论文精度") && autoFormerTask?.detail.includes("项目总体验收仍未完成"), "AutoFormer 历史 smoke 或当前科学边界缺失");
  assert(autoFormerTask?.detail.includes("full-data 5 epoch") && autoFormerTask?.detail.includes("one-percent-data 500 epoch") && autoFormerTask?.detail.includes("last.pt/best.pt"), "AutoFormer 双重 1% 完成边界缺失");
  const dualOnePercentPolicy = data.evidence.find((entry) => entry.id === "EV-DUAL-ONE-PERCENT-ZCP-ONLY-POLICY-20260804");
  assert(dualOnePercentPolicy?.result.includes("一个 zcp-selected 架构") && dualOnePercentPolicy?.result.includes("全数据 × 1% epoch") && dualOnePercentPolicy?.result.includes("1% 数据 × 完整 schedule"), "未来双重 1% 单架构两协议政策缺失");
  assert(dualOnePercentPolicy?.result.includes("不能证明 ZCP 优于") && dualOnePercentPolicy?.result.includes("约 3 倍资源") && dualOnePercentPolicy?.result.includes("退出工程 gate"), "短训科学边界或三候选资源理由缺失");
  assert(dualOnePercentPolicy?.result.includes("已完成的历史产物只读保留") && dualOnePercentPolicy?.result.includes("尚未启动的 queued baseline 必须取消"), "历史产物保留或 queued baseline 取消政策缺失");
  assert(dualOnePercentPolicy?.result.includes("另行预声明") && dualOnePercentPolicy?.result.includes("充分训练") && dualOnePercentPolicy?.result.includes("多 seed"), "验收比较研究前置条件缺失");
  for (const taskId of ["H2", "H3"]) {
    const policyTask = data.tasks.find((entry) => entry.id === taskId);
    assert(policyTask?.evidence.includes(dualOnePercentPolicy?.id), `任务 ${taskId} 未引用双重 1% 新政策`);
    assert(policyTask?.detail.includes("zcp-selected") && policyTask?.detail.includes("充分训练") && policyTask?.detail.includes("多 seed"), `任务 ${taskId} 未同步单候选或比较研究边界`);
  }
  const policyCleanupEvidence = data.evidence.find((entry) => entry.id === "EV-AUTOFORMER-POLICY-CLEANUP-20260804");
  const policySchedulingRisk = data.risks.find((entry) => entry.id === "R-AUTOFORMER-POLICY-SCHEDULING");
  assert(policyCleanupEvidence?.result.includes("不是新手工下发") && policyCleanupEvidence?.result.includes("调度判断错误") && policyCleanupEvidence?.result.includes("不应继续"), "旧 supervisor 自动续启基线的责任边界缺失");
  assert(policyCleanupEvidence?.result.includes("task5 fixed-random") && policyCleanupEvidence?.result.includes("task6 params-flops-matched") && policyCleanupEvidence?.result.includes("manifest=interrupted"), "task5/6 定向终止状态缺失");
  assert(policyCleanupEvidence?.result.includes("约 89/15 MiB") && policyCleanupEvidence?.result.includes("利用率均为 0%") && policyCleanupEvidence?.result.includes("两把 kernel flock 已释放"), "task5/6 GPU 与锁释放证据缺失");
  assert(policyCleanupEvidence?.result.includes("task4 zcp-selected 继续在 GPU5") && policyCleanupEvidence?.result.includes("旧 main supervisor 已 STOP"), "task4 保留或旧 supervisor 阻断状态缺失");
  assert(policyCleanupEvidence?.result.includes("zcp-test-autoformer-policy-cleanup.service") && policyCleanupEvidence?.result.includes("等待 task4 terminal") && policyCleanupEvidence?.result.includes("policy-override.json") && policyCleanupEvidence?.result.includes("KILL 旧服务"), "cleanup watcher 生命周期缺失");
  assert(policySchedulingRisk?.status === "关闭" && policySchedulingRisk?.description.includes("调度判断错误") && policySchedulingRisk?.mitigation.includes("禁止 fixed-random") && policySchedulingRisk?.mitigation.includes("params-flops-matched"), "AutoFormer 政策调度终态或缓解措施缺失");
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
  assert(autoFormerRisk?.status === "关闭" && autoFormerRisk?.description.includes("11:02–11:03") && autoFormerRisk?.description.includes("主动中断") && autoFormerRisk?.description.includes("不记科学失败"), "AutoFormer 风险未保留首次 real dual-1% 主动中断");
  assert(autoFormerRisk?.description.includes("5 epoch") && autoFormerRisk?.description.includes("500 epoch") && autoFormerRisk?.description.includes("fixed-random") && autoFormerRisk?.description.includes("interrupted"), "AutoFormer 风险未同步当前完成与基线中断状态");
  const cohortEvidence = data.evidence.find((entry) => entry.id === "EV-AUTOFORMER-COHORT-RECONCILED");
  assert(cohortEvidence?.result.includes("24,000/24,000 candidate rows") && cohortEvidence?.result.includes("23,999 unique evaluations") && cohortEvidence?.result.includes("1 cache hit"), "AutoFormer cohort 归并证据缺失");
  assert(cohortEvidence?.result.includes("workload 状态为 completed") && cohortEvidence?.result.includes("独立 orchestration warning") && cohortEvidence?.result.includes("总体验收仍未完成"), "AutoFormer cohort 证据科学边界缺失");
  const frozenCandidatesEvidence = data.evidence.find((entry) => entry.id === "EV-AUTOFORMER-CANDIDATES-FROZEN");
  assert(frozenCandidatesEvidence?.result.includes("2026-08-04T10:55:31+08:00") && frozenCandidatesEvidence?.result.includes("42e6457ccb580a092454") && frozenCandidatesEvidence?.result.includes("d904aacf51d2b0867df6") && frozenCandidatesEvidence?.result.includes("41b5e6d4dc3279909487"), "AutoFormer 候选冻结时间或身份缺失");
  assert(frozenCandidatesEvidence?.result.includes("42dc72f29e141fa97c042c1979f390486962a97fa34cdbcd3394b556148bdb4a") && frozenCandidatesEvidence?.result.includes("只写入 provenance"), "AutoFormer manifest SHA 或 supporting provenance 边界缺失");
  const memorySmokeEvidence = data.evidence.find((entry) => entry.id === "EV-AUTOFORMER-FULL-BATCH-MEMORY-SMOKE");
  assert(memorySmokeEvidence?.result.includes("zcp-test-autoformer-smoke-{zcp-selected,fixed-random,params-flops-matched}.service") && memorySmokeEvidence?.result.includes("GPU0/1/3") && memorySmokeEvidence?.result.includes("batch=256"), "AutoFormer 三候选 smoke unit、GPU 或 batch 缺失");
  assert(memorySmokeEvidence?.result.includes("completed/exit 0") && memorySmokeEvidence?.result.includes("不使用真实 ImageNet") && memorySmokeEvidence?.result.includes("双重 1% 训练尚未启动"), "AutoFormer smoke 终态或科学边界缺失");
  const immutableLauncherEvidence = data.evidence.find((entry) => entry.id === "EV-LAUNCHER-IMMUTABLE-SNAPSHOT");
  assert(immutableLauncherEvidence?.result.includes("仅复制 Shell") && immutableLauncherEvidence?.result.includes("git archive") && immutableLauncherEvidence?.result.includes("Shell、Python 与 configs"), "Launcher 完整 commit 快照边界缺失");
  assert(immutableLauncherEvidence?.result.includes("只读 launcher-snapshots") && immutableLauncherEvidence?.result.includes("不导入主仓新代码") && immutableLauncherEvidence?.result.includes("lock_acquired/lock_released"), "Launcher 快照执行或结构化锁日志证据缺失");
  const interruptedTrainingEvidence = data.evidence.find((entry) => entry.id === "EV-AUTOFORMER-REAL-DUAL-1PCT-SNAPSHOT-INTERRUPT");
  assert(interruptedTrainingEvidence?.result.includes("11:02") && interruptedTrainingEvidence?.result.includes("11:03") && interruptedTrainingEvidence?.result.includes("interrupted"), "AutoFormer real dual-1% 首次中断时间或状态缺失");
  assert(interruptedTrainingEvidence?.result.includes("不记科学失败") && interruptedTrainingEvidence?.result.includes("所有 GPU 锁已释放") && interruptedTrainingEvidence?.result.includes("待修复提交后重新启动"), "AutoFormer 首次中断科学边界、锁释放或重启条件缺失");
  const budgetRisk = data.risks.find((entry) => entry.id === "R-BUDGET");
  assert(budgetRisk?.description.includes("允许增加高成本验收时间预算"), "高成本验收追加时间预算授权缺失");
  const ddpRankRngRisk = data.risks.find((entry) => entry.id === "R-DDP-RANK-RNG");
  assert(ddpRankRngRisk?.status === "关闭" && ddpRankRngRisk?.description.includes("commit 53ddd69") && ddpRankRngRisk?.description.includes("rng_by_rank"), "DDP rank-local RNG 风险未关闭或实现身份缺失");
  assert(ddpRankRngRisk?.description.includes("2 进程 Gloo") && ddpRankRngRisk?.description.includes("fail-closed"), "DDP rank-local RNG 真实往返或拒绝边界缺失");
  assert(ddpRankRngRisk?.mitigation.includes("不证明跨 GPU 型号、CUDA 版本、kernel 或 world-size"), "DDP rank-local RNG 科学边界缺失");
  const ddpRankRngEvidence = data.evidence.find((entry) => entry.id === "EV-DDP-RANK-LOCAL-RNG-ROUNDTRIP");
  assert(ddpRankRngEvidence?.result.includes("两个 rank 状态不同") && ddpRankRngEvidence?.result.includes("下一组随机值精确恢复"), "DDP rank-local RNG 往返结果缺失");
  assert(ddpRankRngEvidence?.result.includes("legacy checkpoint 缺字段") && ddpRankRngEvidence?.result.includes("world-size 不匹配") && ddpRankRngEvidence?.result.includes("fail-closed"), "DDP rank-local RNG 旧格式或 world-size 拒绝证据缺失");
  assert(ddpRankRngEvidence?.command.includes("docs/evidence/ddp_rank_local_rng_roundtrip.json"), "DDP rank-local RNG JSON 证据链接缺失");
  const gpuOwnerTask = data.tasks.find((entry) => entry.id === "J6");
  assert(gpuOwnerTask?.status === "进行中" && gpuOwnerTask?.detail.includes("kernel flock 仍是唯一锁权威") && gpuOwnerTask?.detail.includes("陈旧 lock 文件不阻塞获取"), "GPU owner 实时巡检任务或锁权威边界缺失");
  assert(gpuOwnerTask?.detail.includes("task5 fixed-random") && gpuOwnerTask?.detail.includes("task6 params-flops-matched") && gpuOwnerTask?.detail.includes("两个 manifest 均为 interrupted"), "GPU owner 任务未同步基线终止状态");
  assert(gpuOwnerTask?.detail.includes("两把 kernel flock") && gpuOwnerTask?.detail.includes("已释放") && gpuOwnerTask?.detail.includes("task4 zcp-selected 随后在 GPU5 完成"), "GPU owner 任务未同步历史锁释放或 task4 当前终态");
  const gpuOwnerRisk = data.risks.find((entry) => entry.id === "R-GPU-LOCK-RUNTIME-OWNER");
  assert(gpuOwnerRisk?.status === "监控" && gpuOwnerRisk?.mitigation.includes("非阻塞 flock") && gpuOwnerRisk?.mitigation.includes("唯一权威"), "GPU runtime owner 监控风险或处置边界缺失");
  assert(gpuOwnerRisk?.description.includes("task5/6 对应锁已释放") && gpuOwnerRisk?.description.includes("task4 zcp-selected 已完成") && gpuOwnerRisk?.description.includes("PlainNet preflight") && gpuOwnerRisk?.description.includes("释放锁"), "GPU runtime owner 风险未同步当前锁状态");
  const gpuLeaseEvidence = data.evidence.find((entry) => entry.id === "EV-GPU-LOCK-LEASE-RUNTIME-20260804");
  assert(gpuLeaseEvidence?.result.includes("ImageNet candidate") && gpuLeaseEvidence?.result.includes("DARTS resume") && gpuLeaseEvidence?.result.includes("AutoFormer 8,000-candidate"), "GPU lock wrapper launcher 接入范围缺失");
  assert(gpuLeaseEvidence?.result.includes("内核 flock 是唯一权威") && gpuLeaseEvidence?.result.includes(".lease heartbeat") && gpuLeaseEvidence?.result.includes("supervisor 可继续存活"), "GPU lock/lease 权威或释放边界缺失");
  assert(gpuLeaseEvidence?.result.includes("80 passed") && gpuLeaseEvidence?.result.includes("不热修改") && gpuLeaseEvidence?.command.includes("docs/evidence/gpu_lock_lease_runtime_20260804.json"), "GPU lock 专项结果、immutable snapshot 或 JSON 证据引用缺失");
  const gpuOwnerEvidence = data.evidence.find((entry) => entry.id === "EV-GPU-LOCK-OWNER-DIAGNOSTIC-GUIDE");
  assert(gpuOwnerEvidence?.result.includes("lslocks") && gpuOwnerEvidence?.result.includes("禁止 rm lock pathname") && gpuOwnerEvidence?.result.includes("不声称当前机器全部锁"), "GPU owner 诊断手册证据边界缺失");
  assert(autoFormerTask?.status === "已完成" && autoFormerTask?.progress === 100 && autoFormerTask?.evidence.includes("EV-DUAL-ONE-PERCENT-ZCP-ONLY-POLICY-20260804") && autoFormerTask?.evidence.includes("EV-AUTOFORMER-SINGLE-CANDIDATE-DUAL-1PCT-COMPLETE"), "AutoFormer 任务未关联新政策、完成证据或状态错误");
  assert(dualOnePercentPolicy?.result.includes("尚未启动的 queued baseline 必须取消") && dualOnePercentPolicy?.result.includes("只允许已在运行的 zcp-selected 单候选继续"), "单候选政策未明确 queued baseline 取消边界");
  assert(!dualOnePercentPolicy?.result.includes("自然完成"), "单候选政策仍包含 queued baseline 自然完成的过时表述");
  const documentationTask = data.tasks.find((entry) => entry.id === "F1");
  assert(documentationTask?.status === "进行中" && documentationTask?.detail.includes("文档 P1 补丁已完成、待主线验收"), "文档任务未同步 P1 补丁状态");
  assert(documentationTask?.detail.includes("reconcile-search-cohort") && documentationTask?.detail.includes("convert-vit"), "文档任务未列出 CLI 或数据 P1 修订范围");
  assert(documentationTask?.evidence.includes("EV-DOC-P1-PATCH-20260804"), "文档任务未关联 P1 补丁证据");
  const documentationPatchEvidence = data.evidence.find((entry) => entry.id === "EV-DOC-P1-PATCH-20260804");
  assert(documentationPatchEvidence?.result.includes("legacy import") && documentationPatchEvidence?.result.includes("migration manifest v1 schema"), "文档 P1 证据缺少 legacy 或 manifest 范围");
  assert(documentationPatchEvidence?.result.includes("待主线测试与内容验收") && documentationPatchEvidence?.result.includes("不等同于整个文档审计完成"), "文档 P1 证据未保持验收边界");
  const plainnetSearchTask = data.tasks.find((entry) => entry.id === "J7");
  assert(plainnetSearchTask?.status === "进行中" && plainnetSearchTask?.detail.includes("source-aligned controller 已完成"), "PlainNet source-aligned controller 阶段任务缺失");
  assert(plainnetSearchTask?.detail.includes("15,165.56 秒") && plainnetSearchTask?.detail.includes("4.21 小时"), "PlainNet CPU full-history rerank 估计缺失");
  const plainnetWaitingEvidence = data.evidence.find((entry) => entry.id === "EV-PLAINNET-GPU-PREFLIGHT-WAITING-20260804");
  assert(plainnetWaitingEvidence?.result.includes("12:08:31") && plainnetWaitingEvidence?.result.includes("immutable commit 17f54d7") && plainnetWaitingEvidence?.result.includes("waiting_for_gpu_lock"), "PlainNet GPU preflight V2 历史启动身份或等待状态缺失");
  assert(plainnetWaitingEvidence?.result.includes("尚未取得 flock") && plainnetWaitingEvidence?.result.includes("未占 GPU") && plainnetWaitingEvidence?.result.includes("未开始任何候选"), "PlainNet GPU preflight 历史等待资源边界缺失");
  assert(plainnetSearchTask?.detail.includes("3 accepted") && plainnetSearchTask?.detail.includes("2 evaluations") && plainnetSearchTask?.detail.includes("1 cache hit") && plainnetSearchTask?.detail.includes("formal_search_completed=false") && plainnetSearchTask?.detail.includes("正式 100k 搜索尚未启动"), "PlainNet GPU preflight 当前终态或正式 100k 边界缺失");
  assert(plainnetSearchTask?.detail.includes("formal_valid_candidates=100000") && plainnetSearchTask?.detail.includes("stop_after=3") && plainnetSearchTask?.detail.includes("batch=64") && plainnetSearchTask?.detail.includes("input=224"), "PlainNet GPU preflight 协议字段缺失");
  const plainnetRerankEvidence = data.evidence.find((entry) => entry.id === "EV-PLAINNET-RERANK-SCALING-20260804");
  assert(plainnetRerankEvidence?.command.includes("docs/evidence/plainnet_rerank_scaling_20260804.json") && plainnetRerankEvidence?.result.includes("15,165.56 秒"), "PlainNet rerank scaling JSON 证据缺失");
  const plainnetControllerEvidence = data.evidence.find((entry) => entry.id === "EV-PLAINNET-SOURCE-ALIGNED-CONTROLLER-STAGE");
  assert(plainnetControllerEvidence?.result.includes("source_aligned_control_flow_port") && plainnetControllerEvidence?.result.includes("正式 100k 搜索未启动"), "PlainNet controller fidelity 或正式搜索边界缺失");
  const plainnetPreflightEvidence = data.evidence.find((entry) => entry.id === "EV-PLAINNET-GPU-PREFLIGHT-WAITING-20260804");
  assert(plainnetPreflightEvidence?.result.includes("zcp-test-plainnet-source-preflight-v2.service") && plainnetPreflightEvidence?.result.includes("status=waiting_for_gpu_lock"), "PlainNet GPU preflight unit 或 manifest 状态证据缺失");
  assert(plainnetPreflightEvidence?.result.includes("未占 GPU") && plainnetPreflightEvidence?.result.includes("未开始任何候选") && plainnetPreflightEvidence?.result.includes("不是正式 100k 搜索"), "PlainNet GPU preflight 科学边界缺失");
  assert(plainnetPreflightEvidence?.result.includes("两次更早启动") && plainnetPreflightEvidence?.result.includes("不记科学失败") && plainnetPreflightEvidence?.result.includes("冗余目录已删除"), "PlainNet 更早 preflight 启动失败处置边界缺失");
  for (const [source, label] of [[operationsCnSource, "中文"], [operationsSource, "英文"]]) {
    assert(source.includes("flock -n") && source.includes("lslocks") && source.includes("fuser"), `${label} OPERATIONS 缺少真实 flock owner 诊断命令`);
    assert(source.includes("pstree") && source.includes("pgrep") && source.includes("kill -TERM"), `${label} OPERATIONS 缺少 supervisor child/TERM 诊断流程`);
    assert(source.includes("rm *.lock") && source.includes("inode"), `${label} OPERATIONS 缺少禁止删除锁文件边界`);
  }
  assert(data.risks.find((entry) => entry.id === "R-DARTS-IMAGENET-DATA")?.status === "监控", "DARTS 完成后 BatchNorm 粒度风险应保持监控");
  const launcherExitRisk = data.risks.find((entry) => entry.id === "R-LAUNCHER-EXIT-CODE");
  assert(launcherExitRisk?.status === "关闭", "旧 launcher exit 127 风险未关闭");
  assert(launcherExitRisk?.description.includes("运行中的 launcher 脚本被工作树改写") && launcherExitRisk?.description.includes("只读 snapshot") && launcherExitRisk?.description.includes("SHA-256"), "旧 launcher exit 127 根因或治理缺失");
  assert(launcherExitRisk?.mitigation.includes("lock_acquired") && launcherExitRisk?.mitigation.includes("lock_released"), "结构化锁 acquire/release 日志治理缺失");
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
