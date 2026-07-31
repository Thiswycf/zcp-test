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
assert(appSource.includes('`数据更新时间：${data.updatedAt} · schema v${data.schemaVersion}`'), "未明确显示数据更新时间");
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
  for (const task of [dartsTask, highCostTask, longScheduleTask, dartsLiveTask]) {
    assert(task?.status === "进行中", `DARTS ImageNet 任务 ${task?.id || "?"} 未保持进行中`);
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
  assert(dartsNextSix?.result.includes("延长预算") && dartsNextSix?.result.includes("持久日志") && dartsNextSix?.result.includes("detached 模式"), "DARTS 六项验收启动方式缺失");
  const dartsSixLaunched = data.evidence.find((entry) => entry.id === "EV-DARTS-IMAGENET-SIX-LAUNCHED");
  assert(dartsSixLaunched?.result.includes("2026-07-31 10:12:06 +08 启动"), "DARTS 六项验收启动时间缺失");
  assert(dartsSixLaunched?.result.includes("<audit-root>/training/darts-imagenet-extended.status.json"), "DARTS 六项验收脱敏状态文件缺失");
  assert(dartsSixLaunched?.result.includes("running full-data-3epoch/zcp-selected"), "DARTS 当前运行子项缺失");
  assert(dartsSixLaunched?.result.includes("commit 78d8118") && dartsSixLaunched?.result.includes("四个 UUID 锁已获取"), "DARTS commit 或 GPU 锁状态缺失");
  assert(dartsSixLaunched?.result.includes("detached + persistent supervisor/per-run logs"), "DARTS detached 持久日志模式缺失");
  assert(dartsSixLaunched?.result.includes("允许追加时间预算") && dartsSixLaunched?.result.includes("不得标 completed"), "DARTS 延长预算或未完成边界缺失");
  assert(!/supervisor PID/i.test(dartsSixLaunched?.result || ""), "DARTS 看板不应跟踪本机监督器 PID");
  const fullGate465 = data.evidence.find((entry) => entry.id === "EV-FULL-GATE-465");
  assert(fullGate465?.result.includes("465 tests passed") && fullGate465?.result.includes("pytest 退出码 0"), "最新全仓 465 tests 或 pytest 结果缺失");
  assert(fullGate465?.result.includes("source coverage 87%") && fullGate465?.result.includes("CLI coverage 82%"), "最新全仓 coverage 缺失");
  assert(fullGate465?.result.includes("Ruff") && fullGate465?.result.includes("compileall") && fullGate465?.result.includes("pip check"), "最新全仓静态门禁缺失");
  assert(fullGate465?.result.includes("不表示 integration HEAD 已改变"), "新 worktree 门禁与 integration HEAD 边界缺失");
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
  const autoFormerTask = data.tasks.find((entry) => entry.id === "C1");
  const autoFormerEvidence = data.evidence.find((entry) => entry.id === "EV-AUTOFORMER-TRAINING-PROTOCOL-FIDELITY");
  assert(autoFormerTask?.detail.includes("integration commit 74ee153 完成"), "AutoFormer protocol fidelity 完成状态缺失");
  assert(autoFormerEvidence?.result.includes("133 项专项相关测试通过"), "AutoFormer 专项测试计数缺失");
  assert(autoFormerEvidence?.result.includes("不是全仓测试总数"), "AutoFormer 专项测试范围未明确");
  const budgetRisk = data.risks.find((entry) => entry.id === "R-BUDGET");
  assert(budgetRisk?.description.includes("允许增加高成本验收时间预算"), "高成本验收追加时间预算授权缺失");
  for (const riskId of ["R-DARTS-IMAGENET-DATA", "R-DDP-RANK-RNG", "R-LAUNCHER-EXIT-CODE"]) {
    const risk = data.risks.find((entry) => entry.id === riskId);
    assert(risk && ["开放", "受阻"].includes(risk.status), `关键风险 ${riskId} 未保持开放或受阻`);
  }
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
