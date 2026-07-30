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
assert(indexSource.includes('aria-live="polite"'), "刷新状态缺少 aria-live");
assert(indexSource.includes('aria-atomic="true"'), "刷新状态缺少 aria-atomic");
assert(indexSource.includes('aria-pressed="true"'), "自动刷新按钮缺少 aria-pressed");
assert(indexSource.includes('http-equiv="Cache-Control"'), "页面缺少静态托管 no-store 提示");
assert(indexSource.includes('no-cache, no-store, must-revalidate'), "页面 Cache-Control 未声明 no-store");
assert(indexSource.includes("最后成功刷新"), "刷新栏未显示最后成功刷新时间");
assert(indexSource.includes("上次检查"), "刷新栏未显示上次检查时间");
assert(indexSource.includes('<script src="data.js"></script>'), "file:// 回退缺少直接 data.js 脚本");
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
assert(appSource.includes("可点击“立即刷新”重试"), "刷新失败未提示重试");
assert(readmeSource.includes("python -m http.server 8768 --directory panel"), "README 缺少静态服务器命令");
assert(readmeSource.includes("file://"), "README 缺少 file:// 限制说明");
assert(appSource.includes('status.setAttribute("aria-busy", "true")'), "刷新开始时未设置 aria-busy");
assert(appSource.includes('status.removeAttribute("aria-busy")'), "刷新结束时未清除 aria-busy");
assert(appSource.includes("setRefreshInterval"), "缺少可选自动刷新间隔");
assert(!/\bfetch\s*\(/.test(appSource), "看板刷新不应依赖 fetch（file:// 不兼容）");
assert(!/window\.location\.reload\s*\(/.test(appSource), "看板刷新不应依赖整页 reload/F5");

if (data && Array.isArray(data.tasks) && Array.isArray(data.risks) && Array.isArray(data.evidence)) {
  const now = Date.now();
  const taskIds = assertUnique(data.tasks, "任务");
  const riskIds = assertUnique(data.risks, "风险");
  const evidenceIds = assertUnique(data.evidence, "证据");
  const requiredTaskFields = [
    "id", "phase", "priority", "title", "content", "purpose", "estimate", "startedAt",
    "finishedAt", "status", "progress", "detail", "acceptance", "evidence", "risks", "updatedAt"
  ];

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
