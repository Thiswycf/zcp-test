"use strict";

const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const panelDirectory = __dirname;
const dataPath = path.join(panelDirectory, "data.js");
const htmlPath = path.join(panelDirectory, "index.html");
const appPath = path.join(panelDirectory, "app.js");
const errors = [];

function assert(condition, message) {
  if (!condition) errors.push(message);
}

function isObject(value) {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function parseBeijingTime(value, label) {
  const format = /^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$/;
  assert(typeof value === "string" && format.test(value), `${label} 必须使用北京时间格式 YYYY-MM-DD HH:mm`);
  if (typeof value !== "string" || !format.test(value)) return;
  const timestamp = Date.parse(`${value.replace(" ", "T")}:00+08:00`);
  assert(Number.isFinite(timestamp), `${label} 不是有效的北京时间`);
  if (!Number.isFinite(timestamp)) return;
  const beijingDate = new Date(timestamp + 8 * 60 * 60 * 1000);
  const normalized = [
    beijingDate.getUTCFullYear(),
    String(beijingDate.getUTCMonth() + 1).padStart(2, "0"),
    String(beijingDate.getUTCDate()).padStart(2, "0"),
    String(beijingDate.getUTCHours()).padStart(2, "0"),
    String(beijingDate.getUTCMinutes()).padStart(2, "0")
  ];
  assert(`${normalized[0]}-${normalized[1]}-${normalized[2]} ${normalized[3]}:${normalized[4]}` === value, `${label} 不是有效的北京时间`);
}

function validateEta(value) {
  assert(typeof value === "string" && value.trim().length > 0, "ETA 必须是非空字符串");
  if (typeof value !== "string" || value.trim().length === 0) return;
  if (/^\d/.test(value)) {
    parseBeijingTime(value, "ETA");
  } else {
    assert(value.length <= 40, "ETA 待更新说明不能超过 40 个字符");
  }
}

let data;
try {
  const source = fs.readFileSync(dataPath, "utf8");
  const context = { window: {} };
  vm.createContext(context);
  vm.runInContext(source, context, { filename: dataPath });
  data = context.window.ZCP_PANEL_DATA;
} catch (error) {
  errors.push(`无法载入 data.js：${error.message}`);
}

if (data) {
  assert(isObject(data), "顶层数据必须是对象");
  assert(data.schemaVersion === 3, "schemaVersion 必须为 3");
  assert(data.timeZone === "Asia/Shanghai", "timeZone 必须为 Asia/Shanghai");

  const audit = data.audit;
  assert(isObject(audit), "audit 必须是对象");
  if (isObject(audit)) {
    const requiredFields = ["percentage", "phase", "conclusion", "eta", "updatedAt"];
    for (const field of requiredFields) assert(Object.hasOwn(audit, field), `audit 缺少必需字段 ${field}`);
    assert(Number.isFinite(audit.percentage) && audit.percentage >= 0 && audit.percentage <= 100, "percentage 必须是 0..100 的有限数值");
    assert(typeof audit.phase === "string" && audit.phase.trim().length > 0, "phase 必须是非空字符串");
    assert(typeof audit.conclusion === "string" && audit.conclusion.trim().length > 0, "conclusion 必须是非空字符串");
    validateEta(audit.eta);
    parseBeijingTime(audit.updatedAt, "最后更新时间");
  }
}

try {
  const html = fs.readFileSync(htmlPath, "utf8");
  const cards = html.match(/<article\b/gi) || [];
  assert(cards.length === 1, "页面必须且只能包含一张卡片");
  for (const id of ["audit-percentage", "audit-phase", "audit-conclusion", "audit-eta", "refresh-data", "auto-refresh-toggle", "refresh-interval", "refresh-status"]) {
    assert(html.includes(`id="${id}"`), `页面缺少 #${id}`);
  }
  assert(!html.includes('id="audit-updated-at"'), "页面不应显示内部更新时间字段");
  for (const legacyLabel of ["历史任务", "证据", "风险", "GPU", "实时运行", "状态概览"]) {
    assert(!html.includes(legacyLabel), `页面仍包含旧模块：${legacyLabel}`);
  }
} catch (error) {
  errors.push(`无法检查 index.html：${error.message}`);
}

try {
  const app = fs.readFileSync(appPath, "utf8");
  const refreshConnections = [
    ["elements.refreshButton.addEventListener(\"click\"", "手动刷新按钮未接线"],
    ["elements.autoRefreshButton.addEventListener(\"click\"", "自动刷新开关未接线"],
    ["elements.refreshInterval.addEventListener(\"change\"", "刷新间隔选择器未接线"],
    ["window.setTimeout", "自动刷新定时器缺失"],
    ["data.js?refresh=", "刷新请求缺少缓存规避参数"],
    ["reloadForFileProtocol", "file:// 刷新回退缺失"]
  ];
  for (const [source, message] of refreshConnections) assert(app.includes(source), message);
} catch (error) {
  errors.push(`无法检查 app.js：${error.message}`);
}

if (errors.length > 0) {
  console.error(`面板结构检查失败（${errors.length} 项）：`);
  for (const error of errors) console.error(`- ${error}`);
  process.exitCode = 1;
} else {
  console.log("面板结构检查通过：单卡数据与刷新功能有效");
}
