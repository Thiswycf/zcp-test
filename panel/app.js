(() => {
  "use strict";

  const rawData = window.ZCP_PANEL_DATA;
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const state = {
    query: "",
    status: "all",
    phase: "all",
    priority: "all",
    sort: "priority"
  };
  let dialogOpener = null;
  let refreshPromise = null;
  let liveData = null;
  let liveError = "live.json 尚未加载";
  let refreshRequestId = 0;
  let refreshInitialized = false;
  let autoRefreshEnabled = true;
  let autoRefreshTimer = null;
  let refreshCountdownTimer = null;
  let nextRefreshAt = null;
  let refreshIntervalMs = 30_000;
  const dataScriptTimeoutMs = 10_000;
  const liveFetchTimeoutMs = 8_000;
  const refreshIntervalStorageKey = "zcp-panel-refresh-interval";
  const reloadStateStorageKey = "zcp-panel-reload-state";
  const panelDateTimeFormatter = new Intl.DateTimeFormat("zh-CN", {
    timeZone: "Asia/Shanghai",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false
  });
  const panelClockFormatter = new Intl.DateTimeFormat("zh-CN", {
    timeZone: "Asia/Shanghai",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false
  });

  function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>'"]/g, (character) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      "'": "&#39;",
      "\"": "&quot;"
    })[character]);
  }

  function formatPanelTime(value) {
    if (value === "—" || value == null || value === "") return "—";
    const match = String(value).match(/^(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2})/);
    if (!match) return `${value}（北京时间）`;
    const date = new Date(`${match[1]}-${match[2]}-${match[3]}T${match[4]}:${match[5]}:00+08:00`);
    return `${panelDateTimeFormatter.format(date)}（北京时间）`;
  }

  function validateData(data) {
    if (!data || !Array.isArray(data.tasks) || !data.statuses || !data.priorities) {
      throw new Error("看板数据缺少 tasks、statuses 或 priorities");
    }
    const required = [
      "id", "phase", "priority", "title", "content", "purpose", "estimate",
      "startedAt", "finishedAt", "status", "progress", "detail", "acceptance",
      "evidence", "risks", "updatedAt"
    ];
    data.tasks.forEach((task, index) => {
      const missing = required.filter((field) => task[field] === undefined);
      if (missing.length) {
        throw new Error(`任务 ${index + 1} 缺少字段：${missing.join("、")}`);
      }
      if (!data.statuses[task.status] || !data.priorities[task.priority]) {
        throw new Error(`任务 ${task.id} 使用了未知状态或优先级`);
      }
      if (!Number.isFinite(task.progress) || task.progress < 0 || task.progress > 100) {
        throw new Error(`任务 ${task.id} 的 progress 必须在 0–100`);
      }
    });
    return data;
  }

  function showFatalError(error) {
    document.body.innerHTML = `<main class="fatal-error"><h1>看板数据无法加载</h1><p>${escapeHtml(error.message)}</p></main>`;
  }

  let data;
  try {
    data = validateData(rawData);
  } catch (error) {
    showFatalError(error);
    return;
  }

  const evidenceById = new Map();
  const riskById = new Map();

  function rebuildIndexes() {
    evidenceById.clear();
    riskById.clear();
    data.evidence.forEach((entry) => evidenceById.set(entry.id, entry));
    data.risks.forEach((risk) => riskById.set(risk.id, risk));
  }

  rebuildIndexes();

  function announce(message) {
    $("#announcer").textContent = "";
    window.setTimeout(() => { $("#announcer").textContent = message; }, 20);
  }

  function statusColor(status) {
    return data.statuses[status]?.color || "#68766f";
  }

  function badge(status) {
    const color = statusColor(status);
    return `<span class="badge" style="--badge-color:${escapeHtml(color)}">${escapeHtml(status)}</span>`;
  }

  function priorityBadge(priority) {
    const metadata = data.priorities[priority];
    return `<span class="priority priority-${escapeHtml(priority.toLowerCase())}">${escapeHtml(metadata.label)}</span>`;
  }

  function renderSummary() {
    const total = data.tasks.length;
    const statusCards = Object.entries(data.statuses).map(([status, metadata]) => ({
      label: status,
      value: data.tasks.filter((task) => task.status === status).length,
      note: status === "已完成" ? "满足当前验收条件" : "选择以筛选任务",
      status,
      color: metadata.color
    }));
    const cards = [
      { label: "全部任务", value: total, note: `${new Set(data.tasks.map((task) => task.phase)).size} 个阶段`, status: "all", color: "var(--accent)" },
      ...statusCards
    ];
    $("#summary").innerHTML = cards.map((card) => `
      <button class="summary-card${state.status === card.status ? " is-active" : ""}"
              type="button" data-summary-status="${escapeHtml(card.status)}"
              aria-pressed="${state.status === card.status}"
              style="--card-accent:${escapeHtml(card.color)}">
        <span>${escapeHtml(card.label)}</span>
        <strong>${card.value}</strong>
        <small>${escapeHtml(card.note)}</small>
      </button>`).join("");

    $$("[data-summary-status]").forEach((button) => {
      button.addEventListener("click", () => {
        state.status = button.dataset.summaryStatus;
        $("#status-filter").value = state.status;
        renderAllTaskViews();
        announce(`已筛选${button.textContent.trim()}`);
      });
    });

    const percent = total
      ? Math.round(data.tasks.reduce((sum, task) => sum + task.progress, 0) / total)
      : 0;
    $("#progress-label").textContent = `${percent}%`;
    $("#progress-bar").style.width = `${percent}%`;
    $("#progress-track").setAttribute("aria-valuenow", String(percent));
    $("#status-legend").innerHTML = Object.entries(data.statuses).map(([status, metadata]) => {
      const count = data.tasks.filter((task) => task.status === status).length;
      return `<span class="legend-item"><i class="legend-dot" style="--dot:${escapeHtml(metadata.color)}"></i>${escapeHtml(status)} <strong>${count}</strong></span>`;
    }).join("");
  }

  function searchableText(task) {
    const evidenceText = task.evidence.map((id) => {
      const entry = evidenceById.get(id);
      return entry ? `${entry.title} ${entry.result} ${entry.command}` : id;
    });
    const riskText = task.risks.map((id) => {
      const risk = riskById.get(id);
      return risk ? `${risk.title} ${risk.description} ${risk.mitigation}` : id;
    });
    return [
      task.id, task.phase, task.priority, task.title, task.content, task.purpose,
      task.detail, task.status, ...task.acceptance, ...evidenceText, ...riskText
    ].join(" ").toLocaleLowerCase("zh-CN");
  }

  function compareTasks(left, right) {
    if (state.sort === "updated") {
      return right.updatedAt.localeCompare(left.updatedAt) || left.id.localeCompare(right.id);
    }
    if (state.sort === "status") {
      return data.statuses[left.status].order - data.statuses[right.status].order
        || data.priorities[left.priority].order - data.priorities[right.priority].order
        || left.id.localeCompare(right.id);
    }
    if (state.sort === "phase") {
      return left.phase.localeCompare(right.phase, "zh-CN")
        || data.priorities[left.priority].order - data.priorities[right.priority].order
        || left.id.localeCompare(right.id);
    }
    return data.priorities[left.priority].order - data.priorities[right.priority].order
      || data.statuses[left.status].order - data.statuses[right.status].order
      || left.id.localeCompare(right.id);
  }

  function filteredTasks() {
    const query = state.query.trim().toLocaleLowerCase("zh-CN");
    return data.tasks.filter((task) => (
      (!query || searchableText(task).includes(query))
      && (state.status === "all" || task.status === state.status)
      && (state.phase === "all" || task.phase === state.phase)
      && (state.priority === "all" || task.priority === state.priority)
    )).sort(compareTasks);
  }

  function linkedEvidence(task) {
    return task.evidence.map((id) => evidenceById.get(id)).filter(Boolean);
  }

  function linkedRisks(task) {
    return task.risks.map((id) => riskById.get(id)).filter(Boolean);
  }

  function listMarkup(items, emptyText) {
    if (!items.length) return `<p class="muted">${escapeHtml(emptyText)}</p>`;
    return `<ul class="detail-list">${items.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`;
  }

  function openTask(task, opener) {
    if (!task) return;
    dialogOpener = opener || document.activeElement;
    $("#dialog-kicker").textContent = `${task.phase} · ${task.id} · ${task.priority}`;
    $("#dialog-title").textContent = task.title;
    const evidence = linkedEvidence(task);
    const risks = linkedRisks(task);
    $("#dialog-content").innerHTML = `
      <div class="dialog-status-row">${badge(task.status)}${priorityBadge(task.priority)}<span class="progress-chip">完成 ${task.progress}%</span></div>
      <dl class="detail-grid">
        <dt>任务内容</dt><dd>${escapeHtml(task.content)}</dd>
        <dt>工作目的</dt><dd>${escapeHtml(task.purpose)}</dd>
        <dt>预计时间</dt><dd>${escapeHtml(task.estimate)}</dd>
        <dt>开始时间（北京时间）</dt><dd>${escapeHtml(formatPanelTime(task.startedAt))}</dd>
        <dt>完成时间（北京时间）</dt><dd>${escapeHtml(formatPanelTime(task.finishedAt))}</dd>
        <dt>最近更新（北京时间）</dt><dd>${escapeHtml(formatPanelTime(task.updatedAt))}</dd>
        <dt>完成情况</dt><dd>${escapeHtml(task.detail)}</dd>
      </dl>
      <section class="dialog-section" aria-labelledby="criteria-${escapeHtml(task.id)}">
        <h3 id="criteria-${escapeHtml(task.id)}">验收标准</h3>
        ${listMarkup(task.acceptance, "暂无验收标准")}
      </section>
      <section class="dialog-section" aria-labelledby="evidence-${escapeHtml(task.id)}">
        <h3 id="evidence-${escapeHtml(task.id)}">关联证据</h3>
        ${evidence.length ? evidence.map((entry) => `
          <article class="detail-evidence">
            <strong>${escapeHtml(entry.id)} · ${escapeHtml(entry.title)}</strong>
            <span>${escapeHtml(formatPanelTime(entry.time))}</span>
            <p>${escapeHtml(entry.result)}</p>
            <code>${escapeHtml(entry.command)}</code>
          </article>`).join("") : "<p class=\"muted\">尚无验收证据</p>"}
      </section>
      <section class="dialog-section" aria-labelledby="risks-${escapeHtml(task.id)}">
        <h3 id="risks-${escapeHtml(task.id)}">关联风险</h3>
        ${risks.length ? risks.map((risk) => `
          <article class="detail-risk severity-${escapeHtml(risk.severity)}">
            <strong>${escapeHtml(risk.id)} · ${escapeHtml(risk.title)}</strong>
            <p>${escapeHtml(risk.description)}</p>
            <small>缓解：${escapeHtml(risk.mitigation)}</small>
          </article>`).join("") : "<p class=\"muted\">无已知关联风险</p>"}
      </section>`;
    const dialog = $("#task-dialog");
    if (typeof dialog.showModal === "function") dialog.showModal();
    else dialog.setAttribute("open", "");
    $("#dialog-close").focus();
  }

  function closeDialog() {
    const dialog = $("#task-dialog");
    if (typeof dialog.close === "function") dialog.close();
    else dialog.removeAttribute("open");
    if (dialogOpener && document.contains(dialogOpener)) dialogOpener.focus();
  }

  function renderTasks() {
    const tasks = filteredTasks();
    $("#visible-count").textContent = `显示 ${tasks.length} / ${data.tasks.length} 项`;
    $("#filter-summary").textContent = tasks.length === data.tasks.length
      ? "当前显示全部任务"
      : `筛选后剩余 ${tasks.length} 项`;
    const board = $("#task-board");
    board.innerHTML = tasks.length ? tasks.map((task) => `
      <button class="task-card" type="button" data-task="${escapeHtml(task.id)}"
              aria-label="查看 ${escapeHtml(task.id)} ${escapeHtml(task.title)} 的详情">
        <span class="task-code">${escapeHtml(task.id)}</span>
        <span class="task-name">
          <strong>${escapeHtml(task.title)}</strong>
          <small>${escapeHtml(task.phase)} · ${priorityBadge(task.priority)}</small>
        </span>
        <span class="task-purpose">
          ${escapeHtml(task.purpose)}
          <small>${escapeHtml(task.detail)}</small>
        </span>
        <span class="task-time">
          <small>预计 ${escapeHtml(task.estimate)}</small>
          <small>完成 ${escapeHtml(formatPanelTime(task.finishedAt))}</small>
        </span>
        <span class="task-state">${badge(task.status)}<small>${task.progress}%</small></span>
      </button>`).join("") : `
      <div class="empty" role="status">
        <strong>没有符合条件的任务</strong>
        <span>请调整关键词或筛选条件。</span>
      </div>`;
    $$("[data-task]", board).forEach((button) => {
      button.addEventListener("click", () => {
        openTask(data.tasks.find((task) => task.id === button.dataset.task), button);
      });
    });
  }

  function renderRisks() {
    const severityOrder = { "高": 0, "中": 1, "低": 2 };
    const risks = [...data.risks].sort((left, right) => (
      (severityOrder[left.severity] ?? 9) - (severityOrder[right.severity] ?? 9)
      || left.id.localeCompare(right.id)
    ));
    $("#risk-count").textContent = `${risks.length} 项`;
    $("#risk-list").innerHTML = risks.map((risk) => `
      <article class="risk severity-${escapeHtml(risk.severity)}">
        <div class="risk-heading">
          <strong>${escapeHtml(risk.id)} · ${escapeHtml(risk.title)}</strong>
          <span>${escapeHtml(risk.severity)} · ${escapeHtml(risk.status)}</span>
        </div>
        <p>${escapeHtml(risk.description)}</p>
        <small>缓解：${escapeHtml(risk.mitigation)}</small>
        <div class="task-links">${risk.taskIds.map((id) => `<button type="button" data-risk-task="${escapeHtml(id)}">${escapeHtml(id)}</button>`).join("")}</div>
      </article>`).join("");
    $$("[data-risk-task]").forEach((button) => {
      button.addEventListener("click", () => openTask(
        data.tasks.find((task) => task.id === button.dataset.riskTask), button
      ));
    });
  }

  function renderEvidence() {
    $("#evidence-count").textContent = `${data.evidence.length} 条`;
    $("#evidence-list").innerHTML = data.evidence.map((entry) => `
      <li>
        <time>${escapeHtml(formatPanelTime(entry.time))}</time>
        <strong>${escapeHtml(entry.id)} · ${escapeHtml(entry.title)}</strong>
        <p>${escapeHtml(entry.result)}</p>
        <code>${escapeHtml(entry.command)}</code>
        <div class="task-links">${entry.taskIds.map((id) => `<button type="button" data-evidence-task="${escapeHtml(id)}">${escapeHtml(id)}</button>`).join("")}</div>
      </li>`).join("");
    $$("[data-evidence-task]").forEach((button) => {
      button.addEventListener("click", () => openTask(
        data.tasks.find((task) => task.id === button.dataset.evidenceTask), button
      ));
    });
  }

  function appendOptions(select, entries) {
    select.insertAdjacentHTML("beforeend", entries.map(([value, label]) => (
      `<option value="${escapeHtml(value)}">${escapeHtml(label)}</option>`
    )).join(""));
  }

  function populateFilters() {
    $("#status-filter").innerHTML = '<option value="all">全部状态</option>';
    $("#phase-filter").innerHTML = '<option value="all">全部阶段</option>';
    $("#priority-filter").innerHTML = '<option value="all">全部优先级</option>';
    appendOptions($("#status-filter"), Object.keys(data.statuses).map((status) => [status, status]));
    const phases = [...new Set(data.tasks.map((task) => task.phase))].sort((a, b) => a.localeCompare(b, "zh-CN"));
    appendOptions($("#phase-filter"), phases.map((phase) => [phase, phase]));
    appendOptions($("#priority-filter"), Object.entries(data.priorities)
      .sort(([, left], [, right]) => left.order - right.order)
      .map(([priority, metadata]) => [priority, metadata.label]));
    if (state.status !== "all" && !data.statuses[state.status]) state.status = "all";
    if (state.phase !== "all" && !phases.includes(state.phase)) state.phase = "all";
    if (state.priority !== "all" && !data.priorities[state.priority]) state.priority = "all";
    $("#status-filter").value = state.status;
    $("#phase-filter").value = state.phase;
    $("#priority-filter").value = state.priority;
  }

  function renderAllTaskViews() {
    renderSummary();
    renderTasks();
  }

  function bindControls() {
    $("#search").addEventListener("input", (event) => {
      state.query = event.target.value;
      renderTasks();
    });
    [["status-filter", "status"], ["phase-filter", "phase"], ["priority-filter", "priority"], ["sort-order", "sort"]]
      .forEach(([id, key]) => {
        $(`#${id}`).addEventListener("change", (event) => {
          state[key] = event.target.value;
          renderAllTaskViews();
        });
      });
    $("#filters").addEventListener("reset", () => {
      window.setTimeout(() => {
        Object.assign(state, { query: "", status: "all", phase: "all", priority: "all", sort: "priority" });
        renderAllTaskViews();
        announce("筛选条件已重置");
      }, 0);
    });
    $("#dialog-close").addEventListener("click", closeDialog);
    $("#task-dialog").addEventListener("click", (event) => {
      if (event.target === $("#task-dialog")) closeDialog();
    });
    $("#task-dialog").addEventListener("close", () => {
      if (dialogOpener && document.contains(dialogOpener)) dialogOpener.focus();
    });
  }

  function storedTheme() {
    try { return window.localStorage.getItem("zcp-panel-theme"); }
    catch { return null; }
  }

  function storedRefreshInterval() {
    try {
      const value = Number(window.localStorage.getItem(refreshIntervalStorageKey));
      return [5, 15, 30, 60].includes(value) ? value : 30;
    } catch {
      return 30;
    }
  }

  function formatClock(date = new Date()) {
    return `${panelClockFormatter.format(date)}（北京时间）`;
  }

  function formatDuration(seconds) {
    if (!Number.isFinite(seconds) || seconds < 0) return "—";
    const rounded = Math.round(seconds);
    const hours = Math.floor(rounded / 3600);
    const minutes = Math.floor((rounded % 3600) / 60);
    if (hours > 0) return `${hours}小时${minutes}分`;
    return `${minutes}分${rounded % 60}秒`;
  }

  function formatLiveTimestamp(value) {
    const date = new Date(value);
    return Number.isFinite(date.getTime()) ? `${panelDateTimeFormatter.format(date)}（北京时间）` : "—";
  }

  function renderLiveStatus() {
    const content = $("#live-status-content");
    const freshness = $("#live-freshness");
    const warning = $("#live-warning");
    if (!liveData) {
      freshness.textContent = "实时源不可用";
      warning.className = "live-warning is-stale";
      warning.textContent = `${liveError}；静态看板不受影响。`;
      content.innerHTML = '<div class="live-empty">未取得 live.json，继续显示静态任务、风险与证据。</div>';
      return;
    }

    const generatedAt = new Date(liveData.generated_at);
    const ageSeconds = Number.isFinite(generatedAt.getTime()) ? Math.max(0, Math.round((Date.now() - generatedAt.getTime()) / 1000)) : Infinity;
    const staleAfter = Number(liveData.stale_after_seconds) || 45;
    const staleSeeds = liveData.autoformer?.seeds?.filter((seed) => seed.stale) || [];
    const isStale = ageSeconds > staleAfter;
    freshness.textContent = `更新 ${formatLiveTimestamp(liveData.generated_at)}`;
    warning.className = `live-warning${isStale || liveError ? " is-stale" : ""}`;
    const warnings = [...(liveData.warnings || [])];
    if (liveError) warnings.unshift(liveError);
    if (isStale) warnings.unshift(`live.json 已 ${ageSeconds} 秒未更新`);
    if (staleSeeds.length) warnings.push(`${staleSeeds.length} 个 seed 的 partial state 较旧`);
    warning.textContent = warnings.length ? `数据提示：${warnings.join("；")}` : "实时状态源正常。";

    const autoformer = liveData.autoformer || {};
    const seeds = Array.isArray(autoformer.seeds) ? autoformer.seeds : [];
    const targetUuids = new Set(autoformer.gpu_uuids || []);
    const gpus = (liveData.gpus || []).filter((gpu) => !targetUuids.size || targetUuids.has(gpu.uuid));
    const seedMarkup = seeds.length ? seeds.map((seed) => {
      const percent = Math.min(100, Math.max(0, seed.total ? seed.evaluations / seed.total * 100 : 0));
      const rate = Number.isFinite(seed.rate_per_second) ? `${seed.rate_per_second.toFixed(2)} eval/s` : "速率 —";
      return `<div class="live-seed">
        <div class="live-seed-line"><strong>seed ${escapeHtml(seed.seed)}</strong><span>${escapeHtml(seed.evaluations)}/${escapeHtml(seed.total)}</span></div>
        <div class="live-progress" aria-hidden="true"><span style="width:${percent.toFixed(2)}%"></span></div>
        <small>${escapeHtml(rate)} · ETA ${escapeHtml(formatDuration(seed.eta_seconds))} · ${escapeHtml(formatLiveTimestamp(seed.updated_at))}</small>
      </div>`;
    }).join("") : '<div class="live-empty">search-state.json 暂缺，等待下一次原子保存。</div>';
    const gpuMarkup = gpus.length ? gpus.map((gpu) => `
      <div class="live-gpu">
        <div class="live-gpu-line"><strong>GPU ${escapeHtml(gpu.index)}</strong><span>SM ${escapeHtml(gpu.utilization_percent)}%</span></div>
        <small>显存 ${escapeHtml(gpu.memory_used_mib)} / ${escapeHtml(gpu.memory_total_mib)} MiB</small>
      </div>`).join("") : '<div class="live-empty">GPU 采样暂不可用。</div>';

    content.innerHTML = `
      <article class="live-run">
        <div class="live-run-heading"><h3>DARTS ImageNet 六项</h3><span class="live-status status-${escapeHtml(liveData.darts?.status || "unknown")}">${escapeHtml(liveData.darts?.status || "unknown")}</span></div>
        <p class="live-run-meta">${escapeHtml(liveData.darts?.detail || "状态详情暂缺")}</p>
        <div class="live-facts"><span class="live-fact">状态更新时间 ${escapeHtml(formatLiveTimestamp(liveData.darts?.updated_at))}</span></div>
      </article>
      <article class="live-run">
        <div class="live-run-heading"><h3>AutoFormer AZ-NAS 3×8,000</h3><span class="live-status status-${escapeHtml(autoformer.status || "unknown")}">${escapeHtml(autoformer.status || "unknown")}</span></div>
        <p class="live-run-meta">总进度 ${escapeHtml(autoformer.evaluations_total || 0)}/${escapeHtml(autoformer.target_total || 24_000)}</p>
        <div class="live-seeds">${seedMarkup}</div>
        <div class="live-gpus">${gpuMarkup}</div>
      </article>`;
  }

  function setTheme(theme, persist = false) {
    document.documentElement.dataset.theme = theme;
    const dark = theme === "dark";
    $("#theme-toggle").setAttribute("aria-pressed", String(dark));
    $("#theme-label").textContent = dark ? "切换为浅色" : "切换为深色";
    if (persist) {
      try { window.localStorage.setItem("zcp-panel-theme", theme); }
      catch { /* Storage can be unavailable for local files. */ }
    }
  }

  function initializeTheme() {
    const preferred = window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
    setTheme(storedTheme() || preferred);
    $("#theme-toggle").addEventListener("click", () => {
      const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
      setTheme(next, true);
      announce(`已切换为${next === "dark" ? "深色" : "浅色"}主题`);
    });
  }

  function loadFreshDataScript() {
    return new Promise((resolve, reject) => {
      const script = document.createElement("script");
      const url = new URL("data.js", window.location.href);
      let settled = false;
      let timeoutId = null;
      const cleanup = () => {
        if (timeoutId !== null) window.clearTimeout(timeoutId);
        script.onload = null;
        script.onerror = null;
        script.remove();
      };
      const settle = (callback, value) => {
        if (settled) return;
        settled = true;
        cleanup();
        callback(value);
      };
      url.searchParams.set("refresh", `${Date.now()}-${++refreshRequestId}`);
      script.src = url.href;
      script.async = true;
      script.dataset.panelRefresh = "true";
      script.onload = () => {
        settle(resolve, window.ZCP_PANEL_DATA);
      };
      script.onerror = () => {
        settle(reject, new Error("无法取得最新 data.js"));
      };
      timeoutId = window.setTimeout(() => {
        settle(reject, new Error("加载 data.js 超时（10 秒）"));
      }, dataScriptTimeoutMs);
      document.head.append(script);
    });
  }

  async function loadFreshLiveStatus() {
    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => controller.abort(), liveFetchTimeoutMs);
    const url = new URL("live.json", window.location.href);
    url.searchParams.set("refresh", `${Date.now()}-${++refreshRequestId}`);
    try {
      const response = await fetch(url.href, { cache: "no-store", signal: controller.signal });
      if (!response.ok) throw new Error(`live.json HTTP ${response.status}`);
      const candidate = await response.json();
      if (!candidate || candidate.schema_version !== 1 || candidate.time_zone !== "Asia/Shanghai") {
        throw new Error("live.json schema 无效");
      }
      return candidate;
    } catch (error) {
      if (error.name === "AbortError") throw new Error("加载 live.json 超时");
      throw error;
    } finally {
      window.clearTimeout(timeoutId);
    }
  }

  function saveReloadState() {
    const reloadState = {
      state,
      scrollX: window.scrollX,
      scrollY: window.scrollY
    };
    try { window.sessionStorage.setItem(reloadStateStorageKey, JSON.stringify(reloadState)); }
    catch { /* Storage can be unavailable for local files. */ }
  }

  function restoreReloadState() {
    let reloadState = null;
    try {
      reloadState = JSON.parse(window.sessionStorage.getItem(reloadStateStorageKey));
      window.sessionStorage.removeItem(reloadStateStorageKey);
    } catch { /* Ignore unavailable or malformed session state. */ }
    if (!reloadState?.state) return;
    Object.assign(state, reloadState.state);
    $("#search").value = state.query;
    $("#status-filter").value = state.status;
    $("#phase-filter").value = state.phase;
    $("#priority-filter").value = state.priority;
    $("#sort-order").value = state.sort;
    window.setTimeout(() => window.scrollTo(reloadState.scrollX || 0, reloadState.scrollY || 0), 0);
  }

  function reloadPageWithCacheBusting() {
    saveReloadState();
    const url = new URL(window.location.href);
    url.searchParams.set("refresh", `${Date.now()}-${++refreshRequestId}`);
    window.location.replace(url.href);
  }

  function renderCurrentData() {
    rebuildIndexes();
    populateFilters();
    $("#project-status").textContent = `项目状态：${data.project.status}`;
    $("#project-purpose").textContent = data.project.purpose;
    $("#last-updated").textContent = `数据更新时间（北京时间）：${formatPanelTime(data.updatedAt)} · schema v${data.schemaVersion}`;
    renderRisks();
    renderEvidence();
    renderAllTaskViews();
    renderLiveStatus();
  }

  function updateRefreshCountdown() {
    const countdown = $("#refresh-countdown");
    if (!autoRefreshEnabled) {
      countdown.textContent = "自动刷新已暂停";
    } else if (document.visibilityState !== "visible") {
      countdown.textContent = "返回页面后立即检查";
    } else if (refreshPromise) {
      countdown.textContent = "正在检查";
    } else if (nextRefreshAt === null) {
      countdown.textContent = "即将检查";
    } else {
      const seconds = Math.max(0, Math.ceil((nextRefreshAt - Date.now()) / 1000));
      countdown.textContent = `${seconds} 秒后检查`;
    }
  }

  function cancelScheduledRefresh() {
    if (autoRefreshTimer !== null) window.clearTimeout(autoRefreshTimer);
    autoRefreshTimer = null;
    nextRefreshAt = null;
  }

  function scheduleAutoRefresh() {
    cancelScheduledRefresh();
    if (!autoRefreshEnabled || document.visibilityState !== "visible") {
      updateRefreshCountdown();
      return;
    }
    nextRefreshAt = Date.now() + refreshIntervalMs;
    autoRefreshTimer = window.setTimeout(() => {
      nextRefreshAt = null;
      updateRefreshCountdown();
      refreshData(false).finally(scheduleAutoRefresh);
    }, refreshIntervalMs);
    updateRefreshCountdown();
  }

  function setAutoRefresh(enabled, notify = false) {
    autoRefreshEnabled = enabled;
    const toggle = $("#auto-refresh-toggle");
    toggle.setAttribute("aria-pressed", String(enabled));
    $("#auto-refresh-label").textContent = `自动刷新：${enabled ? "开" : "关"}`;
    if (enabled) {
      $("#refresh-status").textContent = "自动刷新已开启";
      scheduleAutoRefresh();
    } else {
      cancelScheduledRefresh();
      $("#refresh-status").textContent = "自动刷新已暂停";
      updateRefreshCountdown();
    }
    if (notify) announce(`${enabled ? "已开启" : "已暂停"}自动刷新`);
  }

  function setRefreshInterval(seconds, persist = false) {
    const normalized = [5, 15, 30, 60].includes(Number(seconds)) ? Number(seconds) : 30;
    refreshIntervalMs = normalized * 1000;
    $("#refresh-interval").value = String(normalized);
    if (persist) {
      try { window.localStorage.setItem(refreshIntervalStorageKey, String(normalized)); }
      catch { /* Storage can be unavailable for local files. */ }
    }
    if (autoRefreshEnabled) scheduleAutoRefresh();
    else updateRefreshCountdown();
  }

  function refreshData(manual = false) {
    if (refreshPromise) {
      if (manual) announce("数据刷新正在进行中");
      return refreshPromise;
    }

    const button = $("#refresh-data");
    const status = $("#refresh-status");
    const previousData = data;
    const viewport = { left: window.scrollX, top: window.scrollY };
    nextRefreshAt = null;
    updateRefreshCountdown();

    refreshPromise = (async () => {
      button.disabled = true;
      button.classList.add("is-refreshing");
      status.dataset.state = "loading";
      status.setAttribute("aria-busy", "true");
      status.textContent = "正在检查更新…";
      try {
        const [dataResult, liveResult] = await Promise.allSettled([loadFreshDataScript(), loadFreshLiveStatus()]);
        if (liveResult.status === "fulfilled") {
          liveData = liveResult.value;
          liveError = "";
        } else {
          liveError = liveResult.reason?.message || "无法取得 live.json";
        }
        if (dataResult.status === "rejected") throw dataResult.reason;
        const candidate = validateData(dataResult.value);
        data = candidate;
        window.ZCP_PANEL_DATA = candidate;
        renderCurrentData();
        window.scrollTo(viewport.left, viewport.top);
        const changed = data.updatedAt !== previousData.updatedAt;
        status.dataset.state = "success";
        status.textContent = changed ? "已载入更新" : "已是最新";
        $("#refresh-success").textContent = `最后成功刷新：${formatClock()}`;
        if (manual || changed) announce(status.textContent);
      } catch (error) {
        if (window.location.protocol === "file:") {
          status.dataset.state = "loading";
          status.textContent = "局部刷新受限，正在重新载入页面…";
          announce(status.textContent);
          reloadPageWithCacheBusting();
          return;
        }
        data = previousData;
        window.ZCP_PANEL_DATA = previousData;
        renderCurrentData();
        window.scrollTo(viewport.left, viewport.top);
        status.dataset.state = "error";
        status.textContent = `刷新失败，继续显示上次数据：${error.message}；可点击“立即刷新”重试`;
        announce(status.textContent);
      } finally {
        $("#refresh-checked").textContent = `上次检查：${formatClock()}`;
        status.removeAttribute("aria-busy");
        button.classList.remove("is-refreshing");
        button.disabled = false;
        refreshPromise = null;
        updateRefreshCountdown();
      }
    })();

    return refreshPromise;
  }

  function initializeRefresh() {
    if (refreshInitialized) return;
    refreshInitialized = true;
    setRefreshInterval(storedRefreshInterval());
    $("#refresh-data").addEventListener("click", () => {
      refreshData(true).finally(scheduleAutoRefresh);
    });
    $("#auto-refresh-toggle").addEventListener("click", () => {
      setAutoRefresh(!autoRefreshEnabled, true);
    });
    $("#refresh-interval").addEventListener("change", (event) => {
      setRefreshInterval(event.target.value, true);
      announce(`自动刷新间隔已设为${event.target.selectedOptions[0].textContent}`);
    });
    document.addEventListener("visibilitychange", () => {
      if (document.visibilityState === "visible") {
        if (autoRefreshEnabled) refreshData(false).finally(scheduleAutoRefresh);
      } else {
        cancelScheduledRefresh();
        updateRefreshCountdown();
      }
    });
    if (refreshCountdownTimer === null) {
      refreshCountdownTimer = window.setInterval(updateRefreshCountdown, 1000);
    }
    const loadedAt = formatClock();
    $("#refresh-success").textContent = `最后成功刷新：页面载入 ${loadedAt}`;
    $("#refresh-checked").textContent = `上次检查：页面载入 ${loadedAt}`;
    const isFileProtocol = window.location.protocol === "file:";
    $("#refresh-file-hint").hidden = !isFileProtocol;
    if (isFileProtocol) {
      setAutoRefresh(true);
      $("#refresh-status").textContent = "file:// 模式：局部刷新失败时自动重载页面";
    } else {
      setAutoRefresh(true);
    }
    loadFreshLiveStatus().then((candidate) => {
      liveData = candidate;
      liveError = "";
      renderLiveStatus();
    }).catch((error) => {
      liveError = error.message;
      renderLiveStatus();
    });
  }

  $("#project-status").textContent = `项目状态：${data.project.status}`;
  $("#project-purpose").textContent = data.project.purpose;
  $("#last-updated").textContent = `数据更新时间（北京时间）：${formatPanelTime(data.updatedAt)} · schema v${data.schemaVersion}`;
  initializeTheme();
  populateFilters();
  bindControls();
  restoreReloadState();
  initializeRefresh();
  renderLiveStatus();
  renderRisks();
  renderEvidence();
  renderAllTaskViews();
})();
