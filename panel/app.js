(() => {
  "use strict";

  const allowedIntervals = new Set([5, 15, 30, 60]);
  const intervalStorageKey = "zcp-panel-refresh-interval";
  const elements = {
    percentage: document.getElementById("audit-percentage"),
    progress: document.getElementById("audit-progress"),
    progressBar: document.getElementById("audit-progress-bar"),
    phase: document.getElementById("audit-phase"),
    conclusion: document.getElementById("audit-conclusion"),
    eta: document.getElementById("audit-eta"),
    refreshButton: document.getElementById("refresh-data"),
    autoRefreshButton: document.getElementById("auto-refresh-toggle"),
    refreshInterval: document.getElementById("refresh-interval"),
    refreshStatus: document.getElementById("refresh-status")
  };

  let autoRefreshEnabled = true;
  let refreshTimer = null;
  let countdownTimer = null;
  let nextRefreshAt = null;
  let refreshing = false;

  function readStoredInterval() {
    try {
      const value = Number(window.localStorage.getItem(intervalStorageKey));
      return allowedIntervals.has(value) ? value : 30;
    } catch {
      return 30;
    }
  }

  function validateData(data) {
    if (!data || data.schemaVersion !== 3 || data.timeZone !== "Asia/Shanghai") {
      throw new Error("数据结构或时区无效");
    }
    const audit = data.audit;
    if (!audit || audit.title !== "代理忠实度审计") throw new Error("审计对象无效");
    if (!Number.isFinite(audit.percentage) || audit.percentage < 0 || audit.percentage > 100) {
      throw new Error("审计百分比无效");
    }
    for (const field of ["phase", "conclusion", "eta", "updatedAt"]) {
      if (typeof audit[field] !== "string" || !audit[field].trim()) throw new Error(`审计字段 ${field} 无效`);
    }
    return audit;
  }

  function render(data) {
    const audit = validateData(data);
    elements.percentage.textContent = `${audit.percentage}%`;
    elements.progress.setAttribute("aria-valuenow", String(audit.percentage));
    elements.progressBar.style.width = `${audit.percentage}%`;
    elements.phase.textContent = audit.phase;
    elements.conclusion.textContent = audit.conclusion;
    elements.eta.textContent = audit.eta;
  }

  function setRefreshStatus(message, isError = false) {
    elements.refreshStatus.textContent = message;
    elements.refreshStatus.classList.toggle("error", isError);
  }

  function updateCountdown() {
    if (!autoRefreshEnabled) {
      setRefreshStatus("自动刷新已关闭");
      return;
    }
    if (document.visibilityState === "hidden") {
      setRefreshStatus("页面隐藏，自动刷新已暂停");
      return;
    }
    if (nextRefreshAt === null) return;
    const remainingSeconds = Math.max(0, Math.ceil((nextRefreshAt - Date.now()) / 1000));
    setRefreshStatus(`${remainingSeconds} 秒后自动刷新`);
  }

  function clearSchedule() {
    if (refreshTimer !== null) window.clearTimeout(refreshTimer);
    if (countdownTimer !== null) window.clearInterval(countdownTimer);
    refreshTimer = null;
    countdownTimer = null;
    nextRefreshAt = null;
  }

  function scheduleRefresh() {
    clearSchedule();
    if (!autoRefreshEnabled || document.visibilityState === "hidden") {
      updateCountdown();
      return;
    }
    const intervalSeconds = Number(elements.refreshInterval.value);
    nextRefreshAt = Date.now() + intervalSeconds * 1000;
    updateCountdown();
    countdownTimer = window.setInterval(updateCountdown, 1000);
    refreshTimer = window.setTimeout(async () => {
      await refreshData();
      scheduleRefresh();
    }, intervalSeconds * 1000);
  }

  function reloadForFileProtocol() {
    const url = new URL(window.location.href);
    url.searchParams.set("refresh", String(Date.now()));
    window.location.replace(url.href);
  }

  function loadFreshData() {
    return new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = `data.js?refresh=${Date.now()}`;
      script.async = true;
      script.onload = () => {
        script.remove();
        try {
          resolve(validateData(window.ZCP_PANEL_DATA));
        } catch (error) {
          reject(error);
        }
      };
      script.onerror = () => {
        script.remove();
        reject(new Error("无法读取 data.js"));
      };
      document.head.appendChild(script);
    });
  }

  async function refreshData() {
    if (refreshing) return;
    refreshing = true;
    elements.refreshButton.disabled = true;
    setRefreshStatus("正在刷新…");
    try {
      await loadFreshData();
      render(window.ZCP_PANEL_DATA);
      setRefreshStatus("刷新成功");
    } catch (error) {
      if (window.location.protocol === "file:") {
        reloadForFileProtocol();
        return;
      }
      setRefreshStatus(`刷新失败：${error.message}`, true);
    } finally {
      refreshing = false;
      elements.refreshButton.disabled = false;
    }
  }

  function setAutoRefresh(enabled) {
    autoRefreshEnabled = enabled;
    elements.autoRefreshButton.setAttribute("aria-pressed", String(enabled));
    elements.autoRefreshButton.textContent = `自动刷新：${enabled ? "开" : "关"}`;
    scheduleRefresh();
  }

  const initialInterval = readStoredInterval();
  elements.refreshInterval.value = String(initialInterval);

  try {
    render(window.ZCP_PANEL_DATA);
  } catch (error) {
    setRefreshStatus(`载入失败：${error.message}`, true);
  }

  elements.refreshButton.addEventListener("click", async () => {
    await refreshData();
    scheduleRefresh();
  });
  elements.autoRefreshButton.addEventListener("click", () => setAutoRefresh(!autoRefreshEnabled));
  elements.refreshInterval.addEventListener("change", () => {
    const value = Number(elements.refreshInterval.value);
    if (!allowedIntervals.has(value)) return;
    try {
      window.localStorage.setItem(intervalStorageKey, String(value));
    } catch {}
    scheduleRefresh();
  });
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible" && autoRefreshEnabled) {
      refreshData().finally(scheduleRefresh);
    } else {
      scheduleRefresh();
    }
  });

  scheduleRefresh();
})();
