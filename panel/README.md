# 代理忠实度审计看板

页面只展示一张“代理忠实度审计”卡片，数据来自 `panel/data.js`。不运行 GPU/历史 run 扫描器，
浏览器只按所选间隔重新加载这个小文件。

## 启动方式

保留原有纯静态 HTTP 启动方式。在项目根目录运行：

```bash
python -m http.server 8768 --bind 127.0.0.1 --directory panel
```

访问：

```text
http://127.0.0.1:8768/
```

也可以直接打开 `panel/index.html`。页面默认每 30 秒自动刷新，可切换刷新间隔、关闭自动刷新或点击“立即刷新”。在 `file://` 下若浏览器阻止动态加载，页面会通过带时间戳的整页重载完成刷新。

## 更新数据

编辑 `panel/data.js` 中的 `audit` 对象。时间统一使用北京时间，格式为 `YYYY-MM-DD HH:mm`。

提交前运行面板自身检查：

```bash
node --check panel/data.js
node --check panel/app.js
node --check panel/check-data.js
node panel/check-data.js
git diff --check -- panel
```
