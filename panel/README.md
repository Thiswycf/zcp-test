# zcp-test 看板操作说明

## 推荐启动方式

在项目根目录运行纯静态 HTTP 服务：

```bash
python -m http.server 8768 --directory panel
```

然后访问：

```text
http://127.0.0.1:8768/
```

该方式不需要额外服务端依赖，并能稳定支持自动重新加载 `data.js`。

## 启动真实运行状态源

先生成一次 `panel/live.json`：

```bash
python panel/live-status.py --once
python -m json.tool panel/live.json >/dev/null
```

持续模式默认每 15 秒原子更新一次：

```bash
python panel/live-status.py --watch --interval 15
```

也可以用 user systemd transient service 托管：

```bash
systemd-run --user --unit=zcp-test-panel-live --collect \
  --property=Restart=on-failure --working-directory="$(pwd)" \
  python panel/live-status.py --watch --interval 15
```

生成器读取 `runs/acceptance` 下 DARTS、AutoFormer 和各 seed 的最新 search state，并在可用时采样 `nvidia-smi`。输出使用临时文件与 `os.replace` 原子替换，状态文件暂缺或写入中的尾部不完整不会终止 watch。`panel/live.json` 是本机运行态产物，已加入 `.gitignore`。

## 无需 F5 的刷新

- 页面默认每 30 秒自动检查一次，可切换为 5、15、30 或 60 秒。
- “立即刷新”会重新请求 `data.js`，并额外 fetch `live.json`；两个请求都包含唯一 cache-busting 参数。
- 自动刷新可以暂停；页面隐藏时也会暂停，重新可见后立即检查。
- 页面分别显示数据更新时间、最后成功刷新时间、上次检查时间、刷新状态和下次刷新倒计时。
- 刷新会保留当前筛选、搜索、排序和滚动位置。
- 加载失败时继续显示现有内容，可点击“立即刷新”重试。
- `live.json` 缺失或过期时，“实时运行”卡会显示降级/陈旧警告，但静态任务看板继续正常工作。

## `file://` 回退

直接打开 `panel/index.html` 时，页面仍会按所选间隔尝试用唯一查询参数动态加载 `data.js`。若浏览器的本地文件策略阻止动态脚本，页面会保存当前搜索、筛选、排序与滚动位置，并使用唯一查询参数重新载入整个页面；无需手动按 F5。页面会明确显示当前处于 `file://` 回退模式。

不同浏览器对本地文件缓存和脚本加载的实现并不一致，因此推荐使用上面的静态 HTTP 命令，以获得稳定的局部重渲染和明确的失败状态。该命令只使用 Python 标准库，不是看板的常驻服务依赖，也不要通过禁用浏览器安全策略来绕过限制。

刷新间隔保存在浏览器 `localStorage` 中；重新载入页面后继续使用最近选择的 5、15、30 或 60 秒。数据刷新只重建数据视图，保留当前搜索、状态/阶段/优先级筛选、排序和滚动位置。

## 更新看板数据

编辑 `panel/data.js` 后无需修改页面代码。已通过 HTTP 服务打开的页面会在下一次自动检查时取得新数据，也可点击“立即刷新”。

提交前运行：

```bash
node --check panel/data.js
node --check panel/app.js
node --check panel/check-data.js
node panel/check-data.js
python panel/live-status.py --once
python -m json.tool panel/live.json >/dev/null
git diff --check -- panel
```
