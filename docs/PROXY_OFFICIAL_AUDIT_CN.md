# ZCP 官方实现一致性审计（2026-08-05，北京时间）

## 已纠正的来源事实

CVPR 2026 论文《Vision-Oriented Lightweight Neural Architecture Search with Budget-Adaptive Evaluation》及 CVF Open Access 页面均存在：

- 论文：<https://openaccess.thecvf.com/content/CVPR2026/html/Fan_Vision-Oriented_Lightweight_Neural_Architecture_Search_with_Budget-Adaptive_Evaluation_CVPR_2026_paper.html>
- 官方仓库：<https://github.com/fanyi-plus/tf-nas>
- 核验 commit：`58e38062d617e242a7fe915a37ef6db3eeb90085`

但 `tf-nas@58e3806` 仍是占位仓库，未提供可执行官方实现，也未声明软件许可证。因此只能把它用于来源追踪，不能复制其源码，也不能据此声称已复现论文代理。

## 请求代理的解释边界

论文表 1 列出 `AC`、`HI`、`HC` 和 `DSS++`，但固定官方仓库没有提供这些指标的可执行实现：

- `ac`、`hi`、`hc` 按 ACL 2023 官方 `training-free-nas@2d76e01` 移植到 ViT。它们是跨域端口，**不是 CVPR 2026 精确复现**。
- `dss` 是 CVPR 2022 `TF_TAS@42616bc` 的 DSS 论文公式/代码协议端口，不是 `DSS++`。
- `dss_pp`/`DSS++` 因没有官方代码和足够精确的公开协议而处于 **blocked**，不实现；不得用 `dss` 冒充 `DSS++`。

## 当前注册表

当前注册 **23 个 ID**：

`ac`、`az_nas`、`az_nas_autoformer`、`az_nas_plainnet`、`dss`、`er`、`flops`、`gradnorm`、`hc`、`hi`、`jacob_cov`、`meco`、`meco_opt`、`naswot`、`near`、`params`、`swap`、`synflow`、`te_nas`、`ter`、`vkdnw`、`zen`、`zico`。

`ntkt`、`er_pr`、`er_conn`、`er_deg`、`er_dist` 已删除，新运行不得再请求这些 ID。

## 实现忠实度矩阵

| ID | 来源/判定 | 当前边界 |
|---|---|---|
| `gradnorm`, `synflow`, `naswot`, `jacob_cov`, `near`, `swap`, `zen`, `zico` | 固定来源公式端口 | 旧实现已被替换；NEAR 按可核验官方代码处理 |
| `meco`, `meco_opt`, `vkdnw` | 稳定化论文公式端口 | 数值稳定差异由版本标识保留 |
| `ac`, `hi`, `hc` | ACL 2023 official `2d76e01` 的 ViT 跨域端口 | 不是 CVPR 2026 精确复现 |
| `dss` | CVPR 2022 `TF_TAS@42616bc` | DSS 端口，不等于 DSS++ |
| `er` | `TER-Score@a646c5a` 第一方端口 | 只需要语义边的 4-D 激活 |
| `ter` | `TER-Score@a646c5a` 第一方端口 | 需要有向边端点/拓扑和对应 4-D 激活 |
| `te_nas` | `TER-Score@a646c5a` 第一方适配 | RN 减 NTK condition number 的单标量，只允许主分数选择器 |
| `az_nas` | AZ-NAS 搜索空间分派 | 正式多组件排序使用 log-rank，不使用单组件冒充总分 |
| `az_nas_autoformer`, `az_nas_plainnet` | AZ-NAS 空间专用稳定化端口 | 只在各自声明的搜索空间使用 |
| `params`, `flops` | 结构计量 | 不是论文 ZCP，资源方向与精度方向分开 |

## 分数与重复协议

- `--score-selector primary`：使用代理声明的主分数。
- `--score-selector component:NAME`：显式组件消融；AZ-NAS 搜索还必须显式允许组件消融。
- `--score-selector aggregate:az_nas_log_rank`：AZ-NAS 正式 cohort 排序。
- `--score-selector aggregate:mean_percentile_rank`：通用 cohort 百分位聚合，不能冒充 AZ-NAS 官方 log-rank。
- `--proxy-batches`：记录每次代理评估使用的 batch 数。
- `--proxy-repetitions`：记录重复初始化/重复测量次数。

`te_nas` 只暴露一个标量，必须使用 `--score-selector primary`。组件代理不能在未声明 selector 的情况下被静默压成单值。

## 自适应可行性与历史边界

正式 sweep 不再固定宣称覆盖 1%。先测 pilot，再运行：

```bash
zcp-test acceptance plan-feasibility \
  --total-architectures TOTAL \
  --pilot-architectures PILOT_N \
  --pilot-seconds PILOT_SECONDS \
  --max-seconds 600
```

规划器在 10 分钟上限内依次尝试 1%、1‰、1‱，必要时降为 1 个架构；输出始终为 `coverage_claim=false`。规划结果只证明时间可行性，不证明搜索空间覆盖、相关性稳定或论文复现。

2026-08-05 实测选择 NB201 1‱（2 架构、22/22 调用成功）和 AutoFormer 1%（5 架构、20/20
调用成功）。摘要见 `docs/evidence/adaptive_feasibility_20260805.json`；其中 2 点相关性只证明
canonical-ID join 可运行，不能解释代理质量。

`docs/evidence/**` 中旧 22-proxy、旧数量、旧测试统计和被替换公式的结果保持只读，不改写原数值；它们是**已撤销旧版本的审计证据**。当前正式报告必须排除已删除 ID 和被替换版本，不能把历史行数解释为当前 23 ID 的覆盖证明。
