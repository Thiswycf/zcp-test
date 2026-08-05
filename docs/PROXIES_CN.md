# ZCP 代理口径、能力与研究使用说明

## 1. 当前注册 ID

当前 registry 共 23 个 ID：

`ac`、`az_nas`、`az_nas_autoformer`、`az_nas_plainnet`、`dss`、`er`、`flops`、`gradnorm`、`hc`、`hi`、`jacob_cov`、`meco`、`meco_opt`、`naswot`、`near`、`params`、`swap`、`synflow`、`te_nas`、`ter`、`vkdnw`、`zen`、`zico`。

已删除：`ntkt`、`er_pr`、`er_conn`、`er_deg`、`er_dist`。旧 artifact 可保留这些名称，但只能按已撤销旧版本的只读证据解释。

注册 ID 不等于独立论文方法。`params/flops` 是结构计量；空间专用端口、稳定化端口、跨域端口和第一方协议端口必须分别表述。

## 2. 官方来源边界

- CVPR 2026《Vision-Oriented Lightweight Neural Architecture Search with Budget-Adaptive Evaluation》及 [CVF 页面](https://openaccess.thecvf.com/content/CVPR2026/html/Fan_Vision-Oriented_Lightweight_Neural_Architecture_Search_with_Budget-Adaptive_Evaluation_CVPR_2026_paper.html) 存在。
- 官方 `tf-nas@58e3806` 是无许可证占位仓库，没有可执行实现；不得复制源码或声称代码复现。
- `ac/hi/hc` 使用 ACL 2023 官方 `training-free-nas@2d76e01` 的 ViT 跨域端口，不是 CVPR 2026 精确复现。
- `dss` 来自 CVPR 2022 `TF_TAS@42616bc`；`DSS++` 因无官方代码和精确公开协议而处于 **blocked**，不实现。
- NEAR 按可核验官方代码处理。

详细来源和忠实度见 [PROXY_OFFICIAL_AUDIT_CN.md](PROXY_OFFICIAL_AUDIT_CN.md)。

## 3. ER、TER 与 TE-NAS

- `er`：需要 `EdgeActivationBatch` 提供语义边 4-D 激活，不需要端点拓扑参与计分。
- `ter`：除 4-D 激活外，还需要唯一的有向 `source/target` 端点和拓扑。
- `te_nas`：按 `TER-Score@a646c5a` 计算 RN 减 NTK condition number，只返回一个标量；它不是已删除的 `ntkt`。

没有注册语义边 provider 的模型应返回 `unsupported`，不得退回 FX 模块节点或普通层输出近似。

## 4. 分数选择

评估、相关性和搜索必须显式记录分数语义：

```bash
zcp-test search --space autoformer --proxy az_nas_autoformer \
  --score-selector aggregate:az_nas_log_rank \
  --proxy-batches 1 --proxy-repetitions 1
```

- `primary`：主分数。
- `component:NAME`：组件消融；不能冒充正式综合分数。
- `aggregate:az_nas_log_rank`：AZ-NAS 正式多组件 cohort log-rank。
- `aggregate:mean_percentile_rank`：通用百分位聚合，与 AZ-NAS log-rank 含义不同。

`te_nas` 只能选择 `primary`。AZ-NAS 组件代理默认必须使用 `aggregate:az_nas_log_rank`；单组件搜索还需 `--allow-component-ablation`。

## 5. 重复批次协议

`--proxy-batches` 控制一次候选评估的 batch 数，`--proxy-repetitions` 控制重复初始化/测量次数。二者必须进入 artifact 身份，不能把不同设置的分数直接合并。默认值均为 1，不代表论文推荐值。

## 6. 正式可行性计划

先运行小 pilot，记录架构数和总秒数，再生成 10 分钟上限计划：

```bash
zcp-test acceptance plan-feasibility \
  --total-architectures 15625 \
  --pilot-architectures 10 \
  --pilot-seconds 42.0 \
  --max-seconds 600
```

规划器依次尝试 1%、1‰、1‱，必要时只计划 1 个架构。输出 `coverage_claim=false`，所以该流程只能称“自适应可行性 sweep”，不能称搜索空间覆盖或固定 1% 验收。

## 7. 结果与历史证据

1. 按 canonical architecture ID join，不按行位置对齐。
2. 常数、NaN、Inf、failed、unsupported、skipped 和 ties 均保留计数。
3. `params/flops` 的资源方向与 accuracy 方向分开。
4. validation 决定搜索或融合，test 只进入最终报告。
5. `docs/evidence/**` 的旧数值不改写；含旧 ID、旧公式或旧计数的文件是已撤销版本的只读证据，不证明当前 23 ID 覆盖。

## 8. 新增代理

```bash
zcp-test proxy scaffold my_proxy
zcp-test proxy validate my_proxy --device cpu
zcp-test proxy matrix
```

新增代理必须声明主分数、组件、方向、模型族、输入/标签/损失需求、来源、commit、许可证状态和版本；无可核验论文或官方代码时必须明确标为项目扩展。
