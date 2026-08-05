# ZCP 代理口径、能力与研究使用说明

## 1. 注册 ID 不等于独立论文方法

`zcp-test proxy matrix` 当前展示 **24 个可调用 ID**。历史 benchmark sweep 使用其中 22 个，排除
搜索空间专用的 `az_nas_autoformer`、`az_nas_plainnet`。这 22 个名称不是 22 个相互独立、已经逐值
复现论文的 ZCP。

| fidelity | ID | 可主张范围 |
|---|---|---|
| `structural_measure` | `params`, `flops` | 结构资源指标；仍须注明 parameter/MAC 口径 |
| `alias` | `ter→er` | 兼容名称；不得作为独立方法重复计数 |
| `project_extension` | `er`, `er_pr`, `er_conn`, `er_deg`, `er_dist` | 项目公式/拓扑推广，不是外部论文复现 |
| `portable_composite_approximation` | `te_nas`, `az_nas` | 便携近似；不得宣称正式论文公式一致 |
| `paper_formula_port_unverified` | `gradnorm`, `synflow`, `naswot`, `jacob_cov`, `near`, `swap`, `zen`, `ntkt`, `zico` | 接口和有限值已验证，但缺固定上游逐值/排序 golden |
| `paper_formula_port_stabilized` | `meco`, `meco_opt`, `vkdnw`, `az_nas_autoformer`, `az_nas_plainnet` | 固定 commit 的公式 port；稳定化处理使其不声称逐位一致 |

报告应写“22 个注册名称完成 sweep”，不能写“复现 22 个独立论文 ZCP”。alias 必须折叠；常数输出
应报告 ties 与未定义相关性，不能用 0 替代。

## 2. 使用前检查

```bash
zcp-test proxy list
zcp-test proxy inspect zen
zcp-test proxy matrix
zcp-test proxy validate zen --device cpu
```

`validate` 证明返回结构、有限值、状态/RNG/hook 清理等工程契约，不证明论文公式一致。正式研究还要
核对 `implementation_fidelity`、`source`、`version`、`alias_of`、输入协议和 benchmark evidence。
能力矩阵现在分别公开 `requires_data`、`requires_inputs`、`requires_labels` 和
`requires_loss_fn`。evaluator 会在进入代理实现前检查 tensor、标签、损失函数和可选依赖；缺失时返回
结构化 `unsupported`，不会调用 `compute` 后再归类为普通失败。`requires_data=false` 只表示不依赖
真实数据分布，不表示不需要输入 tensor；例如 SynFlow 仍为 `requires_inputs=true`。

### MeCo 语义修复（2026-08-04）

官方实现固定为 [`HamsterMimi/MeCo@0d830dd`](https://github.com/HamsterMimi/MeCo/tree/0d830dd2f639f9d1ba3b5831a65df768d70fc93b)：
`meco` 对每个可用层取首个样本的 feature map，按通道构造相关矩阵，取最小实特征值并跨层求和；
`meco_opt` 是官方独立近似变体，会随机抽取 8 个通道并乘以 `channel_count / 8`，不是 `meco` alias。
项目 port 跳过不具备二维通道特征的 logits/标量输出、把 NaN/Inf 相关项置零，并始终清理 hook，故标为
`paper_formula_port_stabilized`。旧 `portable-v1` 实际计算跨 batch 拼接激活的 log-determinant，公式错误；
此前 evidence 中 `meco == meco_opt` 的相关性只能作为历史错误实现结果，不得引用为 MeCo 论文复现，需用
`hamstermimi-0d830dd-v2` 重新评估。

### VKDNW 语义修复（2026-08-05）

官方依据为 CVPR 2025 论文与作者仓库
[`ondratybl/VKDNW@d2ff276`](https://github.com/ondratybl/VKDNW/tree/d2ff276d37d8ba2e9f8c04beb71499d0bd346146)。
`vkdnw` 从首 128 个可训练参数张量各取一个权重，计算分类预测关于这些权重的 Jacobian，以平滑类别
概率协方差构造经验 Fisher，再取 Fisher 谱的 10%–90% 九个分位数。`entropy` 是这些分位数归一化
后的熵，`dimension` 记录可训练参数张量数；论文 Eq. 12 的正式单代理排名
`single = dimension + entropy` 是默认主分数。项目使用对称协方差特征分解代替上游手写 Cholesky，
并统一支持普通 logits/`(features, logits)` 输出，因此标为 `paper_formula_port_stabilized`。

旧 `portable-v1` 使用的是**输入 Jacobian**奇异值的对数组合，不包含参数 Fisher、类别概率协方差或
谱分位数熵，不能解释为 VKDNW。此前所有 `vkdnw` 相关性 evidence 仅保留为历史错误实现结果，必须用
`ondratybl-d2ff276-v2` 分 benchmark 重跑后才能引用为 VKDNW 论文复现。当前固定实现只声明 CNN；
Transformer 需独立上游协议与 golden 后再开放。

## 3. 相关性与对比规则

1. 按 canonical architecture ID join，不按列表位置对齐。
2. alias 只保留一个代表；项目扩展和 approximation 单独分组。
3. 默认使用 `ProxyOutput.score`；辅助组件必须显式指定 `--component`。
4. `params/flops` 的 accuracy direction 与 resource direction 分开记录。
5. 常数、NaN、Inf、failed、unsupported 和 ties 必须保留数量。
6. 搜索/聚合权重只由 validation 协议决定，test 仅进入最终报告。

## 4. 当前开放风险

- `te_nas` 和通用 `az_nas` 是组合近似；正式 AZ-NAS 搜索使用空间专用 ID。
- 9 个 `paper_formula_port_unverified` 仍需固定论文/官方 commit 和逐值 golden。
- `zen` 的 Proxyless 使用属于 project transfer，不能冒充官方 OFA controller 或直接论文协议。
- 输入契约字段与 fail-fast evaluator 已完成；剩余风险是论文公式 golden 与 alias/approximation 强制折叠。

## 5. 新增代理

```bash
zcp-test proxy scaffold my_proxy
zcp-test proxy validate my_proxy --device cpu
zcp-test proxy matrix
```

模板和安全边界见 [ADD_PROXY_CN.md](ADD_PROXY_CN.md)。新增代理必须声明主分数、组件、方向、模型族、
输入/标签/损失需求、来源和版本；无论文依据时必须标记为项目扩展。
