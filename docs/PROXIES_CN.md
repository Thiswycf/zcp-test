# ZCP 代理口径、能力与研究使用说明

## 1. 注册 ID 不等于独立论文方法

`zcp-test proxy matrix` 当前展示 **24 个可调用 ID**。历史 benchmark sweep 使用其中 22 个，排除
搜索空间专用的 `az_nas_autoformer`、`az_nas_plainnet`。这 22 个名称不是 22 个相互独立、已经逐值
复现论文的 ZCP。

| fidelity | ID | 可主张范围 |
|---|---|---|
| `structural_measure` | `params`, `flops` | 结构资源指标；仍须注明 parameter/MAC 口径 |
| `alias` | `ter→er`, `meco_opt→meco` | 兼容名称；不得作为独立方法重复计数 |
| `project_extension` | `er`, `er_pr`, `er_conn`, `er_deg`, `er_dist` | 项目公式/拓扑推广，不是外部论文复现 |
| `portable_composite_approximation` | `te_nas`, `az_nas` | 便携近似；不得宣称正式论文公式一致 |
| `paper_formula_port_unverified` | `gradnorm`, `synflow`, `naswot`, `meco`, `jacob_cov`, `vkdnw`, `near`, `swap`, `zen`, `ntkt`, `zico` | 接口和有限值已验证，但缺固定上游逐值/排序 golden |
| `paper_formula_port_stabilized` | `az_nas_autoformer`, `az_nas_plainnet` | 固定 commit 的专用 port；稳定化处理使其不声称逐位一致 |

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

## 3. 相关性与对比规则

1. 按 canonical architecture ID join，不按列表位置对齐。
2. alias 只保留一个代表；项目扩展和 approximation 单独分组。
3. 默认使用 `ProxyOutput.score`；辅助组件必须显式指定 `--component`。
4. `params/flops` 的 accuracy direction 与 resource direction 分开记录。
5. 常数、NaN、Inf、failed、unsupported 和 ties 必须保留数量。
6. 搜索/聚合权重只由 validation 协议决定，test 仅进入最终报告。

## 4. 当前开放风险

- `te_nas` 和通用 `az_nas` 是组合近似；正式 AZ-NAS 搜索使用空间专用 ID。
- 11 个 `paper_formula_port_unverified` 仍需固定论文/官方 commit 和逐值 golden。
- `zen` 的 Proxyless 使用属于 project transfer，不能冒充官方 OFA controller 或直接论文协议。
- capability 尚不能完整表达 `loss_fn` 等输入契约；统一字段和 evaluator 检查完成前保持开放风险。

## 5. 新增代理

```bash
zcp-test proxy scaffold my_proxy
zcp-test proxy validate my_proxy --device cpu
zcp-test proxy matrix
```

模板和安全边界见 [ADD_PROXY_CN.md](ADD_PROXY_CN.md)。新增代理必须声明主分数、组件、方向、模型族、
输入/标签/损失需求、来源和版本；无论文依据时必须标记为项目扩展。
