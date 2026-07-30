# NB201 1% × 22 ZCP 单 seed 证据

## 结论

H1 当前只能判定为：**NB201 单 seed 完成，整体进行中**。本次对 NAS-Bench-201 v1.1 的
15,625 个架构做 1% 分层无放回抽样，seed 2026 得到 157 个架构；22 个代理共应产生且实际产生
3,454 个“架构 × 代理”键，其中 `ok=3,451`、`failed=3`、重复键为 0。机器可读摘要见
[`nb201_one_percent_22zcp_summary.json`](nb201_one_percent_22zcp_summary.json)。该文件只保留摘要、
SHA-256 和小型统计，不包含原始 score、checkpoint 或本机绝对路径。

## 抽样和运行协议

- sample manifest SHA-256：
  `9b9e7b0e8b7e59b76cee386cf6221bdac3f9b463a9a4729f68faffcd671391bc`
- 抽样：`proportional_feature_stratified`，population 15,625，210 个 strata，157 个样本，
  四个 shard 为 `40/39/39/39`。
- benchmark/space：`nasbench201@1.1` / `nb201_topology`；模型 fidelity 为
  `reference_topology_pytorch_port`。
- 真值：`cifar10-valid`、`valid-accuracy`、`valid`、200 epoch、repeat `mean`、方向 maximize。
- 输入：真实 dataset batch，batch size 2，32×32，10 类；四个 shard 共用输入指纹
  `5de04e6a61157306b00dc80b673de420ecd45a843b1564f78acaa8bbfab1ceaf`。
- 权重：`independent_scratch`；评估/input seed 均为 2026；代码 commit 为
  `308a91debcba8681766fd1ca76cea36ddc7fa7b3`。

| Shard | Run ID | 架构 | 行数 | ok | failed | `scores.jsonl` SHA-256 |
|---:|---|---:|---:|---:|---:|---|
| 0 | `f40abba1d7fb` | 40 | 880 | 880 | 0 | `bfd4fe7ce82357f15a54f3942078240200029fa8bf33437ddc341964c3d451e4` |
| 1 | `1724f6b53624` | 39 | 858 | 858 | 0 | `add72538eae75e7a8e3c8c615a835bf22daf348fcf724d95b96cff7bf3e63e0d` |
| 2 | `43960d0a061a` | 39 | 858 | 855 | 3 | `2a85ad301611b52a63a4222533608e5f31393841c9cbc276a2c7a3f8f65d90f6` |
| 3 | `d0950b062418` | 39 | 858 | 858 | 0 | `b0ff3476bbc1fb08d30b68362b6a6d882d1e6257962cd16a61b60d7a85a6dc64` |

## 完整失败键

三条失败都位于 benchmark index `3943`、architecture ID
`nb201_topology:839da408774c5a50b88c`、run `43960d0a061a`，代理分别为 `az_nas`、
`naswot`、`te_nas`。三条记录的完整协议键均为：

```text
nasbench201@1.1 / nb201_topology / cifar10-valid / valid-accuracy / valid /
epoch=200 / repeat=mean / evaluation_seed=2026 / benchmark_index=3943 /
architecture_id=nb201_topology:839da408774c5a50b88c / proxy_id=<上述代理>
```

状态均为 `failed`，`ValueError: proxy returned NaN or infinity`。失败未被删除、改写为 0 或替换为
其他代理结果，因此这三个代理的有效样本数为 156，其余代理为 157。

## 22 个代理的主组件相关性

下表严格使用成功记录声明的 `primary_component`，而非任取第一个 component。系数对应相同的
157 架构真值协议；`near` 和 `swap` 的输出为常数，三个系数不可辨识，记为“未定义”而不是 0。

| Proxy | 主组件 | n | Spearman | Kendall tau-b | Pearson |
|---|---|---:|---:|---:|---:|
| `az_nas` | `expressivity` | 156 | 0.606714 | 0.422333 | 0.658976 |
| `er` | `mean` | 157 | 0.132390 | 0.088818 | 0.343959 |
| `er_conn` | `score` | 157 | 0.214253 | 0.144864 | 0.407375 |
| `er_deg` | `score` | 157 | 0.230611 | 0.153771 | 0.402715 |
| `er_dist` | `score` | 157 | 0.236500 | 0.156459 | 0.405631 |
| `er_pr` | `score` | 157 | 0.330047 | 0.229789 | 0.480089 |
| `flops` | `score` | 157 | 0.619446 | 0.431723 | 0.396008 |
| `gradnorm` | `score` | 157 | 0.348334 | 0.258370 | 0.133073 |
| `jacob_cov` | `score` | 157 | 0.609405 | 0.425841 | 0.750554 |
| `meco` | `score` | 157 | 0.143196 | 0.106808 | 0.497817 |
| `meco_opt` | `score` | 157 | 0.143196 | 0.106808 | 0.497817 |
| `naswot` | `score` | 156 | 0.606714 | 0.422333 | 0.658976 |
| `near` | `score` | 157 | 未定义（常数） | 未定义（常数） | 未定义（常数） |
| `ntkt` | `score` | 157 | 0.540347 | 0.385759 | 0.508663 |
| `params` | `score` | 157 | 0.654632 | 0.479513 | 0.393627 |
| `swap` | `score` | 157 | 未定义（常数） | 未定义（常数） | 未定义（常数） |
| `synflow` | `score` | 157 | 0.341958 | 0.220154 | 0.023240 |
| `te_nas` | `synflow` | 156 | 0.329766 | 0.210587 | 0.017026 |
| `ter` | `mean` | 157 | 0.132390 | 0.088818 | 0.343959 |
| `vkdnw` | `score` | 157 | 0.655772 | 0.467908 | 0.718931 |
| `zen` | `score` | 157 | 0.616712 | 0.434918 | 0.790791 |
| `zico` | `score` | 157 | 0.471291 | 0.353095 | 0.419862 |

Params/FLOPs 已从原始验收 scores 重新计算，而非只翻转旧报告符号：accuracy 相关性使用原始资源值
`identity`，二者的 accuracy direction 均为 `maximize`；资源优化语义另存为
`resource_direction=minimize`。旧 artifact 的 `proxy_version=1` 与 `direction=minimize` 是历史 schema
把资源偏好误写入 accuracy 方向所致；当前 reader 只读派生迁移为 Params `count-v2`、FLOPs
`thop-v2`，不改写 raw JSONL。

## Topology 报告规模

修复 shard grouping 后，专属 topology 分析实际生成：157 条 architecture、942 条 edge、5 类
operation、6,720 条 feature correlation、840 条 operation effect、588 条 matched pair 和 504 条
matched-pair summary。旧报告曾把 `run_id` 错当科学协议，产生四组 n=39/40 的相关性并漏掉跨 shard
matched pair；现已改为按 evaluation seed 分组并合并四个互斥 shard。多组件分析会展开代理组件，
因此报告行数不能误解为重新执行了更多架构。

## 限制与后续审计

1. 本文件的 22 代理口径只有 seed 2026；核心 11 代理的 seed 2027/2028 后续已经完成，跨 seed
   稳定性见 [`NB201_CORE_THREE_SEED_CN.md`](NB201_CORE_THREE_SEED_CN.md)。
2. NB201 与 NATS-TSS 虽共享 topology codec，但 adapter、benchmark 身份和真值来源不同；本结果
   **不能外推或合并到 NATS-TSS**。
3. 旧报告错误地把 `params`/`flops` 的资源优化方向 `minimize` 用作 accuracy 相关性方向并执行
   `negated`。本文件已按原始资源值与 accuracy 做 `identity` 重算；资源偏好与“模型规模—精度”
   关联仍是两个问题，必须分别报告。
4. `az_nas` 与 `naswot`、`er` 与 `ter`、`meco` 与 `meco_opt` 等出现相同结果，不足以证明这些名称
   对应独立算法；需继续核验 provenance、公式、别名和实现版本。
5. 这是单 benchmark、单 dataset、单 budget、单 seed 的 1% 证据，不是论文数值复现，也不能代表
   其余 benchmark 或完整搜索/训练验收。

进一步的注册表与逐架构值审计确认：`ter → er`、`meco_opt → meco` 是显式 alias；`az_nas`
当前是 `portable_composite_approximation`，其主组件 `expressivity` 直接使用 NASWOT，因此两者在 156
个共同成功架构上逐值完全相同，不应算作两个独立公式证据。`near` 与 `swap` 在本协议上都只有一个
唯一值，二者相等只说明本次输入下均退化为常数，不能推出公式等价。22 个注册名按 artifact 中的
fidelity 分为 2 个 structural measure、2 个 alias、2 个 portable composite approximation、5 个
project extension 和 11 个尚未完成论文公式核验的 port；因此“22 个命令均执行”不等于“22 个论文
独立实现均已复现”。机器可读明细已写入摘要的 `provenance_audit`。

历史失败记录中的 `az_nas`、`te_nas` 仍按当时 schema 保存 `primary_component=score`；原始文件保持
只读。当前 evaluator 已改为在 failed/unsupported 路径保留代理声明的主组件，相关性派生表同时写出
失败覆盖率、ties、常数原因和方向变换，避免后续报告继续丢失该语义。
