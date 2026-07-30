# NATS-TSS 1% × 22 ZCP 与核心三 seed 证据

## 判定

NATS-TSS v1.0 的既定 H1 协议已经完成：15,625 个 topology 候选按 operation 计数做
1% 比例分层无放回抽样，得到 157 个架构；全部 22 个注册代理完成 seed 2026，核心 11 个代理完成
seed 2026/2027/2028。H1 **仍为进行中**，因为 NATS-SSS、NB101、NB301、TNB101 和 ViT-Bench
尚未完成各自协议。

机器可读摘要见 [`nats_tss_one_percent_summary.json`](nats_tss_one_percent_summary.json)。原始
JSONL、运行 manifest、CSV、PNG、SVG 和 HTML 只保存在外部 audit 目录，不进入 Git。

## 独立 benchmark 协议

- adapter：`nats_bench.create(..., "tss")`，不是 NAS-Bench-201 API；
- benchmark/space：`nats_tss@1.0` / `nb201_topology`；
- 真值：`cifar10-valid / valid / accuracy / 200 epoch / repeat mean`；
- 模型：独立初始化的 `reference_topology_pytorch_port`，不读取 benchmark 已训练权重；
- 输入：真实 CIFAR-10 batch，batch size 2，32×32，seed 2026/2027/2028；
- sample manifest SHA-256：
  `c8280222f5d51a534124f2ed58f104ecb0d5593797481e7c3acc4a6338d18a5c`；
- 分层：210 strata，四片为 `40/39/39/39`，157 个 index 与 architecture ID 均唯一。

NATS-TSS manifest 与 NB201 manifest 使用相同 topology codec、seed 和分层规则，因此抽中的 index 和
architecture ID 相同；但二者的 `benchmark_id`、版本、API、manifest SHA 和真值来源不同。157 个共同
架构中有 31 个真值数值不同，最大绝对差约 1.02；两组真值的 Spearman 为 0.999243。高相关不等于
接口或标准答案相同，结果不得合表或互相替代。

## 22 代理单 seed

seed 2026 共 `157 × 22 = 3,454` 行，其中 `ok=3,451`、`failed=3`、重复稳定键为 0。三个失败都位于
benchmark index 3943、architecture `nb201_topology:839da408774c5a50b88c`，代理为 `az_nas`、
`naswot` 和 `te_nas`，错误均为非有限输出；失败没有被删除或替换为 0。

四个 run 及 `scores.jsonl` SHA-256：

| Shard | Run ID | 行数 | SHA-256 |
|---:|---|---:|---|
| 0 | `c076cb82e8ad` | 880 | `8bcb7ffa86023c90292a930e4261c2e2bcae3f88bcb02fd1f4f2eb70bcd946f9` |
| 1 | `2ab84a130287` | 858 | `fae095a371ec9009c89068c5745541a45de99a73deceaab01e53616aa6506398` |
| 2 | `d9e55fc868a0` | 858 | `226e075bb57851dcbf8855a7c7c848837f35accc31f02481c7d2cc2c03aaa4b4` |
| 3 | `5a2600186842` | 858 | `4c85fb70d6d0c19076c262c15bab40dfaf144b6f43b4ce81a6063baae217e90a` |

所有 run manifest 均记录 `CUDA_DEVICE_ORDER=PCI_BUS_ID`、物理 GPU UUID、PCI bus ID、4090/4090D
型号和内部 `cuda:0`。本次先用 `gpu list` 审计空闲卡，再显式指定 UUID；未使用易混淆的物理数字索引。

## 主组件相关性

下表只使用代理声明的 `primary_component`。Params/FLOPs 已从原始验收 scores 按资源原值与
accuracy 使用 `identity` 重新计算；accuracy direction 为 `maximize`，资源优化方向另记为
`resource_direction=minimize`。

| Proxy | 主组件 | n | Spearman | Kendall tau-b | Pearson |
|---|---|---:|---:|---:|---:|
| `az_nas` | `expressivity` | 156 | 0.611133 | 0.429777 | 0.656976 |
| `er` | `mean` | 157 | 0.047147 | 0.037431 | 0.308555 |
| `er_conn` | `score` | 157 | 0.164247 | 0.107300 | 0.393665 |
| `er_deg` | `score` | 157 | 0.170601 | 0.112363 | 0.388740 |
| `er_dist` | `score` | 157 | 0.148230 | 0.094725 | 0.394706 |
| `er_pr` | `score` | 157 | 0.229892 | 0.153193 | 0.457067 |
| `flops` | `score` | 157 | 0.619762 | 0.432384 | 0.394601 |
| `gradnorm` | `score` | 157 | 0.380733 | 0.276825 | 0.144184 |
| `jacob_cov` | `score` | 157 | 0.583101 | 0.400065 | 0.751349 |
| `meco` | `score` | 157 | 0.161344 | 0.119769 | 0.498241 |
| `meco_opt` | `score` | 157 | 0.161344 | 0.119769 | 0.498241 |
| `naswot` | `score` | 156 | 0.611133 | 0.429777 | 0.656976 |
| `near` | `score` | 157 | 未定义（常数） | 未定义（常数） | 未定义（常数） |
| `ntkt` | `score` | 157 | 0.553920 | 0.402580 | 0.502317 |
| `params` | `score` | 157 | 0.654763 | 0.479855 | 0.392208 |
| `swap` | `score` | 157 | 未定义（常数） | 未定义（常数） | 未定义（常数） |
| `synflow` | `score` | 157 | 0.315181 | 0.197126 | 0.023232 |
| `te_nas` | `synflow` | 156 | 0.302002 | 0.186931 | 0.017142 |
| `ter` | `mean` | 157 | 0.047147 | 0.037431 | 0.308555 |
| `vkdnw` | `score` | 157 | 0.668532 | 0.479177 | 0.715007 |
| `zen` | `score` | 157 | 0.635895 | 0.460069 | 0.814596 |
| `zico` | `score` | 157 | 0.463780 | 0.347705 | 0.423417 |

旧 artifact 的 Params/FLOPs `proxy_version=1`、`direction=minimize` 来自历史资源方向错误。当前
reader 只读派生为 Params `count-v2`、FLOPs `thop-v2`，并分离 accuracy 与资源方向；raw JSONL
未改写。

## 核心代理三 seed

核心 11 代理三 seed 共 5,181 行、5,172 成功、9 失败、0 重复键；合并 score SHA-256 为
`9efbe925701b34490b0904ef01ee6f0d50625a489044de78f73fbac2cf6101e9`。每个 seed 的相同三个
代理在同一架构上失败，覆盖率为 156/157。

| Proxy | Spearman 均值 ± 总体标准差 | 跨 seed score Spearman 均值 |
|---|---:|---:|
| `az_nas` | 0.635949 ± 0.034805 | 0.955145 |
| `flops` | 0.619762 ± 0.000000 | 1.000000 |
| `gradnorm` | 0.381165 ± 0.008713 | 0.959639 |
| `jacob_cov` | 0.520780 ± 0.065977 | 0.459861 |
| `meco` | 0.145095 ± 0.051162 | 0.486406 |
| `naswot` | 0.635949 ± 0.034805 | 0.955145 |
| `params` | 0.654763 ± 0.000000 | 1.000000 |
| `synflow` | 0.322403 ± 0.010651 | 0.900543 |
| `te_nas` | 0.309202 ± 0.010859 | 0.898676 |
| `zen` | 0.635519 ± 0.004976 | 0.953469 |
| `zico` | 0.467700 ± 0.007981 | 0.978045 |

## 专属研究与未通过项

修复 shard grouping 后，Topology 报告生成 157 条 architecture、942 条 edge、5 类 operation、
6,720 条 feature correlation、840 条 operation effect、588 条 matched pair 和 504 条
matched-pair summary。旧报告曾把 `run_id` 错当科学协议，现已改为合并互斥 shard、按 evaluation
seed 分组；因此正式相关性使用完整 n=157，并包含跨 shard matched pair。
通用 correlation、proxy–proxy、top-k、rank、scatter、静态 HTML 和 report bundle 均已生成。

本轮真实运行还暴露三项不能掩盖的缺口：

1. NATS adapter 的 `seed_reduction=min|max` 过去会静默落到官方 mean。本轮已修复为枚举官方 seeds
   并明确归约；真实 index-0 验证得到 mean `81.982667`、min `81.616000`、max `82.240000`，
   不支持枚举时会明确失败。当前 H1 使用的 mean 数值未受该旧缺陷影响。
2. catalog 资产版本为 `1.0-3ffb9`，而 benchmark protocol 只记录 `1.0`；后续应另存资产修订标识，
   不能把下载资产 revision 与 benchmark API version 混成一个字段。
3. topology 专属表主要分析成功记录；正式页面必须同时链接通用报告中的 failed/coverage 诊断，
   后续再增加专属表覆盖率字段。

因此，本证据证明 NATS-TSS 既定相关性协议已执行，并证明 repeat reduction 缺陷已经修复；它不证明
全部代理都是论文公式复现，也不证明剩余两个实现缺口已经完成。
