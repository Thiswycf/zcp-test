# NATS-SSS 1% × 22 ZCP 与核心三 seed 证据

## 判定

NATS-SSS v1.0 的 CIFAR-10-valid/90-epoch 既定 H1 协议已经完成：32,768 个 size 候选按总通道区间、
唯一宽度数和单调性做 1% 比例分层抽样，得到 328 个架构；全部 22 个注册代理完成 seed 2026，
核心 11 个代理完成 seed 2026/2027/2028。H1 整体仍进行中，NB101、NB301、TNB101 和 ViT-Bench
尚未执行；NATS-SSS 的 CIFAR-100/ImageNet16-120 跨数据集 rank transfer 也仍是扩展验收项。

机器摘要见 [`nats_sss_one_percent_summary.json`](nats_sss_one_percent_summary.json)。原始 JSONL、
manifest、CSV、PNG、SVG 和 HTML 只位于外部 audit 目录。

## 协议与抽样

- adapter：`nats_bench.create(..., "sss")`；
- benchmark/space：`nats_sss@1.0` / `nats_size`；
- 真值：`cifar10-valid / valid / accuracy / 90 epoch / repeat mean`；
- 模型：真实逐 stage channel 的 `reference_topology_pytorch_port`；
- 输入：真实 CIFAR-10，batch size 2，32×32；
- manifest SHA-256：`07767985afbad7d498acf062620aea5ef7b66b2bfc8b3db3fea3a0fc768e1992`；
- population 32,768，sample 328，66 strata，四个互斥 shard 均为 82；index 与 architecture ID
  均无重复。

index 0 的独立 `query → build_model → params` CPU smoke 已通过，90-epoch CIFAR-10-valid mean
accuracy 为 `76.88799998779297`。这证明 SSS 使用 `nats_size` 与逐 stage channel，而不是 TSS topology
或 `TinyConvNet`。

## 22 代理单 seed

seed 2026 共 `328 × 22 = 7,216` 行，全部 `ok`，failed/unsupported/重复稳定键均为 0。

| Shard | Run ID | 行数 | SHA-256 |
|---:|---|---:|---|
| 0 | `d988cd7df0cf` | 1,804 | `f7f3152874710817ef4405a622f715f8e4ca06cc7e452c3e56fafd6e4388b896` |
| 1 | `e7dc84a32893` | 1,804 | `b92875dd69ad7269fbcf309d68fa72bf6b7eb32009796a84528b0a81a00857d9` |
| 2 | `b56aaf841ad5` | 1,804 | `077c9f1d33b446e6b7c2dec71b49655b31c64e51921550fa677c6911e32889a0` |
| 3 | `dffe62ce66be` | 1,804 | `d32bac44a516af22d15e048a9d29821f55ace543dad4f28a02f1bf52dc46e4be` |

所有 run 使用 GPU UUID，manifest 记录 PCI bus ID、4090/4090D 型号、
`CUDA_DEVICE_ORDER=PCI_BUS_ID` 和内部 `cuda:0`。

## 主组件相关性

| Proxy | 主组件 | n | Spearman | Kendall tau-b | Pearson |
|---|---|---:|---:|---:|---:|
| `az_nas` | `expressivity` | 328 | 0.566985 | 0.394712 | 0.551463 |
| `er` | `mean` | 328 | 0.105231 | 0.070725 | 0.109821 |
| `er_conn` | `score` | 328 | 0.273052 | 0.183273 | 0.270898 |
| `er_deg` | `score` | 328 | 0.277873 | 0.186276 | 0.273702 |
| `er_dist` | `score` | 328 | 0.280863 | 0.186290 | 0.271076 |
| `er_pr` | `score` | 328 | 0.170850 | 0.113806 | 0.168663 |
| `flops` | `score` | 328 | 0.579883 | 0.411009 | 0.535706 |
| `gradnorm` | `score` | 328 | 0.457247 | 0.318331 | 0.415618 |
| `jacob_cov` | `score` | 328 | 0.068540 | 0.044675 | 0.099318 |
| `meco` | `score` | 328 | 0.005812 | 0.004489 | 0.018920 |
| `meco_opt` | `score` | 328 | 0.005812 | 0.004489 | 0.018920 |
| `naswot` | `score` | 328 | 0.566985 | 0.394712 | 0.551463 |
| `near` | `score` | 328 | 未定义（常数） | 未定义（常数） | 未定义（常数） |
| `ntkt` | `score` | 328 | 0.473246 | 0.330900 | 0.468356 |
| `params` | `score` | 328 | 0.876103 | 0.696503 | 0.848040 |
| `swap` | `score` | 328 | 未定义（常数） | 未定义（常数） | 未定义（常数） |
| `synflow` | `score` | 328 | 0.381016 | 0.259256 | 0.398400 |
| `te_nas` | `synflow` | 328 | 0.381064 | 0.259293 | 0.398401 |
| `ter` | `mean` | 328 | 0.105231 | 0.070725 | 0.109821 |
| `vkdnw` | `score` | 328 | 0.017929 | 0.012065 | 0.034064 |
| `zen` | `score` | 328 | 0.014240 | 0.011543 | -0.042985 |
| `zico` | `score` | 328 | 0.074003 | 0.053798 | 0.096289 |

Params/FLOPs 已从原始验收 scores 按资源原值与 accuracy 使用 `identity` 重新计算；accuracy
direction 为 `maximize`，资源优化语义另记为 `resource_direction=minimize`。旧 artifact 的
`proxy_version=1`、`direction=minimize` 来自历史资源方向错误；当前 reader 只读派生为 Params
`count-v2`、FLOPs `thop-v2`，raw JSONL 未改写。常数代理记为不可辨识，不写成 0。

## 核心三 seed

核心 11 代理三 seed 共 10,824 行，全部成功且无重复键；合并 SHA-256 为
`81622da200341d5d025086e5a8da849ae1e12b5b66d9cf78f8a0e12badf47d4d`。

| Proxy | Spearman 均值 ± 总体标准差 | 跨 seed score Spearman 均值 |
|---|---:|---:|
| `az_nas` | 0.567858 ± 0.001633 | 0.999635 |
| `flops` | 0.579883 ± 0.000000 | 1.000000 |
| `gradnorm` | 0.483399 ± 0.026219 | 0.515150 |
| `jacob_cov` | 0.018584 ± 0.077330 | 0.011972 |
| `meco` | 0.044842 ± 0.038670 | 0.080990 |
| `naswot` | 0.567858 ± 0.001633 | 0.999635 |
| `params` | 0.876103 ± 0.000000 | 1.000000 |
| `synflow` | 0.385096 ± 0.003083 | 0.981771 |
| `te_nas` | 0.385124 ± 0.003065 | 0.981771 |
| `zen` | -0.005651 ± 0.014077 | 0.016798 |
| `zico` | 0.131568 ± 0.042709 | 0.519380 |

这些稳定性只适用于当前 SSS/CIFAR-10-valid/input 协议。例如 `jacob_cov`、`meco`、`zen` 的跨 seed
稳定性很低，不得外推为其他 benchmark 的固有结论。

## Size 专属研究与修复

修复 shard grouping 后，size 报告生成 328 条 architecture、1,640 条 stage、12 条描述统计、
3,528 条 feature correlation、840 条 stage sensitivity、672 条 size-controlled correlation 和
112 条 size strata。正式相关性使用完整 n=328。

真实报告最初暴露：`run_id` 被错误纳入专属研究的 protocol grouping，导致四个 shard 分别计算
n=82 的相关性。现已删除 `run_id/source_run` 分组、加入 evaluation `seed` 分组，并用测试证明
“互斥 shard 合并、不同 seed 分离”。NB201 与 NATS-TSS 的既有 topology 报告也已按相同修复重建。

## 仍待完成

1. CIFAR-100 与 ImageNet16-120 的真实输入、独立目标和跨数据集 rank transfer 尚未执行；当前只能
   判定 CIFAR-10-valid/90-epoch 协议完成。
2. catalog 的 SSS 资产 revision `1.0-50262` 尚未作为独立于 API version 的 score 字段保存。
3. 22 个注册名不等于 22 个独立论文公式；alias、portable approximation 和 unverified port 的边界
   继续沿用代理 provenance 审计。
