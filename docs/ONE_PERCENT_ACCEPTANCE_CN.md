# 1% Benchmark 相关性验收

本协议只用于有可查询标准答案或明确 surrogate 的 benchmark。DARTS、MobileNetV2 和开放
AutoFormer 搜索空间没有完整 tabular 真值，不运行这种相关性验收；它们使用 validation-only 搜索和
候选完整训练。NAS-Bench-301 的结论必须写成 surrogate association。

## 1. 生成确定性分层清单

`benchmark sample` 遍历 benchmark 的 canonical architecture，按 benchmark 特征建立 strata，按
总体占比使用 largest-remainder 分配样本，再在每层内使用固定 seed 无放回抽样。manifest 保存
benchmark/version、总体规模、sample fraction、architecture ID、benchmark index、stratum 和 shard。

```bash
zcp-test benchmark sample nasbench201 --trusted \
  --catalog ~/.config/zcp-test/data.json \
  --fraction 0.01 --seed 2026 --shards 4 \
  --output /path/to/audit/samples/nasbench201-seed2026.json
```

当前特征层定义：

| benchmark | 分层特征 | 说明 |
|---|---|---|
| NAS-Bench-101 | vertex、edge 与三类中间 operation 数量 | 不按 module hash 或文件位置分层 |
| NAS-Bench-201 / NATS-TSS | 六条边上的 operation 计数 | 两者可用相同 codec，但必须分别生成 manifest |
| NATS-SSS | 总通道 bin、不同宽度数、是否非递减 | 用于避免只抽到相邻 size 编码 |
| NAS-Bench-301 | normal/reduction 的 skip、pool、sep、dil 与重复 parent 数 | 结果仍是 surrogate association |
| TransNAS-Bench-101 | base width 与编码 digit 计数 | micro/macro 分别生成 |
| ViT-Bench-101 | depth 与 hidden/base dimension | AutoFormer main/ext/PiT 分切片生成 |

这是**比例分层**，不是每层等量抽样；极稀有层在小样本时可能分配为 0。manifest 是抽样真源，后续
不得重新按 seed 临时抽样。

## 2. 最低规模

| 协议 | manifest 参数 | 最低样本 |
|---|---|---:|
| NB101 full | `--fraction 0.01` | 4,237 |
| NB201 v1.1 | `--fraction 0.01` | 157 |
| NATS-TSS | `--fraction 0.01` | 157 |
| NATS-SSS | `--fraction 0.01` | 328 |
| NB301 deterministic | `--count 1000` | 1,000 |
| TNB101 micro | `--fraction 0.01` | 41 |
| TNB101 macro | `--fraction 0.01` | 33 |
| ViT 每个正式切片 | `--count 5` | 5 |

NB101 的 4/12/36/108 budget、TNB101 的各任务、ViT 的 vanilla/KD/inherited 以及不同 dataset/split
必须分别报告；禁止对协议直接平均。AutoFormer extension 不能并入 main。

## 3. 四卡分片执行

每个 worker 使用 manifest 中一个 shard。`CUDA_VISIBLE_DEVICES` 必须是 GPU UUID；程序内部使用
`cuda:0`。以下命令以 NB201 为例：

```bash
SAMPLE=/path/to/audit/samples/nasbench201-seed2026.json
GPU_UUIDS=(GPU-UUID-0 GPU-UUID-1 GPU-UUID-2 GPU-UUID-3)
for SHARD in 0 1 2 3; do
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${GPU_UUIDS[$SHARD]}" \
  zcp-test evaluate --benchmark nasbench201 --trusted \
    --catalog ~/.config/zcp-test/data.json \
    --sample-manifest "$SAMPLE" --sample-shard "$SHARD" \
    --dataset cifar10-valid --target-metric valid-accuracy --target-split valid \
    --epoch-budget 200 --metric-seed-reduction mean \
    --proxies az_nas,er,er_conn,er_deg,er_dist,er_pr,flops,gradnorm,jacob_cov,meco,meco_opt,naswot,near,ntkt,params,swap,synflow,te_nas,ter,vkdnw,zen,zico \
    --input-source dataset --data-root /path/to/cifar10 \
    --batch-size 2 --input-size 32 --classes 10 --device cuda:0 \
    --output /path/to/audit/nb201/shard-$SHARD &
done
wait
```

`evaluate` 会校验 manifest benchmark/version、索引唯一性及每个索引对应的 architecture ID。run 的
`config.yaml` 记录 strategy、seed、population、当前 shard 样本数、manifest SHA-256 和 shard index。
`--start` 不能与 `--sample-manifest` 同用。

## 4. Seed 协议

- 全部 22 ZCP：一个固定初始化/input seed；
- 核心代理 `params,flops,gradnorm,jacob_cov,naswot,synflow,zen,zico,meco,te_nas,az_nas`：至多三个 seed；
- 所有 seed 必须复用同一 sample manifest；只改变模型初始化/输入 batch seed；
- 同一 benchmark 的不同 seed run 可交给 `analyze sensitivity --parameter seed`，但不能把 seed 行当作额外架构扩大样本数。

## 5. 完成判定

每个协议必须保留：sample manifest 及 SHA-256、每 shard manifest/config/scores、成功/failed/NaN 数、
输入指纹、每代理有效样本数、相关性与 bootstrap CI、top-k、耗时/显存，以及稳定任务键去重后的总行数。
任何 OOM、NaN 或 unsupported 都保留原状态；不得回退随机输入、近似模型或伪造分数。

本机低成本 smoke 已对 NB201 生成 16/15,625 的 0.1% manifest：210 个 strata、4 个互斥 shard，
每 shard 4 个索引。shard 0 的 params evaluate 完成 4/4 行，run 记录 manifest SHA-256。该 smoke 只
证明抽样和分片协议，不计入 1% 科学验收。

## 6. NB201 单 seed 实际验收（H1）

H1 当前状态为：**NB201 单 seed 完成，整体进行中**。seed 2026 的正式 1% manifest 从 15,625
个架构按 210 个 feature strata 抽取 157 个架构，SHA-256 为
`9b9e7b0e8b7e59b76cee386cf6221bdac3f9b463a9a4729f68faffcd671391bc`。四个 shard 分别为
40/39/39/39 个架构，对应 run `f40abba1d7fb`、`1724f6b53624`、`43960d0a061a`、
`d0950b062418`。

22 个代理严格产生 3,454 个架构—代理键：3,451 条 `ok`、3 条 `failed`、0 个重复键。三条失败
均为 index 3943、architecture `nb201_topology:839da408774c5a50b88c` 上的 `az_nas`、`naswot`
和 `te_nas`，错误为非有限输出；失败保留原状态。22 个主组件系数、四个 score SHA、完整失败键
和 topology 表规模见
[`evidence/NB201_ONE_PERCENT_22ZCP_CN.md`](evidence/NB201_ONE_PERCENT_22ZCP_CN.md) 与
[`evidence/nb201_one_percent_22zcp_summary.json`](evidence/nb201_one_percent_22zcp_summary.json)。

上述 22 代理证据覆盖 seed 2026。核心 11 代理现已补齐 seed 2027/2028；三 seed 共 5,181 行、
5,172 成功、9 失败，跨 seed 排名稳定性与八个新增 run 的 SHA 见
[`evidence/NB201_CORE_THREE_SEED_CN.md`](evidence/NB201_CORE_THREE_SEED_CN.md)。
`params`/`flops` 的负号来自 `minimize → negated` 方向转换，资源方向与“规模—精度原始关联”应分开
报告；名称不同但结果相同的代理不得据此认定算法独立。NB201 和 NATS-TSS 必须分别运行和报告，
即使 topology codec 相同。

## 7. NATS-TSS 实际验收（H1）

NATS-TSS 已使用独立 `nats_bench.create(..., "tss")` 真值、独立 v1.0 manifest 和相同最低规模完成：
22 代理 seed 2026 为 3,454 行、3,451 成功、3 失败；核心 11 代理三 seed 为 5,181 行、5,172
成功、9 失败，均无重复稳定键。共同 topology 不代表共同真值：与 NB201 的 157 个共同架构中，
31 个 target 数值不同。完整 run SHA、主组件相关性、三 seed 稳定性和专属 topology 表规模见
[`evidence/NATS_TSS_ONE_PERCENT_CN.md`](evidence/NATS_TSS_ONE_PERCENT_CN.md) 与
[`evidence/nats_tss_one_percent_summary.json`](evidence/nats_tss_one_percent_summary.json)。

因此 H1 当前判定为 **“NB201 与 NATS-TSS 既定 seed 协议完成，整体进行中”**；其余 benchmark
仍需独立执行。真实运行发现的 NATS `min/max` repeat reduction 与 shard grouping 已修复并回归；
专属表仍须与通用 failed/coverage 报告配套读取，不能因 mean 协议通过而忽略失败分母。

## 8. NATS-SSS 实际验收（H1）

NATS-SSS 的 CIFAR-10-valid/90-epoch 协议已完成：328 架构 × 22 代理 seed 2026 为 7,216 行，
核心 11 代理三 seed 为 10,824 行，全部成功且无重复键。size 专属报告按 5 个 stage、总通道、
stage sensitivity 和 size-controlled strata 分析。真实运行发现并修复了 `run_id` 导致四片各自
n=82 的错误分组；正式结果合并四个互斥 shard 为 n=328，并按 evaluation seed 分离。详见
[`evidence/NATS_SSS_ONE_PERCENT_CN.md`](evidence/NATS_SSS_ONE_PERCENT_CN.md)。

当前判定更新为 **“NB201、NATS-TSS 与 NATS-SSS/CIFAR-10-valid 既定协议完成，H1 整体进行中”**。
NATS-SSS 的 CIFAR-100/ImageNet16-120 rank transfer 属于尚未完成的定制扩展。

## 9. NB101 实际验收（H1）

NB101 正式 1% 既定协议已完成：从 `nasbench101@full` 的 423,624 个架构中，按 seed 2026 的
250 个特征 strata 无放回抽取 4,237 个架构。22 代理 seed 2026 应有并成功
`4,237 × 22 = 93,214` 个稳定任务键，失败 0、重复键 0；核心 11 代理在 seed
2026/2027/2028 应有并成功 `4,237 × 11 × 3 = 139,821` 个稳定任务键，失败 0、重复键 0。

预算研究覆盖 4/12/36/108 epoch，并分别完成 benchmark repeat 的 `mean`、`min`、`max` 聚合；
三种聚合不可互换。结构控制覆盖 vertices、edges、DAG longest-path depth、conv3/conv1/max-pool
计数，样本内一编辑邻居共 306 对；这些结果只描述 4,237 个抽中架构，不是 423,624 个架构的
全空间枚举。

旧 SynFlow `v1` 与 TE-NAS `portable-v1` 失败行保持不可变，正式有效集使用完整补跑的 SynFlow
`double-v2` 与 TE-NAS `portable-v2`。其中 TE-NAS `portable-v2` 是本仓库的可移植近似，**不是
官方完整 TE-NAS**，不得据此声称完成官方方法复现。完整协议、替换边界、SHA-256 和相关性摘要见
[`evidence/NB101_ONE_PERCENT_CN.md`](evidence/NB101_ONE_PERCENT_CN.md) 与
[`evidence/nb101_one_percent_summary.json`](evidence/nb101_one_percent_summary.json)。

当前判定更新为 **“NB201、NATS-TSS、NATS-SSS/CIFAR-10-valid 与 NB101 的既定协议完成，H1
整体进行中”**。NB301、TNB101、ViT-Bench 等 benchmark 的 1% 协议仍待独立执行，完整项目 H1
尚未完成。
