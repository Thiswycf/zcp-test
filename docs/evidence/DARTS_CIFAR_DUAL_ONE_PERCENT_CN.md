# DARTS CIFAR 双重 1% 训练验收

## 结论边界

本证据关闭的是 DARTS CIFAR-10/100 的双重 1% **工程与限定协议验收**，不是 600 epoch
全数据精度复现，也不是多 seed 搜索收益证明。所有结果使用训练 seed `20260731`；ER 候选由一个
固定 CIFAR-10 batch 和一个初始化 seed 选择，因此只能称为本项目既定候选。

## 候选冻结

真实 CIFAR-10 ER 进化搜索使用 population 32、16 generations，共写入 416 条候选记录、336 次
唯一评估。冻结三类候选：

| 角色 | architecture ID | CIFAR profile 参数量 |
|---|---|---:|
| ER 搜索候选 | `8229ba5055865c0aaf8d` | 3,708,912 |
| 固定随机候选 | `e6f106292da6dfb2c91b` | 3,820,152 |
| 参数匹配随机池候选 | `107e3944e6a3710eca1a` | 3,707,400 |

参数匹配候选来自 256 个固定 seed 随机架构，和 ER 候选只差 1,512 个参数。它不是“参数代理搜索
最优”，而是控制模型规模的随机基线。

## 预检发现与修复

真实单 epoch 预检依次发现并修复：模型初始化未受 `--seed` 控制、CUDA deterministic 状态未锁定、
短程运行把 600-epoch cosine/drop-path 压缩到 1/6 epoch、训练 JSONL 缺显存和吞吐指标。
最终同卡同 seed 两次预检满足：

- loss、top-1/top-5、LR、显存等科学字段完全相同；
- 模型 state SHA-256 均为
  `a5f1cca5fb75ec4e82bc534cf5e44d556583044e2d06ca748c5720a9ffa9e14d`；
- `schedule_epochs=600`，epoch 0 后 LR 为 `0.024999828653092835`，不是 0；
- 峰值 allocated 显存约 3.06 GiB。

## 全数据 × 6 epoch

该模式标识为 `full_data_one_percent_epochs`，运行正式 600-epoch schedule 的前 6 个 epoch。

| 数据集 | ER 候选 | 固定随机 | 参数匹配随机 |
|---|---:|---:|---:|
| CIFAR-10 best valid top-1 | 78.62 | 77.28 | 64.67 |
| CIFAR-100 best valid top-1 | 46.19 | 44.13 | 26.81 |

六个 manifest 均为 `completed`，每个 `training.jsonl` 恰好 6 行，且均有 `last.pt`、`best.pt`。
ER 候选的可信 checkpoint 恢复生成新 run，并恢复 6 条历史记录。

## 恰好 1% 数据 × 600 epoch

该模式标识为 `one_percent_data_protocol`；每个 split 先计算全局 `round(N×0.01)`，再按最大余数法
确定性分配类别配额。

| 数据集 | ER 候选 | 固定随机 | 参数匹配随机 |
|---|---:|---:|---:|
| CIFAR-10 best valid top-1 | 42.0 | 45.0 | 46.0 |
| CIFAR-100 best valid top-1 | 15.0 | 12.0 | 11.0 |

六个 manifest 均为 `completed`，每个 `training.jsonl` 恰好 600 行，末 epoch 为 599，最终 LR 为
0，最终 drop-path 为 `0.2×599/600`。ER 候选的可信 checkpoint 恢复生成新 run，并恢复 600 条
历史记录。

两个协议给出的候选排序不同，禁止求平均后宣称 ER 稳定优于基线。它们分别测试早期全数据学习和
小数据完整 schedule，科学含义不同。

## 典型命令

```bash
zcp-test train --config configs/training/darts_cifar10.yaml \
  --real-data-preflight --epochs 1 --data-fraction 1.0 \
  --architecture CANDIDATE.json --data-root /path/to/cifar10 --gpu auto

zcp-test train --config configs/training/darts_cifar10.yaml \
  --acceptance-smoke --epochs 6 --data-fraction 1.0 \
  --architecture CANDIDATE.json --data-root /path/to/cifar10 --gpu auto

zcp-test train --config configs/training/darts_cifar10.yaml \
  --acceptance-smoke --epochs 600 --data-fraction 0.01 \
  --architecture CANDIDATE.json --data-root /path/to/cifar10 --gpu auto
```

原始数据、run、checkpoint 和大型报告只保存在外部审计目录。可提交的脱敏摘要见
[`darts_cifar_dual_one_percent_summary.json`](darts_cifar_dual_one_percent_summary.json)。
