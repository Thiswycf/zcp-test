# AutoFormer source-profile fidelity 审计

审计日期：2026-07-31。审计对象包括锁定的 Auto-Prox `90ed458eff6948a6f0d23e440a8d21bbec50d091`、
AZ-NAS `5e6683a2cfa5c6d0dc34a1317a842497ba7eae47` 与 Microsoft Cream
`b799630a29995163f282b15e2f38701160272fd1`。

## 结论

同一个架构编码不能证明 forward semantics 相同。旧项目把 ViT-Bench 发布候选和 AZ-NAS scratch
开放搜索共用同一个 Cream/AZ 风格静态模型，导致 ViT-Bench 的模型依赖 ZCP 并非在发布 GT 所用
Auto-Prox 静态模型上计算。GT JSONL、canonical architecture ID 与指标查询本身仍然有效；旧 ZCP
原始记录保持只读，但必须标记为 `legacy_cross_implementation`，不得静默覆盖或继续称 reference。

## 已拆分 profile

| Profile | 用途 | QKV / scale | Relative position | LayerNorm | stochastic depth |
|---|---|---|---|---|---|
| `vitbench-autoprox-90ed458` | ViT-Bench AutoFormer 发布切片 | QKV=`3×embed_dim`；每头维度 scale | 无 | `eps=1e-6` | 按实际 depth |
| `aznas-scratch-5e6683` | AZ-NAS AutoFormer 开放搜索/从头训练 | QKV=`3×heads×64`；scale=`(embed_dim//heads)^-0.5` | key/value | `eps=1e-5` | 14 层母 schedule 前缀 |

构造函数强制显式 profile；跨来源选项会 fail closed。ViT-Bench adapter 路由到 Auto-Prox profile，
开放 `autoformer` space 路由到 AZ-NAS profile。结果 provenance 分别记录 model profile、来源和完整
commit，不再用搜索空间级来源覆盖 benchmark 构模来源。

## 自动验证

- Auto-Prox AF-Zero fixture：QKV shape、无 relative-position 参数、LN epsilon、drop-path schedule 与
  参数量 golden；
- AZ-NAS fixture：固定 head width、attention scale、relative K/V 与 super-depth schedule；
- 同一 architecture 经 ViT-Bench adapter 和开放 space 构造时必须得到不同 profile；
- Cream/AZ `get_complexity` 只允许 AZ profile 调用，不得冒充通用 FLOPs 或 ViT-Bench 成本。

## 2026-07-31 时尚未关闭

在该日审计时，AZ-NAS scratch 的 optimizer/scheduler 仍须逐项验证：timm 等价 no-weight-decay 参数组、
`warmup_lr=1e-6`、`min_lr=1e-5`、validation plain CE 与正式 epoch trace。在这些 golden 完成前，
当时 `configs/training/autoformer_imagenet.yaml` 必须保持 `formal_training_ready: false`，不得启动正式双重
1% 或完整训练。ViT-Bench 三公开切片也必须使用新 profile 重新计算代表性架构和 5×22 预验收。

## 2026-08-04 状态更新

上述 optimizer/scheduler/no-decay/validation golden 已补齐，单候选 `zcp-selected` 又完成全 ImageNet
5 epoch 与严格 1% 数据 500 epoch 两项限定协议，分别保留 5/500 行训练记录、终态 manifest 和
`last.pt`/`best.pt`。因此仓库 AutoFormer profile 的启动门禁现已显式解除；checksum 与科学边界见
`autoformer_single_candidate_dual_one_percent_completion_20260804.json`。这只表示 reference profile
具备启动完整训练的工程条件，不表示已完成 500 epoch 全数据精度复现，也不证明 ZCP 优于随机或资源基线。
ViT-Bench 的固定候选协议仍与该 scratch training profile 分开。

## 权威来源

- <https://github.com/lliai/Auto-Prox-AAAI24/tree/90ed458eff6948a6f0d23e440a8d21bbec50d091>
- <https://github.com/cvlab-yonsei/AZ-NAS/tree/5e6683a2cfa5c6d0dc34a1317a842497ba7eae47/ImageNet_AutoFormer>
- <https://github.com/microsoft/Cream/tree/b799630a29995163f282b15e2f38701160272fd1/AutoFormer>
