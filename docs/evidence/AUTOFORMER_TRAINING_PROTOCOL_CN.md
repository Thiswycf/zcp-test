# AutoFormer scratch 训练协议审计

日期：2026-07-31。

## 权威来源

本项目开放 `autoformer` 搜索空间以 AZ-NAS commit
`5e6683a2cfa5c6d0dc34a1317a842497ba7eae47` 为实现来源，不与 ViT-Bench inherited-supernet
协议合并。主要依据：

- `ImageNet_AutoFormer/train_searched_result_az.sh`：Tiny/Small 使用 500 epoch、20 epoch warmup、
  每卡 batch 256、8 卡；Base 使用 300 epoch。本项目搜索空间对应 Tiny YAML 的
  `embed_dim={192,216,240}`、`depth={12,13,14}`，因此锁定 500 epoch profile。
- `ImageNet_AutoFormer/train_subnet.py`：AdamW、base LR `5e-4`、linear global-batch scaling、
  `warmup_lr=1e-6`、`min_lr=1e-5`、mixup/cutmix 和 validation 流程。
- `ImageNet_AutoFormer/model/autoformer_subnet.py`：`no_weight_decay()` 排除 position embedding、
  class token 和 relative-position 参数；当前静态 port 使用等价的本地参数名。

来源 URL 和 commit 固定在 `configs/training/autoformer_imagenet.yaml`，不得用本地 fork 的未提交
修改替代。

## 已锁定实现

- AdamW 对矩阵/卷积权重应用 weight decay，对 bias、Norm、class token 与 position embedding
  使用 `weight_decay=0` 参数组。
- 500 epoch cosine schedule 在 epoch 0 从 `1e-6` 开始，warmup 结束达到 `5e-4`，完整 schedule
  下界为 `1e-5`；LR 仍按有效 global batch 相对 512 线性缩放。
- 训练 criterion 保留 label smoothing 或 mixup soft target；validation 始终使用 plain cross
  entropy，不把训练平滑带入验证指标。
- 上述字段进入版本化训练 profile schema、resolved config 和 checkpoint config identity；未知或
  拼错的训练键在创建 run 前 fail closed。

CPU 协议 smoke 位于
`<audit-root>/training/autoformer-protocol-cpu-smoke/20260731T005436Z_c556c4739ed8`：run 状态为
`completed`，epoch 0 的 LR 为 `1e-6`，下一 epoch LR 为 `2.595e-5`，并同时生成 `last.pt`
和 `best.pt`。该 smoke 使用合成数据，只验证调度、优化器和 artifact 路径，不属于真实精度证据。

## 验证与边界

单元测试锁定 no-decay 参数成员、LR golden 点、plain validation CE、YAML schema 和 profile 字段。
当前 `formal_training_ready` 仍为 `false`：这些测试证明代码协议，不证明 500 epoch ImageNet 精度，
也不替代“全数据至少 1% epoch”和“严格 1% 数据完整 schedule”的真实 GPU 验收。完成双重 1%
并核验 checkpoint 恢复、吞吐和最终 validation 后，才能单独评估是否解除正式训练门禁。
