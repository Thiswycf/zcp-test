# PlainNet-MBV2 fidelity 审计

## 结论

当前 `zennas_plainnet_mbv2` 不能标记为 `reference_model`。项目中的实现是固定五 stage 的
MBConv 编码，字段为 kernel、expand、depth、width multiplier 和 resolution；而 ZenNAS 与
AZ-NAS 的公开 MobileNetV2 实验使用 PlainNet structure string、`MasterNet` 和可变
`SuperResIDWE*K*` block。两者不是同一个 canonical 搜索空间。

因此当前实现降级为 `proxy_approximation`：默认禁止正式 ZCP evaluate、进化搜索和训练，仅能在
显式 `--allow-approximation` 的方法消融中使用，并必须在报告中保留 fidelity 标签。

## 上游锁定来源

- ZenNAS：[`idstcv/ZenNAS@d1d617e`](https://github.com/idstcv/ZenNAS/tree/d1d617e0352733d39890fb64ea758f9c85b28c1a)
- AZ-NAS MobileNetV2：[`cvlab-yonsei/AZ-NAS@5e6683a`](https://github.com/cvlab-yonsei/AZ-NAS/tree/5e6683a2cfa5c6d0dc34a1317a842497ba7eae47/ImageNet_MBV2)
- ZiCo 上游：[`SLDGroup/ZiCo@b0fec65`](https://github.com/SLDGroup/ZiCo/tree/b0fec65923a90e84501593f675b1e2f422d79e3d)

AZ-NAS README 明确说明其 MobileNetV2 代码由 ZenNAS 和 ZiCo 修改而来。150-epoch 官方脚本使用：

- 8 GPU × 每卡 batch 64，即 global batch 512；
- SGD、momentum 0.9、Nesterov、weight decay `4e-5`；
- `lr_per_256=0.4`，因此 global batch 512 时初始 LR 为 `0.8`；
- cosine、5 epoch warmup、label smoothing 0.1；
- BN momentum `0.01`、custom Kaiming initialization；
- `use_se`、`target_downsample_ratio=16`；
- 224×224 ImageNet-1k、150 epoch。

480-epoch 路线另含 teacher-student distillation、random erase、mixup、auto augmentation 和
EfficientNet-B3 teacher，不能与 150-epoch simplified recipe 合并。

## 当前实现差异

当前项目实现尚未提供以下 reference 语义：

1. PlainNet structure-string parser 与稳定 canonical architecture ID；
2. `SuperResIDWE*K*` block 类型、可变 bottleneck channel、sub-layer 和输出 channel；
3. `use_se` 与 target downsample ratio 16；
4. 上游 custom Kaiming 初始化和 BN momentum 0.01；
5. 官方 150-epoch 的 per-device batch、global batch、LR scaling 和 golden architecture；
6. 上游结构字符串对应的参数量、FLOPs、输出 shape 和 stage golden fixture。

## 升级门槛

只有完成上述结构 port，并至少满足以下条件后，才能恢复 `reference_model`：

- 三个上游发布/搜索结构的 architecture ID、参数量和 FLOPs golden；
- 每个 block 类型的输入输出 shape、stride、SE 和 residual 语义测试；
- 150-epoch recipe 的解析配置、LR、warmup、BN 和初始化 golden；
- sample、canonicalize、mutate、crossover、build、forward 全契约测试；
- 真实 ImageNet 双重 1% 与可信 checkpoint 恢复验收。

本审计不否定现有固定-stage模型作为消融近似的用途，但禁止把它的结果写成 ZenNAS、ZiCo 或
AZ-NAS PlainNet 搜索空间的正式结论。
