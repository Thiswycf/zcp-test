# PlainNet-MBV2 fidelity 与训练协议审计

## 当前结论

`zennas_plainnet_mbv2` 已替换为 ZenNAS/AZ-NAS 的 PlainNet structure-string 搜索空间和可执行
PyTorch reference port，不再使用旧固定五 stage `{kernel, expand, depth, width, resolution}` 近似。
当前模型 fidelity 为 `reference_model`，可用于正式 ZCP evaluate 和 validation-only 搜索；这只证明
结构语义，不证明论文精度复现。

150-epoch scratch profile 已代码锁定，但 `formal_training_ready` 保持 `false`。在真实 ImageNet
双重 1% GPU、分布式 validation 和 checkpoint resume 验收完成前，只允许 `--smoke`、
`--real-data-preflight` 或 `--acceptance-smoke`。

## 锁定来源

- [ZenNAS `d1d617e`](https://github.com/idstcv/ZenNAS/tree/d1d617e0352733d39890fb64ea758f9c85b28c1a)
- [AZ-NAS MobileNetV2 `5e6683a`](https://github.com/cvlab-yonsei/AZ-NAS/tree/5e6683a2cfa5c6d0dc34a1317a842497ba7eae47/ImageNet_MBV2)
- [ZiCo `b0fec65`](https://github.com/SLDGroup/ZiCo/tree/b0fec65923a90e84501593f675b1e2f422d79e3d)

支持的安全白名单 block 为 `SuperConvK{1,3}BNRELU` 和
`SuperResIDWE{1,2,4,6}K{3,5,7}`。parser 不使用 `eval`，并验证参数数目、通道连续性、stride、
固定 2048-channel head 和 224 分辨率。canonical architecture ID 由规范 structure string 生成。

## 结构与 golden

AZ-NAS 450M 搜索脚本的初始结构为：

```text
SuperConvK3BNRELU(3,8,2,1)SuperResIDWE6K3(8,32,2,8,1)SuperResIDWE6K3(32,48,2,32,1)SuperResIDWE6K3(48,96,2,48,1)SuperResIDWE6K3(96,128,2,96,1)SuperConvK1BNRELU(128,2048,1,1)
```

对 clean upstream clone 与项目 port 使用相同输入 `1×3×224×224` 得到：

| 协议 | 参数量 | Conv/Linear MAC |
|---|---:|---:|
| 搜索模型，`use_se=false` | 2,824,264 | 159,334,080 |
| 训练模型，`use_se=true` | 3,579,232 | 160,081,728 |

两种模型输出均为 `1×1000`。上游 `MasterNet.get_FLOPs()` 还计入部分 BN、ReLU、residual 和 SE
算术，因此分别报告 `162,396,776` 与 `164,511,512`；不得与 Conv/Linear MAC 混列。

实现复现上游每个 `SuperResIDWE` sub-layer 的两段 inverted-depthwise residual、首段强制 projection、
第二段按 shape projection、可选 SE、BN epsilon `1e-3`，并在初始化后将 residual branch 最后 BN
权重置零。sample、mutate、crossover 会重连相邻 block 的输入通道，固定最终 head。

## 150-epoch scratch profile

配置文件为 `configs/training/zennas_plainnet_mbv2_imagenet.yaml`：

- ImageNet-1k、224、150 epoch；
- 8 GPU × 每卡 batch 64，即 global batch 512；
- `lr_per_256=0.4`，线性缩放后的有效 LR 为 `0.8`；
- SGD、momentum 0.9、Nesterov、weight decay `4e-5`；
- Conv/Linear weight decay，bias 与 BN 不 decay；
- 按 optimizer step 的 5-epoch linear warmup，之后 cosine 到 0；
- label smoothing 0.1、bicubic random resized crop、horizontal flip、ColorJitter(0.4) 和
  AlexNet-style PCA lighting；
- BN momentum `0.01`、custom Kaiming、`use_se=true`。

公开 480-epoch teacher-student 路线含 EfficientNet-B3 teacher、mixup、random erase 和 auto augment，
是不同协议，未混入 150-epoch profile。搜索阶段按上游脚本使用 `use_se=false`，scratch retrain 才启用
SE；结果 manifest 必须保留该协议差异。

## 尚未解除的门禁

1. 全数据至少 2 epoch 和固定 1% 数据完整 150 epoch 的三候选 GPU 验收；
2. 8-rank 或等价可证明的 global batch 512 分布式 validation 归约；
3. 中断后 optimizer、step scheduler、scaler、RNG 和 JSONL 的恢复一致性；
4. 搜索 best、固定随机、资源匹配候选的最终对比报告。

完成前不得把 CPU golden、合成 smoke 或短程 preflight 写成论文精度复现。
