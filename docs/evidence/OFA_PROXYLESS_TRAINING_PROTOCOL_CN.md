# OFA / Proxyless-MBV2 scratch 协议审计

日期：2026-07-31。

## 权威来源与协议边界

开放搜索空间 `ofa_proxyless_mbv2` 的静态子网、inherited supernet 与 scratch retrain 是三种不同协议。
本文件只约束 scratch retrain；inherited 权重导出和 BN recalibration 结果不得写成 scratch accuracy。

- 静态子网与 inherited 来源：`mit-han-lab/once-for-all@f03b2673db313b9167e2a1c2b7a5cad540cc1313`。
- scratch recipe 来源：`mit-han-lab/ProxylessNAS@b23018c9c369d22931f7422b71ca6a7eaa354c46`。
- 训练 profile：ImageNet-1k、150 epoch、SGD/Nesterov、LR `0.05`、weight decay `4e-5`、
  label smoothing `0.1`、train batch 256、test batch 500、resize scale `0.08`。

`ProxylessNAS` 上游 `_calc_learning_rate()` 使用
`t_cur = epoch * nBatch + batch`，因此 cosine 必须按 batch 更新，不能用每 epoch 的 `LambdaLR`
替代。上游 `distort_color=normal` 对应
`ColorJitter(brightness=32/255, saturation=0.5)`；本项目配置用更明确的 `color_distortion: tf`
标识该协议。

## 已实现并锁定

- `scheduler: cosine_step` 在每次成功 optimizer step 后推进；epoch 0 batch 0 使用 base LR，完整
  schedule 的下一未使用 step 到达 0。短程 1% epoch 验收仍使用正式 150-epoch 总步数。
- TensorFlow 风格颜色扰动、`RandomResizedCrop(scale=(0.08, 1.0))` 和验证尺寸
  `ceil(input_size / 0.875)` 与固定上游语义一致。
- scratch 初始化采用官方 `he_fout`：卷积 fan-out 正态、BN weight/bias 为 1/0、Linear 按输入
  fan-in 均匀初始化并清零 bias。
- optimizer 只对 normalization 参数关闭 weight decay，不把普通卷积/Linear bias 一并排除。
- BN recalibration 清空旧 running statistics，按实际 batch size 加权累计每批有偏均值/方差，并在
  结束后恢复 module momentum 和 train/eval 状态。
- profile validator 固定来源 commit、增强、scheduler、优化器、batch 和初始化字段；拼写错误或
  语义漂移在创建 run 前失败。

## 官方 MAC golden

固定子网为 `width_mult=1.3`、`depth=[2,2,2,2,2]`、21 个位置均为 `ks=3,e=3`、输入
`1x3x224x224`、分类数 1000。审计使用本机 clean clone，并先执行：

```bash
git -C /tmp/once-for-all-f03b267 rev-parse HEAD
git -C /tmp/zcp_nas_research/proxyless rev-parse HEAD
```

输出分别为 `f03b2673db313b9167e2a1c2b7a5cad540cc1313` 和
`b23018c9c369d22931f7422b71ca6a7eaa354c46`。OFA 侧通过
`OFAProxylessNASNets.set_active_subnet(ks=[3]*21,e=[3]*21,d=[2]*5+[1])`、
`get_active_subnet(preserve_weight=False)` 后执行
`ofa.utils.flops_counter.profile(copy.deepcopy(subnet),(1,3,224,224))`；ProxylessNAS 侧用相同
逐层配置构造 `ProxylessNASNets` 后执行 `net.get_flops(torch.zeros(1,3,224,224))`。

计数结果：

- 精确整数 golden / ProxylessNAS `get_flops`：`265,526,256 MACs`；其中 Conv2d
  `263,862,256`，Linear `1,664,000`。
- OFA `profile` 原样报告：`265,526,240 ops`。其逐层 `total_ops` 和总和使用 float32 tensor，
  在该数量级累计舍入损失 16；这不是结构或公式差异。
- 两个上游均把一次乘加计为 1，并在变量或输出中称为 FLOPs/ops。本文统一称 MACs；若采用常见
  `1 MAC = 2 FLOPs` 约定，则对应 `531,052,512 FLOPs`。
- 计入 Conv2d 和最终 Linear 的权重乘加；不计 BN、ReLU6、dropout、global average pooling、
  residual add，也不单独计 Linear bias add。测试使用卷积实际输出 shape，避免额外尺寸近似。

精确 fixture 位于 `tests/test_mobile_models.py::test_ofa_proxyless_224_official_mac_golden`。

## 尚未解除的门禁

`formal_training_ready` 保持 `false`。仍需完成：

1. 真实 ImageNet 双重 1% GPU 验收和 checkpoint resume；
2. distributed validation batch 语义与吞吐审计；
3. inherited accuracy、scratch accuracy 和 predictor 输出的报告分栏验收；
4. 若对外展示 FLOPs，需在结果 schema 中显式记录 `MAC=1` 或 `FMA=2 FLOPs` 的换算口径；当前
   golden 不覆盖 BN/activation/pooling/residual add 的硬件级算术或访存成本。

现有单元测试与 CPU smoke 只证明实现契约，不证明论文精度复现。
