# ZCP 官方实现一致性审计（2026-08-05，北京时间）

## 结论边界

本审计区分“论文公式”“作者代码”“固定 commit 的独立端口”“项目扩展”和“接口可运行”。接口返回有限值不等于论文复现。旧 JSONL 保持只读；被判定为 `wrong` 的历史分数、相关性、排序、聚合权重和搜索候选选择均不得再作为相应论文代理的证据。

## 用户给定五种名称的事实核验

题名 `Vision-Oriented Lightweight Neural Architecture Search with Budget-Adaptive Evaluation` 未在 Crossref、OpenAlex、Semantic Scholar 或 arXiv 检索到正式记录。最吻合的可核验来源是 CVPR 2022《Training-free Transformer Architecture Search》及作者仓库 [`decemberzhou/TF_TAS@42616bc`](https://github.com/decemberzhou/TF_TAS/tree/42616bcf1b6bb643bf968a8342f8aaddc4f53f32)。该仓库注册的五项是：

- `DSS`
- `GraSP`
- `SNIP`
- `NASWOT`
- `TE-NAS`

并非 `AC/HI/HC/DSS/DSS++`。后者只有四个名称，且：

- `DSS++` 在目标论文和固定仓库中不存在；
- `AC/HI/HC` 最接近 ACL 2023《Training-free Neural Architecture Search for RNNs and Transformers》及 [`training-free-nas@2d76e01`](https://github.com/aaronserianni/training-free-nas/tree/2d76e01b9586cad7340e8268dadba3056efd070b)；
- `Attention Confidence` 与发布表中的 `Head Confidence` 属同类命名，不能无证据作为两个独立代理；
- `HC` 若指 `Hidden Covariance`，它是该论文的 RNN 指标，并非 Transformer-only。

因此项目不伪造第五项，也不把跨论文缩写强行归并。本次只加入来源和公式均无歧义的 `dss`；其余名称需用户提供准确论文 DOI/URL、表格行或公式编号后再实现。

## 当前注册代理审计矩阵

| ID | 判定 | 官方/第一方来源 | 当前实现结论 |
|---|---|---|---|
| `params`, `flops` | structural | 框架结构计量 | 不是论文 ZCP；资源方向与精度方向分离 |
| `gradnorm` | **wrong** | `mohsaied/zero-cost-nas@b5059bc` | 当前为所有参数梯度的全局 L2；官方只对 Conv/Linear weight 逐层 norm 后求和 |
| `synflow` | partial | `mohsaied/zero-cost-nas@b5059bc` | 线性化和全 1 输入基本一致；层选择与 BN 语义不一致 |
| `naswot` | partial | `BayesWatch/nas-without-training@b3a82a6` | 核公式一致；当前扩展到多种激活且缺官方路径筛选 |
| `jacob_cov` | partial | `mohsaied/zero-cost-nas@b5059bc` | 主公式接近；特征值平移、截断与失败语义不同 |
| `meco`, `meco_opt` | stabilized port | `HamsterMimi/MeCo@0d830dd` | 已按作者公式修复；有限值和 hook 清理属于稳定化差异 |
| `vkdnw` | stabilized port | `ondratybl/VKDNW@d2ff276` | 已按参数 Fisher 谱分位数熵修复；使用稳定对称分解 |
| `near` | **no official code + wrong** | 仅论文 arXiv:2408.08776 | 当前拼接普通 rank，不是逐层 pre/post effective rank |
| `swap` | **wrong** | `pym1024/SWAP@0853fc8` | 当前统计 sample pattern；官方转置后统计 neuron-wise sample pattern |
| `zen` | **wrong** | `idstcv/ZenNAS@d1d617e` | 当前 logits 扰动缺失重初始化、pre-GAP、BN 项和重复平均 |
| `ntkt` | **no official code + wrong** | ETAS 论文 ACL Findings 2024 | 当前 sample Gram logdet，不是 token-mean NTK trace；应仅限 Transformer |
| `zico` | **wrong** | `SLDGroup/ZiCo@b0fec65` | 当前逐样本观测和全参数聚合不同于官方跨 mini-batch、逐层 Conv/Linear 公式 |
| `te_nas` | **wrongly named approximation** | `VITA-Group/TENAS@9df78ff` | 当前为 SynFlow/NASWOT/GradNorm 组合，不是 NTK 条件数与线性区域协议 |
| `az_nas` | no official generic implementation | AZ-NAS `5e6683a` | 当前通用三项组合只是项目近似，不能称官方 AZ-NAS |
| `az_nas_autoformer` | stabilized port | AZ-NAS `5e6683a` | 公式接近；输入、异常和模型状态协议有稳定化差异 |
| `az_nas_plainnet` | partial stabilized port | AZ-NAS `5e6683a` | 主公式接近；初始化和恒等转移处理存在实质差异 |
| `er` | project extension | 本地 `TER-Score@a646c5a` 可作第一方对照 | 当前计算模块输出 ER，不是本地 edge-feature ER |
| `ter` | **wrong legacy alias** | 本地 `TER-Score@a646c5a` | 当前只是 `er` alias；本地 TER 是双向 edge PageRank 加权 ER |
| `er_pr`, `er_conn`, `er_deg`, `er_dist` | name-colliding project extension | 本地 `TER-Score@a646c5a` 可作第一方对照 | 当前基于 FX 模块节点，均不等价于本地同名 edge 公式 |
| `dss` | paper-formula stabilized port | `TF_TAS@42616bc` | 独立实现 MSA 核范数多样性 + MLP 突触显著性；仅支持项目 StaticAutoFormer/StaticPiT |

## 单独列出的“无官方实现”

1. `near`：论文未给出作者代码；项目当前实现还与论文公式冲突。
2. `ntkt`：ETAS 论文未给出作者代码；项目当前实现与论文公式冲突。
3. 通用 `az_nas`：不存在任意 CNN/Transformer 共用的作者官方函数；官方按搜索空间提供专用实现。
4. 当前 `er` 与 `er_*`：是项目扩展，不是本地 TER-Score 同名实现。
5. `DSS++`：截至审计日期，在目标论文及固定作者仓库中没有该方法或函数。
6. `AC/HI/HC` 这组请求：存在跨论文和缩写歧义，不能据当前信息确定五个互异官方代理。

ER/TER 的 `${TER_SCORE_ROOT}` 与公开仓库 [`Thiswycf/TER-Score`](https://github.com/Thiswycf/TER-Score) 是可核验的第一方项目实现，但没有 tag 或独立发布包；仓库固定 commit 为 `a646c5a6e0b4633d06a153fe3cdc9b6ca3d9f06f`。项目只读使用本地路径作逐文件审计，并在 metadata 中引用公开固定 commit，不复制或修改原项目。

## DSS 使用

```bash
zcp-test proxy inspect dss
zcp-test evaluate --space autoformer --proxies dss --count 10 --input-source random
zcp-test evaluate --space pit --proxies dss --count 10 --input-source random
```

`dss` 使用单个全 1 图像并对输出和反向，不使用标签；主分数为：

```text
score = attention_diversity + mlp_saliency + auxiliary_saliency
```

`auxiliary_saliency` 对应作者代码额外计入的 patch embedding、分类头和 PiT stage class projection。
当前版本为 `tf-tas-42616bc-code-protocol-port-v2`，模型族严格为 `transformer`。它是依据论文和公开代码独立编写的公式端口，不复制无许可证的 TF_TAS 源码，也不声称与其 sampled-weight 超网逐位一致。

## 历史结果失效范围

- 必须重算：`gradnorm`, `near`, `swap`, `zen`, `ntkt`, `zico`, `te_nas`, `ter`。
- 若声称作者实现可比则必须重算：`synflow`, `naswot`, `jacob_cov`。
- `er_pr/conn/deg/dist` 只能按项目 FX 扩展解释，不能引用为本地 TER-Score 对照。
- 受影响的旧 NB101/NB201/NATS/NB301/TNB 22-ZCP 汇总保留作审计，不再作为对应论文方法的有效相关性证据。
- `params`, `flops`, benchmark 真值和训练准确率不因本次代理公式审计失效。

常见错误：`autoformer` reference profile 固定使用 224×224；若传入 `--input-size 32`，模型会明确失败
`AutoFormer inputs must be 224x224`。正式 smoke 使用 `--input-size 224 --batch-size 1`。

CPU CLI 典型实例见 [`evidence/dss_cli_smoke_20260805.json`](evidence/dss_cli_smoke_20260805.json)：
1 个 AutoFormer 架构产生严格 1 行成功记录，三个组件与主分数一致。该实例只验收工程链路，不是相关性或论文精度复现。
