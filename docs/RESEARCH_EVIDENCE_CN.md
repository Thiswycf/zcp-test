# ZCP 研究实验依据与推广边界

## 口径

- **直接依据**：论文在该 benchmark 或其官方同名搜索空间上实际计算 ZCP、相关性或搜索结果。
- **部分直接**：论文覆盖同一搜索空间/模型族，但没有使用本项目对应的发布工件或切片。
- **推广**：依据架构编码或任务特点设计的新分析；不得写成论文已经验证。

“高引”会随日期和检索服务变化，文档不固定容易过期的引用次数，而选择领域中被广泛复用的
基础或系统性工作，并提供论文和官方代码链接。

## 证据矩阵

| 工作 | 常用实验 | 对本项目的直接范围 |
|---|---|---|
| [NASWOT](https://proceedings.mlr.press/v139/mellor21a.html) / [代码](https://github.com/BayesWatch/nas-without-training) | Kendall/Spearman、随机采样选优、初始化/batch/随机输入稳健性、时间 | NB101、NATS-SSS 直接；其余推广 |
| [Zero-Cost Proxies for Lightweight NAS](https://arxiv.org/abs/2101.08134) / [代码](https://github.com/mohsaied/zero-cost-nas) | 全局 Spearman、top-region 命中、编辑距离 1 局部排序、搜索 warmup/proposal、初始化和 batch 消融 | NB101 直接；其余推广 |
| [TE-NAS](https://arxiv.org/abs/2102.11535) / [代码](https://github.com/VITA-Group/TENAS) | trainability/expressivity、逐操作剪枝、搜索成本和组件消融 | DARTS 搜索空间部分直接；不是 NB301 surrogate 实验 |
| [NAS-Bench-Suite-Zero](https://arxiv.org/abs/2210.03230) / [代码](https://github.com/automl/naslib/tree/zerocost) | 代理间相关、跨任务/benchmark、条件熵/信息增益、结构偏置、运行时间、代理加入 predictor | NB101、NB301、TNB micro/macro 直接 |
| [ZiCo](https://arxiv.org/abs/2301.11300) / [代码](https://github.com/SLDGroup/ZiCo) | Spearman/Kendall、Params/FLOPs 基线、NATS/TNB 搜索和多次运行 | NB101、NATS-TSS/SSS、TNB micro 直接 |
| [MeCo](https://papers.nips.cc/paper_files/paper/2023/hash/bfa815ac6f08f4ada34fe22be054f2b9-Abstract-Conference.html) / [代码](https://github.com/HamsterMimi/MeCo) | 多 benchmark Spearman、输入/channel 消融、ZC-PT 搜索、失败任务披露 | NB101、NATS-TSS/SSS、NB301、TNB micro/macro 直接 |
| [AZ-NAS](https://openaccess.thecvf.com/content/CVPR2024/html/Lee_AZ-NAS_Assembling_Zero-Cost_Proxies_for_Network_Architecture_Search_CVPR_2024_paper.html) / [代码](https://github.com/cvlab-yonsei/AZ-NAS) | 多代理组件消融、非线性 rank aggregation、Kendall/Spearman/时间、MobileNet/AutoFormer 搜索 | AutoFormer 搜索空间部分直接；不是 ViT-Bench GT，PiT 无直接依据 |

## 从论文实验到本项目接口

| 实验范式 | 本项目产物 | 约束 |
|---|---|---|
| 全局 rank | `correlations.csv`、protocol heatmap | benchmark/task/budget/split/seed 协议独立 |
| top-region 检索 | `top_k.csv`、`proxy_proxy_top_k.csv` | 稳定 ties 策略，报告 effective-k 和候选集 |
| 局部排序 | NB101 budget、NATS topology matched pairs、NB301 DARTS interaction | matched contrast 仍非因果 |
| 结构偏置 | feature correlations、size-controlled partial Spearman | 保留 Params/FLOPs/规模朴素基线 |
| 代理互补 | proxy-pair correlation、residual、union recall、holdout fusion | 低互相关不等于互补；组合只能由 validation 学习 |
| 跨任务 | TNB task-transfer matrix | micro/macro、metric、split 独立，报告交集覆盖率 |
| 成本 | protocol-specific Pareto | 同时报 wall time、显存和相关性，不用 FLOPs 替代时间 |
| 稳健性 | sensitivity/sample-size convergence | 初始化、batch、输入源、分辨率至少 3–5 seeds |

## 本项目明确属于推广的分析

- NATS-TSS 固定其余五条边的 operation matched-pair；
- NATS-SSS 控制 channel sum 的 stage partial Spearman 与 size-strata；
- NB301 的 `cell × node × source class × operation` 条件效应；
- TNB101 macro module 与 micro edge operation 的统一 factor effect；
- ViT-Bench-101 的 layer/stage 参数展开，尤其 PiT 分析。

这些分析利用 benchmark 的可枚举结构提高可解释性，但不能声称操作替换产生因果性能变化。
