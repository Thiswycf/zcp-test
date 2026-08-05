# ZCP Research Evidence and Extension Boundaries

The project distinguishes **direct evidence** (the paper evaluates the same benchmark/search
space), **partial evidence** (the same model family or search space but a different released
artifact), and **project extension** (a new benchmark-driven analysis).

| Work | Common experimental pattern | Direct scope here |
|---|---|---|
| [NASWOT](https://proceedings.mlr.press/v139/mellor21a.html) | rank correlation, selection, initialization/batch/random-input stability, wall time | NB101 and NATS-SSS |
| [Zero-Cost Proxies](https://arxiv.org/abs/2101.08134) | global and top-region rank, edit-distance-one neighborhoods, search warmup/proposals | NB101 |
| [TE-NAS](https://arxiv.org/abs/2102.11535) | trainability/expressivity, operation pruning, ablations and search cost | DARTS space only; not NB301 surrogate data |
| [NAS-Bench-Suite-Zero](https://arxiv.org/abs/2210.03230) | proxy-proxy relations, task transfer, structural bias, cost, predictor augmentation | NB101, NB301, TNB micro/macro |
| [ZiCo](https://arxiv.org/abs/2301.11300) | Spearman/Kendall, Params/FLOPs baselines, repeated search | NB101, NATS-TSS/SSS, TNB micro |
| [MeCo](https://papers.nips.cc/paper_files/paper/2023/hash/bfa815ac6f08f4ada34fe22be054f2b9-Abstract-Conference.html) / [code `0d830dd`](https://github.com/HamsterMimi/MeCo/tree/0d830dd2f639f9d1ba3b5831a65df768d70fc93b) | cross-benchmark Spearman, input/channel ablations, ZC-PT search; the project uses the minimum-eigenvalue formula only from `hamstermimi-0d830dd-v2` onward | NB101, NATS-TSS/SSS, NB301, TNB micro/macro; legacy `portable-v1` results are invalid |
| [VKDNW](https://openaccess.thecvf.com/content/CVPR2025/html/Tybl_Training-free_Neural_Architecture_Search_through_Variance_of_Knowledge_of_Deep_CVPR_2025_paper.html) / [code `d2ff276`](https://github.com/ondratybl/VKDNW/tree/d2ff276d37d8ba2e9f8c04beb71499d0bd346146) | parameter-Fisher spectrum decile entropy, random/real-input and FIM-dimension ablations, nDCG and size orthogonality | NB201 and MobileNetV2 direct; legacy project `portable-v1` results are invalid |
| [AZ-NAS](https://openaccess.thecvf.com/content/CVPR2024/html/Lee_AZ-NAS_Assembling_Zero-Cost_Proxies_for_Network_Architecture_Search_CVPR_2024_paper.html) | multi-proxy rank aggregation, component ablations, correlation/time, AutoFormer search | AutoFormer search space, not ViT-Bench GT; no direct PiT evidence |

Project extensions include NATS one-edge matched contrasts, NATS-SSS size-controlled stage
analysis, NB301 operation/topology interactions, unified TNB macro/micro factors, and ViT/PiT
layer-stage parameter analysis. These are observational diagnostics and are never presented as
causal effects or as experiments already reported by the cited papers.

See [RESEARCH_EVIDENCE_CN.md](RESEARCH_EVIDENCE_CN.md) for the detailed experiment-to-artifact
mapping and official code links.
