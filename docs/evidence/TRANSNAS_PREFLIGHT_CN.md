# TransNAS-Bench-101 正式验收预检

## 判定

当前状态为 **partial / blocked pending licensed Taskonomy data**，不是 H1 完成：

- tabular 标准答案、转换表、adapter、七任务真值、reference topology port 和 1% 抽样已验收；
- 真实 Taskonomy contract manifest/provider 已实现并通过安全 fixture 测试；
- 本机尚无用户依法取得的 TransNAS 使用 Taskonomy 5k 图像与标签，因此没有运行正式 22-ZCP
  真实输入 GPU sweep；
- random/CIFAR 输入和项目 fixture 均不得替代正式结果。

论文说明正式实验随机选择 24 栋建筑、120K 图像并按 80K/20K/20K 划分；公开代码与发布目录没有
提供可验证的 building split、最终训练 config 或完整逐任务 transform。因此即使用户注册合法的
Taskonomy debug/tiny/custom split，也只能标为 `taskonomy_contract_input`，不能升级为正式 H1。

Taskonomy 数据的新访问必须遵守
[官方获取方式](https://docs.omnidata.vision/starter_dataset_download.html#Examples)及
[Taskonomy Dataset EULA](https://github.com/StanfordVL/taskonomy/blob/master/data/LICENSE)，项目不自动
下载或再分发该数据。

## 锁定资产

| 资产 | SHA-256 |
|---|---|
| `transnas-bench_v10141024.pth` | `1974b0ba21872494fabb541616e9bdae740242eecd6df5c5893747291364bc10` |
| `transnas_micro.jsonl` | `cc6c9fb2cd62e9394deecdb24f86c58536976a5315c7ebe8dfdbe57b93753ee2` |
| `transnas_macro.jsonl` | `4818b9e6f6b72a0f089eb82797ade56c6626531911c8f6edde61a83c66ae6bd4` |

- 上游实现：`yawen-d/TransNASBench@6d4231b1eb04e95750a5b2b6cf391db770bc25d6`。
- micro：4,096 条、4,096 个唯一编码、index 0–4095 连续。
- macro：3,256 条、3,256 个唯一编码、index 0–3255 连续。
- 两个转换表均只含各自 `transnasbench101-{space}-final` 协议，source SHA 与原始文件一致。

## 正式 validation target

以下目标在 micro 4,096/4,096、macro 3,256/3,256 中均唯一匹配且为有限值：

| task | split | metric | budget | direction |
|---|---|---|---:|---|
| `class_scene` | valid | `valid_top1` | 25 | maximize |
| `class_object` | valid | `valid_top1` | 25 | maximize |
| `room_layout` | valid | `valid_loss` | 25 | minimize |
| `jigsaw` | valid | `valid_top1` | 10 | maximize |
| `segmentsemantic` | valid | `valid_mIoU` | 30 | maximize |
| `normal` | valid | `valid_ssim` | 30 | maximize |
| `autoencoder` | valid | `valid_ssim` | 30 | maximize |

`valid_neg_loss` 如被使用必须是 maximize；CLI 已修复自动方向推断。正式主协议优先使用语义更直接的
`valid_loss + minimize`。

## 1% 清单

| space | population | sample | strata | manifest SHA-256 |
|---|---:|---:|---:|---|
| micro | 4,096 | 41 | 84 | `4bb5793fd50ae85e260acd763237c8e93dcf118af6307c56e6a008fe3b4863cc` |
| macro | 3,256 | 33 | 81 | `d58a74138bcf1d2e624a932695e0aa50f49d421ad6b7c1114c9ce9e26d233ede` |

分层字段将 base channel、macro module 与 micro 六条 cell edge operation 分开，不再混算编码数字。
manifest 同时保存 `search_space_id`、variant、source SHA 和 converted SHA。

## 输入与分析边界

- manifest 只允许相对 POSIX 路径，拒绝 `..`、绝对路径、逃逸 symlink、重复 sample ID 和缺失文件；
- classification 使用官方 final5k mask；Jigsaw 使用官方 1,000 permutation；dense/regression target
  从外部数据读取；
- deterministic evaluation transform 明确记录 `training_augmentation_match=false`；
- manifest 明确记录 `official_transnas_24_building_split=false` 与
  `official_transnas_input_protocol_match=false`；
- 标签依赖代理只对 class-object、class-scene 和 jigsaw 开放；回归/dense loss 契约尚无充分上游配置
  证据，继续记为 `unsupported`；
- 专属 transfer 报告新增 `score_coverage.csv`，保留 `ok/failed/unsupported/skipped` 与 paired coverage，
  禁止只报告成功幸存样本。

严格 adapter 上的 index-0 structural smoke 已分别完成：micro/class-object `params=57,963`，
macro/segmentsemantic `params=9,802,833`，真值查询同步成功。两次 smoke 的 `input_source=random`，只
证明 `load → query → build → params`，不进入相关性结论；对应 score SHA-256 为
`80158d99…227d5e8` 和 `19fde61a…26bf5e`。

机器本地完整预检保存在审计目录，不进入 Git；脱敏摘要见
`docs/evidence/transnas_preflight_summary.json`。
