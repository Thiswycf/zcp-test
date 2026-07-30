# 可复现研究实例

`data/` 保存七组小型、确定性、无外部数据依赖的 score JSONL：通用多代理、NB101、NATS-TSS、
NATS-SSS、NB301、TNB101 和 ViT101。完整命令、预期表和解释见：

- `docs/ANALYSIS_CN.md`
- `docs/BENCHMARK_STUDIES_CN.md`
- `docs/RESEARCH_EVIDENCE_CN.md`

运行输出统一写到 `/tmp/zcp-test-examples` 或用户指定的 run 目录。PNG/SVG/HTML/CSV 都能由
JSONL 重建，因此不提交生成物；这避免把验收中间内容长期保留在 Git。

NB101 示例包含官方完整集前四个 canonical architecture ID，执行 budget 视图时仍需通过
`--benchmark-path /path/to/data/nasbench101/converted/full` 指向转换后的本机数据。
