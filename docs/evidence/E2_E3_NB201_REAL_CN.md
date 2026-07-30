# NB201 通用分析与拓扑研究真实工作流证据

本证据用于关闭“CLI 是否能对真实 benchmark 真值和真实 CIFAR 输入生成报告”的接线问题，**不用于**
声明 1% 相关性或论文数值复现。机器外原始 run 位于审计目录；仓库只保存脱敏摘要、哈希和复现命令。

## 协议

- benchmark：NAS-Bench-201 v1.1；不得与 NATS-TSS 合并；
- 架构：index 0–19，共 20 个连续架构；
- 真值：`cifar10-valid / valid / valid-accuracy / 200 epoch / repeat mean`；
- ZCP：`params,naswot,synflow`；
- 输入：真实 CIFAR-10 dataset batch，batch size 2；
- 初始化/输入 seed：7 和 8；
- 代码基线：`f62693d91abfca871c92e0bcae5c85d65b3db5d6`。

可移植命令：

```bash
for SEED in 7 8; do
  zcp-test evaluate \
    --benchmark nasbench201 --trusted --catalog ~/.config/zcp-test/data.json \
    --start 0 --count 20 --seed "$SEED" \
    --dataset cifar10-valid --target-metric valid-accuracy --target-split valid \
    --epoch-budget 200 --metric-seed-reduction mean --target-direction maximize \
    --proxies params,naswot,synflow --input-source dataset \
    --data-root /path/to/cifar10 --batch-size 2 --input-size 32 --classes 10 \
    --gpu auto --output /path/to/audit/evaluate-seed-$SEED
done
```

对命令打印的精确 timestamp run 执行：

```bash
zcp-test analyze correlation --scores "$RUN7/scores.jsonl" \
  --bootstrap-samples 200 --output "$RUN7/reports/correlation"
zcp-test analyze compare --scores "$RUN7/scores.jsonl" \
  --top-k 5 10 --bootstrap-samples 200 --output "$RUN7/reports/compare"
zcp-test analyze sensitivity --scores "$RUN7/scores.jsonl" "$RUN8/scores.jsonl" \
  --parameter seed --bootstrap-samples 200 --output /path/to/audit/seed-sensitivity
zcp-test analyze benchmark --scores "$RUN7/scores.jsonl" \
  --benchmark nasbench201 --view topology --top-k 5 10 \
  --bootstrap-samples 200 --output "$RUN7/reports/topology"
```

## 结果边界

- 每个 run 60 行；seed 7 为 59 成功、1 失败。失败是 index 12 的 NASWOT 返回非有限值，原始
  `scores.jsonl` 保留 `failed` 状态和错误，不伪造分数。
- correlation、compare 和 seed sensitivity 均生成 CSV、PNG、SVG 与静态 HTML。
- topology 生成 20 条 architecture、120 条 edge、5 类 operation、720 条 correlation 和
  90 条 operation-effect 记录；20 个连续架构没有形成合法 matched pair，因此该表为 0 行。
- 详细哈希、有效样本数和数值见 `nb201_real_analysis_summary.json`。
- 20 个连续架构只验证真实工作流；科学结论必须等待确定性 feature-stratified 1% 样本。
