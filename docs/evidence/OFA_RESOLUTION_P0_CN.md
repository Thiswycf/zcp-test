# OFA Proxyless 候选分辨率 P0 修复证据

## 修复范围

本轮仅修复 `ofa_proxyless_mbv2` 的候选输入分辨率语义，不调整候选编码、训练 recipe、
`he_fout`、TF color jitter 或官方 BN recalibration 算法。

修复前，architecture spec 中的 `resolution` 只进入 architecture ID 和模型 metadata；
`evaluate` 与 `search` 在运行开始时按全局 `input_size` 构造一个 batch，所有候选共用该
空间尺寸。因此两个 genotype 相同、仅 `resolution` 不同的候选可能实际接收完全相同的输入。

## 当前协议

- `ofa_proxyless_mbv2` 的 `evaluate` 和 `search` 使用
  `input_size_policy=architecture_resolution`。
- 每个候选从 architecture spec 解析 `resolution`，输入 tensor 的高宽必须与其完全一致。
- 相同分辨率、相同 seed 和输入协议复用同一个确定性 batch；同一候选的多个 ZCP 共享该
  batch，不会为每个代理重新抽样。
- 不同分辨率分别执行 resize/生成输入，并产生不同 `input_fingerprint`。
- search cache identity 同时包含 architecture ID 和实际输入 fingerprint。
- `scores.jsonl` 与候选级 `search.jsonl` 明确记录：
  `requested_input_size`、`candidate_resolution`、`actual_input_size`、
  `input_size_policy`、`input_fingerprint` 和完整 `input_protocol`。
- inherited-supernet BN recalibration 在启用时也按候选实际分辨率构造校准 stream；其算法仍是
  `project_deterministic`，本轮不将其升级为官方 OFA 校准协议。

普通固定模型空间继续使用原有固定 `input_size` batch、fingerprint 和 manifest 协议。

## 显式 input_size 边界

- 普通固定模型允许并继续使用用户显式 `input_size`。
- OFA evaluate 若用户通过 CLI 或 YAML 显式指定的 `input_size` 与候选 `resolution` 不同，
  在创建 run 目录前失败，不静默覆盖。
- OFA search 的分辨率是可变候选字段，因此拒绝固定的显式 `input_size`；用户应省略该字段，
  由每个候选的 `resolution` 主导。
- 模型的 `image_size`、architecture resolution 和实际 tensor shape 任一不一致时 fail closed。

## 回归覆盖

`tests/test_ofa_resolution_protocol.py` 覆盖：

1. 同一 genotype 的两个 resolution 使用同一 ImageNet 样本 ID，但 tensor shape、transform
   protocol、fingerprint 和 cache key 不同。
2. 两个静态 OFA 模型分别接收正确空间尺寸，forward 输出 shape 正确且结果不相同。
3. 同一候选的多个代理收到同一 tensor storage；不同 resolution 不复用 tensor。
4. evaluate score 行和 search candidate 行均保存候选级分辨率与 fingerprint。
5. evaluate 的 input-size mismatch 和 search 的固定显式 input-size 均在 run 创建前失败。

定向门禁命令：

```bash
PYTHONPATH=src conda run -n zcp-test pytest -q \
  tests/test_ofa_resolution_protocol.py \
  tests/test_reference_models.py tests/test_core.py \
  tests/test_workflow.py tests/test_cli_commands.py
```
