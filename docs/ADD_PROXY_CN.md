# 新增 ZCP

自定义代理位于受信任的 `src/zcp_test/proxies/custom/` 包，不支持从任意外部 Python 路径动态执行。
`scaffold` 只适用于可写源码 checkout 或 editable install；普通只读 wheel/site-packages 不是支持
目标，因为命令会同时写源码包和仓库测试目录。

```bash
zcp-test proxy scaffold my_proxy
```

命令生成：

- `src/zcp_test/proxies/custom/my_proxy.py`
- `tests/test_proxy_my_proxy.py`

前置目录必须存在；任一目标文件已存在时命令 fail closed，不覆盖。两个文件先写入同目录临时文件，
再以原子“不覆盖”方式发布；若第二个目标发生并发冲突，第一个目标会回滚。成功 stdout 为：

```json
{"module": ".../my_proxy.py", "test": ".../test_proxy_my_proxy.py", "next": "zcp-test proxy validate my_proxy"}
```

初始模板故意抛出 `NotImplementedError`，evaluator 会将其分类为 `unsupported`；因此在实现公式前执行
`proxy validate` 预期非零退出，不是脚手架损坏。

实现公式后声明 `ProxyCapability`：模型族、版本、是否需要数据/标签、方向、组件和主组件。标量代理直接返回 `float`；多组件代理推荐返回 `ProxyOutput`：

`direction` 表示“该 score 越大还是越小越可能对应更高目标性能”，用于 accuracy 相关性和代理搜索；
它不是成本偏好。若代理同时输出 Params/FLOPs/latency 等资源量，使用
`resource_direction=minimize` 单独声明约束方向。禁止因为资源越少越好而把规模—accuracy
相关性取负；任何方向语义变化都必须升级代理版本并为旧结果写显式 migration。

```python
return ProxyOutput(
    score=mean,
    primary_component="mean",
    components={"mean": mean, "sum": total},
)
```

不得依赖字典第一个字段作为主分数。验证命令：

```bash
zcp-test proxy validate my_proxy
zcp-test proxy inspect my_proxy
zcp-test proxy matrix
pytest -q tests/test_proxy_my_proxy.py
```

`validate` 使用固定 CPU 合成 CNN，检查有限数值、声明的主组件、模型权重、train/eval mode、
`requires_grad`、hook 以及 Python/NumPy/Torch RNG 均恢复。命令总是先向 stdout 打印详细 JSON；任一
检查失败后再以非零状态退出，因此自动化应同时保存 stdout 和退出码。`NotImplementedError` 记为
`unsupported`，其他公式异常记为 `failed`；不要返回伪造分数。

`proxy matrix` 只是按 proxy ID 排序的**静态 capability inventory**，不会在每种 benchmark、模型族、
GPU 或依赖组合上实际运行。每行稳定字段为 `proxy_id`、`version`、`model_families`、
`requires_data`、`requires_labels`、`supports_cpu`、`direction`、`components`、`primary_component`、
`dependencies`、`implementation_fidelity`、`source`、`alias_of`、`resource_direction`。运行时兼容性必须
另做 `validate`、真实模型 smoke 和支持矩阵 sweep。

该验证只使用小型合成模型，证明接口和隔离约束，不证明公式复现论文数值，也不替代 22-proxy
sweep、真实 benchmark 相关性或跨模型族兼容性验收。
