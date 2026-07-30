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

`validate` 检查有限数值、声明的主组件、模型权重不变性和 hook 清理。异常应正常抛出，由 evaluator 记录为 `failed`；不要返回伪造分数。

该验证只使用小型合成模型，证明接口和隔离约束，不证明公式复现论文数值，也不替代 22-proxy
sweep、真实 benchmark 相关性或跨模型族兼容性验收。
