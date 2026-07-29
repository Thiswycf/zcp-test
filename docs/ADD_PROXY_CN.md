# 新增 ZCP

自定义代理位于受信任的 `src/zcp_test/proxies/custom/` 包，不支持从任意外部 Python 路径动态执行。

```bash
zcp-test proxy scaffold my_proxy
```

命令生成：

- `src/zcp_test/proxies/custom/my_proxy.py`
- `tests/test_proxy_my_proxy.py`

实现公式后声明 `ProxyCapability`：模型族、版本、是否需要数据/标签、方向、组件和主组件。标量代理直接返回 `float`；多组件代理推荐返回 `ProxyOutput`：

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

