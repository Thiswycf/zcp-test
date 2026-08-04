# Adding a ZCP

Custom proxies live only in the trusted `src/zcp_test/proxies/custom/` package. Arbitrary external
Python paths are never imported. Scaffolding requires a writable source checkout or editable install:

```bash
zcp-test proxy scaffold my_proxy
```

It creates `src/zcp_test/proxies/custom/my_proxy.py` and `tests/test_proxy_my_proxy.py`. Both parent
directories must already exist. Existing targets are never overwritten; files are prepared in local
temporary files and published with no-replace semantics, with rollback if the second target races.
Success emits `module`, `test`, and `next` JSON fields. The initial stub intentionally raises
`NotImplementedError`, so evaluation is `unsupported` until the formula is implemented.

Declare a `ProxyCapability` with version, model families, data/label requirements, score direction,
components, and one explicit primary component. Return a scalar for a one-component formula or a
`ProxyOutput` for named components:

```python
return ProxyOutput(
    score=mean,
    primary_component="mean",
    components={"mean": mean, "sum": total},
)
```

`direction` describes how the score relates to higher target accuracy; use `resource_direction`
separately for Params/FLOPs/latency constraints. Never infer the primary score from dictionary order.

```bash
zcp-test proxy validate my_proxy
zcp-test proxy inspect my_proxy
zcp-test proxy matrix
pytest -q tests/test_proxy_my_proxy.py
```

Validation uses a fixed synthetic CPU CNN and checks finite output, primary-component agreement,
weights, train/eval modes, gradient flags, hooks, and Python/NumPy/Torch RNG restoration. It prints a
JSON report first and exits nonzero if any check fails. `NotImplementedError` is `unsupported`; other
formula errors are `failed`.

`proxy matrix` is a proxy-ID-sorted static capability inventory, not an empirical compatibility sweep.
Stable fields are `proxy_id`, `version`, `model_families`, `requires_data`, `requires_labels`,
`supports_cpu`, `direction`, `components`, `primary_component`, `dependencies`,
`implementation_fidelity`, `source`, `alias_of`, and `resource_direction`. Paper fidelity, real-data
correlation, accelerator behavior, and each model family still require dedicated validation.
