from zcp_test.registry import Registry

PROXIES: Registry = Registry("proxy")

for _retired_proxy in ("ntkt", "er_pr", "er_conn", "er_deg", "er_dist"):
    PROXIES.retire(
        _retired_proxy,
        "withdrawn after fidelity audit; historical JSONL remains read-only",
    )


def load_builtin_proxies() -> None:
    from zcp_test.proxies import builtin  # noqa: F401

    load_custom_proxies()


def load_custom_proxies() -> None:
    import importlib
    import pkgutil

    from zcp_test.proxies import custom

    for module in pkgutil.iter_modules(custom.__path__):
        if module.name.startswith("_") or not module.name.isidentifier():
            continue
        importlib.import_module(f"{custom.__name__}.{module.name}")
