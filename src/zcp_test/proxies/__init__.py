from zcp_test.registry import Registry

PROXIES: Registry = Registry("proxy")


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
