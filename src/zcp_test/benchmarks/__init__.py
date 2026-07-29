from zcp_test.registry import Registry

BENCHMARKS: Registry = Registry("benchmark")


def load_builtin_benchmarks() -> None:
    from zcp_test.benchmarks import adapters  # noqa: F401


load_builtin_benchmarks()
