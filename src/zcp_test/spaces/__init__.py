from zcp_test.registry import Registry

SPACES: Registry = Registry("search space")


def load_builtin_spaces() -> None:
    from zcp_test.spaces import builtin  # noqa: F401

