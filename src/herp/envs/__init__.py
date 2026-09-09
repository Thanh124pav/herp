"""Benchmark-family adapters registered here; HERP core imports only from here."""
from __future__ import annotations

from .base import EnvAdapter


def make_adapter(benchmark: str, **kwargs) -> EnvAdapter:
    """Return an unopened adapter; call ``.make()`` to create the underlying env."""
    if benchmark == "maniskill":
        from .maniskill import ManiSkillAdapter
        return ManiSkillAdapter(**kwargs)
    if benchmark == "metaworld":
        from .metaworld import MetaWorldAdapter
        return MetaWorldAdapter(**kwargs)
    if benchmark == "fetch":
        from .fetch import FetchAdapter
        return FetchAdapter(**kwargs)
    raise ValueError(f"Unknown benchmark {benchmark!r}; expected maniskill/metaworld/fetch")


__all__ = ["EnvAdapter", "make_adapter"]
