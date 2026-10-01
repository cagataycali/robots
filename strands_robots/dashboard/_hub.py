"""How the dashboard says why a Hub search is unavailable."""

from __future__ import annotations


def hub_unavailable_reason(exc: BaseException) -> str:
    """A reason an operator can act on: the missing module and the extra that ships it, or the error.

    A bare exception class ("ModuleNotFoundError") named neither what was missing
    nor how to get it, on an install that followed the dashboard page to the
    letter.
    """
    if isinstance(exc, ImportError):
        module = getattr(exc, "name", None) or "huggingface_hub"
        return f"{module} is not installed: pip install 'strands-robots[dashboard]'"
    return f"{type(exc).__name__}: {exc}"[:200]
