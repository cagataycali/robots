"""Pin: ``ImportError`` raised when gr00t is absent carries ``name='gr00t'``.

AGENTS.md > Key Conventions > 7 requires that a hand-rolled
``raise ImportError(...)`` reporting an absent dependency leaves the module
recoverable from the exception.  When the raise is outside an ``except``
handler, ``name=`` is the only shape that achieves it.

This test constructs a ``Gr00tPolicy`` with ``model_path`` set (the local-
inference path) while gr00t is not importable, and asserts:

  1. An ``ImportError`` is raised (the detection fires).
  2. ``exc.name`` is ``"gr00t"`` (the module is recoverable).
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest


def _detect_returns_none() -> None:
    """Stub that makes ``_detect_groot_version`` return ``None``."""
    return None


@pytest.fixture(autouse=True)
def _hide_groot(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure gr00t cannot be imported, regardless of what is installed."""
    monkeypatch.setitem(sys.modules, "gr00t", None)
    monkeypatch.setitem(sys.modules, "gr00t.model", None)


class TestAbsentGrootImportNamesTheModule:
    """The absent-gr00t ``ImportError`` carries ``name='gr00t'``."""

    def test_import_error_carries_name(self) -> None:
        from strands_robots.policies.groot.policy import Gr00tPolicy

        with patch(
            "strands_robots.policies.groot.policy._detect_groot_version",
            side_effect=_detect_returns_none,
        ):
            with pytest.raises(ImportError) as exc_info:
                Gr00tPolicy(model_path="/tmp/nonexistent_checkpoint")

            assert exc_info.value.name == "gr00t", (
                f"ImportError.name should be 'gr00t', got {exc_info.value.name!r}. "
                "AGENTS.md > Key Conventions > 7 requires name= on a raise "
                "outside an except handler."
            )

    def test_message_is_actionable(self) -> None:
        from strands_robots.policies.groot.policy import Gr00tPolicy

        with patch(
            "strands_robots.policies.groot.policy._detect_groot_version",
            side_effect=_detect_returns_none,
        ):
            with pytest.raises(ImportError, match="no gr00t entry point"):
                Gr00tPolicy(model_path="/tmp/nonexistent_checkpoint")
