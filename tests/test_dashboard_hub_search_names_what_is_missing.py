"""The dashboard's Hub search says what is missing and how to get it.

The dashboard page installs ``strands-robots[dashboard,sim-mujoco]``; neither
extra shipped ``huggingface_hub``, so the Train tab read "Hub search
unavailable (ModuleNotFoundError)" - an exception class, no module, no remedy -
on an install that followed the page to the letter. The extra now ships it, and
a missing module is named with the extra that supplies it.
"""

from __future__ import annotations

import builtins
import tomllib
from pathlib import Path

import pytest

for _module in ("fastapi", "webauthn", "jwt"):
    pytest.importorskip(_module)

from strands_robots.dashboard import checkpoints, training  # noqa: E402
from strands_robots.dashboard._hub import hub_unavailable_reason  # noqa: E402

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def test_the_dashboard_extra_ships_the_hub_client() -> None:
    with PYPROJECT.open("rb") as fh:
        extra = tomllib.load(fh)["project"]["optional-dependencies"]["dashboard"]
    assert any(req.replace("-", "_").startswith("huggingface_hub") for req in extra), extra


def test_a_missing_module_is_named_with_the_extra_that_ships_it() -> None:
    reason = hub_unavailable_reason(ModuleNotFoundError("No module named 'huggingface_hub'", name="huggingface_hub"))
    assert "huggingface_hub is not installed" in reason
    assert "strands-robots[dashboard]" in reason


def test_any_other_failure_keeps_its_message() -> None:
    assert hub_unavailable_reason(TimeoutError("read timed out")) == "TimeoutError: read timed out"


@pytest.mark.parametrize(
    ("search", "cache"),
    [(training.hub_datasets, training._HUB_DS_CACHE), (checkpoints.hub_search, checkpoints._CACHE)],
)
def test_both_searches_report_it(monkeypatch, search, cache) -> None:
    real_import = builtins.__import__

    def no_hub(name, *args, **kwargs):
        if name.startswith("huggingface_hub"):
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_hub)
    rows, warning = search("so101-missing-hub-probe")
    assert rows == []
    assert "huggingface_hub is not installed" in warning and "strands-robots[dashboard]" in warning
