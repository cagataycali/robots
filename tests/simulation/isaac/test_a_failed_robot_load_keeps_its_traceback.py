"""A failed ``add_robot`` load says where it failed, not only what.

On Isaac Sim 6.1 every robot load failed with exactly this, and nothing else::

    Failed to load USD robot 'so100': 'NoneType' object has no attribute 'is_homogeneous'

The loaders' ``except`` logged ``%s`` of the exception - no traceback, no
exception type - so the line that raised (deep inside
``isaacsim.core.prims``' eager tensor-view bind) could only be found by
monkeypatching the logger. The log record now carries ``exc_info`` and the
envelope names the exception type.
"""

from __future__ import annotations

import logging
import types
from typing import Any

import pytest

pytest.importorskip("strands_robots.simulation.isaac")

from strands_robots.simulation.isaac.simulation import IsaacConfig, IsaacSimulation  # noqa: E402
from tests.simulation._isaac_engine import isaac_engine  # noqa: E402

_LOGGER = "strands_robots.simulation.isaac.simulation"


def _engine() -> Any:
    engine = isaac_engine(IsaacConfig())
    engine._world = types.SimpleNamespace(physics_sim_view=object())
    engine._world_created = True
    return engine


def _boom(*args: Any, **kwargs: Any) -> Any:
    raise AttributeError("'NoneType' object has no attribute 'is_homogeneous'")


@pytest.mark.parametrize(
    ("loader", "kwargs", "kind"),
    [
        ("_load_usd_robot", {"usd_path": "/robots/so100.usda"}, "USD"),
        ("_load_urdf_robot", {"urdf_path": "/robots/arm.urdf"}, "URDF"),
    ],
)
class TestAFailedLoad:
    def test_the_log_record_carries_the_traceback(self, monkeypatch, caplog, loader, kwargs, kind) -> None:
        monkeypatch.setattr(IsaacSimulation, loader, _boom)
        with caplog.at_level(logging.ERROR, logger=_LOGGER):
            result = _engine().add_robot("r", **kwargs)

        assert result["status"] == "error"
        records = [r for r in caplog.records if f"Failed to load {kind} robot" in r.getMessage()]
        assert records, caplog.text
        assert records[0].exc_info is not None, "the loader failure was logged without its traceback"
        assert records[0].exc_info[0] is AttributeError

    def test_the_envelope_names_the_exception_type(self, monkeypatch, loader, kwargs, kind) -> None:
        monkeypatch.setattr(IsaacSimulation, loader, _boom)
        result = _engine().add_robot("r", **kwargs)

        text = result["content"][0]["text"]
        assert text.startswith(f"Failed to load {kind} robot 'r': AttributeError: ")
        assert "is_homogeneous" in text
