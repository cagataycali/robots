"""Every native driver's ``status`` verb answers with ``get_status``'s envelope, one level deep.

Six drivers wrapped the envelope ``get_status()`` already returns in a second
one, so the fields sat at ``content[0].json.content[0].json`` and an agent
reading ``content[0]["json"]["port"]`` got a ``KeyError`` (#4151); the other five
returned it directly. First-robot.md promises one envelope shape for every
method. The population is the whole fleet, read off the registry, so a driver
added later is graded too.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

import strands_robots.drivers as drivers_pkg
from strands_robots.drivers import get_native_driver_class

pytestmark = pytest.mark.usefixtures("_reachy_discovery_stays_on_this_machine")


@pytest.fixture
def _reachy_discovery_stays_on_this_machine(monkeypatch: pytest.MonkeyPatch) -> None:
    from strands_robots.drivers.reachy_vocabulary import ENV_HOST

    monkeypatch.setenv(ENV_HOST, "127.0.0.1")
    monkeypatch.setattr("strands_robots.drivers.reachy_vocabulary.discovery_candidates", lambda: ["127.0.0.1"])


def _driver_classes() -> dict[str, type[Any]]:
    return {
        cls.__name__: cls
        for cls in (get_native_driver_class(robot) for robot in drivers_pkg.list_native_drivers())
        if cls is not None
    }


def _first_robot_for(cls: type[Any]) -> str:
    for candidate in cls.__mro__:
        for robot in drivers_pkg.list_native_drivers():
            if get_native_driver_class(robot) is candidate:
                return robot
    raise AssertionError(cls.__name__)


def _status(driver: Any) -> dict[str, Any]:
    async def _drive() -> dict[str, Any]:
        results = [
            event
            async for event in driver.stream({"toolUseId": "call-1", "name": "t", "input": {"action": "status"}}, {})
        ]
        assert len(results) == 1
        return results[0]

    return asyncio.run(_drive())


_CLASSES = sorted(_driver_classes().items())


def test_the_population_is_the_whole_fleet() -> None:
    assert len(_CLASSES) >= 10, [name for name, _ in _CLASSES]


@pytest.mark.parametrize(("name", "cls"), _CLASSES, ids=[name for name, _ in _CLASSES])
def test_status_fields_sit_one_level_down(name: str, cls: type[Any]) -> None:
    driver = cls(tool_name=_first_robot_for(cls), cameras=None, data_config=None)
    try:
        envelope = _status(driver)
        own = asyncio.run(driver.get_status())
    finally:
        cleanup = getattr(driver, "cleanup", None)
        if callable(cleanup):
            try:
                cleanup()
            except Exception:  # noqa: BLE001 - off hardware, teardown is best effort
                pass
    assert envelope["status"] in ("success", "error"), envelope
    payload = envelope["content"][0]["json"]
    # The payload is the driver's status dict, not another envelope.
    assert not ({"status", "content"} <= set(payload) and isinstance(payload.get("content"), list)), (
        f"{name}: the status verb nests an envelope inside its envelope"
    )
    # And it is the same shape ``get_status`` hands the mesh.
    assert set(payload) == set(own["content"][0]["json"]), name
