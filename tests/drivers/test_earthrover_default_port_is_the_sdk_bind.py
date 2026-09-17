"""With ``port`` omitted the EarthRover driver dials where the SDK listens.

The earth-rovers-sdk has one bind: its Dockerfile runs ``hypercorn main:app
--bind 0.0.0.0:8000``, every README example is ``localhost:8000`` and lerobot's
``earthrover_mini_plus`` config defaults ``sdk_url`` to the same. The driver
shipped with ``DEFAULT_SDK_URL`` on ``:8001``, so the documented
``Robot("earthrover", mode="real", driver="strands")`` with no ``port=`` sent
``GET /data`` to a port nothing serves and reported the SDK unreachable.
"""

from __future__ import annotations

import sys
import types
from typing import Any
from urllib.parse import urlsplit

import pytest

from strands_robots.drivers.earthrover import DEFAULT_SDK_URL, EarthRoverDriver

#: ``earth-rovers-sdk/Dockerfile``: ``--bind 0.0.0.0:8000``.
SDK_BIND_PORT = 8000


class _Recorder:
    def __init__(self) -> None:
        self.gets: list[str] = []

    def get(self, url: str, timeout: float = 0.0, **_: Any) -> Any:
        self.gets.append(url)
        return types.SimpleNamespace(status_code=200, text="", json=lambda: {"battery": 90})

    def close(self) -> None:
        pass


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    rec = _Recorder()
    fake = types.ModuleType("requests")
    fake.Session = lambda: rec  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "requests", fake)
    return rec


def test_default_sdk_url_is_the_port_the_sdk_binds() -> None:
    assert urlsplit(DEFAULT_SDK_URL).port == SDK_BIND_PORT


def test_omitted_port_dials_the_sdk_bind(recorder: _Recorder) -> None:
    driver = EarthRoverDriver()
    assert driver.connect_eagerly() is None
    assert recorder.gets == [f"http://localhost:{SDK_BIND_PORT}/data"]
