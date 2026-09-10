"""``create_policy("remote")`` without the [inference] extra must fail on the first
line that needs the transport with the extra named, not with a bare
``ModuleNotFoundError: websockets`` (measured on a [sim-mujoco,lerobot]-only venv).

Construction itself stays transport-free: the pre-wire contracts (port and host
domains, key-list refusals, handshake validation) are exercised by ~100 tests
that never open a socket, and they run on a venv without ``websockets``.
"""

from __future__ import annotations

import asyncio
import sys

import pytest

from strands_robots import utils
from strands_robots.inference import RemotePolicy


@pytest.fixture
def no_websockets(monkeypatch):
    monkeypatch.setitem(sys.modules, "websockets", None)  # makes ``import websockets`` raise ImportError
    monkeypatch.delitem(utils._lazy_modules, "websockets", raising=False)  # defeat require_optional's cache


def test_construction_does_not_need_the_transport(no_websockets):
    client = RemotePolicy(host="127.0.0.1", port=8765)
    assert client.uri == "ws://127.0.0.1:8765"


def test_first_use_without_websockets_names_the_extra(no_websockets):
    client = RemotePolicy(host="127.0.0.1", port=8765)
    with pytest.raises(ImportError, match=r"strands-robots\[inference\]") as info:
        asyncio.run(client.get_actions({"observation.state": [0.0]}, instruction="reach"))
    assert info.value.name == "websockets"
