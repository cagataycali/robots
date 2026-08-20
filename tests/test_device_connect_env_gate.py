"""The SDK gate must not be fooled by a sibling test's mock.

Modules in this directory install ``MagicMock``s for ``device_connect_edge`` in
``sys.modules``. If the gate consulted ``sys.modules`` (as ``importlib.util.find_spec``
does), a sweep would run the SDK-backed security tests against mocks and report
green - the worst possible outcome for a test whose whole job is proving a guard.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

from tests.device_connect_env import real_device_connect_edge_on_disk, skip_reason_if_sdk_missing


def test_a_mock_in_sys_modules_does_not_count(monkeypatch) -> None:
    truth = real_device_connect_edge_on_disk()
    monkeypatch.setitem(sys.modules, "device_connect_edge", MagicMock())
    assert real_device_connect_edge_on_disk() is truth, "the gate read sys.modules instead of the disk"


def test_the_reason_is_present_exactly_when_the_sdk_is_absent() -> None:
    if real_device_connect_edge_on_disk():
        assert skip_reason_if_sdk_missing() == ""
    else:
        assert "device_connect_edge" in skip_reason_if_sdk_missing()
        assert "pip install -e" in skip_reason_if_sdk_missing(), "a skip must say how to un-skip it"
