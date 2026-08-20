"""Is the REAL ``device_connect_edge`` package available to this venv?

Two test modules in this directory exercise the genuine Device Connect edge SDK -
``test_device_connect_hardening.py`` says so in its own docstring, because the
``@rpc`` caller-identity contextvar hook it pins only exists in the real package -
and several OTHER modules replace that package with ``MagicMock`` in
``sys.modules`` at import time. Those two facts make "can I import it?" the wrong
question: in a sweep, a sibling's mock answers yes, and a security test that passes
against a MagicMock has proven nothing at all.

So this asks the FILESYSTEM instead. ``PathFinder`` walks ``sys.path`` and never
consults ``sys.modules``, which is exactly the property needed here -
``importlib.util.find_spec`` would return the mock's auto-created ``__spec__``
attribute and report success.
"""

from __future__ import annotations

import sys
from importlib.machinery import PathFinder

#: Why the SDK-backed tests cannot run here - empty string when they can.
MISSING_REASON = (
    "the real device_connect_edge package is not installed in this venv, and these "
    "tests exercise it directly (the sibling modules' MagicMocks cannot stand in: a "
    "security test that passes against a mock has proven nothing). Install the SDK "
    "editable - pip install -e path/to/device-connect-edge - to run them."
)


def real_device_connect_edge_on_disk() -> bool:
    """True when the genuine package can be found on ``sys.path``.

    Deliberately blind to ``sys.modules``, so a sibling test's mock cannot make
    this say yes.
    """
    try:
        return PathFinder().find_spec("device_connect_edge") is not None
    except (ImportError, ValueError):
        return False


def skip_reason_if_sdk_missing() -> str:
    """``MISSING_REASON`` when the SDK is absent, otherwise an empty string."""
    return "" if real_device_connect_edge_on_disk() else MISSING_REASON
