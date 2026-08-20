"""The two device_connect test modules must not sabotage each other.

Both mock ``device_connect_edge`` in ``sys.modules`` at import time and restore it
in ``teardown_module``. Pytest imports every collected module BEFORE running any
test and tears down per FILE, so whichever runs second used to find the mocks
gone - and 30 tests failed with ``ModuleNotFoundError: No module named
'device_connect_edge'`` in a file that passes perfectly alone.

That is the worst shape a test failure can take: it appears only in a sweep, it
names a module nobody touched, and it makes the suite useless as a gate, because a
real regression cannot be told apart from the noise. So the pairing is pinned here,
in BOTH orders, as a subprocess run - the only way one pytest session can assert
something about another session's collection order.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_A = "tests/test_device_connect_all_robots.py"
_B = "tests/test_device_connect_drivers.py"


def _run(*files: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "pytest", *files, "-q", "-p", "no:cacheprovider", "--no-cov"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )


@pytest.mark.parametrize("order", [(_A, _B), (_B, _A)], ids=["all_robots_first", "drivers_first"])
def test_the_pair_passes_in_either_order(order: tuple[str, str]) -> None:
    done = _run(*order)
    tail = "\n".join(done.stdout.strip().splitlines()[-15:])
    assert done.returncode == 0, f"{' then '.join(order)} failed:\n{tail}"
    assert "No module named 'device_connect_edge'" not in done.stdout, (
        "a sibling's teardown removed the mocked module while this one still needed it:\n" + tail
    )
