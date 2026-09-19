"""A captured frame lands inside the frame root, whatever the agent asked for.

``save_path`` on the ``camera`` verb is a tool parameter a model fills in, and
the value reaches a filesystem write (``cv2.imwrite``) with this process's
privileges. Unconfined, a prompt-injected caller overwrites any file the process
can write with JPEG bytes - a shell profile, a cron fragment - without leaving
the documented API at all.

So the parameter names a file *inside* :func:`~strands_robots.drivers.microduck.frame_root`
and nothing else, checked through the package's own path sandbox
(:mod:`strands_robots._path_validation`) before mediad is dialed. Containment
rather than a character allowlist is the guarantee, for the reason recorded with
that module: the resolved target has to lie under the resolved root, which is
exactly statable, while an allowlist has to guess which spellings are dangerous
and refuses working names.

The refusal is asserted to land *before* the transport: these cases point the
driver at a socket that does not exist, so a case that reached the dial would
report mediad's absence instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from strands_robots.drivers.microduck import FRAME_ROOT_ENV, MicroduckDriver

#: Values that must not be written to, with what each one is trying to reach.
ESCAPES: list[str] = [
    "/etc/cron.d/strands",  # absolute, a protected system directory
    "/var/tmp/operator-notes.png",  # absolute, an ordinary file the process can write
    "../../../.bashrc",  # traversal out of the root
    "sub/../../escape.jpg",  # traversal through a legitimate-looking prefix
    "\x00frame.jpg",  # a NUL byte, which truncates the path for the C layer
]


def _camera(driver: MicroduckDriver, **params: Any) -> dict[str, Any]:
    from strands_robots.drivers.microduck import _act_camera

    return _act_camera(driver, params)


def _text(envelope: dict[str, Any]) -> str:
    return " ".join(block.get("text", "") for block in envelope["content"])


@pytest.fixture
def driver(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MicroduckDriver:
    """A driver with the frame root at ``tmp_path`` and no mediad to dial."""
    monkeypatch.setenv(FRAME_ROOT_ENV, str(tmp_path))
    duck = MicroduckDriver(port="/nowhere.sock", timeout=0.5)
    duck._media_socket = str(tmp_path / "absent.sock")
    return duck


@pytest.mark.parametrize("value", ESCAPES, ids=repr)
def test_a_path_that_leaves_the_frame_root_is_refused_before_the_dial(
    driver: MicroduckDriver, tmp_path: Path, value: str
) -> None:
    envelope = _camera(driver, save_path=value)

    assert envelope["status"] == "error"
    assert "save_path" in _text(envelope) and str(tmp_path) in _text(envelope)
    assert "mediad" not in _text(envelope), "the escape was carried as far as the transport"


def test_a_name_inside_the_root_is_accepted_and_only_the_transport_refuses(driver: MicroduckDriver) -> None:
    """The premise: the guard refuses the escape, not every name."""
    envelope = _camera(driver, save_path="sub/frame.jpg")

    assert envelope["status"] == "error"
    assert "mediad" in _text(envelope), "a contained name was refused by the guard"
