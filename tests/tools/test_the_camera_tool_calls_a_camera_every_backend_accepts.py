"""Every camera call the tool makes must bind on every camera it can be handed.

``lerobot_camera`` opens a device through ``_create_camera``, which returns an
``OpenCVCamera`` or a ``RealSenseCamera`` behind the ``Camera`` contract it
declares, and a handler does not know which one it holds. So every call it makes
has to be one all of them accept.

Nothing checked that. lerobot ships no ``py.typed``, so every camera symbol this
module imports is ``Any`` to mypy; and each test that drives a handler replaces
the factory with a stand-in, which can only grade the calls its own cells reach.
A read spelled with a keyword only one backend declares - or the ``timeout_ms``
that was once dropped from the asynchronous read - would reach a device and fail
there.

This pin reads the call sites out of the module instead of driving them: every
camera the module builds, from the factory or from a backend class directly, and
every call made on it, bound against the signature of the method it names. The
receiver has to be resolvable from an assignment, which is how this module
builds cameras; ``strands_robots.hardware_robot`` reaches its cameras by
iterating lerobot's own ``cameras`` mapping, so its calls are not in this basis
and the non-vacuity floor below is over this module alone.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
from dataclasses import dataclass
from typing import Any

import pytest

import strands_robots.tools.lerobot_camera as cam_mod
from tests.tools._camera_stand_in import CONTRACTS

#: The module whose camera calls are graded here.
_SOURCE = pathlib.Path(cam_mod.__file__)

#: Every way this module gets hold of a camera, and the contracts a call on it
#: must satisfy. A camera from the factory can be either backend, so it must
#: satisfy both and the type the factory declares; one built from a backend class
#: directly is that backend.
_SOURCES: dict[str, tuple[Any, ...]] = {
    "_create_camera": CONTRACTS,
    "OpenCVCamera": (cam_mod.OpenCVCamera,),
    "RealSenseCamera": (cam_mod.RealSenseCamera,),
}

#: What a handler does with a camera, so an omission in the scan is visible.
_EXPECTED_METHODS = frozenset({"connect", "disconnect", "read", "async_read"})

#: Camera calls in the module when this pin was written. A scan rooted at the
#: wrong node, or a receiver shape it cannot resolve, reads as compliance
#: otherwise.
_MINIMUM_CALLS = 20

#: The builders the module currently calls a camera through. A RealSense camera is
#: built inside the factory and returned, never assigned and driven here, so it is
#: a contract this scan must honor rather than a source it must find.
_RESOLVED_SOURCES = frozenset({"_create_camera", "OpenCVCamera"})


@dataclass(frozen=True)
class CameraCall:
    """One call on a camera, as the module spells it."""

    line: int
    source: str
    method: str
    positional: int
    keywords: tuple[str, ...]

    def __str__(self) -> str:
        spelling = ", ".join(["<value>"] * self.positional + [f"{name}=..." for name in self.keywords])
        return f"{_SOURCE.name}:{self.line} {self.source} camera.{self.method}({spelling})"


def _camera_calls() -> list[CameraCall]:
    """Every call on a camera this module builds, in source order."""
    tree = ast.parse(_SOURCE.read_text(encoding="utf-8"))
    calls: list[CameraCall] = []
    for scope in ast.walk(tree):
        if not isinstance(scope, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        cameras: dict[str, str] = {}
        for node in ast.walk(scope):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                builder = node.value.func
                if isinstance(builder, ast.Name) and builder.id in _SOURCES:
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            cameras[target.id] = builder.id
        for node in ast.walk(scope):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            receiver = node.func.value
            if not isinstance(receiver, ast.Name) or receiver.id not in cameras:
                continue
            calls.append(
                CameraCall(
                    line=node.lineno,
                    source=cameras[receiver.id],
                    method=node.func.attr,
                    positional=len(node.args),
                    keywords=tuple(keyword.arg or "**" for keyword in node.keywords),
                )
            )
    # A nested handler is walked once as itself and once inside its enclosing
    # function, so the same call is reached twice; it is one claim either way.
    return sorted(dict.fromkeys(calls), key=lambda call: call.line)


CAMERA_CALLS = _camera_calls()


def test_the_scan_found_the_calls_it_grades() -> None:
    """A basis that resolved nothing would pass every cell below."""
    assert len(CAMERA_CALLS) >= _MINIMUM_CALLS, [str(call) for call in CAMERA_CALLS]
    assert {call.method for call in CAMERA_CALLS} == _EXPECTED_METHODS
    sources = {call.source for call in CAMERA_CALLS}
    assert _RESOLVED_SOURCES <= sources <= set(_SOURCES), sources
    assert len(CONTRACTS) == 3, [contract.__name__ for contract in CONTRACTS]


@pytest.mark.parametrize("call", CAMERA_CALLS, ids=str)
def test_a_camera_call_binds_on_every_camera_it_can_reach(call: CameraCall) -> None:
    for contract in _SOURCES[call.source]:
        method = getattr(contract, call.method, None)
        assert method is not None, f"{contract.__name__} declares no {call.method}"
        arguments = [object()] * (call.positional + 1)  # + the receiver
        inspect.signature(method).bind(*arguments, **dict.fromkeys(call.keywords, object()))


def test_the_factory_is_called_the_way_it_declares() -> None:
    """The seam itself: every ``_create_camera`` call binds on its signature."""
    tree = ast.parse(_SOURCE.read_text(encoding="utf-8"))
    signature = inspect.signature(cam_mod._create_camera)
    sites = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_create_camera"
    ]
    assert len(sites) >= 6, [site.lineno for site in sites]
    for site in sites:
        arguments = [object()] * len(site.args)
        keywords = {keyword.arg or "**": object() for keyword in site.keywords}
        signature.bind(*arguments, **keywords)
