# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A demo that mounts a camera records that camera, not the free overview view.

``start_recording`` with no ``cameras=`` records every view the scene can
render, which always includes the implicit ``default`` free camera the world
carries whether or not anybody asked for it. The recorder says so at WARNING
level - "The 'default' view is not a sensor any policy declares; it bloats the
dataset and will not match a policy's input_features" - and two shipped demos
ignored their own library's advice.

Measured on ``5b445066``, before this module. ``examples/03_record_dataset.py``
mounts one camera (``front``) and its dataset lands with two image features,
``observation.images.default`` beside ``observation.images.front`` (628 KB for
100 frames instead of 364 KB), under a docstring that calls the result
"training-ready". ``examples/07_post_tune_any_policy.py`` mounts the same one
camera and then trains on what it recorded, so the phantom view becomes a
declared input of the exported checkpoint::

    input_features:
      observation.images.default {'type': 'VISUAL', 'shape': [3, 480, 640]}
      observation.images.front   {'type': 'VISUAL', 'shape': [3, 480, 640]}

and the artifact that "closes the loop" is then refused by any robot outside a
simulator - a ``front``-only observation raises ``Robot supplies 1 camera(s)
['front'] but the policy requires image input(s) [...]; unmatched policy keys:
['observation.images.default']``. Scoping the recording to the mounted sensor
leaves one visual input, and the same observation is accepted.

The rule graded here is the narrow one the recorder's own warning states: a
script that mounts its own sensors names them when it records. A demo that
mounts nothing is not covered - there the overview view is the only view there
is, and recording it is a choice rather than an oversight.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import strands_robots

_EXAMPLES = Path(strands_robots.__file__).resolve().parent.parent / "examples"


def _call_names(tree: ast.AST) -> list[tuple[str, ast.Call]]:
    """Every call in ``tree``, paired with the attribute or function name called."""
    found: list[tuple[str, ast.Call]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        found.append((name, node))
    return found


def _recording_demos() -> list[str]:
    """Example scripts that both mount a camera and start a recording."""
    demos = []
    for path in sorted(_EXAMPLES.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - a demo that does not parse is another test's business
            continue
        names = [name for name, _ in _call_names(tree)]
        if "add_camera" in names and "start_recording" in names:
            demos.append(str(path.relative_to(_EXAMPLES.parent)))
    return demos


DEMOS = _recording_demos()


def test_the_examples_tree_still_has_recording_demos_to_grade() -> None:
    """A rule with nothing left to grade would pass by vacuum."""
    assert DEMOS, f"no example under {_EXAMPLES} both mounts a camera and records"


@pytest.mark.parametrize("relpath", DEMOS)
def test_a_demo_that_mounts_a_camera_records_that_camera(relpath: str) -> None:
    tree = ast.parse((_EXAMPLES.parent / relpath).read_text(encoding="utf-8"))
    mounted = sorted(
        {
            kw.value.value
            for name, call in _call_names(tree)
            if name == "add_camera"
            for kw in call.keywords
            if kw.arg == "name" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str)
        }
    )
    for name, call in _call_names(tree):
        if name != "start_recording":
            continue
        scoped = [kw for kw in call.keywords if kw.arg == "cameras"]
        assert scoped, (
            f"{relpath}:{call.lineno} start_recording does not pass cameras=, so it also records the "
            f"implicit 'default' overview view beside the mounted sensor(s) {mounted}; a policy trained "
            "on that dataset declares an input no robot outside the simulator supplies"
        )
