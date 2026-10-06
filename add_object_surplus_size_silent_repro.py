"""Repro: add_object() silently discards surplus `size` components for shapes
with fewer than 3 consumed slots (sphere / cylinder / capsule / plane).

Context
-------
``MuJoCoSimEngine.add_object`` docstring (simulation.py:4860-4867) explicitly
calls out partial `size` vectors as a footgun and refuses them::

    A partial vector (``size=[0.5]`` on a box) is rejected rather than
    completed from a backend default, because a completed vector compiles a
    differently-sized object while reporting success.

The matching rejection path lives in ``spec_builder._validate_size``:

    if len(size) < required:
        return "add_object: {shape} needs {required} 'size' component(s)..."

The symmetric case -- a vector with MORE components than the shape consumes --
is NOT rejected: ``_MAX_SIZE_COMPONENTS == 3`` and the only upper bound is a
blanket "<= 3". A sphere consumes `size[0]` only (`_SIZE_LAYOUT["sphere"]` =
`(1, "[diameter]")`); the trailing components are silently discarded by
`_normalize_size`.

The user-visible effect is identical to the one the "partial vector" rule was
written to prevent: an input that cannot be honored in full compiles something
else and returns status=success. In particular,

    add_object(name=..., shape="sphere", size=[D, X, Y])

always compiles a uniform sphere of diameter `D`, no matter what X and Y are.
A caller trying to express asymmetric extents (natural tab-complete off a
sibling box call, natural adaptation from an ellipsoid snippet, LLM mimicking
a 3-vector pattern from the README quickstart) gets a sphere and no warning.

The success-text echo (`_compiled_geometry_detail`, simulation.py:400) reads
back the AABB, so a diligent reader sees `size=[D, D, D]` for a sphere -- but
the status is `success` and no part of the response calls out that the input
components beyond index 0 were discarded.

Verifier
--------
This script runs five calls that currently return status=success but which
either (a) describe a differently-sized object than requested or (b) encode a
request the shape cannot carry. The output table lets a reviewer confirm the
drift at a glance. The script exits with code 1 whenever a surplus-vector call
is accepted silently -- a passing exit means the asymmetry has been closed.

Run
---
    MUJOCO_GL=egl python add_object_surplus_size_silent_repro.py

Expected (currently):
    All five rows report status=success; input != compiled extent on four of
    them; the fifth (sphere with uniform surplus) matches by coincidence of
    numbers but still carries two components the shape never read. Exit 1.

After fix:
    Each surplus call returns status=error with a shape-specific message
    naming the exact component count the shape consumes, symmetric with the
    partial-vector rejection. Exit 0.
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    os.environ.setdefault("MUJOCO_GL", "egl")

    from strands_robots import Robot

    robot = Robot("so100")

    # (shape, user-supplied size, what the shape actually consumes,
    #  silent-when-surplus?)
    #
    # cylinder/capsule carry a documented "unused middle component"
    # (``_SIZE_LAYOUT[...] = (3, "[diameter, unused, full height]")``) so a
    # 3-vector is the SHAPE'S OWN POSITIONAL FORM, not surplus. They are kept
    # in the table as a reminder that the rejection must NOT regress onto
    # them; the ``silent_surplus`` column is False and the pass expects them
    # to continue returning status=success.
    cases = [
        ("sphere",    [0.05, 0.10, 0.20], "size[0] only ([diameter])",                     True),
        ("sphere",    [0.20, 0.05, 0.05], "size[0] only ([diameter])",                     True),
        ("cylinder",  [0.05, 0.10, 0.20], "size[0] and size[2] (middle is documented)",    False),
        ("capsule",   [0.05, 0.10, 0.20], "size[0] and size[2] (middle is documented)",    False),
        ("plane",     [1.0, 2.0, 3.0],    "size[0] and size[1] ([x, y] half-widths)",      True),
    ]

    header = f"{'shape':9} | {'input size':24} | {'status':8} | {'echoed size (AABB)':30} | consumed slots"
    print(header)
    print("-" * len(header))

    regression = False
    silent_accepted = False
    for i, (shape, size, consumed, silent_surplus) in enumerate(cases):
        name = f"repro_{shape}_{i}"
        kwargs: dict = {
            "name": name,
            "shape": shape,
            "size": size,
            "color": [1.0, 0.0, 0.0],
        }
        # Plane cannot be dynamic; everything else lands at z=0.5 so it does
        # not collide with the ground on recompile.
        if shape == "plane":
            kwargs["is_static"] = True
            kwargs["position"] = [0.0, -0.5, 0.0]
        else:
            kwargs["position"] = [0.0, -0.5 + 0.1 * i, 0.5]

        r = robot.add_object(**kwargs)
        status = r.get("status", "?")
        text = (r.get("content") or [{}])[0].get("text", "")
        # Extract the "size=[...]" clause from the error/success text, if any.
        echoed = ""
        if "size=" in text:
            tail = text.split("size=", 1)[1]
            if tail.startswith("["):
                end = tail.find("]")
                if end >= 0:
                    echoed = "[" + tail[1:end] + "]"
        if echoed == "":
            echoed = "(absent)"
        print(f"{shape:9} | {str(size):24} | {status:8} | {echoed:30} | {consumed}")

        if silent_surplus and status == "success":
            silent_accepted = True
        if not silent_surplus and status != "success":
            # Would mean the fix regressed onto a documented 3-component call.
            regression = True

    print()
    if silent_accepted:
        print(
            "FAIL: a surplus-component `size` was accepted silently.\n"
            "      Partial vectors are rejected by spec_builder._validate_size\n"
            "      (`len(size) < required`), but surplus vectors with `len(size)\n"
            "      <= _MAX_SIZE_COMPONENTS == 3` pass. The symmetric check\n"
            "      `len(size) > required` for shapes whose `_SIZE_LAYOUT`\n"
            "      consumed-count is below 3 would close the gap and match the\n"
            "      rejection message the partial-vector path already uses."
        )
        return 1
    if regression:
        print(
            "FAIL: the fix regressed onto a shape with a documented three-\n"
            "      component form (cylinder/capsule's `[diameter, unused,\n"
            "      full height]`). The rejection must only cover shapes whose\n"
            "      `_SIZE_LAYOUT` consumed-count is below three."
        )
        return 1
    print("PASS: surplus `size` is rejected on sphere/plane; cylinder/capsule still accept their documented 3-vector form.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
