"""Repro: docs/start/first-robot.md envelope paragraph forgets list_robots() and list_cameras().

Post-#691 (closed), docs/start/first-robot.md:84 reads:

    > Every call but `get_observation()` and `cleanup()` (`None`) returns the same
    > envelope: `status` and a `content` list of `text`, `json` or `image` blocks.

Two sections above (line 15) the SAME page prints `list_robots()`:

    print(robot.list_robots())
    # expected printout: ['so101']

That's a bare `list[str]`, not an envelope. The paragraph forgot it. Same thing
happens for `list_cameras()` from the README quickstart (the user who follows the
headline snippet then naturally reaches for `robot.list_cameras()` to verify the
`add_camera(name="front")` call succeeded).

The signature is enshrined at the ABC level: `base.py:1412 def list_robots(self) -> list[str]`,
and mujoco/rendering.py:2233 `def list_cameras(self) -> list[str]` + three sibling
backends sign it the same way.

A reader following the paragraph writes `robot.list_robots()["status"]` and hits:

    TypeError: list indices must be integers or slices, not str

Fix mirrors #691: name the two additional exceptions in the paragraph (zero code
change). No shape change — these ABC return types are stable and shouldn't
narrow for a line of prose.

Run (no GPU, no network):

    MUJOCO_GL=egl python bugbash_repros/envelope_paragraph_forgets_list_robots_and_list_cameras_repro.py
"""

from __future__ import annotations

import inspect
import os
import sys


def main() -> int:
    os.environ.setdefault("MUJOCO_GL", "egl")
    from strands_robots import Robot

    r = Robot("so101")

    # --- 1. list_robots: ABC-signed bare list[str] (base.py:1412) ---
    sig = inspect.signature(type(r).list_robots)
    annot = sig.return_annotation
    print(f"list_robots signature return: {annot}")
    assert str(annot) == "list[str]", f"expected 'list[str]', got {annot!r}"

    out = r.list_robots()
    print(f"list_robots() -> type={type(out).__name__} value={out!r}")
    assert isinstance(out, list), f"expected list, got {type(out).__name__}"
    assert all(isinstance(x, str) for x in out), "expected all str elements"
    # Follow the paragraph literally:
    try:
        _ = out["status"]  # type: ignore[index]
        print("UNEXPECTED: list_robots()['status'] did not raise")
        return 1
    except TypeError as e:
        print(f"list_robots()['status'] -> TypeError: {e}")

    # --- 2. list_cameras: backend-signed bare list[str] (mujoco/rendering.py:2233) ---
    cams_fn = type(r).list_cameras
    cams_sig = inspect.signature(cams_fn)
    print(f"list_cameras signature return: {cams_sig.return_annotation}")
    assert str(cams_sig.return_annotation) == "list[str]", (
        f"expected 'list[str]', got {cams_sig.return_annotation!r}"
    )

    cams = r.list_cameras()
    print(f"list_cameras() -> type={type(cams).__name__} value={cams!r}")
    assert isinstance(cams, list)
    try:
        _ = cams["status"]  # type: ignore[index]
        print("UNEXPECTED: list_cameras()['status'] did not raise")
        return 1
    except TypeError as e:
        print(f"list_cameras()['status'] -> TypeError: {e}")

    # --- 3. list_objects and list_bodies DO return the envelope ---
    objs = r.list_objects()
    print(f"list_objects() -> type={type(objs).__name__} status={objs.get('status') if isinstance(objs, dict) else '?'}")
    assert isinstance(objs, dict) and "status" in objs, "list_objects IS envelope"
    assert isinstance(objs.get("content"), list), "list_objects content is list"

    bodies = r.list_bodies()
    print(f"list_bodies() -> type={type(bodies).__name__} status={bodies.get('status') if isinstance(bodies, dict) else '?'}")
    assert isinstance(bodies, dict) and "status" in bodies, "list_bodies IS envelope"

    # --- 4. list_cameras_info IS envelope (companion exists for cameras) ---
    cams_info = r.list_cameras_info()
    print(f"list_cameras_info() -> type={type(cams_info).__name__} status={cams_info.get('status') if isinstance(cams_info, dict) else '?'}")
    assert isinstance(cams_info, dict) and "status" in cams_info

    # --- 5. The paragraph claim site: docs/start/first-robot.md (post-#691) ---
    import pathlib
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    doc_path = repo_root / "docs" / "start" / "first-robot.md"
    if doc_path.exists():
        text = doc_path.read_text(encoding="utf-8")
        # Two acceptable shapes: pre-fix (over-promising, filed as this issue) and
        # post-fix (names list_robots + list_cameras as additional exceptions).
        pre_fix = "Every call but `get_observation()` and `cleanup()` (`None`) returns the same envelope"
        post_fix = "list_robots()` / `list_cameras()`"
        if pre_fix in text and post_fix not in text:
            print("paragraph state: PRE-FIX (defect present)")
        elif post_fix in text:
            print("paragraph state: POST-FIX (defect fixed by this branch)")
        else:
            print(f"paragraph state: UNKNOWN - page shape changed")
            return 1
        # The two offenders are used ON the same page (line 16 of the first code block)
        assert "robot.list_robots()" in text, (
            "list_robots() must be present on same page to prove the omission"
        )
    else:
        print(f"(doc not shipped at {doc_path}; skipping doc assertion)")

    print()
    print("FINDING: docs/start/first-robot.md envelope paragraph forgets list_robots() + list_cameras()")
    print("         - both are ABC/backend-signed `-> list[str]` and return bare lists")
    print("         - list_robots() is literally USED on the same page, 68 lines above the paragraph")
    print("         - fix mirrors #691: name the two additional exceptions in the paragraph")
    return 0


if __name__ == "__main__":
    sys.exit(main())
