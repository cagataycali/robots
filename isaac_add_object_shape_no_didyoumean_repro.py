"""Minimal repro: Isaac add_object's shape refusal has no 'Did you mean'
clause, while the two sibling backends (MuJoCo spec_builder._geom_type and
Newton add_object) both do for the same user action.

Static source inspection suffices to pin the asymmetry (an Isaac runtime is
not required here).

Run as:
    python isaac_add_object_shape_no_didyoumean_repro.py

Expected (post-fix): a Did-you-mean clause appears on the Isaac path for
typos such as 'spehre' / 'cilinder' / 'capsul'.

Actual (v0.5.3, upstream HEAD a637553ed): bare 'Valid: (...)' tuple is
printed; sibling MuJoCo (post-#726) and Newton both suggest. The helper
``did_you_mean`` is already imported by a sibling file in the Isaac tree
(strands_robots/simulation/isaac/joint_names.py:41), so the fix is a
3-LOC mirror of newton/simulation.py:1006-1007.
"""
from __future__ import annotations

import difflib
import os
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "strands_robots" / "simulation"

ISAAC = (ROOT / "isaac" / "simulation.py").read_text()
NEWTON = (ROOT / "newton" / "simulation.py").read_text()
MUJOCO = (ROOT / "mujoco" / "spec_builder.py").read_text()


def _refusal_window(src: str, raise_anchor: str, lead: int = 10) -> str:
    """Return the ~15 lines leading up to a refusal site."""
    idx = src.find(raise_anchor)
    assert idx >= 0, f"anchor {raise_anchor!r} not found"
    # Walk back ``lead`` newlines so we capture the pre-raise difflib call.
    start = idx
    for _ in range(lead):
        start = src.rfind("\n", 0, start)
        if start < 0:
            start = 0
            break
    end = src.find("\n", idx + len(raise_anchor))
    return src[start : end if end > 0 else idx + 300]


def _window_suggests(window: str) -> bool:
    return bool(re.search(r"get_close_matches|did_you_mean|Did you mean", window))


# MuJoCo path (post-#726 fix in spec_builder.py:_geom_type)
mujoco_window = _refusal_window(MUJOCO, "Unsupported shape {shape!r}")
mujoco_has_hint = _window_suggests(mujoco_window)

# Newton path (newton/simulation.py add_object)
newton_window = _refusal_window(NEWTON, "Unsupported shape {shape!r}")
newton_has_hint = _window_suggests(newton_window)

# Isaac path (isaac/simulation.py add_object validate shape)
isaac_window = _refusal_window(ISAAC, "Unknown shape: {shape!r}")
isaac_has_hint = _window_suggests(isaac_window)

print(f"MuJoCo add_object shape refusal has did-you-mean: {mujoco_has_hint}")
print(f"Newton add_object shape refusal has did-you-mean: {newton_has_hint}")
print(f"Isaac  add_object shape refusal has did-you-mean: {isaac_has_hint}")

# Reproduce the user-visible message gap for a scoring typo on each backend.
isaac_shapes = ("box", "sphere", "capsule", "cylinder", "mesh")
newton_shapes = ("box", "sphere", "capsule", "cylinder", "mesh")
mujoco_shapes = sorted(
    {"box", "sphere", "cylinder", "capsule", "ellipsoid", "mesh", "plane"}
)

typo = "spehre"

print()
print("-- Isaac (current) refusal for shape=%r --" % typo)
accepted = isaac_shapes + ("cuboid",)
print(f"  Unknown shape: {typo!r}. Valid: {accepted}")

print()
print("-- Newton (sibling) refusal for shape=%r --" % typo)
close = difflib.get_close_matches(typo.lower(), list(newton_shapes), n=1, cutoff=0.6)
hint = f" Did you mean {close[0]!r}?" if close else ""
print(
    f"  Unsupported shape {typo!r} for Newton backend.{hint} "
    f"Supported: {', '.join(newton_shapes)}."
)

print()
print("-- MuJoCo (sibling, post-#726) refusal for shape=%r --" % typo)
close = difflib.get_close_matches(typo.lower(), list(mujoco_shapes), n=1, cutoff=0.6)
hint = f" Did you mean {close[0]!r}?" if close else ""
print(
    f"  Unsupported shape {typo!r}.{hint} Supported: {', '.join(mujoco_shapes)}."
)

# ---------------------------------------------------------------------------
# PIN: static asymmetry.
#
# On upstream v0.5.3 (strands-labs/robots HEAD a637553ed):
#   - MuJoCo and Newton both carry Did-you-mean on their shape refusal.
#   - Isaac is the lone backend without one.
#
# On the branch bugbash/isaac-add-object-shape-no-didyoumean (3-LOC fix):
#   - Isaac now carries the hint too (asymmetry is gone).
#
# Set STRANDS_BUGBASH_EXPECT_FIX=1 to pin the post-fix shape instead.
# ---------------------------------------------------------------------------
assert mujoco_has_hint, (
    "MuJoCo already fixed (#726); if this flips, the fossil has regressed. "
    "window=\n" + mujoco_window
)
assert newton_has_hint, (
    "Newton already carries did-you-mean (newton/simulation.py:1006). "
    "window=\n" + newton_window
)

if os.environ.get("STRANDS_BUGBASH_EXPECT_FIX") in {"1", "true", "yes"}:
    assert isaac_has_hint, (
        "Isaac was expected to carry the hint post-fix, but it does not. "
        "window=\n" + isaac_window
    )
    print()
    print("OK (post-fix): all three backends offer a Did-you-mean on add_object.")
else:
    assert not isaac_has_hint, (
        "Isaac add_object shape refusal unexpectedly contains did-you-mean - "
        "if this fails, the defect is fixed and the test should be inverted "
        "(set STRANDS_BUGBASH_EXPECT_FIX=1). window=\n" + isaac_window
    )
    print()
    print(
        "OK (pre-fix): Isaac add_object shape refusal is the lone backend "
        "without a Did-you-mean hint. Fix is a 3-LOC mirror of "
        "newton/simulation.py:1006-1007."
    )

# Also: the helper did_you_mean is already imported by a sibling isaac file,
# proving the Isaac tree already carries the dependency.
sibling = (ROOT / "isaac" / "joint_names.py").read_text()
assert "did_you_mean" in sibling, (
    "did_you_mean import drift in isaac/joint_names.py - rebase check."
)

# And ``difflib`` is already imported by newton/simulation.py, which is the
# exact 3-LOC lift the Isaac fix mirrors:
assert "import difflib" in NEWTON, "newton/simulation.py drift - rebase check."
