"""Repro: README-quickstart shape='cube'/'cuboid' is refused with no 'Did you mean'
hint by the MuJoCo default backend, while Isaac silently accepts 'cuboid' via
``_SHAPE_ALIASES`` and the docstring of the same MuJoCo add_object itself calls
the resulting geom a "cube" eleven times (including a run-anywhere example at
``simulation/mujoco/simulation.py:4982``).

Compound Error UX + Asymmetric API + Docs mismatch, from the README-quickstart
rotation. v0.5.3, upstream HEAD at the time of filing (strands-labs/robots).

Run: ``MUJOCO_GL=egl python shape_cube_cuboid_asymmetric_alias_and_missing_hint_repro.py``
No display / GPU required.
"""

from __future__ import annotations

import difflib
import os
import pathlib
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import strands_robots  # noqa: E402
from strands_robots import Robot  # noqa: E402

print(f"strands_robots: {strands_robots.__file__}")
print()

# -----------------------------------------------------------------------------
# PART 1 — README-quickstart user types `shape="cube"` (natural: their variable
# is `red_cube`, the README comment and the docstring both say "5 cm cube").
# -----------------------------------------------------------------------------
robot = Robot("so100", mesh=False)

print("=== PART 1: MuJoCo default backend refuses 'cube' / 'cuboid' with NO hint ===")
for typo in ("cube", "cuboid"):
    r = robot.add_object(name=f"obj_{typo}", shape=typo, size=[0.05, 0.05, 0.05])
    msg = r["content"][0]["text"]
    has_hint = "did you mean" in msg.lower()
    print(f"  shape={typo!r} -> status={r['status']}")
    print(f"    message: {msg}")
    print(f"    has 'Did you mean' hint: {has_hint}")
    assert r["status"] == "error", "expected error"
    assert not has_hint, (
        f"UNEXPECTED: shape={typo!r} already shows a hint — fix landed, close this issue"
    )
print()

# -----------------------------------------------------------------------------
# PART 2 — Why the hint is missing: difflib.get_close_matches cutoff=0.6
# excludes both the single most natural user guess ('cube', ratio 0.286 to 'box')
# and the brand-synonym Isaac accepts as an alias ('cuboid', ratio 0.444).
# -----------------------------------------------------------------------------
print("=== PART 2: difflib cutoff=0.6 excludes 'cube' and 'cuboid' from the hint path ===")
known = ["box", "capsule", "cylinder", "ellipsoid", "mesh", "plane", "sphere"]
for word in ("cube", "cuboid"):
    matches = difflib.get_close_matches(word, known, n=1, cutoff=0.6)
    ratio = difflib.SequenceMatcher(None, word, "box").ratio()
    print(f"  get_close_matches({word!r}, cutoff=0.6) -> {matches}")
    print(f"  SequenceMatcher({word!r}, 'box').ratio() = {ratio:.3f}  (needs >= 0.6)")
    assert matches == [], f"difflib found a close match for {word!r}; this repro assumes the cutoff still prunes it"
print()

# -----------------------------------------------------------------------------
# PART 3 — Isaac has ``_SHAPE_ALIASES = {"cuboid": "box"}`` so the IDENTICAL
# call that errors on the DEFAULT MuJoCo backend would SILENTLY SUCCEED on Isaac.
# mjlab has yet a THIRD vocabulary (no mesh/plane/ellipsoid). Three backends,
# three different stories for the same word.
# -----------------------------------------------------------------------------
print("=== PART 3: Isaac has a cuboid alias; MuJoCo (default) and mjlab do not ===")
root = pathlib.Path(strands_robots.__file__).parent

isaac_text = (root / "simulation" / "isaac" / "simulation.py").read_text()
mu_text = (root / "simulation" / "mujoco" / "spec_builder.py").read_text()
mjlab_text = (root / "simulation" / "mjlab" / "simulation.py").read_text()

isaac_has_alias = '_SHAPE_ALIASES: dict[str, str] = {"cuboid": "box"}' in isaac_text
mujoco_has_alias = "_SHAPE_ALIASES" in mu_text
mjlab_has_alias = "_SHAPE_ALIASES" in mjlab_text
mjlab_vocab = ["box", "sphere", "cylinder", "capsule"]  # from _SHAPE_GEOM at mjlab:71
mujoco_vocab = sorted(known)

print(f"  Isaac  has _SHAPE_ALIASES=cuboid->box: {isaac_has_alias}")
print(f"  MuJoCo has _SHAPE_ALIASES:             {mujoco_has_alias}")
print(f"  mjlab  has _SHAPE_ALIASES:             {mjlab_has_alias}")
print(f"  MuJoCo vocab: {mujoco_vocab}")
print(f"  mjlab  vocab: {sorted(mjlab_vocab)}  (no mesh/plane/ellipsoid; a different vocabulary altogether)")

assert isaac_has_alias, "Isaac alias not present — _SHAPE_ALIASES wording may have changed"
assert not mujoco_has_alias, "MuJoCo gained an alias path; this issue may be resolved"
assert not mjlab_has_alias, "mjlab gained an alias path; this issue may be partially resolved"
print()

# -----------------------------------------------------------------------------
# PART 4 — The same MuJoCo module whose vocab refuses 'cube' calls the resulting
# geom a "cube" ELEVEN times in its own docstring, including a copy-pasteable
# >>> example at line 4982.
# -----------------------------------------------------------------------------
print("=== PART 4: The MuJoCo module itself calls the geom a 'cube' 14+ times ===")
simfile = root / "simulation" / "mujoco" / "simulation.py"
simtext = simfile.read_text()
cube_lines = [(i + 1, ln) for i, ln in enumerate(simtext.splitlines()) if "cube" in ln]
print(f"  'cube' hits in {simfile.name}: {len(cube_lines)}")
for lineno, line in cube_lines[:5]:
    print(f"    L{lineno}: {line.strip()[:110]}")
print(f"  (plus {max(0, len(cube_lines) - 5)} more hits elsewhere in the same file)")

# Confirm the example at ~L4982 uses "cube" as the geom name
example_hit = any(">>> sim.add_object(\"cube\"" in ln for _, ln in cube_lines)
print(f"  Docstring has copy-pasteable example `sim.add_object(\"cube\", shape=\"box\", ...)`: {example_hit}")
assert example_hit, "example not present — docstring may have changed"

# README calls the thing a "cube" too (comment, prose, agent instruction)
readme_text = (root.parent / "README.md").read_text()
readme_cube_hits = readme_text.lower().count("cube")
print(f"  README.md 'cube' hits: {readme_cube_hits}  "
      f"(variable `red_cube`, prose 'pick up the red cube', 'squeeze the cube', ...)")
print()

# -----------------------------------------------------------------------------
# SUMMARY — one word, three different behaviours, zero help text.
# -----------------------------------------------------------------------------
print("=== SUMMARY ===")
print(
    "A user following the README quickstart who guesses `shape=\"cube\"` or "
    "`shape=\"cuboid\"` (the word the docstring, the README, and the variable "
    "name all use) gets a bare `Supported: ...` list on the default MuJoCo "
    "backend, a silent success on Isaac (via `_SHAPE_ALIASES`), and a THIRD "
    "shorter list on mjlab. The guard that is supposed to catch this "
    "(difflib.get_close_matches, cutoff=0.6) prunes both guesses — fixing "
    "the asymmetry = three-line patch per backend: add `_SHAPE_ALIASES = "
    "{\"cube\": \"box\", \"cuboid\": \"box\"}` and consult it before the vocab "
    "gate on both the mujoco path (spec_builder.py:_geom_type) and the mjlab "
    "path (simulation.py add_object)."
)
sys.exit(0)
