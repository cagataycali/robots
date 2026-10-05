"""Repro for strands-labs/robots v0.5.3

Defect: ``_unknown_robot_msg`` (``strands_robots/simulation/base.py:1262``
and the MuJoCo override at ``strands_robots/simulation/mujoco/simulation.py:2268``)
builds its "Did you mean" fragment with ``close_match_hint(requested, list_robots())``
only - it consults string-edit-distance against the names currently registered in
the world, never the alias table carried by
``strands_robots/registry/robots.json`` (where ``unitree_g1.aliases`` already
lists ``g1`` and the resolver ``strands_robots/registry/robots.py:55 resolve_name``
documents ``resolve_name("g1") -> "unitree_g1"``).

Result: a user who followed ``docs/robots/unitree_g1.md:19`` ("Aliases: g1,
g1_wbc, ...") and typed ``robot.robot_action_keys("g1")`` after
``Robot("unitree_g1")`` is handed a flat refusal with no "Did you mean", while
the much weaker typo ``"unitreeg1"`` (not promoted as an alias anywhere) *does*
get suggested because its character distance to the canonical name is below the
0.4 difflib cutoff and ``"g1"`` isn't.

The same difflib site at ``strands_robots/simulation/base.py:255 close_match_hint``
is also used by the model-error path at ``base.py:425`` where the sibling
diagnosis correctly calls ``resolve_name`` before difflib - the fix is already in
the file, just not plumbed into the robot-not-found path.
"""
import os, difflib
os.environ.setdefault("MUJOCO_GL", "egl")

from strands_robots import Robot
from strands_robots.registry.robots import resolve_name

print("=" * 60)
print("1) resolve_name knows the alias (direct registry evidence)")
print("=" * 60)
print(f"  resolve_name('g1')             -> {resolve_name('g1')!r}")
print(f"  resolve_name('g1_wbc')         -> {resolve_name('g1_wbc')!r}")
print(f"  resolve_name('unitree_g1_wbc') -> {resolve_name('unitree_g1_wbc')!r}")

print()
print("=" * 60)
print("2) But difflib@0.4 scores 'g1' below cutoff vs the only")
print("   registered name 'unitree_g1' -> close_match_hint returns ''")
print("=" * 60)
known = ["unitree_g1"]
for req in ("g1", "g1_wbc", "unitreeg1", "unitree_g1_wbc", "G1"):
    m = difflib.get_close_matches(req, known, n=4, cutoff=0.4)
    ratio = difflib.SequenceMatcher(None, req, known[0]).ratio()
    print(f"  {req!r:>20}: ratio={ratio:.3f}  difflib@0.4 -> {m}")

print()
print("=" * 60)
print("3) End-user flow: canonical build + documented alias")
print("=" * 60)
r = Robot("unitree_g1")
print(f"  list_robots() -> {r.list_robots()}")
# docs/robots/unitree_g1.md:19 lists 'g1' as an alias
for arg in ("g1", "g1_wbc", "unitree_g1_wbc", "unitreeg1"):
    try:
        r.robot_action_keys(arg)
        print(f"  robot_action_keys({arg!r}): OK (unexpected)")
    except ValueError as e:
        s = str(e)
        has_hint = "Did you mean" in s
        print(f"  robot_action_keys({arg!r})  did-you-mean? {has_hint}")
        print(f"    msg: {s}")
r.cleanup()

print()
print("=" * 60)
print("4) The sibling model-error path at base.py:425 ALREADY")
print("   calls resolve_name before difflib -- the fix is in the file,")
print("   just not plumbed into _unknown_robot_msg.")
print("=" * 60)
import inspect
import strands_robots.simulation.base as _base
src = open(_base.__file__).read()
print(f"  resolve_name imported in base.py: {'resolve_name' in src}")
# inspect the actual methods
print(f"  _unknown_robot_msg uses resolve_name? {'resolve_name' in inspect.getsource(_base.SimEngine._unknown_robot_msg)}")
