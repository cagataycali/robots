"""
Repro: `_spawn_burial_warning` suggests `position=` that never clears the burial
when the robot is loaded via `urdf_path=` + `keyframe=`.

Context (microduck_sim_quickstart rotation, v0.5.3 bug-bash fire #122):
`docs/learn/policies/microduck.md:81` instructs:

    Robot("microduck", urdf_path=str(scene), keyframe="STAND")

for the `scene_rollers.xml` variant. That exact call emits:

    WARNING strands_robots.simulation.mujoco.simulation:
      'microduck' starts 20.7 mm inside the ground, so the contact solver
      pushes it from the first step.
      Pass position=[0.0, 0.0, 0.0207] to spawn it resting on the ground.

Applying the fix verbatim produces the SAME 20.7 mm burial and a
larger suggestion (`position=[0.0, 0.0, 0.0414]`). Divergent.

Root cause (strands_robots/simulation/mujoco/simulation.py:3362-3410):
`_spawn_burial_warning` builds `lift = [x, y, z + buried]` using
`robot.position`, assuming the attach-frame `position=` kwarg shifts the
measured base. On the `urdf_path=` + `keyframe=` path the model carries
its own scene and the top-level attach frame has no effect on the
base's measured world-z after the keyframe restore. So:

  * `buried` is independent of `position=` (always 20.7 mm here);
  * the suggestion monotonically grows by 20.7 mm each call;
  * the actual corrective sign is NEGATIVE for this path, which the
    warning never names.

The result is a wrong-direction fix hint pointed at a user who is
following docs verbatim.
"""
import contextlib
import io
import os
import re
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

from strands_robots import Robot
from strands_robots.assets import get_search_paths


def _probe(z):
    """Return (burial_mm, suggested_position) for one trial."""
    scene = next(
        p for p in (os.path.join(sp, "microduck", "scene_rollers.xml") for sp in get_search_paths())
        if os.path.exists(p)
    )
    kwargs = {"urdf_path": scene, "keyframe": "STAND"}
    if z is not None:
        kwargs["position"] = [0.0, 0.0, z]
    cap = io.StringIO()
    with contextlib.redirect_stderr(cap):
        Robot("microduck", **kwargs)
    warn = cap.getvalue()
    m_depth = re.search(r"starts\s+([0-9.]+)\s+mm", warn)
    m_sugg = re.search(r"position=\[([^\]]+)\]", warn)
    return (
        float(m_depth.group(1)) if m_depth else 0.0,
        m_sugg.group(1) if m_sugg else None,
    )


def main() -> int:
    print("Repro: `_spawn_burial_warning` wrong-direction suggestion on urdf_path+keyframe path\n")
    print(f"{'position= passed':>18} | burial (mm) | suggested next position=")
    print("-" * 70)

    trials = [None, 0.0, 0.0207, 0.0414, 0.0621, -0.0207]
    for z in trials:
        burial, sugg = _probe(z)
        pz = "(unset)" if z is None else f"[0, 0, {z}]"
        print(f"{pz:>18} | {burial:>10.1f}  | position=[{sugg}]")

    print()
    print("Observed invariants:")
    print(" 1. `burial` is 20.7 mm regardless of `position=`.")
    print(" 2. suggested_z = passed_z + 0.0207 (strictly additive).")
    print(" 3. applying the suggestion verbatim produces the SAME warning")
    print("    with a strictly larger suggestion -> no fixed point.")
    print()
    print("Expected: the warning should either (a) not fire for the")
    print("`urdf_path=` + `keyframe=` path where `position=` cannot change")
    print("the measured base, or (b) name a `position=` that would actually")
    print("clear the burial (likely NEGATIVE on this path).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
