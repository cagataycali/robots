"""
Repro: `add_object(shape='capsule', size=[D, _, H])` creates a capsule whose
total world-space extent is H + D, not H -- contradicting the LLM-visible
`size` tool_spec description ("add_object: FULL extents in meters per axis")
and the shape label `spec_builder.py:212` ("[diameter, unused, full height]").

The simulation.py:4814-4817 docstring tells the truth (capsule = cylindrical
section + two hemispherical caps that add `size[0]` to total height), but the
agent-facing schema and the success-message echo both report the user's
size[2] back as if it were the full extent.

Impact: an LLM following docs/README verbatim and sizing a capsule marker to
20 cm actually plants a 24 cm capsule and lands it that much higher than it
asked. Clean silent-wrong + schema mismatch; distinct from closed #168
(box/ellipsoid half-vs-full) and #725/#724 (README scene content).

Run:
    cd /path/to/robots
    python bugbash_repros/capsule_size_not_full_extent_repro.py

Expected (what the schema promises): capsule_end_to_end == size[2] (0.2)
Actual: capsule_end_to_end == size[2] + size[0] (0.24)
"""

import sys

import mujoco

from strands_robots import Robot


def main() -> int:
    r = Robot("so100")

    # --- Baseline: cylinder honors "full height" promise ---
    r.add_object(
        name="cyl",
        shape="cylinder",
        size=[0.04, 0.0, 0.20],
        position=[0.2, -0.2, 0.1],
        color=[0.0, 1.0, 0.0],
    )
    m = r.mj_model
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cyl_geom")
    cyl_radius, cyl_half_len, _ = m.geom_size[gid].tolist()
    cyl_end_to_end = 2 * cyl_half_len
    print(f"cylinder: user size=[0.04, 0, 0.20]")
    print(f"  mujoco geom_size (radius, half-len): [{cyl_radius}, {cyl_half_len}]")
    print(f"  world end-to-end = {cyl_end_to_end}  (expected 0.20)")
    assert abs(cyl_end_to_end - 0.20) < 1e-9, "cylinder shouldn't drift"

    # --- Defect: capsule silently inflates by one diameter ---
    r.add_object(
        name="cap",
        shape="capsule",
        size=[0.04, 0.0, 0.20],
        position=[0.6, -0.2, 0.1],
        color=[1.0, 0.0, 1.0],
    )
    gid2 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cap_geom")
    cap_radius, cap_half_cyl, _ = m.geom_size[gid2].tolist()
    cap_end_to_end = 2 * cap_half_cyl + 2 * cap_radius
    print(f"capsule : user size=[0.04, 0, 0.20]")
    print(f"  mujoco geom_size (radius, half-cyl): [{cap_radius}, {cap_half_cyl}]")
    print(f"  world end-to-end = {cap_end_to_end}  (schema promise: 0.20)")

    schema_promise = 0.20
    inflated_by = cap_end_to_end - schema_promise
    print()
    print(
        f"ASYMMETRY: schema says 'FULL extents', shape label says "
        f"'[diameter, unused, full height]'. Reality: capsule total extent "
        f"= size[2] + size[0] = {cap_end_to_end:.3f} m, inflating by one "
        f"diameter ({inflated_by:.3f} m, {inflated_by / schema_promise * 100:.0f}% of request)."
    )

    # Pin: cylinder matches schema, capsule does not.
    if abs(cap_end_to_end - 0.20) < 1e-9:
        print("UNEXPECTED: capsule now honors full-extent -- defect fixed?")
        return 0
    print()
    print(
        f"Defect reproduces: capsule total extent = {cap_end_to_end:.3f} m "
        f"!= user size[2] (0.20 m)."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
