"""Three sibling methods, three disagreements on a documented alias.

docs/robots/unitree_g1.md:29 lists eight aliases for the canonical ``unitree_g1``
registry entry: ``g1``, ``g1_wbc``, ``real_g1_relative_eef_relative_joints``,
``unitree_g1_full_body``, ``unitree_g1_locomanip``, ``unitree_g1_real``,
``unitree_g1_sonic``, ``unitree_g1_wbc``.

harness#753 landed a "Did you mean: unitree_g1?" fragment on
``_unknown_robot_msg`` so ``robot_action_keys("g1")`` / ``robot_joint_names("g1")``
after ``Robot("unitree_g1")`` now raise ``ValueError`` with a usable sibling name
instead of a dead-end "Available robots: ['unitree_g1']".

But the TWO sibling introspection methods on the same engine --
``actuator_ranges`` and ``saturated_actuators`` -- gate on the SAME
``registered(self._world.robots, robot_name)`` check and, for the same alias
input, silently return ``{}`` and ``None`` respectively (indistinguishable from
a bona-fide empty actuator set / nothing saturated). A fourth sibling,
``send_action``, routes the same name through the raise site via its error-dict
wrapper, so a caller using the three APIs together receives three different
behaviours (raise / empty-dict / None / error-dict) for ONE registry alias:

  method                           -> alias "g1" on Robot("unitree_g1")
  ------------------------------------------------------------------
  robot_action_keys("g1")          -> raises ValueError (helpful)
  robot_joint_names("g1")          -> raises ValueError (helpful)
  send_action(..., robot_name="g1")-> {"status": "error", "content": [...]}
  actuator_ranges("g1")            -> {}                 <-- SILENT
  saturated_actuators("g1")        -> None               <-- SILENT

``strands_robots.registry.resolve_name("g1")`` already returns ``"unitree_g1"``
(the resolver's own docstring example). The two raise-site methods now at least
hand the caller the canonical name; the two silent-return methods do not --
their empty answer is the same answer a correctly-named robot with zero
ctrllimited actuators (or nothing currently saturated) would produce.

Minimal reproduction (sim-only, no network, ~10 s on CPU)."""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

from strands_robots import Robot
from strands_robots.registry import get_robot, resolve_name


def main() -> int:
    # 1. The registry knows the alias:
    assert resolve_name("g1") == "unitree_g1", "registry sanity check"
    assert get_robot("g1") is not None, "get_robot resolves aliases too"

    # 2. Build the canonical robot the doc page's quickstart shows:
    r = Robot("unitree_g1")
    try:
        # 3. The alias docs/robots/unitree_g1.md:29 explicitly lists.
        alias = "g1"

        # (a) RAISE sites -- helpful post-#753.
        raised = []
        for method, call in (
            ("robot_action_keys", lambda: r.robot_action_keys(alias)),
            ("robot_joint_names", lambda: r.robot_joint_names(alias)),
        ):
            try:
                call()
                raised.append((method, None))
            except ValueError as e:
                raised.append((method, str(e)))

        # (b) ERROR-DICT site -- status=error is honest.
        send_out = r.send_action({"left_hip_pitch_joint": 0.0}, robot_name=alias)

        # (c) SILENT sites -- the defect.
        ranges = r.actuator_ranges(alias)
        pinned = r.saturated_actuators(alias)

        # (d) same two SILENT methods on the construction name -- the shape
        #     the empty answer collides with.
        ranges_canon = r.actuator_ranges("unitree_g1")
        pinned_canon = r.saturated_actuators("unitree_g1")

        # Report:
        print("docs/robots/unitree_g1.md:29 lists 'g1' as an alias.")
        print(f"resolve_name('g1') = {resolve_name('g1')!r}  (registry agrees)\n")
        print("--- RAISE sites (helpful post-harness#753) ---")
        for method, msg in raised:
            print(f"  r.{method}('g1') -> ValueError:")
            print(f"    {msg}")
        print()
        print("--- ERROR-DICT site (status=error, actionable) ---")
        print(f"  r.send_action(..., robot_name='g1') -> status={send_out.get('status')}")
        print(f"    text: {send_out['content'][0]['text']}")
        print()
        print("--- SILENT sites (the defect) ---")
        print(f"  r.actuator_ranges('g1')        -> {ranges!r} (len {len(ranges)})")
        print(f"  r.actuator_ranges('unitree_g1')-> dict with {len(ranges_canon)} entries")
        print(f"  r.saturated_actuators('g1')    -> {pinned!r}")
        print(f"  r.saturated_actuators('unitree_g1')-> list with {len(pinned_canon)} entries")
        print()
        print("Divergent outcomes for ONE documented alias, three gate sites:")
        print("  * two raise sites hand the caller the canonical name,")
        print("  * one error-dict site reports status=error with the hint,")
        print("  * two silent sites return the empty answer a correctly-named robot")
        print("    with zero ctrllimited actuators / nothing saturated would return,")
        print("  * get_body_state (parent of the actuator-level methods) sits on the")
        print("    same gate and is sampled only indirectly here.")

        # Assertions pin the shape:
        assert ranges == {}, "actuator_ranges silently returned empty for a documented alias"
        assert pinned is None, "saturated_actuators silently returned None for a documented alias"
        assert len(ranges_canon) == 29, "ranges on canonical name should have 29 entries"
        assert send_out.get("status") == "error", "send_action should report status=error on alias"
        assert all(msg is not None for _, msg in raised), "raise sites should raise"
    finally:
        r.cleanup()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
