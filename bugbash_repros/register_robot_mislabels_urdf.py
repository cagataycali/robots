"""Minimal repro — register_robot() mislabels URDF-only discoveries as "built-in"
and reports "0 joints" when joints is actually unknown (None).

Expected: refusal says "already discovered as a robot_descriptions URDF" (truth),
          omits "0 joints" when the joint count is unknown,
          and points broken-URDF cases (has refusal) at overwrite=True as the
          intended rescue path.

Actual:   refusal says "built-in robot" with "0 joints" for eve_r3 - which is
          neither built-in (not in robots.json) nor known-to-have-0-joints
          (urdf_robots.json records no joint count because the clone failed).

Upstream pins:
  - strands_robots/registry/user_registry.py:291-298 (the refusal site)
  - strands_robots/registry/robots.py:92-115 (get_robot synthesises URDF entries)
  - strands_robots/registry/discovery.py:375 (sets refusal on broken URDF)
  - strands_robots/registry/urdf_robots.json:eve_r3 (has_sim=false, no 'joints' key)
"""

import os
import sys
import tempfile


def main() -> int:
    # 1. isolate user registry to tempdir so we don't touch ~/.strands_robots
    os.environ["STRANDS_BASE_DIR"] = tempfile.mkdtemp(prefix="eve_r3_bugbash_")

    from strands_robots.registry import get_robot, register_robot

    entry = get_robot("eve_r3")
    assert entry is not None, "eve_r3 should be discoverable via URDF"
    assert entry["source"] == "urdf", "eve_r3 is URDF-synthesized"
    assert entry.get("discovered") is True, "eve_r3 is auto-discovered"
    assert entry.get("joints") is None, "eve_r3 has NO recorded joint count"
    assert "refusal" in entry, "eve_r3 is a broken URDF with a documented refusal"

    print("eve_r3 entry truth:")
    print(f"  source     = {entry['source']!r}")
    print(f"  discovered = {entry.get('discovered')!r}")
    print(f"  joints     = {entry.get('joints')!r}  (unknown, URDF did not build)")
    print(f"  has_sim    = cannot build (upstream repo deleted)")
    print()

    try:
        register_robot(
            name="eve_r3",
            model_xml="scene.xml",
            description="my rescue of a broken URDF entry",
            category="mobile_manip",
            joints=10,
            asset_dir="my_eve",
        )
    except ValueError as exc:
        msg = str(exc)
        print("register_robot refusal:")
        print(f"  {msg}")
        print()

        wrong_label = "is a built-in robot" in msg
        wrong_joints = "0 joints" in msg
        missing_refusal_hint = (
            "refusal" not in msg.lower()
            and "will not build" not in msg.lower()
            and "does not build" not in msg.lower()
        )
        no_rescue_nudge = (
            "overwrite=True to supply" not in msg
            and "overwrite=True is the intended rescue" not in msg
        )

        print("Defects in refusal text:")
        print(f"  calls URDF-only 'built-in'           : {wrong_label}  (eve_r3 is NOT in robots.json)")
        print(f"  reports '0 joints' when unknown     : {wrong_joints}  (joints is None in urdf_robots.json)")
        print(f"  omits the entry's own refusal        : {missing_refusal_hint}  (entry['refusal'] exists, says why it fails)")
        print(f"  doesn't nudge overwrite=True for rescue: {no_rescue_nudge}")

        failed = [wrong_label, wrong_joints, missing_refusal_hint, no_rescue_nudge]
        if any(failed):
            print()
            print("DEFECT CONFIRMED: register_robot refusal mislabels auto-discovered")
            print("URDF entries as 'built-in' and reports fabricated joint counts.")
            return 1
    else:
        print("DID NOT REFUSE - that would be the fix.")
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
