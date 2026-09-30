"""Repro: lekiwi + spot(-with-arm) miscategorized as `mobile` instead of `mobile_manip`.

Both robots ship as "base + arm" (lekiwi: 6-DOF arm on 3-omniwheel base; spot: with-arm
variant, `scene_arm.xml`). Their own registry descriptions say so. The `mobile_manip`
family is defined in docs/hooks/robot_pages.py:67 as "A base that carries an arm."
Yet both are grouped under `mobile` ("Wheeled and legged platforms that move through
a room"), so:

  * `list_robots_by_category()['mobile_manip']` does not surface them.
  * The `docs/robots/mobile_manip/index.md` family page (Mobile manipulators)
    and the `data-family="mobile_manip"` filter button on `docs/robots/index.md`
    both miss them.

Same shape as `stretch`, `stretch3`, `yahboom_m3pro`, `tiago_dual`, `google_robot`,
and the sibling entry `lekiwi_client` - all of which are `mobile_manip`.

Run: python lekiwi_mobile_manip_repro.py
"""

from strands_robots.registry import list_robots_by_category

by_cat = list_robots_by_category()

mobile_manip_names = {r["name"] for r in by_cat.get("mobile_manip", [])}
mobile_names = {r["name"] for r in by_cat.get("mobile", [])}

# Facts that must hold for the family page + filter to behave correctly.
expected_mm = {"lekiwi", "spot"}
found_in_wrong_bucket = expected_mm & mobile_names
missing_from_mm = expected_mm - mobile_manip_names

print(f"mobile_manip members: {sorted(mobile_manip_names)}")
print(f"'lekiwi' in mobile? {'lekiwi' in mobile_names}   in mobile_manip? {'lekiwi' in mobile_manip_names}")
print(f"'spot'   in mobile? {'spot'   in mobile_names}   in mobile_manip? {'spot'   in mobile_manip_names}")

assert missing_from_mm == set(), f"missing from mobile_manip: {sorted(missing_from_mm)}"
assert found_in_wrong_bucket == set(), f"found in mobile (should be mobile_manip): {sorted(found_in_wrong_bucket)}"
print("OK")
