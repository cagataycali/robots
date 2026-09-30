"""Every robot that is a base carrying an arm is grouped under ``mobile_manip``.

The docs family page, the catalog filter and ``list_robots_by_category`` all
read ``category`` from the registry, so a base-with-arm robot filed under
``mobile`` vanishes from the one place a reader looks for it. The set is
pinned exactly, so a robot drifting either way (in or out) is caught.
"""

from __future__ import annotations

from strands_robots.registry import list_robots_by_category

#: Each of these models carries both a moving base and an arm.
BASE_WITH_ARM = {
    "google_robot",
    "lekiwi",
    "lekiwi_client",
    "spot",
    "stretch",
    "stretch3",
    "tiago_dual",
    "yahboom_m3pro",
}


def test_the_mobile_manip_family_is_exactly_the_base_with_arm_robots() -> None:
    groups = list_robots_by_category()
    assert {r["name"] for r in groups["mobile_manip"]} == BASE_WITH_ARM
    assert not BASE_WITH_ARM & {r["name"] for r in groups["mobile"]}
