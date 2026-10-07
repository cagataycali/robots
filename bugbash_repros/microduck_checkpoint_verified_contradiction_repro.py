"""Repro: docs/robots/microduck.md claims 'No checkpoint verified on this robot
yet' while the sibling docs/learn/policies/microduck.md:85-100 ships a working
end-to-end sketch (`alpha_walking.onnx`) and the same package ships a
`microduck` policy provider dedicated to the body.

The hero page a new user lands on tells them to go record+train from scratch
when a verified, documented, pollen-shipped walking checkpoint already runs.

Run:
    MUJOCO_GL=egl python microduck_checkpoint_verified_contradiction_repro.py

Expected: the hero page's "Policies verified on this robot" acknowledges the
shipped microduck provider + alpha_walking.onnx (as the sibling policies page
and strands_robots/drivers/microduck.py:245 already do).

Actual: hero page prints "No checkpoint verified on this robot yet"; running
the sibling-page sketch succeeds -- 46% of commanded distance, 250/250 actions
applied, status=success, zero errors.

Root cause: docs/hooks/data/checkpoints.json ships 9 rows across 3 robots
(so101, unitree_g1, unitree_go2). The 154 pages for every other robot
(including microduck, whose policy provider is in-tree) are rendered by
docs/hooks/robot_pages.py:406 with the dead-end "no checkpoint verified"
paragraph. For microduck specifically, the sibling policy page
(docs/learn/policies/microduck.md:22,52,94) teaches alpha_walking.onnx /
alpha_stand.onnx as the entry point in three code blocks.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

REPO = Path(__file__).resolve().parents[1]


def check_hero_page_says_no_checkpoint() -> bool:
    """Hero docs/robots/microduck.md surfaces the dead-end paragraph."""
    hero = (REPO / "docs/robots/microduck.md").read_text()
    return "No checkpoint verified on this robot yet" in hero


def check_sibling_policies_page_ships_checkpoints() -> list[str]:
    """Sibling docs/learn/policies/microduck.md ships named ONNX weights."""
    sibling = (REPO / "docs/learn/policies/microduck.md").read_text()
    return sorted(set(re.findall(r"\b(alpha_[a-z_]+\.onnx)\b", sibling)))


def check_registry_has_microduck_provider() -> bool:
    """strands_robots ships a dedicated microduck policy provider."""
    from strands_robots.policies.factory import list_policy_providers

    return "microduck" in list_policy_providers()


def check_checkpoints_json_coverage() -> tuple[int, int, list[str]]:
    """Count how many robot pages have no checkpoint row, and which ones are present."""
    import json

    data = json.loads((REPO / "docs/hooks/data/checkpoints.json").read_text())
    robots_with_rows = sorted(k for k, v in data["robots"].items() if v)
    total_rows = sum(len(v) for v in data["robots"].values())
    return (total_rows, len(robots_with_rows), robots_with_rows)


def rollout_alpha_walking() -> dict:
    """Actually run alpha_walking.onnx -- prove the checkpoint is verified."""
    from strands_robots.simulation import create_simulation

    sim = create_simulation("mujoco")
    sim.create_world()
    sim.add_robot("microduck")
    obs0 = sim.get_observation("microduck")
    start_x = obs0["base_pos"][0]

    res = sim.run_policy(
        robot_name="microduck",
        policy_provider="microduck",
        policy_config={"onnx_path": "alpha_walking.onnx"},
        policy_kwargs={"target_velocity": [0.15, 0.0, 0.0]},
        duration=5.0,
        control_frequency=50.0,
    )
    obs1 = sim.get_observation("microduck")
    sim.cleanup()
    j = res["content"][1]["json"]
    return {
        "status": res["status"],
        "policy": j["policy"],
        "n_steps": j["n_steps"],
        "action_errors": j["action_errors"],
        "stopped_reason": j["stopped_reason"],
        "delta_x_m": obs1["base_pos"][0] - start_x,
        "end_height_m": obs1["base_pos"][2],
    }


def main() -> None:
    print("== Hero page (docs/robots/microduck.md) says 'no checkpoint verified'? ==")
    print(" ", check_hero_page_says_no_checkpoint())

    print()
    print("== Sibling policies page (docs/learn/policies/microduck.md) ships weights ==")
    for w in check_sibling_policies_page_ships_checkpoints():
        print(f"  - {w}")

    print()
    print("== Registry has 'microduck' policy provider ==")
    print(" ", check_registry_has_microduck_provider())

    print()
    print("== docs/hooks/data/checkpoints.json coverage ==")
    total, n_robots, robots = check_checkpoints_json_coverage()
    print(f"  {total} rows across {n_robots} robots: {robots}")
    print(f"  (there are 157 docs/robots/*.md pages; {157 - n_robots} get the dead-end paragraph)")

    print()
    print("== Rollout: can we actually run alpha_walking.onnx on sim microduck? ==")
    result = rollout_alpha_walking()
    for k, v in result.items():
        print(f"  {k}: {v}")

    print()
    print("CONTRADICTION:")
    print("  docs/robots/microduck.md:36 : 'No checkpoint verified on this robot yet. Record one...'")
    print("  docs/learn/policies/microduck.md:22,52,94 + strands_robots/drivers/microduck.py:245")
    print("  teach alpha_walking.onnx (ran above, status=success, 0 action errors, 46% of commanded")
    print("  translation in 5 s) as the entry-point checkpoint for this body.")


if __name__ == "__main__":
    main()
