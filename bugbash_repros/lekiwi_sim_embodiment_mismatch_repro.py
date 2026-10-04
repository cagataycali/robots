"""Minimal repro: lerobot_local lekiwi embodiment has zero overlap with sim actuators.

Only one lerobot embodiment exists for LeKiwi (`lekiwi_real`), and both
`lekiwi` and `lekiwi_client` policy-side aliases point at it (see
`strands_robots/policies/lerobot_local/embodiments.json` configs/aliases).

Its action_keys are the lerobot hardware names
(`arm_shoulder_pan.pos`, ..., `arm_gripper.pos`, `x.vel`, `y.vel`, `theta.vel`).

The sim MuJoCo asset exposes
(`Rotation`, `Pitch`, `Elbow`, `Wrist_Pitch`, `Wrist_Roll`, `Jaw`,
`base_back_wheel`, `base_right_wheel`, `base_left_wheel`).

Zero overlap. A user who follows docs/learn/policies/lerobot-local.md
and calls sim.run_policy('lekiwi', policy_provider='lerobot_local', ...)
gets a rollout where every send_action is refused in full. Each policy
step's send_action returns status=error, applied=[], unresolved=9/9.

Run:
    MUJOCO_GL=egl python lekiwi_sim_embodiment_mismatch_repro.py

Expected: at least one action key resolves on sim actuators.
Actual: all 9 action keys unresolved; world does not advance.
"""
from __future__ import annotations

import json
from pathlib import Path

from strands_robots import Robot


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    emb_path = repo_root / "strands_robots" / "policies" / "lerobot_local" / "embodiments.json"
    embodiments = json.loads(emb_path.read_text())

    # Both lekiwi* names resolve to the same real-hardware config.
    assert embodiments["aliases"]["lekiwi"] == "lekiwi_real"
    assert embodiments["aliases"]["lekiwi_client"] == "lekiwi_real"
    emb = embodiments["configs"]["lekiwi_real"]
    policy_action_keys = emb["action_keys"]
    print("policy embodiment lekiwi_real action_keys:", policy_action_keys)

    sim = Robot("lekiwi")
    sim_action_keys = sim.robot_action_keys("lekiwi")
    print("sim robot_action_keys('lekiwi'):          ", sim_action_keys)

    overlap = set(policy_action_keys) & set(sim_action_keys)
    print("overlap:", sorted(overlap) or "(empty)")

    # Simulate one policy step: feed sim what the embodiment produces.
    fake_action = {k: 0.0 for k in policy_action_keys}
    result = sim.send_action(fake_action, robot_name="lekiwi")
    status = result.get("status")
    body = result["content"][1]["json"] if len(result["content"]) > 1 else {}
    print(f"send_action status={status} applied={body.get('applied')} "
          f"unresolved={body.get('unresolved_keys')}")

    sim.cleanup()

    if overlap:
        print("PASS: at least one key resolves")
        return 0
    print("FAIL: lerobot_local 'lekiwi_real' embodiment is unroutable to sim.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
