"""
Repro: `unitree_g1_real` is an alias with cross-namespace collision.

docs/robots/unitree_g1.md lists `unitree_g1_real` as an alias of `unitree_g1`
(a MuJoCo sim robot). strands_robots/registry/robots.json:894 agrees.

But the SAME name is a REAL-hardware embodiment profile in
strands_robots/policies/lerobot_local/embodiments.json:712 — with motor keys
like `kLeftHipPitch.q` (DDS over Ethernet, Unitree SDK) and a `__note__` that
reads: "REAL hardware: lerobot UnitreeG1 driver. _motors_ft keys are
'<G1_29_JointIndex.name>.q'".

Nothing reconciles the two meanings. `Robot('unitree_g1_real', mesh=False)`
returns a MuJoCoSimEngine silently; the user has no way to know this name
also means something very different inside lerobot_local policy routing.

Additionally, the alias block for `unitree_g1` promises five names that are
NOT defined in the lerobot_local embodiments namespace at all:
`g1_wbc`, `real_g1_relative_eef_relative_joints`, `unitree_g1_full_body`,
`unitree_g1_locomanip`, `unitree_g1_wbc`.
These are dead names — the sim resolves them to the base, but no policy
embodiment exists under those keys.

Run:
    MUJOCO_GL=egl python bugbash_repros/g1_real_alias_collision_repro.py
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from strands_robots import Robot


def main() -> int:
    reg_path = REPO / "strands_robots/registry/robots.json"
    emb_path = REPO / "strands_robots/policies/lerobot_local/embodiments.json"
    reg = json.loads(reg_path.read_text())
    emb = json.loads(emb_path.read_text())

    sim_aliases = reg["robots"]["unitree_g1"]["aliases"]
    emb_configs = set(emb["configs"].keys())
    emb_aliases = emb["aliases"]
    emb_targets = set(emb_configs) | set(emb_aliases.keys())

    print("docs/registry sim aliases for unitree_g1:", sim_aliases)
    print()

    # Half 1: collisions — sim alias name is ALSO a real-hardware embodiment.
    collisions = []
    for a in sim_aliases:
        if a in emb_configs:
            note = emb["configs"][a].get("__note__", "")
            if "REAL hardware" in note or "lerobot" in note.lower():
                collisions.append((a, note[:120]))

    print(f"COLLISIONS ({len(collisions)}): same name in both namespaces, "
          "different semantics, no reconciliation.")
    for name, note in collisions:
        print(f"  - {name}")
        print(f"      lerobot_local __note__: {note}")

    # Half 2: dead names — promised in docs/sim-registry but no policy
    # embodiment exists under that key, so a user who follows the "aliases"
    # row to run a lerobot_local policy on that embodiment gets 404.
    dead = [a for a in sim_aliases if a not in emb_targets]
    print()
    print(f"DEAD NAMES ({len(dead)}): sim-registry promises these, "
          "no lerobot_local embodiment defined, no cross-ref.")
    for name in dead:
        print(f"  - {name}")

    # Observable half: construct both and show same-type silence.
    print()
    r1 = Robot("unitree_g1", mesh=False)
    r2 = Robot("unitree_g1_real", mesh=False)
    print(f"Robot('unitree_g1')      -> {type(r1).__name__}")
    print(f"Robot('unitree_g1_real') -> {type(r2).__name__}")
    print("The suffix '_real' is dropped on the floor; no warning, no error.")

    sim_joints = r1.robot_joint_names("unitree_g1")[:4]
    real_cfg = emb["configs"]["unitree_g1_real"]
    print()
    print(f"sim joint names (first 4):      {sim_joints}")
    print(f"real embodiment state keys (0..3): {real_cfg['state_keys'][:4]}")
    print("Two different conventions, same alias name, no bridge.")

    try:
        r1.cleanup()
    except Exception:
        pass
    try:
        r2.cleanup()
    except Exception:
        pass

    assert collisions, "expected unitree_g1_real to collide"
    assert dead, "expected dead names in the alias list"
    print()
    print("Fails loudly in the sense that the user who trusts the alias row "
          "has no way to know some names mean 'hardware profile' and some "
          "names mean nothing at all.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
