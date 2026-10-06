"""Repro: README L97 names `Robot(mesh=True)` + `robot.mesh.tell(...)` as the user-facing API,
but on a fresh environment with [sim-mujoco,mesh] installed and NO extra env vars,
`Robot(mesh=True)` refuses to start the mesh under the permissive-ACL gate.

README L97 (the "What you get" Mesh row, post harness#830 rewrite):

    | **Mesh** every `Robot(mesh=True)` as a Zenoh peer:
      `robot.mesh.tell(peer, instruction, policy_provider=...)`
      asks another robot to run a policy; broadcast an E-STOP,
      bridge fleets over AWS IoT Core | [Mesh](...) |

A reader who copies that in good faith hits:
    1. `Robot("so100", mesh=True)` → ACL banner, mesh.alive == False
    2. `robot.mesh.tell(...)` → {'status': 'error', 'error': 'mesh not running'}

The actual working one-liner (named in docs/learn/mesh/index.md:9 and the
banner's own four-way-out message) is:

    os.environ["STRANDS_MESH_LOCAL_DEV"] = "true"   # single-machine preset
    r = Robot("so100", mesh=True)

The README names Robot(mesh=True) but not the single-machine preset the mesh
needs to actually start. The dev needs to click through to the mesh index to
find the switch; the README gives no hint one is needed.

Repro:
    uv venv --python 3.12 && source .venv/bin/activate
    uv pip install -e ".[sim-mujoco,mesh]"
    MUJOCO_GL=egl python bugbash_repros/readme_mesh_true_refuses_without_local_dev_repro.py

Expected exit code: 0 (if the README-literal path worked)
Actual exit code: 1 (mesh refuses with permissive-ACL banner)
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    # Ensure a clean posture: no LOCAL_DEV, no ACL file, no explicit opt-in.
    for k in (
        "STRANDS_MESH_LOCAL_DEV",
        "STRANDS_MESH_ACCEPT_PERMISSIVE_ACL",
        "STRANDS_MESH_ACL_FILE",
        "STRANDS_MESH_AUTH_MODE",
    ):
        os.environ.pop(k, None)
    os.environ.setdefault("MUJOCO_GL", "egl")

    from strands_robots import Robot

    r = Robot("so100", mesh=True)  # README L97: this is supposed to work

    mesh = getattr(r, "mesh", None)
    alive = getattr(mesh, "alive", None) if mesh is not None else None
    print(f"[1] Robot('so100', mesh=True).mesh is None?  {mesh is None}")
    print(f"[2] robot.mesh.alive                        : {alive}")

    # README L97 claims this call is the mesh API; call it and show the status.
    try:
        out = mesh.tell("some_peer", "ping", policy_provider="mock")
    except Exception as e:  # noqa: BLE001
        print(f"[3] robot.mesh.tell(...)                   : raised {type(e).__name__}: {e}")
        return 1

    print(f"[3] robot.mesh.tell(...)                   : {out}")

    if alive is True and out.get("status") not in {"error"}:
        # If some future fix lifts the ACL gate for the default install path,
        # this repro should pass (exit 0) so the harness knows to close.
        return 0

    # The README-literal path visibly failed.
    print()
    print("README L97 promises `Robot(mesh=True)` + `robot.mesh.tell(...)` out of the box,")
    print("but on this environment the mesh is NOT running (permissive-ACL refusal).")
    print("The one-liner that actually works (per docs/learn/mesh/index.md:9) is")
    print("`STRANDS_MESH_LOCAL_DEV=true` before `Robot(mesh=True)` -- the README")
    print("never names it.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
