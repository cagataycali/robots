"""Repro: set_joint_positions refuses a free-joint 7-vector that docs explicitly
tell the user to write.

docs/learn/policies/microduck.md 'The ball scene places the ball, not the kick
geometry' tells the user how to seat the kick ball before a ``ball_kick_*``
rollout:

    "write the free joint's qpos to that offset in the trunk's yaw frame and
     zero its qvel."

The only exposed kinematic writer is :meth:`SimEngine.set_joint_positions`.
Running the docs' step on the microduck's own trunk free joint (identical
shape, 7 numbers: ``[x, y, z, qw, qx, qy, qz]``) is refused with a scalar-only
error -- while a bare scalar is silently accepted and only writes ``qpos[0]``.

Root cause: :func:`strands_robots.simulation.base.SimEngine._coerce_joint_state_map`
calls ``float(value)`` on every entry, so a free-joint vector lands in the
``TypeError`` branch and the error text says 'must be a number'. The
scalar-success path (writes 1 of 7 slots, leaves y/z/quat untouched) is the
silent-wrong half: the user's documented teleport reaches qpos[0] and reports
``status='success'``.

Run (fresh clone, repo root)::

    MUJOCO_GL=egl python bugbash_repros/set_joint_positions_refuses_free_joint_vec.py
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402

from strands_robots import Robot  # noqa: E402


def main() -> None:
    sim = Robot("microduck")
    sim.reset()
    m, d = sim.mj_model, sim.mj_data

    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "microduck/trunk_base_freejoint")
    adr = int(m.jnt_qposadr[jid])
    print(f"before: qpos[{adr}:{adr+7}] = {list(d.qpos[adr:adr+7])}")

    # 1. The docs path: write the free joint's 7-vector qpos.
    r = sim.set_joint_positions(
        {"trunk_base_freejoint": [0.5, 0.0, 0.12, 1.0, 0.0, 0.0, 0.0]}
    )
    print("[free, 7-vec]  status:", r.get("status"))
    for c in r.get("content", [])[:2]:
        if "text" in c:
            print("              text :", c["text"][:200])

    # 2. The silent-wrong half: a bare scalar is accepted and writes qpos[0]
    # only, leaving y / z / quat untouched.
    r = sim.set_joint_positions({"trunk_base_freejoint": 0.5})
    print("[free, scalar] status:", r.get("status"))
    print(f"after:  qpos[{adr}:{adr+7}] = {list(d.qpos[adr:adr+7])}")

    # 3. The scene_ball.xml angle: the kick ball's free joint has the same
    # shape and the same refusal, so the documented teleport is unreachable.
    import pathlib

    from strands_robots.assets import get_search_paths

    scene = None
    for p in get_search_paths():
        cand = pathlib.Path(p) / "microduck" / "scene_ball.xml"
        if cand.exists():
            scene = cand
            break
    sim2 = Robot("microduck", urdf_path=str(scene))
    r = sim2.set_joint_positions(
        {"ball_free": [0.09, 0.042, 0.035, 1.0, 0.0, 0.0, 0.0]}
    )
    print("[ball, 7-vec]  status:", r.get("status"))
    for c in r.get("content", [])[:2]:
        if "text" in c:
            print("              text :", c["text"][:200])

    assert (
        r.get("status") == "error"
    ), "upstream fixed the refusal -- rebuild the repro"


if __name__ == "__main__":
    main()
