"""README L97 — `tell()` is advertised as a Robot verb but lives on Mesh.

README line 97 (the "What you get" row for Mesh):

    | **Mesh** every robot as a Zenoh peer: `tell()` another robot what to
    do, broadcast an E-STOP, bridge fleets over AWS IoT Core | [Mesh](...)

Subject of the sentence is "every robot" and the verb is written with
parentheses (`tell()`), the shape readers associate with a bound method
on the subject. The README's hero block one screen up constructs the
subject with `Robot("so100")` and never names the `Mesh` type.

But `tell` does not exist on `Robot` / `MuJoCoSimEngine`. It is a
method on `strands_robots.mesh.core.Mesh` and is only reachable via
`Robot(..., mesh=True).mesh.tell(peer_id, instruction, policy_provider=...)`.

This repro pins the three asymmetries a README-literal reader hits:

  1. `robot.tell(...)` → AttributeError, with no "Did you mean?" and no
     hint pointing at `robot.mesh.tell(...)`.

  2. `Robot("so100")` (hero form, mesh default) returns a robot whose
     `.mesh` attribute is `None`, so even once the reader finds the
     `.mesh.tell` path, the next AttributeError is `NoneType.tell`.

  3. `Robot("so100", mesh=True)` finally exposes `.mesh.tell` — a
     fully-qualified call the README never prints anywhere.

Reproduce with:

    MUJOCO_GL=egl python bugbash_repros/readme_l97_tell_not_on_robot_repro.py

The script fails loudly on each asymmetry and prints the fully-qualified
call that currently works, so a fix that renames / aliases / lifts
`tell` to the Robot surface, or rewords the README row to show the
reach-through, will trip these assertions.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
# Keep mesh startup quiet during the repro; neither branch here opens the mesh.
os.environ.setdefault("STRANDS_MESH_LOCAL_DEV", "true")


def main() -> int:
    import inspect

    from strands_robots import Robot

    # --- 1. README-literal: hero form of Robot (no mesh kwarg) ------------
    hero = Robot("so100")
    assert not hasattr(hero, "tell"), (
        "If `tell` lands on Robot the README L97 promise is honoured; "
        "update this repro + the issue."
    )
    try:
        hero.tell("peer-01", "pick up the red cube")  # type: ignore[attr-defined]
    except AttributeError as e:
        print(f"[1] Robot('so100').tell(...) -> AttributeError: {e}")
    else:
        print("[1] Robot('so100').tell(...) returned without error (unexpected)")
        return 1

    # The hero form has mesh=None, so even once a reader finds `.mesh.tell`
    # the next call also fails — a two-step dead end, not one.
    assert hero.mesh is None, f"hero.mesh expected None, got {hero.mesh!r}"
    try:
        hero.mesh.tell("peer-01", "pick up the red cube")  # type: ignore[union-attr]
    except AttributeError as e:
        print(f"[2] Robot('so100').mesh.tell(...) -> AttributeError: {e}")
    else:
        print("[2] Robot('so100').mesh.tell(...) returned (unexpected)")
        return 1

    # --- 3. The actual, undocumented recipe --------------------------------
    mesh_robot = Robot("so100", mesh=True)
    assert not hasattr(mesh_robot, "tell"), (
        "`tell` still absent from Robot even with mesh=True — the README "
        "L97 phrasing requires it on the Robot surface."
    )
    mesh = mesh_robot.mesh
    assert mesh is not None, "mesh=True must materialise a Mesh handle"
    assert hasattr(mesh, "tell"), f"Mesh.tell is the one home for tell(); got dir={dir(mesh)[:10]}"

    sig = inspect.signature(mesh.tell)
    print(
        "[3] Robot('so100', mesh=True).mesh.tell signature: "
        f"{sig}  (home: strands_robots.mesh.core.Mesh.tell)"
    )

    # Pin: what the README prints verbatim is the shortest form that compiles
    # (any-robot.tell(...)); the real recipe is three dots and a required
    # policy_provider= kwarg.
    source_file = inspect.getsourcefile(type(mesh).tell) or "?"
    assert source_file.endswith("mesh/core.py"), (
        f"tell() moved — expected mesh/core.py, got {source_file}"
    )
    print(f"    source: {source_file}")

    print()
    print("README L97 (strands-labs/robots main) says:")
    print("    | **Mesh** every robot as a Zenoh peer: `tell()` another robot ...")
    print()
    print("Actual callable:")
    print("    Robot(name, mesh=True).mesh.tell(peer, instruction, policy_provider=...)")
    print()
    print(
        "Three asymmetries pinned: (1) Robot.tell missing, (2) mesh=False hero form "
        "lands on NoneType.tell, (3) the full form is nowhere in README."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
