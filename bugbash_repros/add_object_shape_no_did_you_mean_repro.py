"""Repro: README quickstart's add_object refuses an unknown ``shape`` with no
'Did you mean' hint, while the same action's unknown-kwarg path (and 14+ other
refusal sites across the package) do give one.

Before fix on strands-labs/robots@82da623:

    $ python add_object_shape_no_did_you_mean_repro.py
    shape='boxx'      -> Unsupported shape 'boxx'. Supported: box, capsule, cylinder, ellipsoid, mesh, plane, sphere.
    shape='sfere'     -> Unsupported shape 'sfere'. Supported: box, capsule, cylinder, ellipsoid, mesh, plane, sphere.
    shape='cilinder'  -> Unsupported shape 'cilinder'. Supported: box, capsule, cylinder, ellipsoid, mesh, plane, sphere.

Contrast - same action, kwarg typo path (strands_robots/simulation/base.py:310):

    shape='box', colour=... -> Unknown parameter 'colour' for action 'add_object'. Did you mean: color? ...

The ``shape`` refusal is raised at
strands_robots/simulation/mujoco/spec_builder.py:194 and is the only one of
the three vocabulary gates in that file that does NOT use ``difflib``:

  * line 22   - ``import difflib``
  * line 151  - ``get_close_matches`` for MATERIAL_KEYS -> ``Did you mean ...``
  * line 194  - ``Unsupported shape ...`` -> no close match

Fix on this branch adds the same ``difflib.get_close_matches(cutoff=0.6)``
line already used 40 lines above, so the typos below produce:

    shape='boxx'      -> Unsupported shape 'boxx'. Did you mean 'box'? ...
    shape='sfere'     -> Unsupported shape 'sfere'. Did you mean 'sphere'? ...
    shape='cilinder'  -> Unsupported shape 'cilinder'. Did you mean 'cylinder'? ...
"""

import os

# Thor shell wedge contract: scrub SYSTEM_PROMPT out of inherited env.
_clean = {k: v for k, v in os.environ.items() if k != "SYSTEM_PROMPT"}
os.environ.clear()
os.environ.update(_clean)
os.environ.setdefault("MUJOCO_GL", "egl")

from strands_robots import Robot


def main() -> int:
    robot = Robot("so101")  # README's "so100" works too; same code path.

    # Sibling kwarg typo for comparison - this one DOES get a hint today.
    sibling = robot(
        action="add_object",
        name="demo_cube",
        shape="box",
        colour=[1.0, 0.0, 0.0, 1.0],  # typo
        size=[0.025, 0.025, 0.025],
        position=[0.0, -0.20, 0.025],
    )
    print(f"[sibling kwarg path] {sibling['content'][0]['text']}")

    # Shape typos: the subject of this repro. Each landed from a README-shaped
    # call that only differs in the 'shape' value.
    for shape in ("boxx", "sfere", "cilinder", "capsul", "elipsoid"):
        res = robot(
            action="add_object",
            name=f"obj_{shape}",
            shape=shape,
            size=[0.025, 0.025, 0.025],
            position=[0.3, 0.0, 0.05],
            color=[1.0, 0.0, 0.0, 1.0],
        )
        text = res["content"][0]["text"]
        marker = "HINT" if "Did you mean" in text else "NO HINT"
        print(f"[shape={shape!r:10}] [{marker}] {text}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
