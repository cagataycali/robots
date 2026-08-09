"""Capture what a benchmark's declared scene and instruction actually do.

Run in a checkout of strands-labs/robots; writes PNGs + facts.json to --out.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import strands_robots.simulation.benchmark_spec as bs
from strands_robots import Simulation
from strands_robots.policies import Policy
from strands_robots.simulation.benchmark import register_benchmark, unregister_benchmark
from strands_robots.simulation.benchmark_spec import DeclarativeBenchmark

ARM = """<mujoco model="arm"><compiler angle="radian"/>
 <worldbody><body name="base" pos="0 0 0.06"><geom type="box" size=".05 .05 .06" rgba=".35 .38 .45 1"/>
  <body name="l1" pos="0 0 .06"><joint name="j1" type="hinge" axis="0 0 1" range="-2 2" damping="4"/>
   <geom type="capsule" fromto="0 0 0 .22 0 0" size=".025" rgba=".30 .55 .85 1"/></body></body></worldbody>
 <actuator><position name="a_j1" joint="j1" kp="30" ctrlrange="-2 2"/></actuator></mujoco>"""

# The scene a benchmark declares: a pedestal with a crate on it. Loading it is
# the whole visible difference between "the scene was honored" and "the scene
# was silently skipped".
SCENE = """<mujoco model="task_scene"><compiler angle="radian"/>
 <visual><headlight ambient=".55 .55 .55" diffuse=".65 .65 .65"/><global offwidth="1600" offheight="1200"/></visual>
 <worldbody>
  <light pos="0 0 2" dir="0 0 -1"/>
  <geom name="floor" type="plane" size="3 3 .1" rgba=".82 .82 .86 1"/>
  <body name="pedestal" pos=".42 0 .10"><geom type="box" size=".11 .11 .10" rgba=".45 .47 .52 1"/></body>
  <body name="crate" pos=".42 0 .28"><freejoint/>
   <geom type="box" size=".08 .08 .08" rgba=".95 .48 .12 1"/></body>
  <body name="marker" pos="-.30 .34 .16"><geom type="cylinder" size=".035 .16" rgba=".20 .70 .35 1"/></body>
  <camera name="look" pos="1.05 -1.05 .78" mode="targetbody" target="pedestal" fovy="40"/>
 </worldbody></mujoco>"""


class Spy(Policy):
    """Records the instruction the eval loop hands it."""

    def __init__(self) -> None:
        self.instrs: list[object] = []

    @property
    def provider_name(self) -> str:
        return "spy"

    def set_robot_state_keys(self, keys):  # noqa: ANN001, ANN201
        self._keys = list(keys)

    async def get_actions(self, observation_dict, instruction, **kwargs):  # noqa: ANN001, ANN003, ANN201
        self.instrs.append(instruction)
        return [{"a_j1": 0.45}]


def render(sim, out: Path) -> np.ndarray | None:
    """Render the scene camera when the scene loaded, else the free camera."""
    cams = sim.list_cameras()
    names = cams if isinstance(cams, list) else []
    cam = "look" if "look" in names else "default"
    r = sim.render(camera_name=cam, width=760, height=620)
    if r.get("status") != "success":
        return None
    png = next(c["image"]["source"]["bytes"] for c in r["content"] if "image" in c)
    out.write_bytes(png)
    import imageio.v3 as iio

    return np.asarray(iio.imread(png))


def run_case(arm_path: Path, scene_path: Path, out: Path, tag: str, **spec) -> dict:
    """Build a benchmark, evaluate it, and record what happened."""
    kw = dict(
        name="probe",
        supported_robots=[],
        default_robot="panda",
        max_steps=25,
        success_fn=lambda _s: False,
        failure_fn=lambda _s: False,
        reward_terms=[],
    )
    kw.update(spec)
    rec: dict = {"tag": tag, "spec": {k: repr(v) for k, v in spec.items()}}
    try:
        bench = DeclarativeBenchmark(**kw)
    except ValueError as exc:
        rec["constructed"] = False
        rec["refusal"] = str(exc)
        return rec
    rec["constructed"] = True
    register_benchmark("probe_b", bench)
    sim = Simulation(backend="mujoco", mesh=False)
    sim.create_world()
    sim.add_robot(name="arm", urdf_path=str(arm_path))
    spy = Spy()
    spy.set_robot_state_keys(sim.robot_action_keys("arm"))
    res = sim.evaluate_benchmark(
        benchmark_name="probe_b", robot_name="arm", policy_object=spy,
        n_episodes=1, control_frequency=50.0,
    )
    rec["status"] = res.get("status")
    if rec["status"] == "error":
        rec["error"] = (res.get("content") or [{}])[0].get("text", "")[:140]
    rec["instruction_seen"] = repr(spy.instrs[0]) if spy.instrs else None
    rec["instruction_type"] = type(spy.instrs[0]).__name__ if spy.instrs else None
    names = sim.list_cameras()
    rec["scene_loaded"] = "look" in (names if isinstance(names, list) else [])
    img = render(sim, out / f"{tag}.png")
    rec["render"] = None if img is None else {"shape": list(img.shape), "mean": float(img.mean())}
    sim.cleanup()
    unregister_benchmark("probe_b")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tree = str(Path(bs.__file__).parents[2])
    print("TREE:", tree)

    arm = out / "arm.xml"
    arm.write_text(ARM)
    scene = out / "scene.xml"
    scene.write_text(SCENE)

    rows = [
        run_case(arm, scene, out, "honored", scene=str(scene), instruction="pick up the crate"),
        run_case(arm, scene, out, "scene_falsy", scene=[], instruction="pick up the crate"),
        run_case(arm, scene, out, "instr_int", scene=str(scene), instruction=42),
        run_case(arm, scene, out, "instr_list", scene=str(scene), instruction=["pick"]),
        run_case(arm, scene, out, "name_int", scene=str(scene), name=7),
    ]
    (out / "facts.json").write_text(json.dumps({"tree": tree, "rows": rows}, indent=2))
    for r in rows:
        print(
            f"  {r['tag']:12s} built={r.get('constructed')} status={r.get('status')} "
            f"scene_loaded={r.get('scene_loaded')} instr={r.get('instruction_seen')}"
        )


if __name__ == "__main__":
    main()
