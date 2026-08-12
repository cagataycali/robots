"""Capture the two real frames + the measured facts for the PR artifact."""

import json
import pathlib
import subprocess
import sys

import numpy as np

import strands_robots

ROOT = pathlib.Path(strands_robots.__file__).parents[1]
print("TREE:", ROOT)

from strands_robots.simulation import Simulation  # noqa: E402

OUT = pathlib.Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
facts: dict = {"tree": str(ROOT)}


def save() -> None:
    (OUT / "facts.json").write_text(json.dumps(facts, indent=2))


ORTHO = """
<mujoco model="orthographic_free_camera">
  <visual><global orthographic="true" fovy="0.35"/></visual>
  <worldbody>
    <light pos="0.4 -0.6 1.2" dir="-0.2 0.4 -1"/>
    <body name="cube" pos="0.4 0 0.05">
      <geom name="cube_g" type="box" size="0.1 0.1 0.05" rgba="0.9 0.4 0.1 1"/>
    </body>
  </worldbody>
</mujoco>
"""
PERSP = ORTHO.replace('orthographic="true" fovy="0.35"', 'fovy="45"')


def run(label: str, scene_xml: str) -> None:
    d = OUT / label
    d.mkdir(exist_ok=True)
    f = d / "scene.xml"
    f.write_text(scene_xml)
    sim = Simulation(mesh=False)
    rec: dict = {}
    try:
        assert sim.create_world(ground_plane=False)["status"] == "success"
        assert sim.load_scene(str(f))["status"] == "success"
        eng = getattr(sim, "_engine", sim)
        rec["orthographic_flag"] = int(eng._world._model.vis.global_.orthographic)

        rgb, depth = sim.get_frame("default")
        rec["get_frame"] = "ok"
        rec["rgb_shape"] = list(rgb.shape)
        rec["depth_shape"] = list(depth.shape)
        np.save(OUT / f"{label}.npy", rgb)
        sat = float(((rgb.max(2).astype(int) - rgb.min(2).astype(int)) > 45).mean())
        rec["saturated_frac"] = sat

        try:
            cam = sim.get_camera_params("default")
            rec["get_camera_params"] = f"ok (fx={float(cam.K[0, 0]):.2f})"
        except Exception as e:  # noqa: BLE001 - classifying the outcome, not recovering
            rec["get_camera_params"] = f"{type(e).__name__}: {e}"

        h, w = depth.shape
        r = sim.get_world_point("default", pixels=[[w // 2, h // 2]])
        rec["gwp_status"] = r["status"]
        rec["gwp_text"] = next(c["text"] for c in r["content"] if "text" in c)
        if r["status"] == "success":
            rec["gwp_point"] = [round(v, 4) for v in r["content"][1]["json"]["point"]]
    finally:
        sim.destroy()
    facts[label] = rec
    print(f"{label}: {json.dumps(rec)[:200]}")
    save()


run("orthographic", ORTHO)
run("perspective", PERSP)

# no-behaviour-change proof: docstring-stripped AST digest of the changed source
code = (
    "import ast,hashlib,pathlib,subprocess\n"
    "def strip(t):\n"
    "    class T(ast.NodeTransformer):\n"
    "        def _d(self,n):\n"
    "            self.generic_visit(n)\n"
    "            b=n.body\n"
    "            if b and isinstance(b[0],ast.Expr) and isinstance(b[0].value,ast.Constant) "
    "and isinstance(b[0].value.value,str): n.body=b[1:] or [ast.Pass()]\n"
    "            return n\n"
    "        visit_Module=_d; visit_ClassDef=_d; visit_FunctionDef=_d; visit_AsyncFunctionDef=_d\n"
    "    return T().visit(t)\n"
    "def dig(s): return hashlib.sha256(ast.dump(strip(ast.parse(s))).encode()).hexdigest()[:16]\n"
    "p='strands_robots/simulation/base.py'\n"
    "base=subprocess.run(['git','show',f'f5442883:{p}'],capture_output=True,text=True).stdout\n"
    "head=pathlib.Path(p).read_text()\n"
    "print(dig(base),dig(head),base!=head)\n"
)
o = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT).stdout.split()
facts["ast_digest_base"], facts["ast_digest_head"], facts["text_differs"] = o[0], o[1], o[2]
facts["numstat"] = subprocess.run(
    ["git", "diff", "--numstat", "f5442883", "--", "strands_robots/"],
    capture_output=True, text=True, cwd=ROOT,
).stdout.strip()
save()
print("AST digest base/head:", facts["ast_digest_base"], facts["ast_digest_head"], "text differs:", facts["text_differs"])
print("numstat:", facts["numstat"])
