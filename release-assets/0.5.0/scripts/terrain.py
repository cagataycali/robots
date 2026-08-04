"""Render the new create_world(terrain=, difficulty=) locomotion ground."""
import os, io, json
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
from PIL import Image
from strands_robots.simulation import Simulation

OUT = "/tmp/relnotes/assets"
W, H = 560, 420

def png(res):
    for c in res["content"]:
        if "image" in c:
            return c["image"]["source"]["bytes"]
    raise RuntimeError(f"no image: {str(res)[:300]}")

def shot(terrain, difficulty, cam_pos, cam_tgt, robot="go2", settle=120):
    sim = Simulation(backend="mujoco", tool_name=f"t_{terrain}", mesh=False)
    try:
        r = sim.create_world(terrain=terrain, difficulty=difficulty)
        assert r["status"] == "success", r
        a = sim.add_robot(name="dog", data_config=robot)
        assert a["status"] == "success", str(a)[:200]
        sim.add_camera(name="look", position=cam_pos, target=cam_tgt, fov=38)
        sim.step(settle)
        st = sim.get_body_state(body_name="dog/base")
        js = next(c["json"] for c in st["content"] if "json" in c) if st["status"] == "success" else {}
        img = np.array(Image.open(io.BytesIO(png(sim.render(camera_name="look", width=W, height=H)))).convert("RGB"))
        gh = next(c["json"] for c in sim.get_ground_height(0.0, 0.0)["content"] if "json" in c)["height"]
        return img, {"terrain": terrain, "difficulty": difficulty,
                     "base_z": round(float(js.get("position", [0,0,0])[2]), 4),
                     "ground_z_at_origin": round(gh, 4)}
    finally:
        sim.cleanup()

facts = []
frames = {}
# Four terrain kinds at nominal difficulty. Camera low + close so 8 cm relief reads.
cams = {
    "rough":   ([1.15, -1.15, 0.42], [0.0, 0.0, 0.14]),
    "stairs":  ([-1.35, -0.95, 0.40], [0.35, 0.0, 0.12]),
    "pyramid": ([1.30, -1.30, 0.46], [0.0, 0.0, 0.14]),
    "slope":   ([1.25, -1.25, 0.44], [0.0, 0.0, 0.14]),
}
for kind, (p, t) in cams.items():
    img, f = shot(kind, 1.0, p, t)
    frames[kind] = img; facts.append(f)
    print("ok", f, flush=True)

# Difficulty curriculum on one kind.
for d in (1.0, 2.5, 4.0):
    img, f = shot("stairs", d, [-1.35, -0.95, 0.40], [0.35, 0.0, 0.12])
    frames[f"stairs_d{d}"] = img; facts.append(f)
    print("ok", f, flush=True)

np.savez_compressed(f"{OUT}/terrain_frames.npz", **frames)
json.dump(facts, open(f"{OUT}/terrain_facts.json", "w"), indent=1)
print("SAVED", len(frames), "frames")
