"""Replay the extent Isaac compiled into a MuJoCo scene and render it."""
import json, sys
from pathlib import Path
import numpy as np
import strands_robots.simulation.base as _b
print("TREE:", Path(_b.__file__).parents[2])
from strands_robots.simulation import Simulation

OUT = Path("/tmp/art"); OUT.mkdir(exist_ok=True)
W, H = 760, 640

def render(extent, tag):
    """Ground + a 0.30 m reference post + (optionally) the crate at `extent`."""
    sim = Simulation(tool_name=f"art_{tag}", mesh=False)
    try:
        sim.create_world(ground_plane=True)
        # A static 0.30 m reference post: the caller's requested size, in the frame.
        sim.add_object(name="ruler", shape="box", size=[0.02, 0.02, 0.30],
                       position=[0.30, 0.0, 0.15], color=[0.20, 0.45, 0.85, 1.0], is_static=True)
        if extent is not None:
            sim.add_object(name="crate", shape="box", size=list(extent),
                           position=[0.0, 0.0, extent[2] / 2.0],
                           color=[0.95, 0.55, 0.12, 1.0], is_static=True)
        sim.add_camera(name="look", position=[0.44, -0.62, 0.34], target=[0.11, 0.0, 0.13], fov=38)
        r = sim.render(camera_name="look", width=W, height=H)
        png = next(c["image"]["source"]["bytes"] for c in r["content"] if "image" in c)
        p = OUT / f"{tag}.png"; p.write_bytes(png)
        import imageio.v3 as iio
        return np.asarray(iio.imread(p))[:, :, :3]
    finally:
        try: sim.cleanup()
        except Exception: pass

which = sys.argv[1]           # "before" | "after"
facts = json.loads(Path(f"/tmp/art/{which}.json").read_text())
honored = facts["cases"]["scale (documented alias)"]["compiled"]
wrongkey = facts["cases"]["extents (plausible, wrong)"]["compiled"]

frames = {"honored": render(honored, f"{which}_honored")}
frames["wrongkey"] = render(wrongkey, f"{which}_wrongkey")
# Framing gate: the crate must be a legible fraction of the frame.
# Warm-hue discriminator, calibrated on the real render: the crate comes out
# as [112,65,14] / [97,56,12] / [231,134,29], never at the requested RGB.
def orange(a):
    r, g, b = (a[:, :, i].astype(int) for i in range(3))
    return float(((r > g) & (g > b) & (r - b > 40)).mean())
meta = {"tree": facts["tree"], "orange_honored": float(orange(frames["honored"]))}
if wrongkey is not None:
    meta["orange_wrongkey"] = float(orange(frames["wrongkey"]))
    d = (np.abs(frames["honored"].astype(int) - frames["wrongkey"].astype(int)).sum(2) > 12).mean()
    meta["diff_honored_vs_wrongkey"] = float(d)
    print(f"  orange honored={meta['orange_honored']:.4f} wrongkey={meta['orange_wrongkey']:.4f} diff={d:.4f}")
    assert meta["orange_honored"] > 0.03, f"crate too small in frame: {meta['orange_honored']}"
    assert d > 0.10, f"panels differ on only {d:.2%} of pixels - reframe"
else:
    assert meta["orange_honored"] > 0.03
    assert orange(frames["wrongkey"]) < 0.002, "the refused panel must carry no crate"
    print(f"  orange honored={meta['orange_honored']:.4f}  (refused panel has no crate)")
Path(f"/tmp/art/{which}_render.json").write_text(json.dumps(meta, indent=2))
