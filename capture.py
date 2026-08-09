"""Render real MuJoCo frames, encode them at each candidate quality, record the outcome."""
import hashlib, json, math, pathlib, sys
import numpy as np
import strands_robots.rendering.video as vid
TREE = str(pathlib.Path(vid.__file__).parents[2])
print("TREE:", TREE, "__debug__ =", __debug__); sys.stdout.flush()

OUT = pathlib.Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
NPY = pathlib.Path("/tmp/art_frames.npy")

if NPY.exists():
    frames = [f for f in np.load(NPY)]
    print("reused", len(frames), "frames")
else:
    from strands_robots import Robot
    sim = Robot("panda", mode="sim", mesh=False)
    sim.add_camera(name="look", position=[0.85, -0.62, 0.62], target=[0.35, 0.0, 0.42], fov=34)
    keys = [f"actuator{i}" for i in (1, 2, 3, 5, 7)]
    frames = []
    for step in range(24):
        f = step / 23.0
        sim.send_action({k: -0.42 + 0.84 * f for k in keys}, robot_name="panda", n_substeps=10)
        r = sim.render(camera_name="look", width=640, height=480)
        png = next(c["image"]["source"]["bytes"] for c in r["content"] if "image" in c)
        import io
        from PIL import Image
        frames.append(np.asarray(Image.open(io.BytesIO(png)).convert("RGB")))
    np.save(NPY, np.stack(frames))
    print("rendered", len(frames), "frames", frames[0].shape)
    sim.cleanup()

def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse == 0 else 10.0 * math.log10(255.0**2 / mse)

facts = {"tree": TREE, "optimized": not __debug__, "rows": {}}
import imageio.v3 as iio
for label, q in [("8", 8), ("1", 1), ("True", True), ("0", 0), ("-5", -5), ("500", 500),
                 ("nan", math.nan), ("'8'", "8"), ("np.int64(8)", np.int64(8))]:
    p = OUT / f"q_{label.replace(chr(39),'').replace('.','_').replace('(','').replace(')','')}.mp4"
    row = {}
    try:
        o = vid.encode_clip(list(frames), p, fps=12, quality=q)
        dec = list(iio.imiter(str(o)))
        row = {"outcome": "encoded", "size": o.stat().st_size,
               "md5": hashlib.md5(o.read_bytes()).hexdigest()[:12],
               "frames": len(dec), "psnr": round(psnr(frames[12], dec[12]), 2), "path": str(o)}
    except Exception as e:
        row = {"outcome": type(e).__name__, "message": str(e)[:150], "file_left": p.exists()}
    facts["rows"][label] = row
    print(f"  quality={label:12s} {row.get('outcome'):16s} {row.get('md5','')} psnr={row.get('psnr','-')}")
    sys.stdout.flush()

(OUT / "facts.json").write_text(json.dumps(facts, indent=1))
print("WROTE", OUT / "facts.json")
