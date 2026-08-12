"""With the guard in place: Cosmos3-Edge must still load and run on Thor."""
import json, pathlib, time
import numpy as np, torch, strands_robots
import importlib.metadata as md
from strands_robots.policies.cosmos3.embodiments import get_embodiment
from strands_robots.policies.cosmos3.policy_diffusers import (
    Cosmos3DiffusersBackend, _unloaded_checkpoint_tensors,
)
TREE = str(pathlib.Path(strands_robots.__file__).parents[1])
print("TREE:", TREE, "diffusers:", md.version("diffusers"), flush=True)

emb = get_embodiment("umi")
out = {"tree": TREE, "diffusers": md.version("diffusers"), "torch": torch.__version__,
       "device_name": torch.cuda.get_device_name(0), "model": "nvidia/Cosmos3-Edge",
       "embodiment": emb.name, "domain": emb.domain_name,
       "raw_action_dim": emb.raw_action_dim, "chunk": emb.action_chunk_size, "fps": emb.fps}

t0 = time.time()
be = Cosmos3DiffusersBackend(embodiment=emb, model="nvidia/Cosmos3-Edge",
                             mode="forward_dynamics", resolution_tier=256,
                             num_inference_steps=4, guidance_scale=6.0)
out["load_seconds"] = round(time.time() - t0, 2)
out["device"] = be.device
out["unloaded_after_guard"] = len(_unloaded_checkpoint_tensors(be._pipeline))
out["n_params_scanned"] = sum(
    1 for c in be._pipeline.components.values()
    if callable(getattr(c, "named_parameters", None)) for _ in c.named_parameters()
)
print(f"LOAD OK {out['load_seconds']}s device={be.device} unloaded={out['unloaded_after_guard']}", flush=True)

rng = np.random.default_rng(0)
frame = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
raw = np.zeros((emb.action_chunk_size, emb.raw_action_dim), dtype=np.float32)
raw[:, 0] = np.linspace(0.0, 0.05, emb.action_chunk_size)
obs = {"prompt": "the robot gripper moves forward", "observation/image": frame}

torch.cuda.reset_peak_memory_stats()
t0 = time.time()
res = be.infer(obs, raw_actions=raw)
out["infer_seconds"] = round(time.time() - t0, 2)
out["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
v = np.asarray(res["video"])
out["video_shape"] = list(v.shape)
out["action_is_none_for_forward_dynamics"] = res["action"] is None
np.save("/tmp/art_video.npy", v)
print(json.dumps(out, indent=2), flush=True)
pathlib.Path("/tmp/art_healthy.json").write_text(json.dumps(out, indent=2))
