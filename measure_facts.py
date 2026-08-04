"""Measure the reward-config asset path. Run once per tree; dumps JSON."""
import json, os, subprocess, sys, time
from pathlib import Path
import strands_robots.training.lerobot as L

TREE = str(Path(L.__file__).parents[2])
out = {"tree": TREE}

from strands_robots.training.base import TrainSpec
from strands_robots.training.lerobot import LerobotTrainer, _reward_friendly_fields

root = Path(sys.argv[1])           # cold HF_HOME
ds = Path(sys.argv[2]); (ds / "meta").mkdir(parents=True, exist_ok=True)
(ds / "meta" / "info.json").write_text(json.dumps({"total_episodes": 10}))

def spec(extra):
    return TrainSpec(dataset_root=str(ds), base_model="", output_dir="/tmp/o_rm",
                     steps=100, extra={"reward_model": extra})

t = LerobotTrainer(device="cpu")

# 1. validate() verdict for the default robometer spec (no network needed)
out["validate_default"] = t.validate(spec({"type": "robometer"}))

# 2. build_config for the default spec on a host that cannot reach the Hub
try:
    t.build_config(spec({"type": "robometer"}))
    out["build_default"] = {"raised": None, "msg": "built"}
except Exception as e:
    out["build_default"] = {"raised": type(e).__name__, "msg": str(e)}

# 3. build_config with the derived field supplied (the remedy)
try:
    cfg = t.build_config(spec({"type": "robometer", "vlm_config": {"text_config": {"vocab_size": 151674}}}))
    out["build_remedy"] = {"raised": None, "msg": f"built type={cfg.reward_model.type}"}
except Exception as e:
    out["build_remedy"] = {"raised": type(e).__name__, "msg": str(e)}

# 4. a bad field VALUE keeps the config's own error
try:
    t.build_config(spec({"type": "robometer", "reward_output": "not-a-mode",
                         "vlm_config": {"text_config": {"vocab_size": 151674}}}))
    out["build_bad_value"] = {"raised": None, "msg": "built"}
except Exception as e:
    out["build_bad_value"] = {"raised": type(e).__name__, "msg": str(e)}

out["vlm_config_is_forwardable"] = "vlm_config" in _reward_friendly_fields("robometer")

# 5. the parity suite itself, cold cache + offline
env = dict(os.environ, HF_HOME=str(root), HF_HUB_OFFLINE="1", MUJOCO_GL="egl")
r = subprocess.run([sys.executable, "-m", "pytest", "tests/training/test_reward_model_parity.py",
                    "-q", "--no-cov", "-p", "no:randomly", "--tb=no", "-ra"],
                   cwd=TREE, capture_output=True, text=True, env=env)
tail = [l for l in r.stdout.splitlines() if " passed" in l or " failed" in l]
out["suite_offline"] = tail[-1].strip() if tail else "?"
out["suite_offline_failed"] = sum(1 for l in r.stdout.splitlines() if l.startswith("FAILED"))

Path(sys.argv[3]).write_text(json.dumps(out, indent=2))
print("TREE:", TREE, "->", sys.argv[3])
print(json.dumps({k: v for k, v in out.items() if k != "tree"}, indent=2)[:900])
