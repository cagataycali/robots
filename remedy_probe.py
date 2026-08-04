from pathlib import Path
import strands_robots.training.lerobot as L
print("TREE:", Path(L.__file__).parents[2])
import transformers
# Make ANY backbone fetch fatal, so a success proves nothing was fetched.
def _boom(*a, **k):
    raise AssertionError("NETWORK: a backbone fetch was attempted")
transformers.AutoConfig.from_pretrained = _boom
transformers.AutoTokenizer.from_pretrained = _boom

from strands_robots.training.base import TrainSpec
from strands_robots.training.lerobot import LerobotTrainer

MIN_VLM = {"text_config": {"vocab_size": 151674}}
for label, rm in [
    ("robometer default", {"type": "robometer"}),
    ("robometer + vlm_config", {"type": "robometer", "vlm_config": MIN_VLM}),
    ("sarm", {"type": "sarm"}),
    ("topreward", {"type": "topreward"}),
    ("reward_classifier", {"type": "reward_classifier"}),
]:
    spec = TrainSpec(dataset_root="/tmp/ds", base_model="", output_dir="/tmp/o", steps=10,
                     extra={"reward_model": rm})
    t = LerobotTrainer(device="cpu")
    problems = t.validate(spec)
    try:
        cfg = t.build_config(spec)
        print(f"  {label:26s} validate={problems} -> BUILT type={cfg.reward_model.type} "
              f"is_rm={cfg.is_reward_model_training} vlm_keys={list(getattr(cfg.reward_model,'vlm_config',{}) or {})[:3]}")
    except Exception as e:
        print(f"  {label:26s} validate={problems} -> {type(e).__name__}: {str(e)[:100]}")
