import contextlib, importlib, io, pathlib, sys
import strands_robots
print("TREE:", pathlib.Path(strands_robots.__file__).parents[1])
importlib.import_module("lerobot.policies")  # register_subclass side effect
import draccus
from lerobot.configs.train import TrainPipelineConfig
import lerobot
print("lerobot", lerobot.__version__)

base = ["--policy.type=act", "--dataset.repo_id=x/y", "--policy.push_to_hub=false"]
cases = {
    "bare push_to_hub=true":   base + ["--push_to_hub=true"],
    "prefixed policy.push_to_hub=true": ["--policy.type=act", "--dataset.repo_id=x/y", "--policy.push_to_hub=true"],
    "bare pretrained_path":    base + ["--pretrained_path=/tmp/x"],
    "control (no extra)":      base,
}
for label, argv in cases.items():
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            cfg = draccus.parse(config_class=TrainPipelineConfig, args=argv)
        print(f"{label:38s} -> PARSED  policy.push_to_hub={getattr(cfg.policy,'push_to_hub','<none>')}")
    except SystemExit as e:
        out = buf.getvalue()
        line = [l for l in out.splitlines() if "unrecognized" in l or "error" in l.lower()]
        print(f"{label:38s} -> SystemExit({e.code}) {line[:1]}")
    except Exception as e:
        print(f"{label:38s} -> {type(e).__name__}: {str(e)[:110]}")
# does the dataclass even have a top-level push_to_hub field?
import dataclasses
tops = {f.name for f in dataclasses.fields(TrainPipelineConfig)}
print("TrainPipelineConfig has top-level 'push_to_hub'? ", "push_to_hub" in tops)
from lerobot.configs.policies import PreTrainedConfig
print("PreTrainedConfig fields incl push_to_hub? ", "push_to_hub" in {f.name for f in dataclasses.fields(type(draccus.parse(config_class=TrainPipelineConfig, args=base).policy))})
