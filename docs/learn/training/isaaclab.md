---
description: GPU-parallel RL through the isaaclab trainer, with Isaac Lab in its own venv.
---

# Isaac Lab training

The `isaaclab` trainer runs `python -m isaaclab train` (rsl_rl PPO) in a separate Isaac Lab venv, reading its log. strands-robots never imports it: its pins conflict, and Kit exits the process that closes it.

```bash
uv venv --python 3.12 ~/il && uv pip install --python ~/il/bin/python --prerelease=allow \
  --index https://pypi.nvidia.com --index-strategy unsafe-best-match "isaaclab[rsl-rl,isaacsim]==3.0.0rc1"
export ISAACLAB_PYTHON=~/il/bin/python OMNI_KIT_ACCEPT_EULA=YES   # the EULA is yours to accept
```

```python title="sketch"
train_policy(action="train", provider="isaaclab", steps=50, output_dir="runs",
             extra={"task": "Isaac-Cartpole", "num_envs": 4096, "physics": "newton_mjwarp", "timeout_s": 600})
```

It returns a `job_id`; `action="status"` reports rewards, `success_rate`, `learning`, a failure's cause and `checkpoint_dir`; `action="stop"` ends a run. `steps` counts PPO iterations. `extra['wait']` blocks until the run ends; `extra['rl_library']` accepts only `rsl_rl`.

Measured on one L40S with 4096 envs: Cartpole 291k env steps/s, G1 flat locomotion 110k.

Caveats: Isaac Lab 3.0 is an RC; first RTX use compiles shaders (~4 min); PhysX and Newton results differ (`strands_run.json` beside the checkpoints names the preset); `create_policy("rl")` cannot load rsl_rl checkpoints yet.
