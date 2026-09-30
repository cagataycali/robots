---
description: GPU-parallel RL through the isaaclab trainer, with Isaac Lab in its own venv.
---

# Isaac Lab training

The `isaaclab` trainer runs `python -m isaaclab train` (rsl_rl PPO) in a separate Isaac Lab venv; strands-robots never imports it (its pins conflict, and Kit exits the process that closes it).

```bash
uv venv --python 3.12 ~/il && uv pip install --python ~/il/bin/python --prerelease=allow \
  --index https://pypi.nvidia.com --index-strategy unsafe-best-match "isaaclab[rsl-rl,isaacsim]==3.0.0rc1"
export ISAACLAB_PYTHON=~/il/bin/python OMNI_KIT_ACCEPT_EULA=YES   # the EULA is yours to accept
```

```python title="sketch"
train_policy(action="train", provider="isaaclab", steps=50, output_dir="runs",
             extra={"task": "Isaac-Cartpole", "num_envs": 4096, "physics": "newton_mjwarp", "timeout_s": 600})
```

It returns a `job_id`; `action="status"` reports rewards, `success_rate`, a failure's cause and `checkpoint_dir`; `action="stop"` ends it; `action="play"` records a video; `action="export"` writes what `create_policy("rl", checkpoint_dir=...)` loads. `extra['rl_library']` accepts only `rsl_rl`. Your own task package trains once the operator sets `STRANDS_ISAACLAB_TASK_PACKAGES=module:register_fn` (importable in the Isaac Lab venv).

One L40S, 4096 envs: Cartpole 291k env steps/s, G1 flat locomotion 110k.

Caveats: Isaac Lab 3.0 is an RC; first RTX use compiles shaders (~4 min); PhysX and Newton differ, even in joint order, so export records a `deploy_contract` that `create_policy("rl")` applies by name.
