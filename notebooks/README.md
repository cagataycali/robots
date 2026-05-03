# PR #85 — MuJoCo sim notebooks

End-to-end learning materials for the simulation backend in
[PR #85 · feat: MuJoCo simulation backend](https://github.com/strands-labs/robots/pull/85).

All notebooks are **committed with their outputs baked in** (including MP4
videos of the actual VLA rollouts) so reviewers can browse on GitHub and
watch the simulation running, without installing MuJoCo or downloading
model weights locally.

## Contents

| # | Notebook | Focus | Runtime |
|---|----------|-------|---------|
| 1 | [`01_mujoco_quickstart.ipynb`](01_mujoco_quickstart.ipynb) | Learn the sim API: create world → add robot → step → render → send_action → record MP4 | ~10s |
| 2 | [`02_vla_inference.ipynb`](02_vla_inference.ipynb) | **Load real [SmolVLA](https://huggingface.co/lerobot/smolvla_base) and watch it drive the SO-101 arm from a text prompt** | ~30s (+ 15s cold model load) |
| 3 | [`03_multi_robot_vla.ipynb`](03_multi_robot_vla.ipynb) | Two robots, two SmolVLA rollouts, one LeRobot dataset with per-robot joint prefixing (the headline feature of this PR) | ~45s |

## Running locally

```bash
# from the PR branch (feat/mujoco-backend)
pip install -e ".[all,dev,sim-mujoco]"
pip install lerobot[smolvla] jupyter ipykernel matplotlib

# register a kernel that points at this repo's venv
python -m ipykernel install --user --name pr85 --display-name "PR85 (strands-robots)"

# set env vars for VLA + headless GL
export STRANDS_TRUST_REMOTE_CODE=true   # needed for HF models with custom code
export MUJOCO_GL=glfw                   # macOS; use osmesa on Linux headless

# execute all cells in place
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=pr85 \
    --ExecutePreprocessor.timeout=1800 \
    notebooks/*.ipynb
```

## What's actually in each notebook (baked outputs)

### nb 01 — Quickstart
* 1 PNG preview render of the initial scene
* **2 embedded MP4 videos** (front cam + wrist cam) of the arm reaching a
  commanded pose
* Observation-shape + dataset-tree printouts

### nb 02 — VLA inference (headline demo)
* 1 PNG scene preview
* **2 embedded MP4 videos** — front cam + wrist cam of the **actual SmolVLA
  rollout** responding to the prompt *"grasp the green cube"*
* Parquet inspection showing the 60 VLA action vectors
* 1 PNG trajectory plot of the 6 commanded joint targets over time

Execution log (M-series MPS, MuJoCo 3.8, lerobot 0.5.1):
```
loading lerobot/smolvla_base on MPS (this takes ~15s cold)...
loaded in 13.1s
running SmolVLA @ 20 Hz for 3 seconds (60 inference steps)...
run_policy: success — wall time 9.5s (~6.3 Hz effective)
```

### nb 03 — Multi-robot VLA + dataset
* 1 PNG top-view of the dual-robot world
* **3 embedded MP4 videos** — top view + alice's wrist cam + bob's wrist cam
* `info.json` schema dump proving per-robot prefixed joint names:
  ```
  observation.state.names (12 = 6 alice + 6 bob):
    alice__shoulder_pan   alice__shoulder_lift   alice__elbow_flex
    alice__wrist_flex     alice__wrist_roll      alice__gripper
    bob__shoulder_pan     bob__shoulder_lift     bob__elbow_flex
    bob__wrist_flex       bob__wrist_roll        bob__gripper
  ```
* 1 matplotlib plot with both robots' action trajectories in stacked subplots
* Backwards-compat control: single-robot scene still produces flat names

## Why these live on a sibling branch

* Output-baked notebooks with embedded videos are **large** (100KB–1.1MB each)
* Pulls in heavy deps (`mujoco`, `lerobot`, `transformers`, `torch`, ffmpeg)
* Reviewers can browse them on GitHub with rendered video output, no
  install needed — just click the notebook on the web UI.

## Hardware / model notes

* Tested on **Apple M-series (MPS)** — SmolVLA runs at ~6 Hz end-to-end
* Also works on **CUDA** (`device="cuda"`) and **CPU** (slow but works)
* First run downloads `lerobot/smolvla_base` (~2 GB) to `~/.cache/huggingface/`
  — subsequent runs are cache hits

## Swap in other VLAs

```python
# π0 flow-matching VLA (Physical Intelligence)
create_policy("lerobot_local", pretrained_name_or_path="lerobot/pi0",
              device="mps", actions_per_step=1)

# NVIDIA GR00T N1.7 (requires a separate ZMQ inference server)
create_policy("groot", host="localhost", port=5555, data_config="so101")
```
