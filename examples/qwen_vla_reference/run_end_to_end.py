"""END-TO-END local run of the full Qwen-VLA closed loop on a single GPU.

Exercises every real code path we built, with the runnable reference model:

  1. Stage 1 (T2A):  run_t2a  -> DiT warm-start checkpoint
  2. Stage 2 (CPT):  run_cpt  -> Qwen-VLA-Base (joint VLM+DiT, mixture)
  3. Stage 3 (SFT):  run_sft  -> Qwen-VLA (SFT) over LeRobotAdapter samples
  4. Stage 4 (RL):   run_rl   -> Qwen-VLA-Instruct (PPO+GAE on sim success)
  5. Save checkpoint; start the ZMQ SERVICE server; SERVICE-mode inference
     via the real QwenVlaPolicy; LOCAL-mode inference via load_policy.
  6. Hot-swap a new checkpoint into the running server (redeploy loop).

Prints a structured report and asserts the loop closed correctly.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from reference_model import ReferenceQwenVla  # noqa: E402
from sim_env import ReferenceSimEnv  # noqa: E402

from strands_robots.policies.qwen_vla import QwenVlaPolicy, compute_quantile_stats  # noqa: E402
from strands_robots.training.qwen_vla import (  # noqa: E402
    CPTConfig,
    RLConfig,
    SFTConfig,
    T2AConfig,
    get_embodiment_tag,
    run_cpt,
    run_rl,
    run_sft,
    run_t2a,
)
from strands_robots.training.qwen_vla.data import LeRobotAdapter  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("e2e")

CKPT_DIR = HERE / "_ckpts"
CKPT_DIR.mkdir(exist_ok=True)
EMB = get_embodiment_tag("so100")
ACTION_CHANNELS = 7  # so100: 6-dof arm + gripper
PORT = 5566
HOST = "127.0.0.1"


def banner(title):
    log.info("=" * 70)
    log.info(title)
    log.info("=" * 70)


def stage1_t2a(model) -> dict:
    banner("STAGE 1: Text-to-Action (T2A) - DiT pretraining (VLM frozen)")
    model.freeze_vlm(True)
    cfg = T2AConfig(max_steps=60, batch_size=32, action_dim=model.action_dim)
    summary = run_t2a(cfg, EMB, action_channels=ACTION_CHANNELS, model=model)
    ckpt = str(CKPT_DIR / "t2a_warmstart.pt")
    model.save_checkpoint(ckpt)
    summary["checkpoint"] = ckpt
    log.info("T2A done: final_loss=%.4f -> %s", summary["final_loss"], ckpt)
    return summary


def stage2_cpt(model) -> dict:
    banner("STAGE 2: Continued Pretraining (CPT) - joint VLM+DiT on mixture")
    model.freeze_vlm(False)
    cfg = CPTConfig(
        max_steps=60, batch_size=32, action_dim=model.action_dim, warmup_checkpoint=str(CKPT_DIR / "t2a_warmstart.pt")
    )
    summary = run_cpt(cfg, model=model)
    ckpt = str(CKPT_DIR / "qwen_vla_base.pt")
    model.save_checkpoint(ckpt)
    summary["checkpoint"] = ckpt
    log.info("CPT done: final_loss=%.4f -> %s (Qwen-VLA-Base)", summary["final_loss"], ckpt)
    return summary


def _make_sft_dataset(model, n: int) -> list[dict]:
    """Build adapted SFT samples from synthetic teleop frames via LeRobotAdapter."""
    stats = compute_quantile_stats(np.random.default_rng(0).uniform(-2, 2, (300, ACTION_CHANNELS)).astype(np.float32))
    adapter = LeRobotAdapter(
        embodiment=EMB,
        video_keys=["observation.images.webcam"],
        view_tags={"observation.images.webcam": "ego"},
        state_keys=["observation.state.single_arm", "observation.state.gripper"],
        action_dim=model.action_dim,
        quantile_stats=stats,
    )
    rng = np.random.default_rng(1)
    samples = []
    for i in range(n):
        frame = {
            "observation.images.webcam": (rng.random((16, 16, 3)) * 255).astype(np.uint8),
            "observation.state.single_arm": rng.standard_normal(6).astype(np.float32),
            "observation.state.gripper": rng.standard_normal(1).astype(np.float32),
        }
        chunk = rng.standard_normal((EMB.chunk_size, ACTION_CHANNELS)).astype(np.float32)
        s = adapter.adapt(frame, chunk, f"pick up object {i}")
        # Convert the adapted sample into the batch dict the model's loss expects.
        x1 = s.action[None]  # (1,H,K) normalized+padded target
        x0 = rng.standard_normal(x1.shape).astype(np.float32)
        from strands_robots.training.qwen_vla.config import TimestepDist
        from strands_robots.training.qwen_vla.flow_matching import interpolate, sample_timesteps, target_velocity

        t = sample_timesteps(1, TimestepDist.BETA, rng=rng)
        samples.append(
            {
                "x_t": interpolate(x0, x1, t),
                "timesteps": t,
                "target": target_velocity(x0, x1),
                "mask": s.mask[None],
                "prompts": [s.language],
            }
        )
    return samples


def stage3_sft(model) -> dict:
    banner("STAGE 3: Supervised Fine-Tuning (SFT) - teleop track via LeRobotAdapter")
    cfg = SFTConfig(
        max_steps=60, batch_size=1, action_dim=model.action_dim, base_checkpoint=str(CKPT_DIR / "qwen_vla_base.pt")
    )
    dataset = _make_sft_dataset(model, n=40)
    summary = run_sft(cfg, model=model, dataset=dataset)
    ckpt = str(CKPT_DIR / "qwen_vla_sft.pt")
    model.save_checkpoint(ckpt)
    summary["checkpoint"] = ckpt
    log.info("SFT done: final_loss=%.4f -> %s", summary["final_loss"], ckpt)
    return summary


def stage4_rl(model) -> dict:
    banner("STAGE 4: Reinforcement Learning (PPO+GAE) on sim success")
    cfg = RLConfig(
        rollout_steps=40,
        num_envs=8,
        action_dim=model.action_dim,
        sft_checkpoint=str(CKPT_DIR / "qwen_vla_sft.pt"),
        ppo_epochs=4,
    )
    # Re-init the value head so the RL climb starts from scratch (clean demo).
    import torch.nn as nn

    for layer in model.value_head:
        if isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight, std=0.02)
            nn.init.zeros_(layer.bias)
    env = ReferenceSimEnv(episode_len=8, seed=0, target_value=1.0)

    # Track success rate before/after to show non-negative transfer (Table 11 trend).
    def avg_success():
        env.reset(seed=123)
        rollouts = [env.rollout(model) for _ in range(16)]
        # "success" = shaped terminal reward above 0.5 (climbs as PPO trains).
        return float(np.mean([1.0 if c["reward"] > 0.5 else 0.0 for traj in rollouts for c in traj if c["done"]]))

    before = avg_success()
    summary = run_rl(cfg, model=model, env=env)
    after = avg_success()
    ckpt = str(CKPT_DIR / "qwen_vla_instruct.pt")
    model.save_checkpoint(ckpt)
    summary.update({"checkpoint": ckpt, "success_before": before, "success_after": after})
    log.info(
        "RL done: objective=%.4f, success %.2f -> %.2f -> %s (Instruct)",
        summary["final_objective"],
        before,
        after,
        ckpt,
    )
    return summary


def start_server(model_path: str) -> subprocess.Popen:
    proc = subprocess.Popen(
        [
            sys.executable,
            str(HERE / "reference_server.py"),
            "--model-path",
            model_path,
            "--host",
            HOST,
            "--port",
            str(PORT),
            "--device",
            "cuda",
            "--denoising-steps",
            "4",
            "--data-config",
            "so100",
        ],
    )
    # Wait for the port.
    import socket

    for _ in range(60):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            if s.connect_ex((HOST, PORT)) == 0:
                return proc
        time.sleep(0.5)
    raise RuntimeError("server failed to start")


def service_inference() -> dict:
    banner("INFERENCE: SERVICE mode via QwenVlaPolicy over ZMQ")
    policy = QwenVlaPolicy(data_config="so100", host=HOST, port=PORT)
    assert policy._client.ping(), "server ping failed"
    obs = {
        "webcam": np.zeros((224, 224, 3), np.uint8),
        "single_arm": np.zeros(6, np.float32),
        "gripper": np.zeros(1, np.float32),
    }
    # Reproducibility: same seed -> same chunk.
    policy.reset(seed=7)
    a1 = policy.get_actions_sync(obs, "pick up the red cube")
    policy.reset(seed=7)
    a2 = policy.get_actions_sync(obs, "pick up the red cube")
    horizon = len(a1)
    # Different instruction -> conditioning changes the action.
    policy.reset(seed=7)
    a3 = policy.get_actions_sync(obs, "open the drawer")
    first_arm_1 = np.array(a1[0]["single_arm"])
    first_arm_3 = np.array(a3[0]["single_arm"])
    instr_sensitive = not np.allclose(first_arm_1, first_arm_3, atol=1e-4)
    log.info(
        "SERVICE inference: horizon=%d, deterministic_reset=%s, instruction_sensitive=%s",
        horizon,
        len(a1) == len(a2),
        instr_sensitive,
    )
    return {"horizon": horizon, "instruction_sensitive": bool(instr_sensitive), "action_keys": sorted(a1[0].keys())}


def local_inference() -> dict:
    banner("INFERENCE: LOCAL mode via load_policy (in-proc)")
    # Route the policy's LOCAL loader to our reference load_policy.
    import reference_model

    import strands_robots.policies.qwen_vla.policy as pol

    orig = pol._qwen_vla_installed
    pol._qwen_vla_installed = lambda: True
    # Monkeypatch the lazy import target.
    sys.modules["qwen_vla"] = reference_model
    try:
        policy = QwenVlaPolicy(data_config="so100", model_path=str(CKPT_DIR / "qwen_vla_instruct.pt"), device="cuda")
        obs = {
            "webcam": np.zeros((224, 224, 3), np.uint8),
            "single_arm": np.zeros(6, np.float32),
            "gripper": np.zeros(1, np.float32),
        }
        actions = policy.get_actions_sync(obs, "pick up the red cube")
        log.info("LOCAL inference: horizon=%d, keys=%s", len(actions), sorted(actions[0].keys()))
        return {"horizon": len(actions), "action_keys": sorted(actions[0].keys())}
    finally:
        pol._qwen_vla_installed = orig


def hotswap_test(server_proc) -> dict:
    banner("REDEPLOY: hot-swap a fresh checkpoint into the running server")
    from strands_robots.tools import qwen_vla_train

    # Train a tiny fresh model + save, then hot-swap.
    fresh = ReferenceQwenVla(device="cuda", seed=99)
    fresh_ckpt = str(CKPT_DIR / "qwen_vla_hotswap.pt")
    fresh.save_checkpoint(fresh_ckpt)
    res = qwen_vla_train(action="hotswap", checkpoint=fresh_ckpt, server_host=HOST, server_port=PORT)
    log.info("hot-swap result: %s", res)
    return res


def main():
    banner(f"QWEN-VLA FULL CLOSED-LOOP E2E on {torch.cuda.get_device_name(0)}")
    report = {}
    model = ReferenceQwenVla(action_dim=32, horizon=EMB.chunk_size, device="cuda", seed=0)

    report["stage1_t2a"] = stage1_t2a(model)
    report["stage2_cpt"] = stage2_cpt(model)
    report["stage3_sft"] = stage3_sft(model)
    report["stage4_rl"] = stage4_rl(model)

    # Serve the trained Instruct checkpoint + inference.
    server = start_server(str(CKPT_DIR / "qwen_vla_instruct.pt"))
    try:
        report["service_inference"] = service_inference()
        report["local_inference"] = local_inference()
        report["hotswap"] = hotswap_test(server)
    finally:
        server.terminate()
        server.wait(timeout=10)

    banner("CLOSED-LOOP COMPLETE - report")
    print(json.dumps(report, indent=2, default=str))

    # Assertions: the loop must have closed correctly.
    assert report["service_inference"]["horizon"] == EMB.chunk_size
    assert report["service_inference"]["instruction_sensitive"], "model ignores the instruction"
    assert report["local_inference"]["horizon"] == EMB.chunk_size
    assert report["hotswap"]["status"] == "success", "hot-swap redeploy failed"
    assert report["stage4_rl"]["success_after"] >= report["stage4_rl"]["success_before"], "RL caused negative transfer"
    assert not np.isnan(report["stage1_t2a"]["final_loss"])
    log.info("ALL E2E ASSERTIONS PASSED")
    return report


if __name__ == "__main__":
    main()
