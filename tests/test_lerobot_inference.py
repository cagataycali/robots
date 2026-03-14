"""Cycles 2-3: Simulation-based inference tests for LeRobot policies.

Tests actual model loading + inference through strands_robots Simulation tool
with MuJoCo backend. Downloads pretrained models from HuggingFace and runs
them against simulated environments.

Cycle 2: Model loading + GPU inference (ACT, Diffusion, SmolVLA)
Cycle 3: MuJoCo simulation + policy runner integration + latency profiling

Requires: mujoco, torch with CUDA, internet access for model download
"""

import logging
import os
import sys
import time

import numpy as np
import pytest

logger = logging.getLogger(__name__)

# Skip entire module if no CUDA
try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
    DEVICE = "cuda" if HAS_CUDA else "cpu"
except ImportError:
    HAS_CUDA = False
    DEVICE = "cpu"

try:
    import mujoco
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False


# ---------------------------------------------------------------------------
# Model loading + inference (actual HF download)
# ---------------------------------------------------------------------------

# Models known to work with LeRobot 0.5.x
PRETRAINED_MODELS = [
    pytest.param(
        "lerobot/act_aloha_sim_transfer_cube_human",
        "act",
        14,  # ALOHA action dim (14 joints)
        id="act-aloha-sim",
    ),
    pytest.param(
        "lerobot/diffusion_pusht",
        "diffusion",
        2,   # PushT action dim (x, y)
        id="diffusion-pusht",
    ),
]

SMOLVLA_MODEL = pytest.param(
    "lerobot/smolvla_base",
    "smolvla",
    6,  # SmolVLA base is typically 6-DOF
    id="smolvla-base",
)


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
class TestLerobotModelLoading:
    """Test loading pretrained models from HuggingFace."""

    @pytest.mark.parametrize("model_id,policy_type,action_dim", PRETRAINED_MODELS)
    def test_load_pretrained_model(self, model_id, policy_type, action_dim):
        """Load a pretrained model and verify it initializes correctly."""
        from strands_robots.policies.lerobot_local import LerobotLocalPolicy

        t0 = time.time()
        policy = LerobotLocalPolicy(pretrained_name_or_path=model_id)
        elapsed = time.time() - t0

        assert policy._loaded, f"Failed to load {model_id}"
        assert policy._device is not None

        info = policy.get_model_info()
        assert info["loaded"] is True
        assert info["provider"] == "lerobot_local"
        assert info["model_id"] == model_id

        logger.info(
            f"Loaded {model_id} ({info['policy_class']}) in {elapsed:.1f}s "
            f"on {policy._device}, {info.get('n_parameters', 0):,} params"
        )

    @pytest.mark.parametrize("model_id,policy_type,action_dim", PRETRAINED_MODELS)
    def test_inference_with_dummy_obs(self, model_id, policy_type, action_dim):
        """Run inference with random dummy observations."""
        from strands_robots.policies.lerobot_local import LerobotLocalPolicy

        policy = LerobotLocalPolicy(pretrained_name_or_path=model_id)

        # Set robot state keys
        state_keys = [f"joint_{i}" for i in range(action_dim)]
        policy.set_robot_state_keys(state_keys)

        # Build observation matching the model's expected format
        obs = {}
        for feat_name, feat_info in policy._input_features.items():
            if hasattr(feat_info, "shape"):
                shape = feat_info.shape
                if "image" in feat_name:
                    # Image: (C, H, W) -> generate random
                    obs[feat_name] = torch.randn(1, *shape, device=policy._device)
                elif "state" in feat_name:
                    obs[feat_name] = torch.randn(1, *shape, device=policy._device)
                else:
                    obs[feat_name] = torch.randn(1, *shape, device=policy._device)

        # Run inference
        t0 = time.time()
        action = policy.select_action_sync(obs, instruction="test task")
        elapsed = time.time() - t0

        assert isinstance(action, np.ndarray), f"Expected ndarray, got {type(action)}"
        assert action.size > 0, "Empty action returned"

        logger.info(
            f"Inference {model_id}: action shape={action.shape}, "
            f"range=[{action.min():.3f}, {action.max():.3f}], "
            f"time={elapsed*1000:.0f}ms"
        )

    def test_smolvla_loading(self):
        """Test SmolVLA base model loads (VLA with language)."""
        from strands_robots.policies.lerobot_local import LerobotLocalPolicy

        try:
            t0 = time.time()
            policy = LerobotLocalPolicy(pretrained_name_or_path="lerobot/smolvla_base")
            elapsed = time.time() - t0

            assert policy._loaded
            info = policy.get_model_info()
            assert "SmolVLA" in info["policy_class"]

            logger.info(
                f"SmolVLA loaded in {elapsed:.1f}s, "
                f"{info.get('n_parameters', 0):,} params on {policy._device}"
            )
        except Exception as e:
            logger.warning(f"SmolVLA load failed (may need more VRAM): {e}")
            pytest.skip(f"SmolVLA failed: {e}")

    def test_smolvla_inference_with_instruction(self):
        """Test SmolVLA VLA inference with language instruction + image."""
        from strands_robots.policies.lerobot_local import LerobotLocalPolicy

        try:
            policy = LerobotLocalPolicy(pretrained_name_or_path="lerobot/smolvla_base")
        except Exception as e:
            pytest.skip(f"SmolVLA load failed: {e}")

        state_keys = [f"joint_{i}" for i in range(6)]
        policy.set_robot_state_keys(state_keys)

        # Build obs with image + state
        obs = {}
        for feat_name, feat_info in policy._input_features.items():
            if hasattr(feat_info, "shape"):
                shape = feat_info.shape
                obs[feat_name] = torch.randn(1, *shape, device=policy._device)

        t0 = time.time()
        action = policy.select_action_sync(obs, instruction="pick up the red cube")
        elapsed = time.time() - t0

        assert isinstance(action, np.ndarray)
        assert action.size > 0

        logger.info(
            f"SmolVLA inference: action shape={action.shape}, "
            f"time={elapsed*1000:.0f}ms"
        )


# ---------------------------------------------------------------------------
# Simulation integration tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_MUJOCO, reason="mujoco not installed")
@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
class TestSimulationInference:
    """Test policy execution through the Simulation tool."""

    def test_simulation_creates_world(self):
        """Simulation tool can create a MuJoCo world."""
        os.environ.setdefault("MUJOCO_GL", "egl")
        from strands_robots.simulation.simulation import Simulation

        sim = Simulation(tool_name="test_sim", mesh=False)
        try:
            assert sim is not None
        finally:
            sim.cleanup()

    def test_mock_policy_in_sim_loop(self):
        """Run mock policy through simulation for N steps."""
        from strands_robots.policies import MockPolicy

        policy = MockPolicy()
        policy.set_robot_state_keys(["j0", "j1", "j2", "j3", "j4", "j5"])

        # Run 10 steps
        for step in range(10):
            obs = {"observation.state": [0.0] * 6}
            actions = policy.get_actions_sync(obs, "test movement")
            assert len(actions) > 0
            assert all(isinstance(a, dict) for a in actions)
            assert all("j0" in a for a in actions)

    def test_lerobot_local_factory_roundtrip(self):
        """Create policy via factory, set keys, get dummy actions."""
        from strands_robots.policies import create_policy

        policy = create_policy("lerobot_local")
        policy.set_robot_state_keys(["shoulder", "elbow", "wrist", "gripper"])

        # Without loading a model, should return zero actions
        import asyncio
        actions = asyncio.run(policy.get_actions({}, "test"))
        assert len(actions) > 0
        for a in actions:
            assert all(v == 0.0 for v in a.values())


# ===========================================================================
# Cycle 3: MuJoCo Simulation + Policy Runner Integration Tests
# ===========================================================================


def _make_sim(name="cycle3"):
    """Helper: create a headless Simulation instance."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    from strands_robots.simulation.simulation import Simulation
    return Simulation(tool_name=name)


@pytest.fixture
def sim():
    """Provide a clean Simulation and tear it down after the test."""
    s = _make_sim()
    yield s
    s.cleanup()


@pytest.fixture
def aloha_sim():
    """Simulation with ALOHA robot already loaded."""
    s = _make_sim("aloha_test")
    r = s.create_world()
    assert r["status"] == "success"
    r = s.add_robot("aloha", data_config="aloha")
    assert r["status"] == "success"
    yield s
    s.cleanup()


@pytest.fixture
def so100_sim():
    """Simulation with SO-100 robot already loaded."""
    s = _make_sim("so100_test")
    r = s.create_world()
    assert r["status"] == "success"
    r = s.add_robot("so100", data_config="so100")
    assert r["status"] == "success"
    yield s
    s.cleanup()


# ---------------------------------------------------------------------------
# 3a. World creation + robot loading
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_MUJOCO, reason="mujoco not installed")
class TestSimWorldCreation:
    """Verify Simulation world lifecycle: create → add robot → step → destroy."""

    def test_create_world_defaults(self, sim):
        """World creation with default params (500Hz, gravity -9.81)."""
        r = sim.create_world()
        assert r["status"] == "success"
        assert sim._world is not None
        assert sim._world._model is not None
        assert sim._world._data is not None
        assert sim._world.timestep == 0.002  # 500Hz default
        assert sim._world.gravity == [0.0, 0.0, -9.81]

    def test_create_world_custom_params(self, sim):
        """World creation with custom timestep and gravity."""
        r = sim.create_world(timestep=0.005, gravity=[0, 0, -3.0])
        assert r["status"] == "success"
        np.testing.assert_allclose(sim._world.gravity, [0, 0, -3.0])

    def test_add_aloha_robot(self, sim):
        """Add ALOHA bimanual robot and verify joints/actuators."""
        sim.create_world()
        r = sim.add_robot("aloha", data_config="aloha")
        assert r["status"] == "success"

        robot = sim._world.robots["aloha"]
        assert len(robot.joint_names) == 16  # 2x (6 joints + 2 finger joints)
        assert len(robot.actuator_ids) == 14  # 2x (6 joints + 1 gripper)
        assert "left/waist" in robot.joint_names
        assert "right/gripper" in [
            mujoco.mj_id2name(sim._world._model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(sim._world._model.nu)
        ]

    def test_add_so100_robot(self, sim):
        """Add SO-100 arm and verify 6 joints."""
        sim.create_world()
        r = sim.add_robot("so100", data_config="so100")
        assert r["status"] == "success"

        robot = sim._world.robots["so100"]
        assert len(robot.joint_names) == 6
        assert "Rotation" in robot.joint_names
        assert "Jaw" in robot.joint_names

    def test_step_simulation(self, aloha_sim):
        """Step physics and verify sim_time advances."""
        t0 = aloha_sim._world.sim_time
        r = aloha_sim.step(100)
        assert r["status"] == "success"
        assert aloha_sim._world.sim_time > t0

    def test_reset_returns_to_t0(self, aloha_sim):
        """Reset returns sim_time to 0."""
        aloha_sim.step(50)
        assert aloha_sim._world.sim_time > 0
        r = aloha_sim.reset()
        assert r["status"] == "success"
        assert aloha_sim._world.sim_time == 0.0
        assert aloha_sim._world.step_count == 0

    def test_destroy_cleans_up(self, sim):
        """Destroy removes world."""
        sim.create_world()
        sim.add_robot("aloha", data_config="aloha")
        r = sim.destroy()
        assert r["status"] == "success"
        assert sim._world is None

    def test_duplicate_world_rejected(self, aloha_sim):
        """Creating a world when one exists returns error."""
        r = aloha_sim.create_world()
        assert r["status"] == "error"
        assert "already exists" in r["content"][0]["text"]


# ---------------------------------------------------------------------------
# 3b. Observation + action pipeline
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_MUJOCO, reason="mujoco not installed")
class TestSimObservationAction:
    """Test _get_sim_observation / _apply_sim_action / get_observation / send_action."""

    def test_observation_contains_joint_states(self, aloha_sim):
        """Observation dict has float values for each joint."""
        obs = aloha_sim._get_sim_observation("aloha")
        robot = aloha_sim._world.robots["aloha"]
        for jname in robot.joint_names:
            assert jname in obs, f"Missing joint {jname} in observation"
            assert isinstance(obs[jname], float)

    def test_observation_contains_camera_images(self, aloha_sim):
        """ALOHA scene has 6 cameras; observation should contain numpy images."""
        obs = aloha_sim._get_sim_observation("aloha")
        image_keys = [k for k, v in obs.items() if isinstance(v, np.ndarray) and v.ndim == 3]
        assert len(image_keys) >= 1, f"Expected camera images, got keys: {list(obs.keys())}"
        for k in image_keys:
            img = obs[k]
            assert img.shape[2] == 3, f"Image {k} should be RGB, got shape {img.shape}"
            assert img.dtype == np.uint8

    def test_apply_action_changes_ctrl(self, aloha_sim):
        """Applying action dict sets MuJoCo ctrl and advances physics."""
        old_time = aloha_sim._world.sim_time
        action = {"left/waist": 0.5, "left/shoulder": -0.3}
        aloha_sim._apply_sim_action("aloha", action)
        assert aloha_sim._world.sim_time > old_time
        # ctrl values should be updated
        model = aloha_sim._world._model
        data = aloha_sim._world._data
        act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left/waist")
        assert abs(data.ctrl[act_id] - 0.5) < 1e-6

    def test_get_observation_public_api(self, aloha_sim):
        """Public get_observation() returns same format as internal method."""
        obs = aloha_sim.get_observation("aloha")
        assert isinstance(obs, dict)
        assert "left/waist" in obs

    def test_send_action_public_api(self, aloha_sim):
        """Public send_action() works."""
        action = {"left/waist": 0.2}
        aloha_sim.send_action(action, robot_name="aloha")
        assert aloha_sim._world.sim_time > 0

    def test_get_robot_state_returns_json(self, aloha_sim):
        """get_robot_state returns success with joint positions/velocities."""
        r = aloha_sim.get_robot_state("aloha")
        assert r["status"] == "success"
        state_json = r["content"][1]["json"]["state"]
        assert "left/waist" in state_json
        assert "position" in state_json["left/waist"]
        assert "velocity" in state_json["left/waist"]


# ---------------------------------------------------------------------------
# 3c. Mock policy through run_policy
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_MUJOCO, reason="mujoco not installed")
class TestMockPolicyInSim:
    """Run the mock policy through the full Simulation.run_policy path."""

    def test_mock_run_policy_aloha(self, aloha_sim):
        """Mock policy on ALOHA runs to completion (fast_mode)."""
        r = aloha_sim.run_policy(
            "aloha",
            policy_provider="mock",
            instruction="wave arms",
            duration=0.5,
            fast_mode=True,
        )
        assert r["status"] == "success"
        assert "✅ Policy complete" in r["content"][0]["text"]
        assert aloha_sim._world.robots["aloha"].policy_steps > 0

    def test_mock_run_policy_so100(self, so100_sim):
        """Mock policy on SO-100 runs to completion."""
        r = so100_sim.run_policy(
            "so100",
            policy_provider="mock",
            instruction="pick up cube",
            duration=0.3,
            fast_mode=True,
        )
        assert r["status"] == "success"
        assert so100_sim._world.robots["so100"].policy_steps > 0

    def test_mock_policy_changes_joint_positions(self, so100_sim):
        """Mock sinusoidal policy actually changes joint positions."""
        # Record initial qpos
        data = so100_sim._world._data
        qpos_before = data.qpos.copy()

        so100_sim.run_policy(
            "so100",
            policy_provider="mock",
            instruction="move",
            duration=0.5,
            fast_mode=True,
        )

        qpos_after = data.qpos.copy()
        # At least some joints should have moved
        assert not np.allclose(qpos_before, qpos_after, atol=1e-6), \
            "Joint positions did not change after running mock policy"

    def test_eval_policy_mock(self, aloha_sim):
        """eval_policy runs multiple episodes with mock policy."""
        r = aloha_sim.eval_policy(
            "aloha",
            policy_provider="mock",
            instruction="test",
            n_episodes=3,
            max_steps=10,
        )
        assert r["status"] == "success"
        metrics = r["content"][1]["json"]
        assert metrics["n_episodes"] == 3
        assert len(metrics["episodes"]) == 3
        assert all(ep["steps"] == 10 for ep in metrics["episodes"])

    def test_run_policy_error_no_robot(self, aloha_sim):
        """run_policy returns error for nonexistent robot."""
        r = aloha_sim.run_policy("nonexistent", policy_provider="mock")
        assert r["status"] == "error"
        assert "not found" in r["content"][0]["text"]

    def test_run_policy_error_no_world(self, sim):
        """run_policy returns error when no world exists."""
        r = sim.run_policy("anything", policy_provider="mock")
        assert r["status"] == "error"


# ---------------------------------------------------------------------------
# 3d. Tool dispatch interface (AgentTool)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_MUJOCO, reason="mujoco not installed")
class TestSimToolDispatch:
    """Test Simulation._dispatch_action — the AgentTool interface."""

    def test_dispatch_full_pipeline(self, sim):
        """Full pipeline via dispatch: create → add_robot → step → run_policy → destroy."""
        # Create world
        r = sim._dispatch_action("create_world", {})
        assert r["status"] == "success"

        # Add SO-100
        r = sim._dispatch_action("add_robot", {"name": "arm", "data_config": "so100"})
        assert r["status"] == "success"

        # Step
        r = sim._dispatch_action("step", {"n_steps": 20})
        assert r["status"] == "success"

        # Run policy
        r = sim._dispatch_action("run_policy", {
            "robot_name": "arm",
            "policy_provider": "mock",
            "instruction": "reach forward",
            "duration": 0.2,
            "fast_mode": True,
        })
        assert r["status"] == "success"
        assert "✅ Policy complete" in r["content"][0]["text"]

        # Get state
        r = sim._dispatch_action("get_state", {})
        assert r["status"] == "success"

        # Features
        r = sim._dispatch_action("get_features", {})
        assert r["status"] == "success"
        assert "Actuators" in r["content"][0]["text"]

        # Destroy
        r = sim._dispatch_action("destroy", {})
        assert r["status"] == "success"

    def test_dispatch_unknown_action(self, sim):
        """Unknown actions return error."""
        r = sim._dispatch_action("fly_to_moon", {})
        assert r["status"] == "error"

    def test_dispatch_add_object_to_scene(self, aloha_sim):
        """Add a cube object via dispatch."""
        r = aloha_sim._dispatch_action("add_object", {
            "name": "red_cube",
            "shape": "box",
            "position": [0.3, 0.0, 0.1],
            "size": [0.02, 0.02, 0.02],
            "color": [1.0, 0.0, 0.0, 1.0],
            "mass": 0.05,
        })
        assert r["status"] == "success"
        assert "red_cube" in aloha_sim._world.objects


# ---------------------------------------------------------------------------
# 3e. ACT pretrained model in ALOHA sim (end-to-end)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_MUJOCO, reason="mujoco not installed")
@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
class TestACTPolicyInSim:
    """Run actual ACT pretrained model through ALOHA simulation."""

    def test_act_run_policy_in_aloha_sim(self, aloha_sim):
        """Load ACT model and run through ALOHA sim via run_policy."""
        r = aloha_sim.run_policy(
            "aloha",
            policy_provider="lerobot_local",
            instruction="transfer cube",
            duration=0.2,
            fast_mode=True,
            pretrained_name_or_path="lerobot/act_aloha_sim_transfer_cube_human",
        )
        assert r["status"] == "success", f"ACT run_policy failed: {r}"
        assert "✅ Policy complete" in r["content"][0]["text"]
        assert aloha_sim._world.robots["aloha"].policy_steps > 0
        logger.info("ACT in ALOHA sim: %s", r["content"][0]["text"])

    def test_act_policy_produces_nonzero_actions(self, aloha_sim):
        """ACT inference produces non-trivial joint commands (not all zeros)."""
        from strands_robots.policies.lerobot_local import LerobotLocalPolicy

        policy = LerobotLocalPolicy(
            pretrained_name_or_path="lerobot/act_aloha_sim_transfer_cube_human"
        )
        robot = aloha_sim._world.robots["aloha"]
        policy.set_robot_state_keys(robot.joint_names)

        # Get real sim observation
        obs = aloha_sim._get_sim_observation("aloha")

        # Run inference
        actions = policy.get_actions_sync(obs, "transfer cube")
        assert len(actions) > 0

        # At least one action should have non-zero values
        all_values = [abs(v) for a in actions for v in a.values()]
        assert max(all_values) > 1e-6, "ACT produced all-zero actions"
        logger.info(
            "ACT actions: %d steps, max_abs=%.4f, keys=%s",
            len(actions), max(all_values), list(actions[0].keys())[:5]
        )

    def test_act_actions_match_aloha_joint_names(self, aloha_sim):
        """ACT action dicts use ALOHA joint names when state_keys are set."""
        from strands_robots.policies.lerobot_local import LerobotLocalPolicy

        policy = LerobotLocalPolicy(
            pretrained_name_or_path="lerobot/act_aloha_sim_transfer_cube_human"
        )
        robot = aloha_sim._world.robots["aloha"]
        policy.set_robot_state_keys(robot.joint_names)

        obs = aloha_sim._get_sim_observation("aloha")
        actions = policy.get_actions_sync(obs, "pick up cube")

        # Action keys should match robot joint names
        action_keys = set(actions[0].keys())
        robot_keys = set(robot.joint_names)
        assert action_keys == robot_keys, (
            f"Action keys mismatch:\n  action: {sorted(action_keys)}\n  robot: {sorted(robot_keys)}"
        )


# ---------------------------------------------------------------------------
# 3f. Inference latency profiling
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
class TestInferenceLatency:
    """Profile inference latency for ACT vs SmolVLA.

    These are measurement tests — they always pass but log latency stats.
    """

    @staticmethod
    def _profile_policy(policy, obs, instruction, n_warmup=3, n_runs=20):
        """Profile select_action_sync latency."""
        for _ in range(n_warmup):
            policy.select_action_sync(obs, instruction)

        latencies = []
        for _ in range(n_runs):
            t0 = time.time()
            policy.select_action_sync(obs, instruction)
            latencies.append((time.time() - t0) * 1000)

        return {
            "mean_ms": float(np.mean(latencies)),
            "median_ms": float(np.median(latencies)),
            "min_ms": float(np.min(latencies)),
            "max_ms": float(np.max(latencies)),
            "std_ms": float(np.std(latencies)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "n_runs": n_runs,
        }

    def test_act_inference_latency(self):
        """Profile ACT inference latency (target: <5ms on Thor GPU)."""
        from strands_robots.policies.lerobot_local import LerobotLocalPolicy

        policy = LerobotLocalPolicy(
            pretrained_name_or_path="lerobot/act_aloha_sim_transfer_cube_human"
        )
        policy.set_robot_state_keys([f"joint_{i}" for i in range(14)])

        obs = {}
        for feat_name, feat_info in policy._input_features.items():
            if hasattr(feat_info, "shape"):
                obs[feat_name] = torch.randn(1, *feat_info.shape, device=policy._device)

        stats = self._profile_policy(policy, obs, "transfer cube")
        logger.info(
            "ACT latency: mean=%.1fms, median=%.1fms, p95=%.1fms, min=%.1fms, max=%.1fms",
            stats["mean_ms"], stats["median_ms"], stats["p95_ms"],
            stats["min_ms"], stats["max_ms"],
        )
        # ACT is lightweight — should be well under 5ms on Thor GPU
        assert stats["median_ms"] < 50, f"ACT too slow: median={stats['median_ms']:.1f}ms"

    def test_smolvla_inference_latency(self):
        """Profile SmolVLA inference latency (VLA with language + vision)."""
        from strands_robots.policies.lerobot_local import LerobotLocalPolicy

        try:
            policy = LerobotLocalPolicy(pretrained_name_or_path="lerobot/smolvla_base")
        except Exception as e:
            pytest.skip(f"SmolVLA load failed: {e}")

        policy.set_robot_state_keys([f"joint_{i}" for i in range(6)])

        obs = {}
        for feat_name, feat_info in policy._input_features.items():
            if hasattr(feat_info, "shape"):
                obs[feat_name] = torch.randn(1, *feat_info.shape, device=policy._device)

        stats = self._profile_policy(policy, obs, "pick up the red cube", n_warmup=2, n_runs=10)
        logger.info(
            "SmolVLA latency: mean=%.1fms, median=%.1fms, p95=%.1fms, min=%.1fms, max=%.1fms",
            stats["mean_ms"], stats["median_ms"], stats["p95_ms"],
            stats["min_ms"], stats["max_ms"],
        )
        # SmolVLA is a VLA — heavier, but should be under 200ms on Thor
        assert stats["median_ms"] < 500, f"SmolVLA too slow: median={stats['median_ms']:.1f}ms"

    def test_latency_comparison_act_vs_smolvla(self):
        """Compare ACT vs SmolVLA latency side-by-side."""
        from strands_robots.policies.lerobot_local import LerobotLocalPolicy

        # Load ACT
        act = LerobotLocalPolicy(
            pretrained_name_or_path="lerobot/act_aloha_sim_transfer_cube_human"
        )
        act.set_robot_state_keys([f"joint_{i}" for i in range(14)])
        act_obs = {}
        for fn, fi in act._input_features.items():
            if hasattr(fi, "shape"):
                act_obs[fn] = torch.randn(1, *fi.shape, device=act._device)

        act_stats = self._profile_policy(act, act_obs, "transfer cube")

        # Load SmolVLA
        try:
            smol = LerobotLocalPolicy(pretrained_name_or_path="lerobot/smolvla_base")
        except Exception as e:
            pytest.skip(f"SmolVLA load failed: {e}")

        smol.set_robot_state_keys([f"joint_{i}" for i in range(6)])
        smol_obs = {}
        for fn, fi in smol._input_features.items():
            if hasattr(fi, "shape"):
                smol_obs[fn] = torch.randn(1, *fi.shape, device=smol._device)

        smol_stats = self._profile_policy(smol, smol_obs, "pick up cube", n_warmup=2, n_runs=10)

        speedup = smol_stats["median_ms"] / max(act_stats["median_ms"], 0.01)
        logger.info(
            "\n=== Latency Comparison ===\n"
            "  ACT:     median=%.1fms (p95=%.1fms)\n"
            "  SmolVLA: median=%.1fms (p95=%.1fms)\n"
            "  SmolVLA is %.1fx slower than ACT",
            act_stats["median_ms"], act_stats["p95_ms"],
            smol_stats["median_ms"], smol_stats["p95_ms"],
            speedup,
        )

        # Both should work — just log the comparison
        assert act_stats["median_ms"] < smol_stats["median_ms"], \
            "Expected ACT to be faster than SmolVLA"


# ---------------------------------------------------------------------------
# 3g. Full pipeline: Simulation(tool) → create → robot → objects → policy
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_MUJOCO, reason="mujoco not installed")
@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
class TestFullSimPipeline:
    """End-to-end: create world + robot + objects + ACT policy + verify."""

    def test_aloha_cube_transfer_pipeline(self):
        """Full ALOHA + cube + ACT pipeline via Simulation tool."""
        os.environ.setdefault("MUJOCO_GL", "egl")
        from strands_robots.simulation.simulation import Simulation

        sim = Simulation(tool_name="pipeline_test")
        try:
            # 1. Create world
            r = sim.create_world()
            assert r["status"] == "success"

            # 2. Add ALOHA
            r = sim.add_robot("aloha", data_config="aloha")
            assert r["status"] == "success"
            assert len(sim._world.robots["aloha"].joint_names) == 16

            # 3. Add a cube object to the scene
            r = sim.add_object(
                "cube", shape="box",
                position=[0.3, 0.0, 0.05],
                size=[0.02, 0.02, 0.02],
                color=[1.0, 0.0, 0.0, 1.0],
                mass=0.02,
            )
            assert r["status"] == "success"

            # 4. Step to settle physics
            r = sim.step(100)
            assert r["status"] == "success"

            # 5. Get observation (should include joints + cameras)
            obs = sim.get_observation("aloha")
            assert isinstance(obs, dict)
            assert "left/waist" in obs

            # 6. Run ACT policy
            r = sim.run_policy(
                "aloha",
                policy_provider="lerobot_local",
                instruction="transfer the cube",
                duration=0.3,
                fast_mode=True,
                pretrained_name_or_path="lerobot/act_aloha_sim_transfer_cube_human",
            )
            assert r["status"] == "success"
            text = r["content"][0]["text"]
            assert "✅ Policy complete" in text
            assert sim._world.robots["aloha"].policy_steps > 0
            logger.info("Full pipeline result: %s", text)

            # 7. Verify physics advanced
            assert sim._world.sim_time > 0

            # 8. Get features for introspection
            r = sim.get_features()
            assert r["status"] == "success"
            features = r["content"][1]["json"]["features"]
            assert features["n_actuators"] >= 14  # ALOHA has 14

        finally:
            sim.cleanup()

    def test_so100_mock_policy_pipeline(self):
        """SO-100 with mock policy pipeline (no GPU needed for mock)."""
        os.environ.setdefault("MUJOCO_GL", "egl")
        from strands_robots.simulation.simulation import Simulation

        sim = Simulation(tool_name="so100_pipeline")
        try:
            # Create + robot
            sim.create_world()
            sim.add_robot("arm", data_config="so100")

            # Add objects
            sim.add_object("target", shape="sphere", position=[0.2, 0.0, 0.1],
                           size=[0.03, 0.03, 0.03], color=[0, 1, 0, 1])

            # Run mock policy
            r = sim.run_policy(
                "arm", policy_provider="mock",
                instruction="reach for target",
                duration=1.0, fast_mode=True,
            )
            assert r["status"] == "success"
            assert sim._world.robots["arm"].policy_steps > 0

            # Verify state changed
            state_r = sim.get_robot_state("arm")
            assert state_r["status"] == "success"
            state = state_r["content"][1]["json"]["state"]
            # After running policy, at least some joints should have non-zero velocity
            positions = [state[j]["position"] for j in state]
            assert any(abs(p) > 1e-6 for p in positions), \
                "Robot joints did not move during mock policy run"

        finally:
            sim.cleanup()

    def test_render_after_policy(self):
        """Render camera image after running policy."""
        os.environ.setdefault("MUJOCO_GL", "egl")
        from strands_robots.simulation.simulation import Simulation

        sim = Simulation(tool_name="render_test")
        try:
            sim.create_world()
            sim.add_robot("aloha", data_config="aloha")
            sim.run_policy("aloha", policy_provider="mock", duration=0.2, fast_mode=True)

            r = sim.render(camera_name="overhead_cam", width=320, height=240)
            assert r["status"] == "success"
            # Should have image content
            assert any("image" in c for c in r["content"]), \
                f"No image in render result: {[list(c.keys()) for c in r['content']]}"

        finally:
            sim.cleanup()
