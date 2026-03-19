"""Integration tests for GR00T N1.6 policy — requires CUDA + Isaac-GR00T.

Run explicitly: hatch run test-integ
Or: pytest tests_integ/ -v --timeout=300

Requirements: CUDA GPU, Isaac-GR00T N1.6, nvidia/GR00T-N1.6-3B, pyzmq, msgpack
"""

import asyncio
import logging
import os
import signal
import subprocess
import sys
import time

import numpy as np
import pytest

logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("GROOT_MODEL_PATH", "nvidia/GR00T-N1.6-3B")
EMBODIMENT_TAG = os.getenv("GROOT_EMBODIMENT_TAG", "GR1")
SERVER_PORT = 15555
SERVER_STARTUP_TIMEOUT = int(os.getenv("GROOT_SERVER_TIMEOUT", "180"))

pytestmark = pytest.mark.gpu


# -- Server fixture ----------------------------------------------------------


@pytest.fixture(scope="module")
def groot_server():
    server_script = _find_server_script()
    cmd = [
        sys.executable,
        server_script,
        "--model-path",
        MODEL_PATH,
        "--embodiment-tag",
        EMBODIMENT_TAG,
        "--port",
        str(SERVER_PORT),
        "--host",
        "0.0.0.0",
    ]
    print(f"\n🤖 Starting GR00T server: {EMBODIMENT_TAG} on :{SERVER_PORT}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,
    )
    _wait_for_server(proc, SERVER_PORT, SERVER_STARTUP_TIMEOUT)
    yield {"port": SERVER_PORT, "process": proc}

    print("\n🛑 Stopping GR00T server...")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=10)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


def _find_server_script():
    """Locate the GR00T inference server script."""
    candidates = []
    env_script = os.getenv("GROOT_SERVER_SCRIPT")
    if env_script:
        candidates.append(env_script)
    try:
        import gr00t

        candidates.append(os.path.join(os.path.dirname(gr00t.__file__), "eval", "run_gr00t_server.py"))
    except ImportError:
        pass
    for path in candidates:
        if os.path.exists(path):
            return path
    pytest.fail("Cannot find GR00T server script. Set GROOT_SERVER_SCRIPT or install Isaac-GR00T.")


def _wait_for_server(proc, port, timeout):
    """Block until the server responds to a ping, or fail."""
    import msgpack
    import zmq

    start = time.time()
    context = zmq.Context()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            stdout = proc.stdout.read() if proc.stdout else ""
            pytest.fail(f"Server exited({proc.returncode}):\n{stdout[-2000:]}")
        try:
            sock = context.socket(zmq.REQ)
            sock.setsockopt(zmq.RCVTIMEO, 2000)
            sock.setsockopt(zmq.SNDTIMEO, 2000)
            sock.setsockopt(zmq.LINGER, 0)
            sock.connect(f"tcp://localhost:{port}")
            sock.send(msgpack.packb({"endpoint": "ping"}))
            reply = msgpack.unpackb(sock.recv())
            if isinstance(reply, dict) and reply.get("status") == "ok":
                sock.close()
                context.term()
                print(f"   ✅ Server ready in {time.time() - start:.1f}s")
                return
            sock.close()
        except Exception:
            try:
                sock.close()
            except Exception:
                pass
            time.sleep(2)
    context.term()
    stdout = proc.stdout.read() if proc.stdout else ""
    pytest.fail(f"Server not ready within {timeout}s.\n{stdout[-2000:]}")


# -- Helpers ------------------------------------------------------------------


def _make_gr1_server_observation(instruction="pick up the cube"):
    """GR1 nested observation for direct server calls (B=1, T=1 shape)."""
    rng = np.random.RandomState(42)
    return {
        "observation": {
            "video": {
                "ego_view_bg_crop_pad_res256_freq20": rng.randint(0, 256, (1, 1, 256, 256, 3), dtype=np.uint8),
            },
            "state": {
                "left_arm": rng.uniform(-1, 1, (1, 1, 7)).astype(np.float32),
                "right_arm": rng.uniform(-1, 1, (1, 1, 7)).astype(np.float32),
                "left_hand": rng.uniform(0, 1, (1, 1, 6)).astype(np.float32),
                "right_hand": rng.uniform(0, 1, (1, 1, 6)).astype(np.float32),
                "waist": rng.uniform(-1, 1, (1, 1, 3)).astype(np.float32),
            },
            "language": {
                "task": [[instruction]],
            },
        },
        "options": None,
    }


def _make_gr1_robot_observation():
    """GR1 robot-side observation (raw sensor values, no batching).

    This is what a robot would produce — single frames and 1D state vectors.
    The policy's mapping layer handles all reshaping.
    """
    rng = np.random.RandomState(42)
    return {
        "ego_view": rng.randint(0, 256, (256, 256, 3), dtype=np.uint8),
        "left_arm": rng.uniform(-1, 1, (7,)).astype(np.float32),
        "right_arm": rng.uniform(-1, 1, (7,)).astype(np.float32),
        "left_hand": rng.uniform(0, 1, (6,)).astype(np.float32),
        "right_hand": rng.uniform(0, 1, (6,)).astype(np.float32),
        "waist": rng.uniform(-1, 1, (3,)).astype(np.float32),
    }


def _extract_action(result):
    """Extract action dict from server result (tuple or dict)."""
    if isinstance(result, (tuple, list)):
        return result[0]
    return result


# -- Tests: Service Mode (ZMQ) -----------------------------------------------


class TestGr00tServiceMode:
    def test_server_ping(self, groot_server):
        from strands_robots.policies.groot import Gr00tInferenceClient

        client = Gr00tInferenceClient(host="localhost", port=groot_server["port"])
        assert client.ping() is True

    def test_get_action(self, groot_server):
        """Send GR1 observation, verify action shapes, dtypes, and finite values."""
        from strands_robots.policies.groot.client import Gr00tInferenceClient

        client = Gr00tInferenceClient(host="localhost", port=groot_server["port"])
        observation = _make_gr1_server_observation("pick up the red cube")
        result = client.call_endpoint("get_action", observation)
        action = _extract_action(result)
        assert isinstance(action, dict), f"Expected dict, got {type(action)}"
        assert len(action) > 0, "Action dict is empty"
        for key, value in action.items():
            assert isinstance(value, np.ndarray), f"'{key}' not ndarray: {type(value)}"
            assert value.size > 0, f"'{key}' is empty"
            assert not np.any(np.isnan(value)), f"NaN values in '{key}'"
            assert not np.any(np.isinf(value)), f"Inf values in '{key}'"
            logger.info("Action key '%s': shape=%s dtype=%s", key, value.shape, value.dtype)

    def test_batch_consistency(self, groot_server):
        from strands_robots.policies.groot.client import Gr00tInferenceClient

        client = Gr00tInferenceClient(host="localhost", port=groot_server["port"])
        shapes = []
        for i in range(3):
            result = client.call_endpoint("get_action", _make_gr1_server_observation(f"task {i}"))
            action = _extract_action(result)
            shapes.append({key: value.shape for key, value in action.items()})
        for i in range(1, len(shapes)):
            assert shapes[i] == shapes[0], f"Inconsistent: {shapes}"

    def test_different_instructions(self, groot_server):
        """Different instructions produce valid but potentially different actions."""
        from strands_robots.policies.groot.client import Gr00tInferenceClient

        client = Gr00tInferenceClient(host="localhost", port=groot_server["port"])
        actions_by_instruction = {}
        for instruction in ["pick up cube", "place in bowl", "wave hello"]:
            result = client.call_endpoint("get_action", _make_gr1_server_observation(instruction))
            action = _extract_action(result)
            assert isinstance(action, dict), f"Non-dict for '{instruction}'"
            for key, value in action.items():
                assert isinstance(value, np.ndarray), f"'{key}' not ndarray for '{instruction}'"
                assert value.dtype in (
                    np.float32,
                    np.float64,
                ), f"'{key}' unexpected dtype: {value.dtype}"
                assert not np.any(np.isnan(value)), f"NaN in '{key}' for '{instruction}'"
                assert not np.any(np.isinf(value)), f"Inf in '{key}' for '{instruction}'"
            actions_by_instruction[instruction] = action

        key_sets = [set(a.keys()) for a in actions_by_instruction.values()]
        assert all(keys == key_sets[0] for keys in key_sets), f"Inconsistent action keys: {key_sets}"


# -- Tests: Version Detection -------------------------------------------------


class TestGr00tVersionDetection:
    def test_detects_n16(self):
        from strands_robots.policies.groot.policy import _detect_groot_version

        assert _detect_groot_version(force=True) == "n1.6"

    def test_detection_is_cached(self):
        import strands_robots.policies.groot.policy as policy_mod

        policy_mod._GROOT_VERSION = None
        from strands_robots.policies.groot.policy import _detect_groot_version

        version1 = _detect_groot_version()
        version2 = _detect_groot_version()
        assert version1 == version2 == policy_mod._GROOT_VERSION


# -- Tests: Local Mode --------------------------------------------------------


class TestGr00tLocalMode:
    @pytest.fixture(scope="class")
    def local_policy(self):
        from strands_robots.policies.groot import Gr00tPolicy

        return Gr00tPolicy(
            data_config="fourier_gr1_arms_waist",
            model_path=MODEL_PATH,
            embodiment_tag="gr1",
            device="cuda",
        )

    def test_local_policy_mode(self, local_policy):
        assert local_policy._mode == "local"
        assert local_policy._local_policy is not None

    def test_mappings_initialized(self, local_policy):
        """Mappings should be auto-inferred from data_config + model."""
        assert local_policy._obs_mapping is not None
        assert local_policy._action_mapping is not None
        assert len(local_policy._obs_mapping.video) > 0
        assert len(local_policy._obs_mapping.state) > 0
        assert len(local_policy._action_mapping.actions) > 0
        logger.info("Obs video mapping: %s", local_policy._obs_mapping.video)
        logger.info("Obs state mapping: %s", local_policy._obs_mapping.state)
        logger.info("Action mapping: %s", local_policy._action_mapping.actions)

    def test_model_state_dof_discovered(self, local_policy):
        """DOF should be discovered from model, not hardcoded."""
        assert len(local_policy._model_state_dof) > 0
        logger.info("Discovered DOF: %s", local_policy._model_state_dof)
        # GR1 should have these keys
        for key in ["left_arm", "right_arm", "left_hand", "right_hand", "waist"]:
            assert key in local_policy._model_state_dof, f"Missing DOF for '{key}'"

    def test_local_get_actions(self, local_policy):
        """Full pipeline: robot obs → mappings → model → action timesteps.

        Uses robot-side keys (no prefixes). The policy's mapping layer
        handles all translation to/from model modality keys.
        """
        robot_obs = _make_gr1_robot_observation()
        actions = local_policy._local_get_actions(robot_obs, "pick up the cube")

        assert isinstance(actions, list), f"Expected list, got {type(actions)}"
        assert len(actions) > 0, "Empty action list"

        for i, step in enumerate(actions):
            assert isinstance(step, dict), f"Step {i} not dict: {type(step)}"
            assert len(step) > 0, f"Step {i} is empty"
            for key, value in step.items():
                assert isinstance(value, np.ndarray), f"Step {i} '{key}' not ndarray: {type(value)}"
                assert value.size > 0, f"Step {i} '{key}' is empty"
                assert not np.any(np.isnan(value)), f"NaN in step {i} '{key}'"
                assert not np.any(np.isinf(value)), f"Inf in step {i} '{key}'"

        logger.info(
            "Local inference: %d timesteps, keys=%s",
            len(actions),
            list(actions[0].keys()),
        )

    def test_get_actions_async(self, local_policy):
        """Public async API should work end-to-end."""
        robot_obs = _make_gr1_robot_observation()
        actions = asyncio.run(local_policy.get_actions(robot_obs, "pick up the cube"))
        assert isinstance(actions, list)
        assert len(actions) > 0
        # Should have robot-side keys (mapped), not model keys
        first_keys = set(actions[0].keys())
        logger.info("Async get_actions keys: %s", first_keys)
        # Should not have model-prefixed keys
        for key in first_keys:
            assert not key.startswith("action."), f"Unmapped model key: {key}"

    def test_action_keys_are_robot_names(self, local_policy):
        """Action keys should be robot actuator names, not model keys."""
        robot_obs = _make_gr1_robot_observation()
        actions = local_policy._local_get_actions(robot_obs, "wave hello")

        mapped_robot_names = set(local_policy._action_mapping.actions.values())
        for key in actions[0]:
            if not key.startswith("unmapped."):
                assert key in mapped_robot_names, f"Action key '{key}' not in mapped robot names: {mapped_robot_names}"
