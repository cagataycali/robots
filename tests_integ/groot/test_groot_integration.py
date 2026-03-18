"""Integration tests for GR00T N1.6 policy — requires CUDA + Isaac-GR00T.

Run explicitly: hatch run test-integ
Or: pytest tests_integ/ -v --timeout=300

Requirements: CUDA GPU, Isaac-GR00T N1.6, nvidia/GR00T-N1.6-3B, pyzmq, msgpack
"""

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


def _make_gr1_observation(instruction="pick up the cube"):
    """GR1 nested observation with B=1, T=1 shape convention.

    Uses a fixed seed for reproducibility across test runs.
    """
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
        observation = _make_gr1_observation("pick up the red cube")
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
            result = client.call_endpoint("get_action", _make_gr1_observation(f"task {i}"))
            action = _extract_action(result)
            shapes.append({key: value.shape for key, value in action.items()})
        for i in range(1, len(shapes)):
            assert shapes[i] == shapes[0], f"Inconsistent: {shapes}"

    def test_different_instructions(self, groot_server):
        """Different instructions should produce valid but potentially different action distributions."""
        from strands_robots.policies.groot.client import Gr00tInferenceClient

        client = Gr00tInferenceClient(host="localhost", port=groot_server["port"])
        actions_by_instruction = {}
        for instruction in ["pick up cube", "place in bowl", "wave hello"]:
            result = client.call_endpoint("get_action", _make_gr1_observation(instruction))
            action = _extract_action(result)
            assert isinstance(action, dict), f"Non-dict result for '{instruction}'"
            for key, value in action.items():
                assert isinstance(value, np.ndarray), f"'{key}' not ndarray for '{instruction}'"
                assert value.dtype in (np.float32, np.float64), f"'{key}' unexpected dtype: {value.dtype}"
                assert not np.any(np.isnan(value)), f"NaN in '{key}' for '{instruction}'"
                assert not np.any(np.isinf(value)), f"Inf in '{key}' for '{instruction}'"
            actions_by_instruction[instruction] = action

        # Verify all instructions produce same set of action keys
        key_sets = [set(a.keys()) for a in actions_by_instruction.values()]
        assert all(keys == key_sets[0] for keys in key_sets), f"Inconsistent action keys: {key_sets}"


# -- Tests: Version Detection -------------------------------------------------


class TestGr00tVersionDetection:
    def test_detects_n16(self):
        import strands_robots.policies.groot.policy as policy_mod

        policy_mod._GROOT_VERSION = None
        from strands_robots.policies.groot.policy import _detect_groot_version

        assert _detect_groot_version() == "n1.6"

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
            data_config="so100",
            model_path=MODEL_PATH,
            embodiment_tag="gr1",
            device="cuda",
        )

    def test_local_policy_mode(self, local_policy):
        assert local_policy._mode == "local"
        assert local_policy._local_policy is not None

    def test_local_inference(self, local_policy):
        """Local inference should produce valid action arrays with finite values."""
        observation = _make_gr1_observation()["observation"]
        result = local_policy._local_policy.get_action(observation)
        action = _extract_action(result)
        assert isinstance(action, dict), f"Expected dict, got {type(action)}"
        assert len(action) > 0, "Action dict is empty"
        for key, value in action.items():
            assert isinstance(value, np.ndarray), f"'{key}' not ndarray: {type(value)}"
            assert value.size > 0, f"'{key}' is empty"
            assert not np.any(np.isnan(value)), f"NaN values in '{key}'"
            assert not np.any(np.isinf(value)), f"Inf values in '{key}'"
            logger.info("Local action '%s': shape=%s dtype=%s", key, value.shape, value.dtype)
