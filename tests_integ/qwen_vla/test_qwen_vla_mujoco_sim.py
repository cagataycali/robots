"""GPU-gated MuJoCo-sim integration test: drive the QwenVla provider end-to-end.

Acts as a real user: ``Robot("so100")`` (MuJoCo sim) + ``run_policy(
policy_provider="qwen_vla", ...)`` against a live Qwen-VLA reference server.
Proves the provider is usable in the sim out of the box - observations bridge
from the per-joint MuJoCo schema, actions flatten to the robot's actuators, and
the arm actually moves (joint deltas are non-zero).

Skipped automatically without CUDA / torch / pyzmq / mujoco.
"""

import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

# Import strands_robots BEFORE mujoco. On headless hosts, strands_robots's
# package init eagerly sets MUJOCO_GL=egl (GPU offscreen) / osmesa. MuJoCo
# locks its GL backend at first ``import mujoco``; if mujoco is imported first
# (as a bare ``importorskip`` would), it locks the GLFW backend and rendering
# dies with a cryptic ``gladLoadGL error`` on headless GPU boxes. This import
# ordering is the user-facing contract documented in strands_robots/__init__.py.
pytest.importorskip("strands_robots", reason="strands-robots not importable")
import strands_robots  # noqa: E402,F401  (eager GL backend config side effect)

pytest.importorskip("torch", reason="qwen-vla-train extra (torch) not installed")
pytest.importorskip("zmq", reason="qwen-vla-service extra (pyzmq) not installed")
pytest.importorskip("mujoco", reason="sim-mujoco extra (mujoco) not installed")

import torch  # noqa: E402

if not torch.cuda.is_available():
    pytest.skip("CUDA GPU required for the live reference server", allow_module_level=True)

_EX = Path(__file__).resolve().parents[2] / "examples" / "qwen_vla_reference"
sys.path.insert(0, str(_EX))

pytestmark = pytest.mark.gpu
_PORT = 5584


@pytest.fixture(scope="module")
def server():
    proc = subprocess.Popen(
        [sys.executable, str(_EX / "reference_server.py"), "--port", str(_PORT), "--device", "cuda"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(60):
        with socket.socket() as s:
            s.settimeout(1)
            if s.connect_ex(("127.0.0.1", _PORT)) == 0:
                break
        time.sleep(0.5)
    else:
        proc.kill()
        pytest.fail("reference server did not start")
    yield _PORT
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=10)


def test_qwen_vla_drives_so100_in_mujoco(server):
    from strands_robots import Robot

    sim = Robot("so100")
    joints = sim.robot_joint_names("so100")
    assert joints, "so100 should expose joints"

    before = sim.get_observation("so100", skip_images=True)
    res = sim.run_policy(
        "so100",
        policy_provider="qwen_vla",
        policy_config={"data_config": "so100", "host": "127.0.0.1", "port": server},
        instruction="wave the arm around",
        duration=1.0,
        control_frequency=20.0,
        action_horizon=8,
        fast_mode=True,
    )
    assert res.get("status") == "success", res
    after = sim.get_observation("so100", skip_images=True)

    # The robot must actually move - the provider drives real actuators.
    total_delta = sum(abs(float(after[j]) - float(before[j])) for j in joints)
    assert total_delta > 1e-3, f"robot did not move (total delta {total_delta})"
