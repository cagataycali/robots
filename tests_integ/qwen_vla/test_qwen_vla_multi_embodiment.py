"""GPU-gated multi-embodiment SERVICE-mode test against the reference server.

Proves the unified K-channel unpack (section 2.4 zero-padding layout) is robust
across heterogeneous embodiments - single-arm, dual-arm, dual-arm+waist+mobile,
and 7-DoF EEF - all driven by one K=32 model through one ZMQ server. Each
embodiment's distinct action_mapping must reproduce its exact actuator keys.

Skipped automatically without CUDA / torch / pyzmq.
"""

import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch", reason="qwen-vla-train extra (torch) not installed")
pytest.importorskip("zmq", reason="qwen-vla-service extra (pyzmq) not installed")

import torch  # noqa: E402

if not torch.cuda.is_available():
    pytest.skip("CUDA GPU required for the live reference server", allow_module_level=True)

_EX = Path(__file__).resolve().parents[2] / "examples" / "qwen_vla_reference"
sys.path.insert(0, str(_EX))

pytestmark = pytest.mark.gpu
_PORT = 5572


@pytest.fixture(scope="module")
def live_server():
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


_CASES = {
    "so100": {"webcam": np.zeros((64, 64, 3), np.uint8), "single_arm": np.zeros(6), "gripper": np.zeros(1)},
    "aloha_bimanual": {
        "cam_high": np.zeros((64, 64, 3), np.uint8),
        "cam_left_wrist": np.zeros((64, 64, 3), np.uint8),
        "cam_right_wrist": np.zeros((64, 64, 3), np.uint8),
        "left_arm": np.zeros(6),
        "right_arm": np.zeros(6),
        "left_gripper": np.zeros(1),
        "right_gripper": np.zeros(1),
    },
    "unitree_g1_mobile": {
        "ego_view": np.zeros((64, 64, 3), np.uint8),
        "left_arm": np.zeros(7),
        "right_arm": np.zeros(7),
        "left_hand": np.zeros(6),
        "right_hand": np.zeros(6),
        "waist": np.zeros(3),
    },
    "widowx": {
        "image_0": np.zeros((64, 64, 3), np.uint8),
        **{k: np.zeros(1) for k in ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]},
    },
}


@pytest.mark.parametrize("cfg_name", list(_CASES))
def test_embodiment_roundtrip(live_server, cfg_name):
    from strands_robots.policies.qwen_vla import QwenVlaPolicy, load_data_config

    cfg = load_data_config(cfg_name)
    pol = QwenVlaPolicy(data_config=cfg_name, host="127.0.0.1", port=live_server)
    pol.reset(seed=11)
    acts = pol.get_actions_sync(_CASES[cfg_name], "do the task")
    assert len(acts) == cfg.chunk_size
    expected = sorted(k.removeprefix("action.") for k in cfg.action_keys)
    assert sorted(acts[0].keys()) == expected
    assert np.isfinite(np.array(acts[0][expected[0]])).all()
