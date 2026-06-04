"""GPU-gated END-TO-END test: full Qwen-VLA closed loop with the reference model.

Runs the complete 4-stage recipe (T2A -> CPT -> SFT -> RL) + SERVICE/LOCAL
inference + hot-swap redeploy against the runnable reference model in
``examples/qwen_vla_reference``. Verifies the loop closes: every stage trains
(finite loss), inference returns H-step instruction-sensitive chunks, and the
redeploy hot-swap succeeds.

Skipped automatically when CUDA / torch / pyzmq are unavailable.
"""

import sys
from pathlib import Path

import pytest

pytest.importorskip("torch", reason="qwen-vla-train extra (torch) not installed")
pytest.importorskip("zmq", reason="qwen-vla-service extra (pyzmq) not installed")

import torch  # noqa: E402

if not torch.cuda.is_available():
    pytest.skip("CUDA GPU required for the e2e reference run", allow_module_level=True)

_EX = Path(__file__).resolve().parents[2] / "examples" / "qwen_vla_reference"
sys.path.insert(0, str(_EX))

pytestmark = pytest.mark.gpu


def test_full_closed_loop():
    import run_end_to_end as e2e

    report = e2e.main()

    # Every training stage produced a finite loss/objective.
    assert report["stage1_t2a"]["final_loss"] == report["stage1_t2a"]["final_loss"]  # not NaN
    assert report["stage2_cpt"]["final_loss"] == report["stage2_cpt"]["final_loss"]
    assert report["stage3_sft"]["final_loss"] == report["stage3_sft"]["final_loss"]
    # Inference closed correctly.
    assert report["service_inference"]["horizon"] == 16
    assert report["service_inference"]["instruction_sensitive"] is True
    assert report["local_inference"]["horizon"] == 16
    # Redeploy loop closed.
    assert report["hotswap"]["status"] == "success"
    # RL: non-negative transfer.
    assert report["stage4_rl"]["success_after"] >= report["stage4_rl"]["success_before"]
