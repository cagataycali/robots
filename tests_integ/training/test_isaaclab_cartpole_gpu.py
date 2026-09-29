"""Isaac Lab trainer smoke: a real Isaac-Cartpole run on the GPU, through the agent tool.

Requirements:
  - An NVIDIA RTX-class GPU, driver 580+
  - Isaac Lab 3.0 installed in its OWN virtual environment (never this one);
    see docs/learn/training/isaaclab.md
  - ``ISAACLAB_PYTHON`` naming that environment's python
  - ``OMNI_KIT_ACCEPT_EULA=YES`` (the operator's acceptance of the Omniverse EULA)
  - ``STRANDS_GPU_TEST=1``

Run with::

    STRANDS_GPU_TEST=1 ISAACLAB_PYTHON=~/il/bin/python OMNI_KIT_ACCEPT_EULA=YES \\
        pytest tests_integ/training/test_isaaclab_cartpole_gpu.py -m gpu -v -s

Trains Isaac-Cartpole for 50 rsl_rl iterations on 4096 environments (about a
minute on one L40S, most of it startup), polling ``train_policy(action="status")``
the way an agent would. Asserts the user-visible verdict: the job succeeds, the
reward rose, throughput was reported, and the run directory holds the final
``model_<it>.pt``.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pytest

from strands_robots.tools.train_policy import train_policy

_GPU_ENABLED = os.environ.get("STRANDS_GPU_TEST", "0") == "1"

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not _GPU_ENABLED or not os.environ.get("ISAACLAB_PYTHON"),
        reason="Requires a GPU and an Isaac Lab venv: set STRANDS_GPU_TEST=1 and ISAACLAB_PYTHON.",
    ),
]

_ITERATIONS = 50
_BUDGET_S = 900


def _json(envelope: dict[str, Any]) -> dict[str, Any]:
    return next(item["json"] for item in envelope["content"] if "json" in item)


@pytest.mark.timeout(_BUDGET_S + 120)
def test_cartpole_trains_and_reports_its_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STRANDS_ISAACLAB_JOBS", str(tmp_path / "jobs"))
    launched = train_policy(
        action="train",
        provider="isaaclab",
        steps=_ITERATIONS,
        seed=1,
        output_dir=str(tmp_path / "out"),
        extra={"task": "Isaac-Cartpole", "num_envs": 4096, "timeout_s": _BUDGET_S},
    )
    assert launched["status"] == "success", launched
    job_id = _json(launched)["job_id"]

    deadline = time.monotonic() + _BUDGET_S + 60
    block = _json(launched)
    while block["status"] == "running" and time.monotonic() < deadline:
        time.sleep(5)
        polled = train_policy(action="status", provider="isaaclab", job_id=job_id)
        block = _json(polled)
        print(
            f"[isaaclab] {block['status']} it={block['metrics'].get('latest_iteration')} "
            f"reward={block['metrics'].get('latest_reward')} sps={block['metrics'].get('steps_per_s')}"
        )

    assert block["status"] == "success", polled
    metrics = block["metrics"]
    assert metrics["latest_iteration"] == _ITERATIONS - 1
    assert metrics["learning"] is True, metrics
    assert metrics["steps_per_s"] and metrics["steps_per_s"] > 1_000, metrics
    run_dir = Path(block["checkpoint_dir"])
    assert run_dir.name.endswith(job_id)
    assert metrics["latest_model"] == str(run_dir / f"model_{_ITERATIONS - 1}.pt")
    assert Path(metrics["latest_model"]).is_file()
