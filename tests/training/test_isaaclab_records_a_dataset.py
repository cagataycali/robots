"""A finished Isaac Lab run's policy is recorded as a LeRobotDataset through strands.

Isaac Lab's ``play`` writes one viewport mp4 and ``capture_env_sensors`` is
train-only, so every one of the 13 published Isaac Lab datasets needed an
out-of-tree harness running strands inside the Isaac Lab interpreter
(ILAC-011). ``IsaacLabTrainer.record`` (``train_policy(action="record")``) now
launches a runner shipped with strands - run by ``ISAACLAB_PYTHON``, importing
no strands - and converts what it writes with ``DatasetRecorder``.
"""

from __future__ import annotations

import ast
import json
import stat
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from strands_robots.training import _isaaclab_runtime as runtime
from strands_robots.training import isaaclab as il
from strands_robots.training.isaaclab import IsaacLabTrainer
from tests._package_ast import parse_file

RUNNER = Path(il.__file__).with_name("_isaaclab_record_runner.py")

# A fake Isaac Lab interpreter: trains like the real one (a model_2.pt under
# logs/rsl_rl/<exp>/<time>_<run_name>/) and, given the record runner, writes the
# episode arrays and meta.json the runner writes.
_FAKE = textwrap.dedent(
    """\
    #!{python}
    import json, os, sys, time
    from pathlib import Path
    import numpy as np
    argv = sys.argv[1:]
    Path(os.environ["FAKE_ARGV"]).write_text(json.dumps(argv))
    if argv and argv[0].endswith("_isaaclab_record_runner.py"):
        out = Path(argv[argv.index("--out") + 1]); out.mkdir(parents=True)
        n = int(argv[argv.index("--episodes") + 1]); cam = argv[argv.index("--camera") + 1] != "none"
        lens = [5, 3][:n] + [4] * max(0, n - 2)
        for i, t in enumerate(lens):
            arrays = dict(joint_pos=np.full((t, 2), 0.1 * i, np.float32), policy_obs=np.zeros((t, 4), np.float32),
                          action=np.full((t, 1), 0.5, np.float32), root_pos=np.zeros((t, 3), np.float32),
                          root_quat=np.tile(np.array([0, 0, 0, 1], np.float32), (t, 1)), reward=np.ones(t, np.float32))
            if cam:
                arrays["image"] = np.full((t, 24, 32, 3), 100, np.uint8)
            np.savez(out / ("episode_%03d.npz" % i), **arrays)
        (out / "meta.json").write_text(json.dumps(dict(task="Isaac-Cartpole", checkpoint="m", fps=30,
            joint_names=["slider_to_cart", "cart_to_pole"], action_names=["joint_effort.slider_to_cart"],
            policy_obs_dim=4, camera=dict(width=32, height=24) if cam else None, episodes=n, episode_len=lens,
            episode_return=[float(t) for t in lens], episode_end=["frames"] * n)))
        sys.exit(int(os.environ.get("FAKE_RECORD_EXIT", "0")))
    run = argv[argv.index("--run_name") + 1]
    d = Path.cwd() / "logs" / "rsl_rl" / "cartpole" / (time.strftime("%Y-%m-%d_%H-%M-%S_") + run)
    d.mkdir(parents=True); (d / "model_2.pt").write_bytes(b"x")
    print("[INFO] Logging experiment in directory: %s" % d.parent, flush=True)
    for it in range(3):
        print("Learning iteration %d/3\\n  Mean reward: %d.0" % (it, it), flush=True)
    print("Training time: 1.0 seconds", flush=True)
    """
)


@pytest.fixture
def trainer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> IsaacLabTrainer:
    script = tmp_path / "il" / "bin" / "python"
    script.parent.mkdir(parents=True)
    script.write_text(_FAKE.format(python=sys.executable))
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv(runtime.ISAACLAB_PYTHON_ENV, str(script))
    monkeypatch.setenv(runtime.EULA_ENV, "YES")
    monkeypatch.setenv(runtime.JOBS_DIR_ENV, str(tmp_path / "jobs"))
    monkeypatch.setenv("FAKE_ARGV", str(tmp_path / "argv.json"))
    return IsaacLabTrainer(poll_interval_s=0.05)


def _trained(trainer: IsaacLabTrainer, tmp_path: Path, **extra) -> str:
    from strands_robots.training.base import TrainSpec

    spec = TrainSpec(output_dir=str(tmp_path / "out"), steps=3, extra={"task": "Isaac-Cartpole", "wait": True, **extra})
    result = trainer.train(spec)
    assert result.status == "success", result.message
    return result.job_id


def test_the_runner_imports_no_strands() -> None:
    """It runs in the Isaac Lab interpreter, which has no strands installed."""
    tree = parse_file(RUNNER)
    imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert "strands_robots" not in imported


def test_the_runner_is_launched_with_the_run_s_checkpoint_and_physics(trainer, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        il, "rollout_to_dataset", lambda raw, ds, **k: {"dataset": ds, "episodes": 2, "frames": 8, "fps": 30}
    )
    job = _trained(trainer, tmp_path, physics="isaacsim_physx")
    rec = trainer.record(job, str(tmp_path / "ds"), episodes=2, frames=50, camera_eye=(1, 2, 3), wait=True)
    assert rec.status == "success", rec.message
    argv = json.loads((tmp_path / "argv.json").read_text())
    assert argv[0] == str(RUNNER)
    assert argv[argv.index("--checkpoint") + 1].endswith("model_2.pt")
    assert argv[argv.index("--episodes") + 1] == "2" and argv[argv.index("--frames") + 1] == "50"
    assert argv[argv.index("--eye") + 1] == "1.0,2.0,3.0"
    assert argv[argv.index("--override") + 1] == "physics=isaacsim_physx"
    assert rec.metrics["dataset"] == str(tmp_path / "ds") and rec.metrics["kind"] == "record"


def test_a_second_status_reads_the_converted_dataset_back(trainer, tmp_path: Path, monkeypatch) -> None:
    calls: list[str] = []

    def _convert(raw: object, ds: str, **k: object) -> dict[str, object]:
        calls.append(ds)
        return {"dataset": ds, "episodes": 1}

    monkeypatch.setattr(il, "rollout_to_dataset", _convert)
    rec = trainer.record(_trained(trainer, tmp_path), str(tmp_path / "ds"), wait=True)
    again = trainer.status(rec.job_id)
    assert again.status == "success" and again.metrics["episodes"] == 1
    assert len(calls) == 1, "the dataset is converted once"


def test_a_failed_rollout_is_an_error_and_writes_no_dataset(trainer, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FAKE_RECORD_EXIT", "3")
    rec = trainer.record(_trained(trainer, tmp_path), str(tmp_path / "ds"), wait=True)
    assert rec.status == "error" and "recording failed" in rec.message
    assert not (tmp_path / "ds").exists()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"episodes": 0}, "episodes"),
        ({"frames": -1}, "frames"),
        ({"camera": "yes"}, "camera"),
        ({"camera_eye": (1, 2)}, "camera_eye"),
        ({"camera_target": (1, 2, float("nan"))}, "camera_target"),
        ({"repo_id": "no-slash"}, "repo_id"),
    ],
)
def test_bad_options_are_refused_before_anything_launches(trainer, tmp_path: Path, kwargs, match) -> None:
    job = _trained(trainer, tmp_path)
    (tmp_path / "argv.json").unlink()
    rec = trainer.record(job, str(tmp_path / "ds"), **kwargs)
    assert rec.status == "error" and match in rec.message
    assert not (tmp_path / "argv.json").exists()


def test_the_tool_records_and_names_what_it_reads(trainer, tmp_path: Path, monkeypatch) -> None:
    from strands_robots.tools.train_policy import train_policy

    monkeypatch.setattr(il, "rollout_to_dataset", lambda raw, ds, **k: {"dataset": ds, "episodes": 2})
    job = _trained(trainer, tmp_path)
    out = train_policy(
        action="record", provider="isaaclab", job_id=job, extra={"dataset_dir": str(tmp_path / "ds"), "wait": True}
    )
    assert out["status"] == "success", out
    bad = train_policy(action="record", provider="isaaclab", job_id=job, extra={"fps": 5})
    assert bad["status"] == "error" and "dataset_dir" in bad["content"][0]["text"]


def test_the_rollout_becomes_a_lerobot_dataset(trainer, tmp_path: Path) -> None:
    pytest.importorskip("lerobot")
    job = _trained(trainer, tmp_path)
    rec = trainer.record(job, str(tmp_path / "ds"), repo_id="local/probe", episodes=2, wait=True)
    assert rec.status == "success", rec.message
    assert (rec.metrics["episodes"], rec.metrics["frames"]) == (2, 8)
    info = json.loads((tmp_path / "ds" / "meta" / "info.json").read_text())
    # 2 joints + 4 policy_obs + 3 root_pos + 4 root_quat
    assert info["features"]["observation.state"]["shape"] == [13]
    assert info["features"]["action"]["shape"] == [1]
    assert info["features"]["observation.images.camera"]["shape"][:2] == [24, 32]
    assert info["total_episodes"] == 2 and info["total_frames"] == 8
    assert np.isfinite(rec.metrics["episode_return"]).all()
