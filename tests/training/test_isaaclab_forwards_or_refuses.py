"""Every TrainSpec field and Isaac Lab knob the isaaclab trainer is given reaches Isaac Lab, or is refused.

Measured on Isaac Lab 3.0 before this change: ``base_model`` (fine-tune from a
checkpoint), ``num_gpus``, ``num_nodes`` and ``save_freq`` were accepted and
silently dropped - a "fine-tune from model_600.pt" trained from scratch and
reported success - while ``resume`` was refused though ``--checkpoint`` does
it. ``extra`` accepted six keys, so no Hydra override (network size, PPO
hyperparameters, reward weights, terrain, episode length), no recurrent or
symmetry agent config, no training video and no ``--device`` could be asked
for. And ``learning_rate`` set only the INITIAL rate: rsl_rl's adaptive
schedule, used by 54 of 57 Isaac Lab PPO configs, replaced it from the first
iteration and capped it at 1e-2 (``learning_rate=0.05`` logged 2.3e-05,
0.00195, 0.01 ...).
"""

from __future__ import annotations

import json
import stat
import sys
import textwrap
from pathlib import Path

import pytest

from strands_robots.training.isaaclab import IsaacLabTrainer
from tests.training.test_isaaclab import _poll, _spec, _trainer, fake_python  # noqa: F401


def _flags(cmd: list[str]) -> list[str]:
    return cmd[4:]


class TestTrainSpecFieldsReachIsaacLab:
    def test_base_model_starts_from_that_checkpoint(
        self,
        fake_python: Path,  # noqa: F811
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("STRANDS_TRAIN_OUTPUT_DIR", str(tmp_path))  # a checkpoint home, per base_model_error
        ckpt = tmp_path / "prev" / "model_600.pt"
        ckpt.parent.mkdir()
        ckpt.write_bytes(b"")
        spec = _spec(tmp_path)
        spec.base_model = str(ckpt)
        assert _trainer().validate(spec) == []
        cmd = _trainer().build_command(spec, "isaaclab-20260929-010000-0123456789ab")
        assert cmd[cmd.index("--checkpoint") + 1] == str(ckpt.resolve())

    def test_a_run_directory_starts_from_its_newest_model(self, fake_python: Path, tmp_path: Path) -> None:  # noqa: F811
        run = tmp_path / "prev"
        run.mkdir()
        for it in (50, 600, 100):
            (run / f"model_{it}.pt").write_bytes(b"")
        spec = _spec(tmp_path)
        spec.base_model = str(run)
        cmd = _trainer().build_command(spec, "isaaclab-20260929-010000-0123456789ab")
        assert cmd[cmd.index("--checkpoint") + 1].endswith("model_600.pt")

    def test_a_base_model_that_is_not_a_checkpoint_is_refused(self, fake_python: Path, tmp_path: Path) -> None:  # noqa: F811
        spec = _spec(tmp_path)
        spec.base_model = "nvidia/some-hub-model"
        [problem] = _trainer().validate(spec)
        assert "is not an rsl_rl model_<iteration>.pt" in problem

    def test_resume_continues_the_newest_run_of_the_task(self, fake_python: Path, tmp_path: Path) -> None:  # noqa: F811
        trainer = _trainer()
        first = _poll(trainer, trainer.train(_spec(tmp_path, steps=3)).job_id)
        assert first.status == "success", first.message
        spec = _spec(tmp_path, steps=3)
        spec.resume = True
        assert trainer.validate(spec) == []
        cmd = trainer.build_command(spec, "isaaclab-20260929-020000-0123456789ab")
        assert Path(cmd[cmd.index("--checkpoint") + 1]).name == "model_2.pt"
        resumed = trainer.train(spec)
        job = json.loads((trainer._jobs_dir / resumed.job_id / "job.json").read_text())
        # rsl_rl continues the count: model_2.pt + 3 iterations ends at 5.
        assert (job["start_iteration"], job["max_iterations"]) == (2, 5)
        _poll(trainer, resumed.job_id)

    def test_resume_with_nothing_to_resume_is_refused(self, fake_python: Path, tmp_path: Path) -> None:  # noqa: F811
        spec = _spec(tmp_path)
        spec.resume = True
        [problem] = _trainer().validate(spec)
        assert "no Isaac-Cartpole run" in problem

    def test_save_freq_becomes_the_save_interval_and_the_default_is_left_alone(
        self,
        fake_python: Path,  # noqa: F811
        tmp_path: Path,
    ) -> None:
        spec = _spec(tmp_path, steps=500)
        assert not [t for t in _trainer().build_command(spec, "j") if t.startswith("agent.save_interval")]
        spec.save_freq = 25
        assert "agent.save_interval=25" in _trainer().build_command(spec, "j")

    @pytest.mark.parametrize(("gpus", "nodes"), [(2, 1), (1, 2)])
    def test_more_than_one_gpu_or_node_is_refused_not_run_on_one(
        self,
        fake_python: Path,  # noqa: F811
        tmp_path: Path,
        gpus: int,
        nodes: int,
    ) -> None:
        spec = _spec(tmp_path)
        spec.num_gpus, spec.num_nodes = gpus, nodes
        [problem] = _trainer().validate(spec)
        assert "would run on one GPU here" in problem and "--distributed" in problem

    def test_a_learning_rate_is_pinned_unless_a_schedule_is_chosen(
        self,
        fake_python: Path,  # noqa: F811
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        spec = _spec(tmp_path)
        spec.learning_rate = 0.05
        cmd = _trainer().build_command(spec, "j")
        assert "agent.algorithm.learning_rate=0.05" in cmd and "agent.algorithm.schedule=fixed" in cmd
        monkeypatch.setattr(IsaacLabTrainer, "_cfg_path_problems", lambda self, spec, paths: [])
        spec.extra = {**spec.extra, "overrides": {"agent.algorithm.schedule": "adaptive"}}
        cmd = _trainer().build_command(spec, "j")
        assert "agent.algorithm.schedule=fixed" not in cmd and "agent.algorithm.schedule=adaptive" in cmd


class TestIsaacLabKnobsReachIsaacLab:
    def test_overrides_agent_device_video_and_determinism(
        self,
        fake_python: Path,  # noqa: F811
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(IsaacLabTrainer, "_cfg_path_problems", lambda self, spec, paths: [])
        spec = _spec(
            tmp_path,
            overrides={"env.episode_length_s": 3.0, "agent.algorithm.gamma": 0.95, "env.scene.terrain.debug_vis": True},
            agent="rsl_rl_recurrent_cfg_entry_point",
            device="cuda:0",
            video=True,
            video_length=200,
            deterministic=True,
        )
        assert _trainer().validate(spec) == []
        cmd = _trainer().build_command(spec, "j")
        assert cmd[cmd.index("--agent") + 1] == "rsl_rl_recurrent_cfg_entry_point"
        assert cmd[cmd.index("--device") + 1] == "cuda:0" and "--deterministic" in cmd
        assert cmd[cmd.index("--video_length") + 1] == "200" and "--video" in cmd
        assert {"env.episode_length_s=3.0", "agent.algorithm.gamma=0.95", "env.scene.terrain.debug_vis=true"} <= set(
            cmd
        )

    def test_the_run_record_replays_the_environment_it_trained_in(
        self,
        fake_python: Path,  # noqa: F811
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(IsaacLabTrainer, "_cfg_path_problems", lambda self, spec, paths: [])
        trainer = _trainer()
        spec = _spec(tmp_path, steps=2, physics="isaacsim_physx",
                     overrides={"env.episode_length_s": 3.0, "agent.algorithm.gamma": 0.95})  # fmt: skip
        job = _poll(trainer, trainer.train(spec).job_id)
        played = trainer.play(job.job_id, video_length=10)
        _poll(trainer, played.job_id)
        argv = json.loads(Path(str(tmp_path / "argv.json") + ".play").read_text())
        assert "physics=isaacsim_physx" in argv and "env.episode_length_s=3.0" in argv
        assert "agent.algorithm.gamma=0.95" not in argv  # shaped the training only

    @pytest.mark.parametrize(
        ("extra", "needle"),
        [
            ({"overrides": {"episode_length_s": 3}}, "is not an 'env.<field>...'"),
            ({"overrides": {"env.x": "a b"}}, "plain token"),
            ({"overrides": {"agent.algorithm.learning_rate": 1e-3}}, "TrainSpec.learning_rate"),
            ({"overrides": {}}, "non-empty dict"),
            ({"agent": "recurrent"}, "entry point"),
            ({"device": "tpu"}, "'cpu', 'cuda'"),
            ({"video_length": 100}, "only with extra['video']=True"),
        ],
    )
    def test_a_malformed_knob_is_refused_naming_it(
        self,
        fake_python: Path,  # noqa: F811
        tmp_path: Path,
        extra: dict,
        needle: str,
    ) -> None:
        problems = _trainer().validate(_spec(tmp_path, **extra))
        assert any(needle in p for p in problems), problems


_CHECKER = textwrap.dedent(
    """\
    #!{python}
    import json, os, sys
    # Answers the config check the way the Isaac Lab interpreter does, from $FAKE_CFG_REPORT.
    assert sys.argv[1] == "-c" and "STRANDS_CFG_CHECK" in sys.argv[2]
    print("[INFO]: Parsing configuration from: ...")
    print("STRANDS_CFG_CHECK " + os.environ["FAKE_CFG_REPORT"])
    """
)


class TestAnOverridePathIsCheckedAgainstTheRealConfig:
    def test_a_misspelt_env_path_is_refused_with_the_field_it_meant(
        self,
        fake_python: Path,  # noqa: F811
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        checker = tmp_path / "checker"
        checker.write_text(_CHECKER.format(python=sys.executable))
        checker.chmod(checker.stat().st_mode | stat.S_IXUSR)
        report = {
            "problems": {"env.episode_lenght_s": {"missing": "env.episode_lenght_s", "close": ["episode_length_s"]}}
        }
        monkeypatch.setenv("FAKE_CFG_REPORT", json.dumps(report))
        trainer = IsaacLabTrainer(python=str(checker), jobs_dir=str(tmp_path / "jobs"))
        problems = trainer._cfg_path_problems(_spec(tmp_path), ["env.episode_lenght_s"])
        assert problems == [
            "isaaclab: extra['overrides'] path 'env.episode_lenght_s' names no field ('env.episode_lenght_s' does not "
            "exist in Isaac-Cartpole's config); did you mean ['episode_length_s']? - Hydra would add it silently and "
            "train the default"
        ]

    def test_an_agent_config_the_task_lacks_is_refused(
        self,
        fake_python: Path,  # noqa: F811
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        checker = tmp_path / "checker"
        checker.write_text(_CHECKER.format(python=sys.executable))
        checker.chmod(checker.stat().st_mode | stat.S_IXUSR)
        monkeypatch.setenv(
            "FAKE_CFG_REPORT", json.dumps({"load_error": "ValueError: no rsl_rl_symmetry_cfg_entry_point"})
        )
        trainer = IsaacLabTrainer(python=str(checker), jobs_dir=str(tmp_path / "jobs"))
        [problem] = trainer._cfg_path_problems(_spec(tmp_path, agent="rsl_rl_symmetry_cfg_entry_point"), [])
        assert "has no loadable 'rsl_rl_symmetry_cfg_entry_point' config" in problem
