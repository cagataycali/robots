"""An exported Isaac Lab actor carries its deploy contract, and ``create_policy("rl")`` honours it.

The actor's weights say nothing about what its numbers mean. Isaac Lab's
``JointPositionAction`` commands ``offset + scale * action`` to joints in the
order the run's articulation reported - type-grouped under PhysX, depth-first
under Newton for the same Go2 task - and an export that recorded none of it
deployed the Go2 rough-terrain policy with joints up to 1.73 rad from where
Isaac Lab would have put them. These tests pin the contract end to end, from
the IO descriptors Isaac Lab writes (the real Go2 layout below is from an
``--export_io_descriptors`` run) through ``export`` to the targets
``get_actions`` returns.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from strands_robots.tools.train_policy import train_policy
from strands_robots.training.isaaclab import RUN_RECORD_FILE
from strands_robots.training.rl.deploy_contract import (
    DeployContractError,
    apply_action_contract,
    complete_obs_layout,
    contract_from_io_descriptors,
    contract_problems,
)
from tests.training.test_isaaclab import _json_block, _poll, _spec, _trainer, fake_python  # noqa: F401

_GO2 = ["FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint", "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"]  # fmt: skip
_GO2_OFFSET = [0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5]


def _term(name: str, width: int, **extra: Any) -> dict[str, Any]:
    return {"name": name, "full_path": f"isaaclab.envs.mdp.observations.{name}", "shape": [width],
            "overloads": {"clip": None, "flatten_history_dim": True, "history_length": 0, "scale": None}, **extra}  # fmt: skip


def go2_io_descriptors() -> dict[str, Any]:
    """The Isaac-Velocity-Flat-UnitreeGo2 descriptors Isaac Lab 3.0 wrote on Newton (trimmed)."""
    return {
        "actions": [
            {
                "name": "joint_position_action",
                "full_path": "isaaclab.envs.mdp.actions.joint_actions.JointPositionAction",
                "joint_names": _GO2,
                "offset": _GO2_OFFSET,
                "scale": 0.25,
                "clip": None,
                "shape": [12],
            }  # fmt: skip
        ],
        "articulations": {"robot": {"joint_names": _GO2, "default_joint_pos": _GO2_OFFSET}},
        "observations": {
            "policy": [
                _term("base_lin_vel", 3, extras={"units": "m/s"}),
                _term("base_ang_vel", 3),
                _term("projected_gravity", 3),
                _term("generated_commands", 3),
                _term("joint_pos_rel", 12, joint_names=_GO2, joint_pos_offsets=_GO2_OFFSET),
                _term("joint_vel_rel", 12, joint_names=_GO2),
                _term("last_action", 12),
            ]
        },
        "scene": {"decimation": 4, "dt": 0.02, "physics_dt": 0.005},
    }


def fake_io_descriptors() -> dict[str, Any]:
    """A two-joint, four-observation layout matching ``write_rsl_rl_run``'s actor."""
    joints = ["slider_to_cart", "cart_to_pole"]
    return {
        "actions": [
            {
                "name": "joint_pos",
                "full_path": "isaaclab.envs.mdp.actions.joint_actions.JointPositionAction",
                "joint_names": joints,
                "offset": [0.1, -0.2],
                "scale": 0.5,
                "clip": None,
                "shape": [2],
            }  # fmt: skip
        ],
        "articulations": {"robot": {"joint_names": joints, "default_joint_pos": [0.1, -0.2]}},
        "observations": {"policy": [_term("joint_pos_rel", 2, joint_names=joints), _term("joint_vel_rel", 2)]},
        "scene": {"decimation": 2, "dt": 0.01, "physics_dt": 0.005},
    }


class TestTheContractIsReadFromTheIoDescriptors:
    def test_the_go2_layout_names_joints_offsets_and_the_obs_vector(self) -> None:
        contract = contract_from_io_descriptors(go2_io_descriptors(), physics="newton_mjwarp")
        assert contract["action_keys"] == _GO2 and contract["physics"] == "newton_mjwarp"
        assert contract["action_terms"][0]["scale"] == [0.25] * 12
        assert contract["default_joint_pos"]["RL_thigh_joint"] == pytest.approx(1.0)
        assert [(t["name"], t["start"], t["width"]) for t in contract["obs_layout"]][-3:] == [
            ("joint_pos_rel", 12, 12),
            ("joint_vel_rel", 24, 12),
            ("last_action", 36, 12),
        ]
        assert contract["num_obs"] == 48 and contract["control_dt"] == pytest.approx(0.02)
        assert (contract["quat_order"], contract["base_velocity_frame"]) == ("xyzw", "body")
        assert contract_problems(contract, num_actor_obs=48, num_actions=12) == []

    def test_a_zero_action_is_the_default_pose_not_zero_radians(self) -> None:
        contract = contract_from_io_descriptors(go2_io_descriptors())
        targets = apply_action_contract(contract, [0.0] * 12)
        assert [targets[j] for j in _GO2] == pytest.approx(_GO2_OFFSET)
        assert apply_action_contract(contract, [1.0] * 12)["FL_calf_joint"] == pytest.approx(-1.25)

    def test_a_clip_bounds_the_processed_target_as_isaac_lab_does(self) -> None:
        """``JointAction.process_actions`` computes ``raw * scale + offset`` and clamps THAT
        (joint_actions.py); with offset 0.1, scale 0.5, clip (-1, 1) a raw 5.0 is 1.0, not 0.6."""
        desc = fake_io_descriptors()
        desc["actions"][0]["clip"] = [-1.0, 1.0]
        contract = contract_from_io_descriptors(desc)
        assert apply_action_contract(contract, [5.0, -5.0]) == pytest.approx(
            {"slider_to_cart": 1.0, "cart_to_pole": -1.0}
        )
        assert apply_action_contract(contract, [1.0, 1.0]) == pytest.approx(
            {"slider_to_cart": 0.6, "cart_to_pole": 0.3}
        )

    def test_the_per_joint_clip_pairs_isaac_lab_writes_are_read(self) -> None:
        """Isaac Lab's IO descriptor serialises ``clip`` as ``self._clip[0].tolist()``:
        one ``[low, high]`` pair PER JOINT, ``action_dim`` long. That is the shape every
        real clipped run ships; refusing it withheld the contract from exactly those runs."""
        desc = fake_io_descriptors()
        desc["actions"][0]["clip"] = [[-1.0, 1.0], [-0.5, 0.25]]
        contract = contract_from_io_descriptors(desc)
        assert contract["action_terms"][0]["clip"] == [[-1.0, 1.0], [-0.5, 0.25]]
        # ...and each joint is bounded by ITS pair, after the affine.
        assert apply_action_contract(contract, [5.0, 5.0]) == pytest.approx(
            {"slider_to_cart": 1.0, "cart_to_pole": 0.25}
        )
        assert apply_action_contract(contract, [-5.0, -5.0]) == pytest.approx(
            {"slider_to_cart": -1.0, "cart_to_pole": -0.5}
        )

    def test_a_clip_list_of_the_wrong_length_or_shape_is_still_refused(self) -> None:
        desc = fake_io_descriptors()
        for bad in ([[-1.0, 1.0]], [[-1.0, 1.0], [0.0]], [[-1.0, 1.0], [-1.0, "x"]], "0 1"):
            desc["actions"][0]["clip"] = bad
            with pytest.raises(DeployContractError, match="unrecognised action clip"):
                contract_from_io_descriptors(desc)

    def test_widths_that_do_not_fit_the_actor_are_named(self) -> None:
        contract = contract_from_io_descriptors(go2_io_descriptors())
        problems = contract_problems(contract, num_actor_obs=40, num_actions=11)
        assert problems == [
            "the contract names 12 action joints, the actor emits 11",
            "the contract's observation terms add up to 48, the actor reads 40",
        ]

    def test_a_term_isaac_lab_does_not_describe_leaves_the_layout_marked_incomplete(self) -> None:
        # The rough-terrain Go2 reads 235 values; Isaac Lab describes 48 (no height_scan).
        contract = contract_from_io_descriptors(go2_io_descriptors())
        assert contract_problems(contract, num_actor_obs=235, num_actions=12) == []
        marked = complete_obs_layout(contract, num_actor_obs=235)
        assert (marked["obs_layout_complete"], marked["obs_unaccounted"]) == (False, 187)
        assert complete_obs_layout(contract, num_actor_obs=48)["obs_layout_complete"] is True

    def test_a_non_position_action_term_is_refused(self) -> None:
        desc = fake_io_descriptors()
        desc["actions"][0]["full_path"] = "isaaclab.envs.mdp.actions.joint_actions.JointEffortAction"
        [problem] = contract_problems(contract_from_io_descriptors(desc), num_actor_obs=4, num_actions=2)
        assert "JointEffortAction" in problem and "joint-position" in problem

    def test_descriptors_without_actions_are_refused(self) -> None:
        with pytest.raises(DeployContractError, match="no action terms"):
            contract_from_io_descriptors({"observations": {"policy": [_term("x", 1)]}})


def _export(tmp_path: Path) -> dict[str, Any]:
    return train_policy(action="export", provider="isaaclab", output_dir=str(tmp_path / "out"), steps=3,
                        extra={"task": "Isaac-Cartpole"})  # fmt: skip


def _train_and_write_actor(tmp_path: Path) -> Path:
    pytest.importorskip("torch")
    from tests.training.test_rsl_rl_actor_export import write_rsl_rl_run

    trainer = _trainer()
    trained = _poll(trainer, trainer.train(_spec(tmp_path, physics="isaacsim_physx")).job_id)
    run = Path(trained.checkpoint_dir)
    write_rsl_rl_run(run, iteration=2)
    return run


class TestExportWritesTheContractAndDeployHonoursIt:
    def test_every_training_run_asks_isaac_lab_for_its_io_descriptors(self, fake_python: Path, tmp_path: Path) -> None:  # noqa: F811
        cmd = _trainer().build_command(_spec(tmp_path), "isaaclab-20260929-010000-0123456789ab")
        assert "--export_io_descriptors" in cmd

    def test_outputs_are_joint_targets_bound_by_name(
        self,
        fake_python: Path,  # noqa: F811
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from strands_robots.policies import create_policy

        monkeypatch.setenv("FAKE_IO_DESCRIPTORS", json.dumps(fake_io_descriptors()))
        _train_and_write_actor(tmp_path)
        path = _json_block(_export(tmp_path))["exported_model"]
        meta = json.loads((Path(path) / "policy_meta.json").read_text())
        assert meta["action_keys"] == ["slider_to_cart", "cart_to_pole"]
        assert meta["deploy_contract"]["action_terms"][0]["offset"] == [0.1, -0.2]

        obs = {"policy_obs": [0.1, -0.2, 0.3, 0.0]}
        raw = create_policy("rl", checkpoint_dir=path, raw_actions=True)
        [raw_action] = asyncio.run(raw.get_actions(obs, ""))
        policy = create_policy("rl", checkpoint_dir=path)
        # The robot lists its joints in another order; binding is by name.
        policy.set_robot_state_keys(["cart_to_pole", "slider_to_cart"])
        [action] = asyncio.run(policy.get_actions(obs, ""))
        assert action["slider_to_cart"] == pytest.approx(0.1 + 0.5 * raw_action["slider_to_cart"])
        assert action["cart_to_pole"] == pytest.approx(-0.2 + 0.5 * raw_action["cart_to_pole"])
        contract = getattr(policy, "deploy_contract", None)
        assert contract is not None and contract["obs_group"] == "policy" and contract["obs_layout_complete"]

    def test_a_robot_without_the_actors_joints_is_refused(
        self,
        fake_python: Path,  # noqa: F811
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from strands_robots.policies import create_policy

        monkeypatch.setenv("FAKE_IO_DESCRIPTORS", json.dumps(fake_io_descriptors()))
        _train_and_write_actor(tmp_path)
        policy = create_policy("rl", checkpoint_dir=_json_block(_export(tmp_path))["exported_model"])
        policy.set_robot_state_keys(["joint_a", "joint_b"])
        with pytest.raises(
            ValueError, match=r"drives joints \['slider_to_cart', 'cart_to_pole'\].*never by position.*joint_map="
        ):
            asyncio.run(policy.get_actions({"policy_obs": [0.0] * 4}, ""))

    def test_a_run_without_descriptors_gets_them_from_a_zero_iteration_launch(
        self,
        fake_python: Path,  # noqa: F811
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        run = _train_and_write_actor(tmp_path)  # trained before descriptors were requested
        assert not (run / "io_descriptors").exists()
        monkeypatch.setenv("FAKE_IO_DESCRIPTORS", json.dumps(fake_io_descriptors()))
        path = _json_block(_export(tmp_path))["exported_model"]
        assert json.loads((Path(path) / "policy_meta.json").read_text())["deploy_contract"]["num_obs"] == 4
        argv = json.loads((tmp_path / "argv.json").read_text())
        assert argv[argv.index("--max_iterations") + 1] == "0" and argv[argv.index("--num_envs") + 1] == "1"
        assert "physics=isaacsim_physx" in argv and (run / "io_descriptors" / "IO_descriptors.yaml").is_file()

    @pytest.mark.parametrize(
        "field, value",
        [
            ("task", "Isaac-Cartpole --hydra.run.dir=/tmp/x"),
            ("overrides", ["physics=isaacsim_physx", "--config-dir=/tmp/evil"]),
            ("overrides", ["hydra.job.chdir=True"]),
            ("overrides", ["physics=isaacsim_physx; rm -rf /"]),
        ],
    )
    def test_a_crafted_run_record_never_reaches_the_relaunch_argv(
        self,
        fake_python: Path,  # noqa: F811
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        field: str,
        value: Any,
    ) -> None:
        """The run record lives in a directory the caller points at (a downloaded checkpoint is a
        realistic input), and export relaunches the operator's Isaac Lab interpreter from it. Its
        ``task`` and ``overrides`` are re-validated against the shapes the train path accepts and
        ``_run_record`` writes; anything else is the reason string, and no launch happens."""
        run = _train_and_write_actor(tmp_path)
        record = json.loads((run / RUN_RECORD_FILE).read_text())
        record[field] = value
        (run / RUN_RECORD_FILE).write_text(json.dumps(record))
        monkeypatch.setenv("FAKE_IO_DESCRIPTORS", json.dumps(fake_io_descriptors()))
        training_argv = (tmp_path / "argv.json").read_text()  # the train launch, before the record was touched
        exported = _export(tmp_path)
        injected = value if isinstance(value, str) else value[-1]
        if field == "task":
            # A record whose task is not the task asked for is refused before the
            # relaunch is even considered: export picks runs by task since #4436.
            assert exported["status"] == "error", exported
            assert "no Isaac-Cartpole run" in exported["content"][0]["text"], exported
        else:
            meta = json.loads((Path(_json_block(exported)["exported_model"]) / "policy_meta.json").read_text())
            reason = meta["deploy_contract_missing"]
            assert RUN_RECORD_FILE in reason and repr(injected) in reason and "not rebuilt" in reason, reason
        assert (tmp_path / "argv.json").read_text() == training_argv, "the crafted record reached a relaunch argv"

    def test_an_export_without_a_contract_is_refused_unless_raw_actions_are_asked_for(
        self,
        fake_python: Path,  # noqa: F811
        tmp_path: Path,
    ) -> None:
        from strands_robots.policies import create_policy

        _train_and_write_actor(tmp_path)  # a direct-workflow task: Isaac Lab writes no descriptors
        path = _json_block(_export(tmp_path))["exported_model"]
        meta = json.loads((Path(path) / "policy_meta.json").read_text())
        assert "direct-workflow" in meta["deploy_contract_missing"]
        with pytest.raises(ValueError, match=r"without its deploy contract.*raw_actions=True"):
            create_policy("rl", checkpoint_dir=path)
        raw = create_policy("rl", checkpoint_dir=path, raw_actions=True)
        raw.set_robot_state_keys(["a", "b"])
        assert set(asyncio.run(raw.get_actions({"policy_obs": [0.0] * 4}, ""))[0]) == {"a", "b"}


class TestContractJointsBindByName:
    def test_the_isaac_lab_go2_binds_to_the_mujoco_go2_actuators(self) -> None:
        from strands_robots.policies.rl import bind_contract_joints

        mujoco = ["FL_hip", "FL_thigh", "FL_calf", "FR_hip", "FR_thigh", "FR_calf",
                  "RL_hip", "RL_thigh", "RL_calf", "RR_hip", "RR_thigh", "RR_calf"]  # fmt: skip
        physx_order = [f"{leg}_{part}_joint" for part in ("hip", "thigh", "calf") for leg in ("FL", "FR", "RL", "RR")]
        binding = bind_contract_joints(physx_order, mujoco, {})
        assert binding["FR_hip_joint"] == "FR_hip" and binding["RR_calf_joint"] == "RR_calf"
        assert sorted(binding.values()) == sorted(mujoco)

    def test_a_joint_map_covers_what_names_cannot(self) -> None:
        from strands_robots.policies.rl import bind_contract_joints

        assert bind_contract_joints(["slider_to_cart"], ["cart_x"], {"slider_to_cart": "cart_x"}) == {
            "slider_to_cart": "cart_x"
        }
        with pytest.raises(ValueError, match="never by position"):
            bind_contract_joints(["slider_to_cart"], ["cart_x"], {})


# What Isaac Lab's dump_yaml really writes for an ObsTerm with clip=(-1.0, 1.0):
# class_to_dict keeps the tuple and the default Dumper tags it. The loader must
# read the SEQUENCE under the tag, not null the node: the clip is the only
# source of height_scan's training range, and a nulled clip is baked into every
# exported contract.
_ENV_YAML_WITH_TAGGED_CLIP = """observations:
  policy:
    height_scan:
      func: isaaclab.envs.mdp.observations:height_scan
      clip: !!python/tuple
      - -1.0
      - 1.0
      scale: 1.0
      params:
        offset: 0.5
    base_lin_vel:
      func: isaaclab.envs.mdp.observations:base_lin_vel
      clip: null
      scale: !!python/tuple [2.0, 2.0, 2.0]
      params: {}
    joint_pos:
      func: isaaclab.envs.mdp.observations:joint_pos_rel
      noise: !!python/object:isaaclab.utils.noise.noise_cfg.UniformNoiseCfg
        n_min: -0.01
        n_max: 0.01
      params: {}
"""


class TestARealEnvYamlKeepsItsClip:
    def test_a_tuple_tagged_clip_survives_the_read_into_the_contract(self, tmp_path: Path) -> None:
        from strands_robots.training.rl.deploy_contract import env_cfg_terms

        path = tmp_path / "env.yaml"
        path.write_text(_ENV_YAML_WITH_TAGGED_CLIP)
        env_cfg = _trainer()._read_yaml(path)

        terms = {t["term"]: t for t in env_cfg_terms(env_cfg)}
        assert terms["height_scan"]["clip"] == [-1.0, 1.0]
        assert terms["base_lin_vel"]["scale"] == [2.0, 2.0, 2.0]  # a tagged flow sequence too
        assert terms["base_lin_vel"]["clip"] is None  # a real null stays null

    def test_a_tagged_object_is_still_never_constructed(self, tmp_path: Path) -> None:
        """Only sequences are read through the tag; an object node reads as None, as before."""
        path = tmp_path / "env.yaml"
        path.write_text(_ENV_YAML_WITH_TAGGED_CLIP)
        env_cfg = _trainer()._read_yaml(path)
        assert env_cfg["observations"]["policy"]["joint_pos"]["noise"] is None

    def test_the_interpreter_fallback_reads_the_same_shape(self, tmp_path: Path) -> None:
        """Without PyYAML in this process the Isaac Lab interpreter parses; same rule."""
        import subprocess
        import sys

        from strands_robots.training.isaaclab import _YAML_TO_JSON

        path = tmp_path / "env.yaml"
        path.write_text(_ENV_YAML_WITH_TAGGED_CLIP)
        out = subprocess.run([sys.executable, "-c", _YAML_TO_JSON, str(path)], capture_output=True, check=True)
        parsed = json.loads(out.stdout)
        assert parsed["observations"]["policy"]["height_scan"]["clip"] == [-1.0, 1.0]
        assert parsed["observations"]["policy"]["joint_pos"]["noise"] is None


class TestAnUnreadableDescriptorFileIsAReasonNotACrash:
    """``_read_yaml`` raises ``ValueError`` for every way a file fails to parse.

    The export path degrades to ``deploy_contract_missing`` on ``(OSError,
    ValueError)``; PyYAML's ``YAMLError`` and the interpreter fallback's
    ``CalledProcessError`` / ``TimeoutExpired`` are neither, so a truncated
    ``IO_descriptors.yaml`` (a run killed during Isaac Lab's startup write)
    crashed ``export`` with a traceback instead of exporting the actor with
    the reason recorded.
    """

    _TRUNCATED = "observations:\n  policy:\n    - name: joint_pos\n      shape: [1, 12\n"  # cut mid-list

    def test_malformed_yaml_is_a_value_error_naming_the_file(self, tmp_path: Path) -> None:
        pytest.importorskip("yaml")
        path = tmp_path / "IO_descriptors.yaml"
        path.write_text(self._TRUNCATED)
        with pytest.raises(ValueError, match=r"IO_descriptors\.yaml.*could not be parsed as YAML"):
            _trainer()._read_yaml(path)

    @pytest.mark.parametrize(
        "failure",
        [
            lambda cmd: (_ for _ in ()).throw(
                __import__("subprocess").CalledProcessError(1, cmd, stderr=b"yaml.scanner.ScannerError")
            ),
            lambda cmd: (_ for _ in ()).throw(__import__("subprocess").TimeoutExpired(cmd, 120)),
        ],
        ids=["child-failed", "child-timed-out"],
    )
    def test_the_interpreter_fallback_failures_are_value_errors_too(self, tmp_path: Path, monkeypatch, failure) -> None:
        import subprocess
        import sys

        monkeypatch.setitem(sys.modules, "yaml", None)  # the no-PyYAML branch
        monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: failure(cmd))
        path = tmp_path / "IO_descriptors.yaml"
        path.write_text(self._TRUNCATED)
        with pytest.raises(ValueError, match=r"IO_descriptors\.yaml"):
            _trainer()._read_yaml(path)

    def test_export_records_the_reason_and_still_writes_the_actor(self, fake_python: Path, tmp_path: Path) -> None:  # noqa: F811
        pytest.importorskip("yaml")
        run = _train_and_write_actor(tmp_path)
        descriptors = run / "io_descriptors" / "IO_descriptors.yaml"
        descriptors.parent.mkdir(parents=True, exist_ok=True)
        descriptors.write_text(self._TRUNCATED)

        meta = json.loads((Path(_json_block(_export(tmp_path))["exported_model"]) / "policy_meta.json").read_text())

        assert "could not be read as IO descriptors" in meta["deploy_contract_missing"]
        assert "could not be parsed as YAML" in meta["deploy_contract_missing"]
