"""An rsl_rl actor - what Isaac Lab trains - loads through ``create_policy("rl")`` as the same function.

No Isaac Lab and no rsl_rl: the checkpoint is written here in the layout rsl_rl
5.x saves (``actor_state_dict`` with ``mlp.<i>`` layers and an optional
``obs_normalizer``), beside the ``params/agent.yaml`` Isaac Lab dumps. The
reference is the forward pass rsl_rl's ``MLPModel`` computes, written out by
hand: whiten with ``(x - mean) / (std + 1e-2)``, then Linear/ELU layers.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from strands_robots.policies.rl import RLCheckpointPolicy  # noqa: E402
from strands_robots.training.rl import rsl_rl  # noqa: E402
from strands_robots.training.rl.checkpoint import load_deployable_actor  # noqa: E402

_AGENT_YAML = """\
seed: 1
class_name: OnPolicyRunner
actor:
  class_name: MLPModel
  hidden_dims:
  - 16
  - 8
  activation: {activation}
  obs_normalization: {normalize}
critic:
  class_name: MLPModel
  hidden_dims:
  - 16
  - 8
  activation: tanh
  obs_normalization: false
algorithm:
  class_name: PPO
"""

OBS, ACT, HIDDEN = 4, 2, (16, 8)


def write_rsl_rl_run(
    run_dir: Path, *, iteration: int = 99, normalize: bool = False, activation: str = "elu"
) -> tuple[Path, dict]:
    """Write ``model_<iteration>.pt`` + ``params/agent.yaml`` as rsl_rl 5.x / Isaac Lab 3.0 do."""
    gen = torch.Generator().manual_seed(iteration)
    dims = [OBS, *HIDDEN, ACT]
    actor: dict = {"distribution.std_param": torch.ones(ACT)}
    for k, (i, o) in enumerate(zip(dims, dims[1:], strict=False)):
        actor[f"mlp.{2 * k}.weight"] = torch.randn(o, i, generator=gen)
        actor[f"mlp.{2 * k}.bias"] = torch.randn(o, generator=gen)
    if normalize:
        std = torch.rand(1, OBS, generator=gen) + 0.5
        actor.update(
            {
                "obs_normalizer._mean": torch.randn(1, OBS, generator=gen),
                "obs_normalizer._var": std**2,
                "obs_normalizer._std": std,
                "obs_normalizer.count": torch.tensor(1000),
            }
        )
    critic = {"mlp.0.weight": torch.randn(HIDDEN[0], OBS + 3, generator=gen)}
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor_state_dict": actor,
            "critic_state_dict": critic,
            "optimizer_state_dict": {},
            "iter": iteration,
            "infos": None,
        },
        run_dir / f"model_{iteration}.pt",
    )
    (run_dir / "params").mkdir(exist_ok=True)
    (run_dir / "params" / "agent.yaml").write_text(
        _AGENT_YAML.format(activation=activation, normalize=str(normalize).lower())
    )
    return run_dir / f"model_{iteration}.pt", actor


def _reference(actor: dict, x: torch.Tensor, activation: str = "elu") -> torch.Tensor:
    """rsl_rl's MLPModel forward, by hand."""
    if "obs_normalizer._mean" in actor:
        x = (x - actor["obs_normalizer._mean"]) / (actor["obs_normalizer._std"] + 1e-2)
    act = {"elu": torch.nn.functional.elu, "tanh": torch.tanh, "relu": torch.relu}[activation]
    layers = sorted({int(k.split(".")[1]) for k in actor if k.startswith("mlp.")})
    for n, i in enumerate(layers):
        x = x @ actor[f"mlp.{i}.weight"].T + actor[f"mlp.{i}.bias"]
        if n < len(layers) - 1:
            x = act(x)
    return x


class TestTheConvertedActorIsTheSameFunction:
    @pytest.mark.parametrize(
        ("normalize", "activation"), [(False, "elu"), (True, "elu"), (True, "tanh"), (False, "relu")]
    )
    def test_actions_match_rsl_rl_forward(self, tmp_path: Path, normalize: bool, activation: str) -> None:
        model, actor = write_rsl_rl_run(tmp_path / "run", normalize=normalize, activation=activation)
        out = rsl_rl.convert_checkpoint(str(model), str(tmp_path / "strands"))
        deployed = load_deployable_actor(out)
        x = torch.randn(32, OBS, generator=torch.Generator().manual_seed(1)) * 3
        assert torch.allclose(deployed.act(x), _reference(actor, x, activation), atol=1e-5)
        assert (deployed.normalizer is not None) is normalize

    def test_the_metadata_describes_the_actor(self, tmp_path: Path) -> None:
        model, _ = write_rsl_rl_run(tmp_path / "run", iteration=1499, normalize=True)
        out = rsl_rl.convert_checkpoint(str(model), str(tmp_path / "strands"), extra_meta={"task": "Isaac-Ant"})
        meta = json.loads((Path(out) / "policy_meta.json").read_text())
        assert meta["provider"] == "rsl_rl" and meta["activation"] == "elu" and meta["hidden_dims"] == list(HIDDEN)
        assert meta["num_actor_obs"] == OBS and meta["num_actions"] == ACT and meta["num_critic_obs"] == OBS + 3
        assert meta["actor_obs_keys"] == [f"policy_obs.{i}" for i in range(OBS)]
        assert meta["iteration"] == 1499 and meta["obs_normalization"] is True and meta["task"] == "Isaac-Ant"

    def test_strands_ppo_would_have_computed_another_function(self, tmp_path: Path) -> None:
        # The bug this module exists for: PPO's Tanh network rebuilt from the
        # same widths gives different actions from the same weights.
        model, actor = write_rsl_rl_run(tmp_path / "run")
        out = rsl_rl.convert_checkpoint(str(model), str(tmp_path / "strands"))
        x = torch.randn(8, OBS, generator=torch.Generator().manual_seed(2))
        tanh_net = rsl_rl.build_actor_critic(OBS, OBS, ACT, hidden_dims=HIDDEN, activation="tanh")
        tanh_net.load_state_dict({k: v for k, v in actor.items() if k.startswith("mlp.")})
        assert not torch.allclose(tanh_net.act_inference(x), load_deployable_actor(out).act(x), atol=1e-3)


class TestCreatePolicyRunsIt:
    def test_a_whole_observation_vector_or_its_elements(self, tmp_path: Path) -> None:
        model, actor = write_rsl_rl_run(tmp_path / "run", normalize=True)
        policy = RLCheckpointPolicy(checkpoint_dir=rsl_rl.convert_checkpoint(str(model), str(tmp_path / "s")))
        policy.set_robot_state_keys(["left", "right"])
        obs = [0.2, -1.0, 0.5, 3.0]
        want = _reference(actor, torch.tensor([obs]))[0]
        whole = asyncio.run(policy.get_actions({"policy_obs": obs}, ""))[0]
        split = asyncio.run(policy.get_actions({f"policy_obs.{i}": v for i, v in enumerate(obs)}, ""))[0]
        assert whole == pytest.approx(split)
        assert [whole["left"], whole["right"]] == pytest.approx(want.tolist(), abs=1e-5)
        assert policy.trained_by == "rsl_rl"

    def test_a_short_vector_is_reported_missing_not_padded(self, tmp_path: Path) -> None:
        model, _ = write_rsl_rl_run(tmp_path / "run")
        policy = RLCheckpointPolicy(checkpoint_dir=rsl_rl.convert_checkpoint(str(model), str(tmp_path / "s")))
        policy.set_robot_state_keys(["left", "right"])
        with pytest.raises(ValueError, match=r"policy_obs\.3"):
            asyncio.run(policy.get_actions({"policy_obs": [0.0, 0.0, 0.0]}, ""))


class TestWhatIsNotAnRslRlActorIsRefused:
    def test_a_file_without_an_actor_state_dict(self, tmp_path: Path) -> None:
        path = tmp_path / "model_1.pt"
        torch.save({"model_state_dict": {}}, path)
        with pytest.raises(ValueError, match="actor_state_dict"):
            rsl_rl.convert_checkpoint(str(path), str(tmp_path / "s"), activation="elu")

    def test_layers_that_do_not_chain(self, tmp_path: Path) -> None:
        path = tmp_path / "model_1.pt"
        torch.save(
            {
                "actor_state_dict": {
                    "mlp.0.weight": torch.zeros(8, 4),
                    "mlp.0.bias": torch.zeros(8),
                    "mlp.2.weight": torch.zeros(2, 5),
                    "mlp.2.bias": torch.zeros(2),
                }
            },
            path,
        )
        with pytest.raises(ValueError, match="do not chain"):
            rsl_rl.convert_checkpoint(str(path), str(tmp_path / "s"), activation="elu")

    def test_an_activation_it_cannot_rebuild(self, tmp_path: Path) -> None:
        model, _ = write_rsl_rl_run(tmp_path / "run", activation="gelu")
        with pytest.raises(ValueError, match="'gelu' is not supported"):
            rsl_rl.convert_checkpoint(str(model), str(tmp_path / "s"))

    def test_a_run_without_its_agent_config_names_the_file(self, tmp_path: Path) -> None:
        model, _ = write_rsl_rl_run(tmp_path / "run")
        (tmp_path / "run" / "params" / "agent.yaml").unlink()
        with pytest.raises(FileNotFoundError, match="agent.yaml"):
            rsl_rl.convert_checkpoint(str(model), str(tmp_path / "s"))

    def test_the_newest_checkpoint_is_the_highest_iteration(self, tmp_path: Path) -> None:
        for it in (0, 50, 1499, 150):
            (tmp_path / f"model_{it}.pt").write_bytes(b"")
        assert rsl_rl.latest_model(str(tmp_path)) == str(tmp_path / "model_1499.pt")
