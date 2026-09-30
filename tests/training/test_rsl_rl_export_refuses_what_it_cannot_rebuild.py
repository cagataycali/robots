"""An rsl_rl actor this reader cannot rebuild is refused at export, and a wrong-width observation at deploy.

Isaac Lab trains three rsl_rl actor classes: ``MLPModel``, ``CNNModel`` (an image
encoder under ``cnns.`` in front of the MLP) and ``RNNModel`` (an LSTM/GRU under
``rnn.``). The exporter kept the ``mlp.`` keys of all three and dropped the rest,
so an Isaac-Cartpole-Camera actor exported as a 1,600-input MLP (the CNN's latent
width), and ``create_policy("rl")`` then read the first 1,600 of ANY longer vector
- a 55,296-value image, or 100,000 values - and answered 0.591 where rsl_rl's own
CNN answers -0.088. A recurrent AnymalD actor exported "OK" without its LSTM. The
key layouts below are the ones those two Isaac Lab 3.0 runs wrote.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from strands_robots.policies.rl import RLCheckpointPolicy  # noqa: E402
from strands_robots.training.rl import rsl_rl  # noqa: E402
from tests.training.test_rsl_rl_actor_export import OBS, write_rsl_rl_run  # noqa: E402


def _retype(run_dir: Path, model: Path, class_name: str, extra: dict) -> None:
    """Turn the MLP run into the given actor class, as Isaac Lab writes it."""
    state = torch.load(model, map_location="cpu", weights_only=True)
    state["actor_state_dict"].update(extra)
    torch.save(state, model)
    yaml = run_dir / "params" / "agent.yaml"
    yaml.write_text(
        yaml.read_text().replace("actor:\n  class_name: MLPModel", f"actor:\n  class_name: {class_name}", 1)
    )


def test_a_cnn_actor_is_refused_naming_its_image_encoder(tmp_path: Path) -> None:
    model, _ = write_rsl_rl_run(tmp_path / "run")
    _retype(tmp_path / "run", model, "CNNModel", {"cnns.policy.0.weight": torch.zeros(8, 3, 3, 3)})
    with pytest.raises(ValueError, match=r"CNNModel with an image \(CNN\) encoder.*\['cnns'\].*play\(\)"):
        rsl_rl.convert_checkpoint(str(model), str(tmp_path / "out"))
    assert not (tmp_path / "out" / "policy.pt").exists()


def test_a_recurrent_actor_is_refused_naming_its_memory(tmp_path: Path) -> None:
    model, _ = write_rsl_rl_run(tmp_path / "run")
    _retype(tmp_path / "run", model, "RNNModel", {"rnn.rnn.weight_ih_l0": torch.zeros(512, OBS)})
    with pytest.raises(ValueError, match=r"RNNModel with a recurrent \(LSTM/GRU\) memory.*\['rnn'\]"):
        rsl_rl.convert_checkpoint(str(model), str(tmp_path / "out"))


def test_an_unknown_part_is_refused_even_without_a_class_name(tmp_path: Path) -> None:
    model, _ = write_rsl_rl_run(tmp_path / "run")
    _retype(tmp_path / "run", model, "MLPModel", {"memory.weight": torch.zeros(2, 2)})
    (tmp_path / "run" / "params" / "agent.yaml").unlink()
    with pytest.raises(ValueError, match=r"\['memory'\]"):
        rsl_rl.convert_checkpoint(str(model), str(tmp_path / "out"), activation="elu")


def test_an_mlp_actor_still_exports(tmp_path: Path) -> None:
    model, _ = write_rsl_rl_run(tmp_path / "run")
    assert rsl_rl.read_agent_actor_class(str(tmp_path / "run")) == "MLPModel"
    assert (Path(rsl_rl.convert_checkpoint(str(model), str(tmp_path / "out"))) / "policy.pt").exists()


@pytest.mark.parametrize("length", [OBS + 1, 100_000, OBS - 1])
def test_a_vector_of_another_width_is_refused_not_truncated(tmp_path: Path, length: int) -> None:
    model, _ = write_rsl_rl_run(tmp_path / "run")
    policy = RLCheckpointPolicy(checkpoint_dir=rsl_rl.convert_checkpoint(str(model), str(tmp_path / "s")))
    policy.set_robot_state_keys(["left", "right"])
    with pytest.raises(ValueError, match=rf"carries {length} values, but the actor reads {OBS}"):
        asyncio.run(policy.get_actions({"policy_obs": [0.0] * length}, ""))
