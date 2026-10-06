# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``randomize`` / ``set_obs_noise`` reject keywords they cannot honor.

Both methods declare ``**kwargs`` to match the ``**kwargs``-typed
:class:`~strands_robots.simulation.base.SimEngine` base signature, and neither
forwards it anywhere. That made every misspelled or invented parameter a silent
no-op reported as success: ``randomize(randomize_position=True)`` (singular)
answered "Domain Randomization applied" with object positions untouched, and
``set_obs_noise(joint_pos_stdev=0.05)`` answered with an all-zero noise config,
so a data-collection run believed it was perturbing a world it never touched.
The action dispatcher's own unknown-parameter guard is skipped for
``**kwargs`` methods, so both the direct Python API and the agent dispatch path
were affected.
"""

import inspect

import numpy as np
import pytest

from strands_robots.simulation import Simulation
from strands_robots.simulation.base import unknown_kwargs_error
from strands_robots.simulation.mujoco.randomization import _RANDOMIZE_PARAMS

pytest.importorskip("mujoco")


@pytest.fixture
def sim():
    s = Simulation()
    s.create_world()
    s.add_object("cube", shape="box", size=[0.03, 0.03, 0.03], position=[0.2, 0.0, 0.02])
    yield s
    s.cleanup()


def _cube_xy(sim) -> np.ndarray:
    """Return the cube's free-joint xy slice straight from MuJoCo data."""
    import mujoco

    model, data = sim._world._model, sim._world._data
    jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
    addr = model.jnt_qposadr[jnt]
    return np.array(data.qpos[addr : addr + 2], dtype=float)


class TestRandomizeRejectsUnknownParams:
    def test_misspelled_axis_errors_instead_of_reporting_success(self, sim):
        before = _cube_xy(sim)
        result = sim.randomize(randomize_position=True, position_noise=0.05, seed=1)

        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "randomize_position" in text
        assert "randomize_positions" in text  # the valid spelling is offered
        assert np.allclose(_cube_xy(sim), before)

    def test_invented_parameters_are_all_named(self, sim):
        result = sim.randomize(objects=["cube"], position_range=0.05)

        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "objects" in text and "position_range" in text

    def test_valid_call_still_randomizes_positions(self, sim):
        before = _cube_xy(sim)
        result = sim.randomize(
            randomize_colors=False,
            randomize_lighting=False,
            randomize_positions=True,
            position_noise=0.05,
            seed=7,
        )

        assert result["status"] == "success"
        assert not np.allclose(_cube_xy(sim), before)

    def test_agent_dispatch_path_rejects_too(self, sim):
        result = sim._dispatch_action("randomize", {"action": "randomize", "randomize_position": True})

        assert result["status"] == "error"
        assert "randomize_position" in result["content"][0]["text"]

    def test_accepted_names_match_the_signature(self):
        declared = {
            name
            for name, p in inspect.signature(Simulation.randomize).parameters.items()
            if name != "self" and p.kind is not inspect.Parameter.VAR_KEYWORD
        }
        assert set(_RANDOMIZE_PARAMS) == declared


class TestSetObsNoiseRejectsUnknownParams:
    def test_valid_call_still_configures_noise(self, sim):
        result = sim.set_obs_noise(joint_pos_std=0.02, seed=0)

        assert result["status"] == "success"
        assert sim._obs_noise["joint_pos_std"] == pytest.approx(0.02)

    def test_agent_dispatch_path_rejects_too(self, sim):
        result = sim._dispatch_action("set_obs_noise", {"action": "set_obs_noise", "joint_pos_stdev": 0.05})

        assert result["status"] == "error"
        assert "joint_pos_stdev" in result["content"][0]["text"]


class TestUnknownKwargsErrorHelper:
    def test_no_residual_kwargs_is_not_an_error(self):
        assert unknown_kwargs_error("randomize", {}, ("seed",)) is None

    def test_residual_key_the_method_honors_is_not_an_error(self):
        # Newton reads randomize_positions out of its own **kwargs to answer
        # with a dedicated unsupported-axis error; it must not be called unknown.
        assert unknown_kwargs_error("randomize", {"randomize_positions": True}, ("randomize_positions",)) is None

    @pytest.mark.parametrize(
        ("kwargs", "accepted", "text"),
        [
            (
                {"randomize_colours": True},
                ("randomize_colors", "seed"),
                "Unknown parameter 'randomize_colours' for action 'randomize'. "
                "Did you mean: randomize_colors? Valid: ['randomize_colors', 'seed']",
            ),
            (
                {"b": 1, "a": 2},
                ("seed", "color_range"),
                "Unknown parameters 'a', 'b' for action 'randomize'. Valid: ['color_range', 'seed']",
            ),
        ],
    )
    def test_message_is_the_dispatcher_sentence_with_its_suggestion(self, kwargs, accepted, text):
        # The same sentence the dispatcher gives a method without **kwargs, so a
        # misspelling gets its suggestion whichever kind of method it reached.
        result = unknown_kwargs_error("randomize", kwargs, accepted)

        assert result == {"status": "error", "content": [{"text": text}]}
