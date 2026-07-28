"""A MuJoCo-only method must name the backend, not raise a bare AttributeError.

The Newton backend implements every abstract ``SimEngine`` method, but the MuJoCo
engine carries ~47 public methods beyond the ABC: teleop, multi-policy, camera
recording, state checkpointing, analytic dynamics queries (jacobian/mass matrix/
inverse dynamics), and MJCF scene surgery. None of them are declared on the ABC,
so Newton is not violating a contract by lacking them - but a caller moving
working code from ``backend="mujoco"`` to ``backend="newton"`` got

    AttributeError: 'NewtonSimEngine' object has no attribute 'set_joint_positions'

which names neither the backend that lacks the method nor the one that has it.

``NewtonSimEngine.__getattr__`` now raises ``NotImplementedError`` with that
information for exactly the MuJoCo-only set, while leaving ordinary typos as
``AttributeError`` so ``hasattr`` probes and Python semantics still work.

Gated on Newton + Warp; the cross-backend derivation additionally needs mujoco.
"""

from __future__ import annotations

import copy
import importlib.util

import pytest

_HAS_NEWTON = importlib.util.find_spec("newton") is not None and importlib.util.find_spec("warp") is not None

pytestmark = pytest.mark.skipif(not _HAS_NEWTON, reason="newton/warp not installed")

# A representative method from each MuJoCo-only subsystem.
_MUJOCO_ONLY = [
    "set_joint_positions",  # state write
    "save_state",  # checkpointing
    "load_state",
    "teleoperate",  # teleop
    "attach_teleop",
    "run_multi_policy",  # multi-policy
    "stop_policy",
    "start_cameras_recording",  # camera recording
    "render_depth",  # rendering
    "stream",
    "get_jacobian",  # analytic dynamics
    "get_mass_matrix",
    "inverse_dynamics",
    "apply_force",
    "raycast",  # scene queries
    "export_xml",  # MJCF surgery
    "replace_scene_mjcf",
]


def _make_engine():
    from strands_robots.simulation.newton.simulation import NewtonSimEngine

    return NewtonSimEngine(solver="mujoco")


@pytest.fixture(scope="module")
def engine():
    sim = _make_engine()
    sim.create_world()
    sim.add_robot("so101")
    yield sim
    sim.destroy()


class TestMuJoCoOnlyMethodsExplainThemselves:
    @pytest.mark.parametrize("name", _MUJOCO_ONLY)
    def test_raises_not_implemented_naming_both_backends(self, engine, name):
        pytest.importorskip("mujoco")

        with pytest.raises(NotImplementedError) as excinfo:
            getattr(engine, name)

        message = str(excinfo.value)
        assert name in message
        assert "MuJoCo" in message
        assert "Newton" in message
        # Actionable: it must say what to switch to.
        assert "backend='mujoco'" in message or "create_simulation('mujoco')" in message

    def test_message_is_plain_ascii(self, engine):
        """AGENTS.md: user-facing strings are plain ASCII only."""
        pytest.importorskip("mujoco")

        with pytest.raises(NotImplementedError) as excinfo:
            engine.get_jacobian  # noqa: B018 - attribute access is the trigger

        assert str(excinfo.value).isascii()


class TestNormalSemanticsPreserved:
    def test_a_typo_is_still_an_attribute_error(self, engine):
        """Only the MuJoCo-only set is special-cased; everything else is a typo."""
        with pytest.raises(AttributeError):
            engine.totally_bogus_method_xyz  # noqa: B018

    def test_hasattr_still_returns_false_for_a_typo(self, engine):
        assert not hasattr(engine, "totally_bogus_method_xyz")

    def test_dunder_lookups_are_not_rerouted(self, engine):
        """copy/pickle/inspect probe dunders; they must raise AttributeError."""
        with pytest.raises(AttributeError):
            engine.__some_missing_dunder__  # noqa: B018

    def test_engine_is_still_copyable(self, engine):
        """A __getattr__ that hijacks __deepcopy__/__getstate__ breaks copy."""
        assert copy.copy(engine) is not None

    def test_implemented_methods_are_unaffected(self, engine):
        """The hook must fire only for ABSENT attributes."""
        assert engine.step(n_steps=1)["status"] == "success"
        assert engine.get_observation(skip_images=True)
        assert engine.list_cameras() == ["default"]


class TestTheSetIsDerivedNotHardcoded:
    def test_derivation_matches_a_live_class_comparison(self):
        """A hardcoded list would go stale the first time either backend changes."""
        pytest.importorskip("mujoco")
        from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine
        from strands_robots.simulation.newton.simulation import (
            NewtonSimEngine,
            _mujoco_only_methods,
        )

        def public(cls):
            return {n for n in dir(cls) if not n.startswith("_") and callable(getattr(cls, n, None))}

        assert _mujoco_only_methods() == public(MuJoCoSimEngine) - public(NewtonSimEngine)

    def test_every_probed_name_is_actually_mujoco_only(self):
        """Guard the test's own fixture list against drift."""
        pytest.importorskip("mujoco")
        from strands_robots.simulation.newton.simulation import _mujoco_only_methods

        derived = _mujoco_only_methods()
        assert set(_MUJOCO_ONLY) <= derived, set(_MUJOCO_ONLY) - derived

    def test_abstract_contract_methods_are_never_in_the_set(self):
        """Newton implements the whole ABC; none of it may be reported missing."""
        pytest.importorskip("mujoco")
        from strands_robots.simulation.newton.simulation import _mujoco_only_methods

        contract = {
            "create_world",
            "add_robot",
            "add_object",
            "send_action",
            "step",
            "get_observation",
            "get_state",
            "list_robots",
            "remove_object",
            "remove_robot",
            "render",
            "reset",
            "robot_joint_names",
            "destroy",
        }
        assert not (contract & _mujoco_only_methods())
