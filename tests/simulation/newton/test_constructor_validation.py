# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""``NewtonSimEngine`` must reject an unusable ``substeps`` at construction.

``self.substeps = substeps`` was stored unvalidated, nine lines above the loop
that DOES validate ``nconmax``/``njmax`` in the same constructor. ``_advance``
then computes ``dt = timestep / substeps`` and loops ``range(substeps)``, so every
bad value failed late and badly - and one of them did not fail at all. Measured::

    substeps=0     ctor / create_world / add_robot all OK, then an uncaught
                   ZeroDivisionError inside send_action
    substeps=-3    send_action -> status="success", sim_time advances 0.3333s,
                   and range(-3) runs ZERO solver steps: the joint stays at
                   0.0000 and nothing reports it
    substeps=1.5   uncaught TypeError deep inside range()
    substeps=True  silently acts as 1 (bool is an int subclass)

The ``substeps=-3`` case is the serious one: a rollout that reports success while
integrating no physics at all produces a dataset of frozen states with a
plausible-looking timeline.

The repo already owns this contract for the same quantity on the rollout side
(``SimEngine._validate_control_substeps`` / ``PolicyRunner._control_substeps``),
whose commit message is "control_substeps is honored or rejected, never silently
clamped". ``default_width`` / ``default_height`` are validated in the same pass.
"""

from __future__ import annotations

import importlib.util

import pytest

_HAS_NEWTON = importlib.util.find_spec("newton") is not None and importlib.util.find_spec("warp") is not None

pytestmark = pytest.mark.skipif(not _HAS_NEWTON, reason="newton/warp not installed")


def _engine_cls():
    from strands_robots.simulation.newton.simulation import NewtonSimEngine

    return NewtonSimEngine


class TestSubstepsIsValidated:
    @pytest.mark.parametrize("bad", [0, -1, -3, 1.5, True, False, None, "4", [4]])
    def test_an_unusable_substeps_is_rejected_at_construction(self, bad):
        """Regression: 0 died in send_action, -3 ran zero physics silently."""
        with pytest.raises(ValueError, match="substeps"):
            _engine_cls()(substeps=bad)

    def test_the_message_names_the_parameter_and_the_value(self):
        with pytest.raises(ValueError) as excinfo:
            _engine_cls()(substeps=-3)

        message = str(excinfo.value)
        assert "substeps" in message, message
        assert "-3" in message, message
        assert "positive integer" in message, message
        assert message.isascii()

    @pytest.mark.parametrize("good", [1, 2, 4, 10, 100])
    def test_a_positive_integer_still_constructs(self, good):
        engine = _engine_cls()(substeps=good)

        assert engine.substeps == good

    def test_the_default_is_unchanged(self):
        engine = _engine_cls()()

        assert engine.substeps == 10

    def test_bool_is_rejected_rather_than_acting_as_one(self):
        """``True`` is an int subclass, so it silently meant substeps=1."""
        with pytest.raises(ValueError, match="substeps"):
            _engine_cls()(substeps=True)


class TestRenderSizesAreValidated:
    @pytest.mark.parametrize("field", ["default_width", "default_height"])
    @pytest.mark.parametrize("bad", [0, -16, 1.5, True, None])
    def test_an_unusable_render_size_is_rejected(self, field, bad):
        with pytest.raises(ValueError, match=field):
            _engine_cls()(**{field: bad})

    def test_valid_render_sizes_still_construct(self):
        engine = _engine_cls()(default_width=320, default_height=240)

        assert engine.default_width == 320
        assert engine.default_height == 240


class TestTheExistingContactValidationStillHolds:
    """The loop this one was added beside must be unchanged."""

    @pytest.mark.parametrize("field", ["nconmax", "njmax"])
    @pytest.mark.parametrize("bad", [0, -5, 1.5, True])
    def test_a_bad_contact_limit_is_rejected(self, field, bad):
        with pytest.raises(ValueError, match=field):
            _engine_cls()(**{field: bad})

    @pytest.mark.parametrize("field", ["nconmax", "njmax"])
    def test_none_is_still_accepted_for_a_contact_limit(self, field):
        """Unlike substeps, these have a legitimate ``None`` (auto-derive) form."""
        engine = _engine_cls()(**{field: None})

        assert engine is not None


class TestThePublicFactoryRejectsItToo:
    """No in-tree caller passes ``substeps``, so the realistic source of a bad
    value is an agent-supplied kwarg arriving through the public factory."""

    def test_create_simulation_rejects_a_bad_substeps(self):
        from strands_robots.simulation import create_simulation

        with pytest.raises(ValueError, match="substeps"):
            create_simulation("newton", substeps=-3)

    def test_create_simulation_passes_a_good_substeps_through(self):
        from strands_robots.simulation import create_simulation

        engine = create_simulation("newton", substeps=4)

        assert engine.substeps == 4


class TestAValidEngineStillSteps:
    """The validation must not have broken the working path."""

    def test_a_two_substep_engine_actually_integrates(self):
        engine = _engine_cls()(substeps=2)
        try:
            engine.create_world()
            assert engine.add_robot("so101")["status"] == "success"
            joints = engine.robot_joint_names("so101")
            before = float(engine.get_observation("so101")[joints[1]])

            assert engine.send_action({joints[1]: 0.8}, robot_name="so101", n_substeps=20)["status"] == "success"

            after = float(engine.get_observation("so101")[joints[1]])
            assert after != pytest.approx(before, abs=1e-3), (
                f"joint did not move ({before:.4f} -> {after:.4f}) - zero physics ran"
            )
        finally:
            engine.destroy()
