"""Value-domain contracts for :class:`WBCConfig`.

:meth:`WBCConfig.__post_init__` has always rejected impossible DIMENSIONS - a
sub-minimal count, a per-joint vector whose length contradicts ``num_actions``.
It did not reject impossible VALUES, and the two failures are not the same
size. Every numeric field of this config is read verbatim into either the PD law
that writes ``data.ctrl`` or the observation the network sees:

* ``target_q = default_angles + action_scale * raw_action`` (``compute_targets``)
* ``tau = (target_q - q) * kps + (0 - dq) * kds`` (``pd_control``)

so an ``action_scale`` of ``0`` (or ``False``) makes ``target_q ==
default_angles`` on every tick - the network's decision is discarded and the
humanoid holds its nominal stance - and a ``nan`` anywhere makes ``tau`` ``nan``
on all 15 driven joints. Neither is checked downstream: ``compute_targets`` runs
per-tick inside ``get_actions``, so a non-real value surfaced as a bare
``TypeError`` from its ``float()`` only after the ONNX sessions had loaded and
the rollout had started, which is precisely the mid-rollout failure this module's
docstring says it exists to convert into a construction-time message.

These tests pin the value domain through the public surfaces a caller reaches -
the constructor, ``from_dict`` and ``from_file`` (the config-FILE path a
checkpoint's ``config.json`` and the upstream ``g1_gear_wbc.yaml`` both take) -
and pin the values that remain first-class, so the guard cannot creep into
refusing a controller a caller may legitimately ask for.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from strands_robots.policies.wbc import WBCConfig
from strands_robots.policies.wbc.config import _non_negative_number_error
from strands_robots.policies.wbc.control import compute_targets, pd_control
from strands_robots.utils import finite_number_error

N = 15
_GOOD_ANGLES = [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0, 0.0, 0.0, 0.0]
_GOOD_KPS = [150.0, 150.0, 150.0, 200.0, 40.0, 40.0, 150.0, 150.0, 150.0, 200.0, 40.0, 40.0, 250.0, 250.0, 250.0]
_GOOD_KDS = [2.0, 2.0, 2.0, 4.0, 2.0, 2.0, 2.0, 2.0, 2.0, 4.0, 2.0, 2.0, 5.0, 5.0, 5.0]

# Values no numeric field of this config can be honored as, whatever its sign
# rule: a non-real one raises from the ``float()`` that consumes it, and a
# non-finite one poisons whatever it multiplies.
UNUSABLE_ANY_SIGN: list[Any] = [float("nan"), float("inf"), float("-inf"), True, False, "0.5", None, [0.5], 10**400]


def _config(**kwargs: Any) -> WBCConfig:
    """Build a config through one funnel.

    These tests deliberately supply values outside the declared field types (a
    string where a ``float`` is annotated, a list where a scalar is), which is
    the point - the runtime is what must refuse them. Splatting through one
    ``**kwargs: Any`` funnel states that intent once instead of scattering a
    suppression over every call.
    """
    base: dict[str, Any] = {
        "policy_path": "p.onnx",
        "num_actions": N,
        "default_angles": list(_GOOD_ANGLES),
        "kps": list(_GOOD_KPS),
        "kds": list(_GOOD_KDS),
    }
    return WBCConfig(**{**base, **kwargs})


class TestActionScaleDomain:
    """``action_scale`` is the only path from the network to the joint targets."""

    @pytest.mark.parametrize("value", [0, 0.0, -0.25, *UNUSABLE_ANY_SIGN])
    def test_an_unusable_action_scale_is_refused_at_construction(self, value: Any) -> None:
        with pytest.raises(ValueError, match=r"action_scale"):
            _config(action_scale=value)

    @pytest.mark.parametrize("value", [0.25, 1.0, 0.05, np.float32(0.25), np.float64(0.5), 1])
    def test_a_usable_action_scale_still_builds(self, value: Any) -> None:
        assert float(_config(action_scale=value).action_scale) == pytest.approx(float(value))

    def test_the_refusal_names_the_field_and_the_value(self) -> None:
        with pytest.raises(ValueError, match=r"WBCConfig: action_scale must be > 0, got 0\.0\."):
            _config(action_scale=0.0)


class TestWhyTheActionScaleDomainIsWhatItIs:
    """The refused values are refused because the PD chain cannot honor them.

    Derived from the two functions the config feeds rather than asserted from a
    message, so the domain stays tied to the consequence that motivates it.
    """

    @staticmethod
    def _torque(action_scale: Any) -> np.ndarray:
        raw = np.full(N, 0.5)  # a network asking every joint 0.5 rad off stance
        q = np.asarray(_GOOD_ANGLES)  # measured == stance
        target_q = compute_targets(np.asarray(_GOOD_ANGLES), raw, action_scale)
        return pd_control(target_q, q, np.asarray(_GOOD_KPS), np.zeros(N), np.zeros(N), np.asarray(_GOOD_KDS))

    def test_a_usable_scale_carries_the_network_decision_into_torque(self) -> None:
        assert np.all(np.abs(self._torque(0.25)) > 1.0)

    def test_a_zero_scale_would_have_produced_exactly_zero_torque(self) -> None:
        # The whole network output discarded: target_q == default_angles, so the
        # PD law has no error to act on. docs/policies/wbc.md states WBC "never
        # falls back to silent zero torques" - this is that fallback.
        assert np.array_equal(self._torque(0.0), np.zeros(N))

    def test_a_negative_scale_would_have_inverted_every_torque(self) -> None:
        assert np.all(np.sign(self._torque(-0.25)) == -np.sign(self._torque(0.25)))

    def test_a_non_finite_scale_would_have_poisoned_every_driven_joint(self) -> None:
        assert np.all(np.isnan(self._torque(float("nan"))))

    def test_a_non_real_scale_would_have_raised_from_the_per_tick_float(self) -> None:
        # compute_targets is called from get_actions, i.e. after the ONNX
        # sessions have loaded and the rollout is already running.
        with pytest.raises(TypeError):
            self._torque(None)


class TestGainDomain:
    """A PD gain may be zero; it may not be negative or non-finite."""

    @pytest.mark.parametrize("name", ["kps", "kds"])
    @pytest.mark.parametrize("value", UNUSABLE_ANY_SIGN)
    def test_an_unusable_gain_component_is_refused(self, name: str, value: Any) -> None:
        with pytest.raises(ValueError, match=rf"{name}\[3\]"):
            _config(**{name: [*(_GOOD_KPS[:3]), value, *(_GOOD_KPS[4:])]})

    @pytest.mark.parametrize("name", ["kps", "kds"])
    def test_a_negative_gain_is_refused_because_it_inverts_the_feedback(self, name: str) -> None:
        with pytest.raises(ValueError, match=rf"WBCConfig: {name}\[0\] must be >= 0, got -150\.0\."):
            _config(**{name: [-150.0, *(_GOOD_KPS[1:])]})

    @pytest.mark.parametrize("name", ["kps", "kds"])
    def test_a_zero_gain_stays_first_class(self, name: str) -> None:
        # kp=0 with kd>0 is a pure-damping joint - a controller a caller may
        # legitimately ask for, so the floor is >= 0 and not > 0.
        assert getattr(_config(**{name: [0.0] * N}), name) == [0.0] * N

    def test_the_upstream_g1_sonic_gains_still_build(self) -> None:
        config = _config()
        assert config.kps == _GOOD_KPS
        assert config.kds == _GOOD_KDS


class TestObservationAndCommandScaleDomain:
    """The fields that scale what the network sees or is commanded to do."""

    @pytest.mark.parametrize("name", ["height_cmd", "freq_cmd"])
    @pytest.mark.parametrize("value", UNUSABLE_ANY_SIGN)
    def test_an_unusable_command_scalar_is_refused(self, name: str, value: Any) -> None:
        with pytest.raises(ValueError, match=rf"WBCConfig: {name} must be"):
            _config(**{name: value})

    @pytest.mark.parametrize("name", ["cmd_scale", "rpy_cmd"])
    @pytest.mark.parametrize("value", UNUSABLE_ANY_SIGN)
    def test_an_unusable_command_vector_component_is_refused(self, name: str, value: Any) -> None:
        with pytest.raises(ValueError, match=rf"{name}\[1\]"):
            _config(**{name: [0.5, value, 0.5]})

    @pytest.mark.parametrize("value", UNUSABLE_ANY_SIGN)
    def test_an_unusable_observation_scale_is_refused_naming_its_key(self, value: Any) -> None:
        with pytest.raises(ValueError, match=r"obs_scales\['dof_vel'\]"):
            _config(obs_scales={"ang_vel": 0.5, "dof_pos": 1.0, "dof_vel": value})

    def test_obs_scales_must_be_a_mapping(self) -> None:
        with pytest.raises(ValueError, match=r"obs_scales must be a mapping"):
            _config(obs_scales="ang_vel")

    def test_a_signed_command_stays_first_class(self) -> None:
        # rpy_cmd carries roll/pitch/yaw and default_angles a stance: negative
        # entries are ordinary, so these fields constrain finiteness only.
        config = _config(rpy_cmd=[-0.2, 0.0, 0.1], cmd_scale=[-2.0, 2.0, 0.5])
        assert config.rpy_cmd == [-0.2, 0.0, 0.1]
        assert config.cmd_scale == [-2.0, 2.0, 0.5]

    def test_the_upstream_defaults_still_build(self) -> None:
        config = WBCConfig(policy_path="p.onnx")
        assert config.obs_scales == {"ang_vel": 0.5, "dof_pos": 1.0, "dof_vel": 0.05}
        assert config.cmd_scale == [2.0, 2.0, 0.5]
        assert (config.height_cmd, config.freq_cmd, config.action_scale) == (0.74, 0.75, 0.25)


class TestTheDimensionChecksStayTotal:
    """A field that carries no readable length gets a reason, not a bare error."""

    @pytest.mark.parametrize("name", ["default_angles", "kps", "kds"])
    @pytest.mark.parametrize("value", [5.0, np.float64(5.0), np.array(5.0)])
    def test_a_scalar_per_joint_field_reports_the_field(self, name: str, value: Any) -> None:
        # ``len()`` raises for a plain float and for a 0-d array (whose __len__
        # exists and refuses), naming neither the field nor the class.
        with pytest.raises(ValueError, match=rf"WBCConfig\.{name} must be a sequence of 15 numbers"):
            _config(**{name: value})

    def test_a_scalar_cmd_scale_reports_the_field(self) -> None:
        with pytest.raises(ValueError, match=r"WBCConfig\.cmd_scale must be a sequence of 3 numbers"):
            _config(cmd_scale=2.0)

    @pytest.mark.parametrize("name", ["default_angles", "kps", "kds"])
    def test_a_numpy_per_joint_vector_of_the_right_width_is_accepted(self, name: str) -> None:
        # Previously ``if vec and ...`` raised the ambiguous-truth ValueError for
        # any multi-element array, so a NumPy vector could not be supplied at all.
        assert len(getattr(_config(**{name: np.full(N, 1.0)}), name)) == N

    def test_the_length_mismatch_message_is_unchanged(self) -> None:
        with pytest.raises(ValueError, match=r"kps has length 3 but num_actions=15"):
            _config(kps=[1.0, 2.0, 3.0])

    def test_a_dimension_mistake_is_reported_before_a_value_mistake(self) -> None:
        # A config paired with the wrong checkpoint is the likelier root cause,
        # so its message stays the one such a pair reports.
        with pytest.raises(ValueError, match=r"has length 2 but num_actions=15"):
            _config(kps=[1.0, float("nan")])


class TestTheConfigFilePathIsCovered:
    """``from_dict`` / ``from_file`` are how a checkpoint's config arrives."""

    def test_from_dict_refuses_an_unusable_action_scale(self) -> None:
        with pytest.raises(ValueError, match=r"action_scale must be > 0"):
            WBCConfig.from_dict({"policy_path": "p.onnx", "action_scale": 0})

    def test_from_dict_refuses_an_unusable_flat_upstream_scale_key(self) -> None:
        # The upstream YAML spells the observation scales flat; the normalised
        # value must face the same domain as an explicit obs_scales map.
        with pytest.raises(ValueError, match=r"obs_scales\['ang_vel'\]"):
            WBCConfig.from_dict({"policy_path": "p.onnx", "ang_vel_scale": float("inf")})

    def test_from_file_refuses_an_unusable_value(self, tmp_path: Path) -> None:
        path = tmp_path / "wbc.json"
        path.write_text(json.dumps({"policy_path": "p.onnx", "kps": [150.0] * 14 + [-1.0]}))
        with pytest.raises(ValueError, match=r"kps\[14\] must be >= 0"):
            WBCConfig.from_file(path)

    def test_from_file_still_loads_a_usable_config(self, tmp_path: Path) -> None:
        path = tmp_path / "wbc.json"
        path.write_text(json.dumps({"policy_path": "p.onnx", "action_scale": 0.25, "kps": _GOOD_KPS}))
        assert WBCConfig.from_file(path).kps == _GOOD_KPS


class TestTheGainFloorIsTheOnlyLocalRule:
    """The local gain guard adds a floor to the shared numeric rule, nothing else."""

    @pytest.mark.parametrize("value", [*UNUSABLE_ANY_SIGN, 0.0, 0, 1.5, np.float32(2.0), -1.5])
    def test_it_agrees_with_the_shared_guard_except_on_the_floor(self, value: Any) -> None:
        shared = finite_number_error(value, "kps[0]", "WBCConfig")
        local = _non_negative_number_error(value, "kps[0]", "WBCConfig")
        if shared is not None:
            assert local == shared, "a value the shared rule rejects must be reported in its words"
        else:
            below_floor = float(value) < 0.0
            assert (local is not None) is below_floor
            if below_floor:
                assert "must be >= 0" in local  # type: ignore[operator]

    def test_it_accepts_the_zero_the_positive_guard_would_refuse(self) -> None:
        assert _non_negative_number_error(0.0, "kds[0]", "WBCConfig") is None

    def test_a_non_finite_value_is_reported_as_non_finite_not_as_below_the_floor(self) -> None:
        message = _non_negative_number_error(float("nan"), "kps[0]", "WBCConfig")
        assert message is not None and "must be a finite number" in message
        assert not math.isfinite(float("nan"))
