"""``set_obs_noise`` is one implementation, and it reports the same way everywhere.

MuJoCo, Isaac and Newton each carried their own copy of ``set_obs_noise`` and
its passes. The copies agreed on validation and drifted on the result: an
all-zero call cleared the config and said so on Isaac, but stored a zero config
and reported "Sensor noise: ..." on the other two, and only Isaac echoed the
config back in a ``json`` block. The pins below hold the backends to the one
shared mixin, then grade that mixin's contract once.
"""

from __future__ import annotations

import inspect
import threading
from typing import Any

import numpy as np
import pytest

from strands_robots.simulation.isaac.randomization import IsaacRandomizationMixin
from strands_robots.simulation.mujoco.randomization import RandomizationMixin
from strands_robots.simulation.newton.randomization import DomainRandomizationMixin
from strands_robots.simulation.obs_noise import OBS_NOISE_PARAMS, ObservationNoiseMixin

_BACKEND_MIXINS = {
    "mujoco": RandomizationMixin,
    "isaac": IsaacRandomizationMixin,
    "newton": DomainRandomizationMixin,
}
_NOISE_METHODS = ("set_obs_noise", "_apply_obs_noise", "_apply_state_noise", "_maybe_jitter_frame")


def _host(mixin: type[ObservationNoiseMixin] = ObservationNoiseMixin) -> Any:
    host = mixin.__new__(mixin)
    host._lock = threading.RLock()
    host._obs_noise = None
    host._obs_noise_rng = None
    return host


@pytest.mark.parametrize("backend", sorted(_BACKEND_MIXINS))
def test_every_backend_uses_the_shared_implementation(backend: str) -> None:
    mixin = _BACKEND_MIXINS[backend]
    own = [m for m in _NOISE_METHODS if getattr(mixin, m) is not getattr(ObservationNoiseMixin, m)]
    assert own == [], f"{backend} carries its own copy of {own}"


@pytest.mark.parametrize("backend", sorted(_BACKEND_MIXINS))
def test_an_identical_call_reports_identically_on_every_backend(backend: str) -> None:
    host = _host(_BACKEND_MIXINS[backend])
    on = host.set_obs_noise(joint_pos_std=0.02, joint_vel_std=0.1, camera_jitter_px=3, seed=0)
    assert on["status"] == "success"
    assert on["content"][1]["json"] == {
        "joint_pos_std": 0.02,
        "joint_vel_std": 0.1,
        "camera_jitter_px": 3.0,
        "seed": 0,
    }
    assert host._obs_noise == {"joint_pos_std": 0.02, "joint_vel_std": 0.1, "camera_jitter_px": 3.0}

    off = host.set_obs_noise()
    assert off["content"][0]["text"] == "Sensor noise cleared."
    assert (host._obs_noise, host._obs_noise_rng) == (None, None)


def test_the_accepted_names_are_the_signature() -> None:
    params = inspect.signature(ObservationNoiseMixin.set_obs_noise).parameters
    declared = {n for n, p in params.items() if n != "self" and p.kind is not inspect.Parameter.VAR_KEYWORD}
    assert set(OBS_NOISE_PARAMS) == declared


@pytest.mark.parametrize(
    ("kwargs", "named"),
    [
        ({"joint_pos_stdev": 0.05}, "joint_pos_stdev"),
        ({"joint_pos_std": -0.1}, "joint_pos_std"),
        ({"joint_vel_std": -1.0}, "joint_vel_std"),
        ({"camera_jitter_px": float("nan")}, "camera_jitter_px"),
        ({"joint_pos_std": float("inf")}, "joint_pos_std"),
        ({"joint_pos_std": "fast"}, "joint_pos_std"),
        ({"joint_pos_std": 0.1, "seed": -1}, "seed"),
    ],
)
def test_a_bad_value_is_refused_by_name_and_configures_nothing(kwargs: dict[str, Any], named: str) -> None:
    host = _host()
    result = host.set_obs_noise(**kwargs)
    assert result["status"] == "error"
    assert named in result["content"][0]["text"]
    assert host._obs_noise is None


def test_the_observation_pass_is_suffix_keyed() -> None:
    """Position std on the plain floats, velocity std on ``.vel``, lists untouched."""
    host = _host()
    host.set_obs_noise(joint_pos_std=0.0, joint_vel_std=0.2, seed=0)
    quat = [1.0, 0.0, 0.0, 0.0]
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    samples = [host._apply_obs_noise({"j": 1.0, "j.vel": 0.0, "cam": frame, "base_quat": quat}) for _ in range(5000)]
    assert {s["j"] for s in samples} == {1.0}
    assert abs(float(np.std([s["j.vel"] for s in samples])) - 0.2) < 0.02
    assert all(s["cam"] is frame and s["base_quat"] is quat for s in samples)

    unconfigured = {"j": 1.0}
    assert _host()._apply_obs_noise(unconfigured) is unconfigured


def test_the_state_pass_draws_each_field_from_its_own_std() -> None:
    host = _host()
    host.set_obs_noise(joint_pos_std=0.05, joint_vel_std=0.0, seed=1)
    states = [host._apply_state_noise({"j": {"position": 0.3, "velocity": 0.2}})["j"] for _ in range(5000)]
    assert abs(float(np.std([s["position"] for s in states])) - 0.05) < 0.005
    assert {s["velocity"] for s in states} == {0.2}

    state = {"j": {"position": 0.1, "velocity": 0.2}}
    assert _host()._apply_state_noise(state) is state


@pytest.mark.parametrize(("px", "moves"), [(0.0, False), (0.5, False), (3, True)])
def test_camera_jitter_rolls_whole_pixels_only(px: float, moves: bool) -> None:
    host = _host()
    host.set_obs_noise(camera_jitter_px=px, seed=2)
    frame = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    out = host._maybe_jitter_frame(frame)
    if not moves:
        assert out is frame
        return
    assert out.shape == frame.shape and not np.array_equal(out, frame)
    assert sorted(out.flatten().tolist()) == sorted(frame.flatten().tolist()), "a roll only relocates pixels"


def test_one_seed_is_one_noise_stream() -> None:
    a, b = _host(), _host()
    for host in (a, b):
        host.set_obs_noise(joint_pos_std=0.1, seed=7)
    assert [a._apply_obs_noise({"j": 0.0})["j"] for _ in range(50)] == [
        b._apply_obs_noise({"j": 0.0})["j"] for _ in range(50)
    ]
