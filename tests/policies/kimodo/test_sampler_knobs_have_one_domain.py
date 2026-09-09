"""One Kimodo sampler knob, one domain, whichever surface sets it.

``diffusion_steps`` and ``guidance_scale`` are validated as config fields, and
they are also documented per-call overrides of ``get_actions``. An override is
read straight from ``kwargs`` and reaches the sampler and the buffered-motion
key without passing through the frozen config, so the config's validation is
not on that path - and ``PolicyRunner.run`` / ``evaluate`` forward
``policy_kwargs`` verbatim to every ``get_actions`` call, so it is a routine
path rather than a private one. The sibling ``seed`` override on the next line
already consults its own domain owner (``test_sampling_seed_has_one_domain``);
these two used to be converted with ``int()`` / ``float()`` instead, which
cannot refuse a value, only reinterpret it.

Both spendings of these knobs are why the domain is not advisory. Each is
handed to the sampler - ``diffusion_steps=0`` asks for a sample that was never
denoised, ``True`` for a single iteration, and a ``guidance_scale`` of ``nan``
poisons every comparison the sampler makes with it. And each is part of the key
the buffered motion is identified by, so a value that differs from the one that
produced the motion in hand discards that motion and re-enters the sampler
mid-rollout - which is what a coerced value did silently, with a number the
caller never named.

The ACCEPTED rows are controls that hold on both sides of the fix: every
spelling of a usable knob the config field takes - including the integral float
and the NumPy scalar the shared domains admit, and the ceiling value itself -
still reaches the sampler as the ``int`` and ``float`` it is declared as.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from strands_robots.policies.kimodo import KIMODO_G1_JOINTS, KimodoConfig, KimodoPolicy

_NUM_JOINTS = len(KIMODO_G1_JOINTS)
_ROOT = 7

# The ceiling the config field applies to the step count, stated as the number
# the reference documents rather than imported from the module under test - a
# pin that reads the bound out of the code it grades cannot disagree with it.
_MAX_STEPS = 500

# One native frame per emitted frame, so a sampler run is countable directly.
_FAST = {"num_frames": 6, "native_fps": 30, "tracker_fps": 30}

#: The surface prefixes the two spellings report under.
_CONFIG = "KimodoConfig"
_PER_CALL = "KimodoPolicy.get_actions"


class _KnobRecordingAgent:
    """Record the knobs every sampler run was handed, and ramp per frame."""

    def __init__(self) -> None:
        self.knobs: list[tuple[object, object]] = []

    def sample(self, prompt, num_frames, diffusion_steps, guidance_scale, seed):
        self.knobs.append((diffusion_steps, guidance_scale))
        out = np.zeros((num_frames, _ROOT + _NUM_JOINTS), dtype=np.float32)
        out[:, 3] = 1.0  # identity quaternion, wxyz
        out[:, _ROOT:] = 0.01 * np.arange(num_frames, dtype=np.float32)[:, None]
        return out


def _policy(**cfg_kwargs):
    agent = _KnobRecordingAgent()
    return KimodoPolicy(config=KimodoConfig(**_FAST, **cfg_kwargs), motion_agent=agent), agent


def _pose(policy, **kwargs) -> tuple[float, ...]:
    action = asyncio.run(policy.get_actions({}, "walk forward", **kwargs))[0]
    return tuple(action[joint] for joint in KIMODO_G1_JOINTS)


# Rosters are lists of (value, id) pairs rather than dicts or sets: ``0 ==
# False`` and ``100 == 100.0``, so a mapping would collapse the exact
# distinctions a coercion defect is about.
#: Step counts no surface can honor. Each is refused as a config field today.
UNUSABLE_STEPS = [
    pytest.param(0, id="zero-a-sample-that-was-never-denoised"),
    pytest.param(-5, id="negative"),
    pytest.param(True, id="bool-an-int-subclass-that-would-denoise-once"),
    pytest.param(2.7, id="fractional-step-count"),
    pytest.param("100", id="numeric-string"),
    pytest.param(_MAX_STEPS + 500, id="above-the-ceiling-the-field-applies"),
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="inf"),
    pytest.param(None, id="None"),
    pytest.param([100], id="a-list"),
]

#: Guidance weights no surface can honor.
UNUSABLE_GUIDANCE = [
    pytest.param(0, id="zero-weight"),
    pytest.param(-1.0, id="negative"),
    pytest.param(True, id="bool-an-int-subclass-that-would-act-as-1"),
    pytest.param("7.5", id="numeric-string"),
    pytest.param(float("nan"), id="nan-poisons-every-comparison"),
    pytest.param(float("inf"), id="inf"),
    pytest.param(None, id="None"),
    pytest.param([7.5], id="a-list"),
]

#: Step counts that survive both uses. Controls: green before and after.
USABLE_STEPS = [
    pytest.param(25, 25, id="low-end-of-the-useful-range"),
    pytest.param(200, 200, id="high-end-of-the-useful-range"),
    pytest.param(_MAX_STEPS, _MAX_STEPS, id="the-ceiling-itself"),
    pytest.param(100.0, 100, id="integral-float-from-a-config-array"),
    pytest.param(np.int64(50), 50, id="numpy-integer"),
    pytest.param(np.float64(50.0), 50, id="numpy-integral-float"),
]

#: Guidance weights that survive both uses.
USABLE_GUIDANCE = [
    pytest.param(1.0, 1.0, id="no-guidance"),
    pytest.param(20.0, 20.0, id="strong-guidance"),
    pytest.param(3, 3.0, id="whole-number-widens-to-the-declared-float"),
    pytest.param(np.float32(7.5), 7.5, id="numpy-float"),
]


def _cases(*rosters):
    """Flatten per-knob rosters into ``(knob, *columns)`` cases, ids prefixed."""
    for knob, roster in rosters:
        for case in roster:
            yield pytest.param(knob, *case.values, id=f"{knob}-{case.id}")


#: Every (knob, value) pair no surface can honor.
UNUSABLE_CASES = list(_cases(("diffusion_steps", UNUSABLE_STEPS), ("guidance_scale", UNUSABLE_GUIDANCE)))

#: Every (knob, value, value-as-spent) triple both surfaces must keep taking.
USABLE_CASES = list(_cases(("diffusion_steps", USABLE_STEPS), ("guidance_scale", USABLE_GUIDANCE)))


class TestBothSurfacesThatSetASamplerKnobShareOneDomain:
    """A knob gets one verdict whether it is a config field or a per-call override."""

    @pytest.mark.parametrize(("knob", "value"), UNUSABLE_CASES)
    def test_the_config_field_refuses_it(self, knob, value):
        with pytest.raises(ValueError, match=f"{knob} must be"):
            KimodoConfig(**{knob: value})

    @pytest.mark.parametrize(("knob", "value"), UNUSABLE_CASES)
    def test_a_per_call_override_refuses_it(self, knob, value):
        """The path a coercion used to take: read from kwargs, spent unchecked."""
        policy, _ = _policy()
        with pytest.raises(ValueError, match=f"{knob} must be"):
            _pose(policy, **{knob: value})

    @pytest.mark.parametrize(("knob", "value"), UNUSABLE_CASES)
    def test_both_surfaces_state_the_same_reason_and_name_themselves(self, knob, value):
        """One value, one verdict - and each message says which surface refused."""
        with pytest.raises(ValueError) as from_field:
            KimodoConfig(**{knob: value})
        policy, _ = _policy()
        with pytest.raises(ValueError) as from_override:
            _pose(policy, **{knob: value})

        field_text, override_text = str(from_field.value), str(from_override.value)
        assert field_text.startswith(f"{_CONFIG}: ")
        assert override_text.startswith(f"{_PER_CALL}: ")
        assert field_text[len(_CONFIG) :] == override_text[len(_PER_CALL) :], (
            "the two surfaces must state one domain, not two that agree today"
        )


class TestARefusedOverrideSpendsNothing:
    """A refusal is not a partial application: the held motion survives it."""

    @pytest.mark.parametrize(("knob", "value"), UNUSABLE_CASES)
    def test_it_leaves_the_held_motion_the_cursor_and_the_sampler_alone(self, knob, value):
        """What a coerced knob cost: a re-sample, and the motion in hand.

        Pre-fix the admitted values here did not raise at all - they keyed a
        different buffer, so the sampler ran again with a number the caller
        never named and the motion being tracked was replaced mid-rollout.
        """
        policy, agent = _policy()
        first = _pose(policy)
        second = _pose(policy)
        assert first != second, "the stub motion must advance, or a rewind is unobservable"

        with pytest.raises(ValueError, match=f"{knob} must be"):
            _pose(policy, **{knob: value})

        assert len(agent.knobs) == 1, "a refused override must not run the sampler"
        assert _pose(policy) != second, "the cursor must not have been rewound"

    def test_the_ceiling_the_field_applies_is_not_escapable_per_call(self):
        """The composite half of the domain: the shared rule plus this ceiling."""
        policy, agent = _policy()
        _pose(policy)
        with pytest.raises(ValueError, match=f"diffusion_steps must be <= {_MAX_STEPS}"):
            _pose(policy, diffusion_steps=_MAX_STEPS + 1)
        assert len(agent.knobs) == 1, "a refused step count must not start a diffusion run"


class TestAKnobTheDomainAcceptsSurvivesBothOfItsUses:
    """The controls: the gate narrows nothing a caller could already ask for."""

    #: What the sampler is declared to take for each knob, and the default the
    #: other knob keeps while one is overridden.
    _DECLARED = {"diffusion_steps": (int, 0, 7.5), "guidance_scale": (float, 1, 100)}

    @pytest.mark.parametrize(("knob", "value", "reaches"), USABLE_CASES)
    def test_an_accepted_override_reaches_the_sampler_as_the_declared_type(self, knob, value, reaches):
        """The gate narrows nothing, and the conversion after it still normalises.

        A NumPy scalar and an integral float are both inside the shared domains,
        so they still pass - and the sampler is still handed the ``int`` and
        ``float`` its signature declares rather than the caller's spelling.
        """
        declared, position, other = self._DECLARED[knob]
        policy, agent = _policy()
        _pose(policy, **{knob: value})

        spent = agent.knobs[0][position]
        assert spent == reaches
        assert isinstance(spent, declared)
        assert agent.knobs[0][1 - position] == other, "the knob not overridden keeps its config value"

    @pytest.mark.parametrize(("knob", "value", "reaches"), USABLE_CASES)
    def test_the_config_field_takes_it_too(self, knob, value, reaches):
        """Neither surface may accept a spelling the other refuses."""
        assert getattr(KimodoConfig(**{knob: value}), knob) == reaches

    def test_a_changed_accepted_override_still_re_samples(self):
        """Re-sampling on a changed knob is what the override is for."""
        policy, agent = _policy()
        _pose(policy)
        _pose(policy, diffusion_steps=25)
        assert agent.knobs == [(100, 7.5), (25, 7.5)]

    def test_repeating_an_accepted_override_still_drains_the_buffer(self):
        """The knob is part of the key, so an unchanged one must not re-sample."""
        policy, agent = _policy()
        _pose(policy, diffusion_steps=25, guidance_scale=3.0)
        _pose(policy, diffusion_steps=25, guidance_scale=3.0)
        assert agent.knobs == [(25, 3.0)]

    def test_an_integral_float_names_the_same_motion_as_the_whole_number(self):
        """``100`` and ``100.0`` are one step count, so they share one key."""
        policy, agent = _policy()
        _pose(policy, diffusion_steps=100)
        _pose(policy, diffusion_steps=100.0)
        assert agent.knobs == [(100, 7.5)], "the same count spelled two ways must not re-run the sampler"
