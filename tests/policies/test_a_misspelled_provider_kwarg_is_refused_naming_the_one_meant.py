"""One rule for every provider: a keyword that misspells a constructor parameter is refused.

Pre-fix each provider had its own behaviour. A constructor with ``**kwargs``
dropped ``create_policy("groot", hots="x")`` silently, so the client dialled
the default host under ``status="success"``; ``remote`` and ``lerobot_async``
logged "ignoring unexpected constructor kwarg(s)" where no agent reads it; a
constructor without a sink raised CPython's ``__init__() got an unexpected
keyword argument 'acton_space'``, naming neither the provider nor the
parameter meant. ``create_policy`` now screens the resolved kwargs against the
constructor's own signature before constructing anything, and judges a
misspelling without reference to the name's length: ``difflib``'s ratio alone
scores one substituted character 0.75 in a four-letter name and 0.90 in a
ten-letter one, so a cutoff on it screens ``pretrained_name_or_path`` and
leaves ``host`` and ``port`` - the parameters the most providers declare - open.
"""

from __future__ import annotations

import difflib
import importlib.util

import pytest

from strands_robots.policies import Policy, create_policy, register_policy
from strands_robots.policies.factory import (
    _MISSPELLING_RATIO,
    _constructor_keywords,
    _misspelling_of,
    _resolve_policy_class,
    policy_kwargs_error,
)
from strands_robots.registry import list_policy_providers

#: ``zmq`` is what resolving the ``groot`` provider imports. Read as a spec
#: rather than imported, so the gate costs nothing and mutes nothing.
_HAS_ZMQ = importlib.util.find_spec("zmq") is not None


class _Tolerant(Policy):
    """A provider whose constructor has a ``**kwargs`` pass-through."""

    def __init__(self, host: str = "localhost", port: int = 5555, action_space: str = "joint", **kwargs):
        self.host, self.port, self.action_space, self.extra = host, port, action_space, kwargs

    async def get_actions(self, observation_dict, instruction, **kwargs):
        return []

    def set_robot_state_keys(self, robot_state_keys):
        pass

    @property
    def provider_name(self):
        return "tolerant_test"


class _Strict(Policy):
    """A provider whose constructor binds exactly what it names."""

    def __init__(self, host: str = "localhost", port: int = 5555):
        self.host, self.port = host, port

    async def get_actions(self, observation_dict, instruction, **kwargs):
        return []

    def set_robot_state_keys(self, robot_state_keys):
        pass

    @property
    def provider_name(self):
        return "strict_test"


@pytest.fixture(scope="module", autouse=True)
def _providers():
    register_policy("tolerant_test", lambda: _Tolerant)
    register_policy("strict_test", lambda: _Strict)


class TestAConstructorWithAPassThrough:
    def test_a_near_miss_of_a_declared_parameter_is_refused_naming_it(self):
        with pytest.raises(TypeError) as info:
            create_policy("tolerant_test", acton_space="ee")
        text = str(info.value)
        assert text.startswith(
            "_Tolerant (policy provider 'tolerant_test') does not accept 'acton_space' (did you mean 'action_space'?)."
        )
        assert "refused rather than dropped" in text
        assert "It accepts: host, port, action_space." in text

    def test_a_transposition_too_short_for_the_ratio_is_still_a_misspelling(self):
        # ``hots`` vs ``host`` scores 0.75 - under the 0.8 cutoff - and is the
        # typo a hand makes most; it is the PC2-009 reproduction verbatim.
        with pytest.raises(TypeError) as info:
            create_policy("tolerant_test", hots="10.0.0.2")
        assert "'hots' (did you mean 'host'?)" in str(info.value)
        with pytest.raises(TypeError) as info:
            create_policy("tolerant_test", prot=6000)
        assert "'prot' (did you mean 'port'?)" in str(info.value)

    def test_every_misspelling_is_named_in_one_pass(self):
        with pytest.raises(TypeError) as info:
            create_policy("tolerant_test", hots="x", prot=1)
        text = str(info.value)
        assert "'hots' (did you mean 'host'?)" in text
        assert "'prot' (did you mean 'port'?)" in text

    def test_an_unrelated_name_still_passes_through(self):
        policy = create_policy("tolerant_test", torch_dtype="bfloat16", num_envs=4)
        assert policy.extra == {"torch_dtype": "bfloat16", "num_envs": 4}

    def test_declared_parameters_construct_as_before(self):
        policy = create_policy("tolerant_test", host="10.0.0.2", port=6000)
        assert (policy.host, policy.port, policy.extra) == ("10.0.0.2", 6000, {})


class TestAConstructorWithoutAPassThrough:
    def test_a_near_miss_gets_the_same_report_as_a_tolerant_one(self):
        with pytest.raises(TypeError) as info:
            create_policy("strict_test", hots="x")
        assert "_Strict (policy provider 'strict_test') does not accept 'hots' (did you mean 'host'?)." in str(
            info.value
        )

    def test_an_unknown_name_is_refused_before_cpython_would_and_lists_what_is_accepted(self):
        with pytest.raises(TypeError) as info:
            create_policy("strict_test", totally_unknown=1)
        text = str(info.value)
        assert "does not accept 'totally_unknown': its constructor declares no **kwargs" in text
        assert "It accepts: host, port." in text
        assert "unexpected keyword argument" not in text


class TestATypoIsJudgedWithoutRegardToTheNameSLength:
    """``host`` is four characters, and one of them being wrong is one typo."""

    @pytest.mark.parametrize(
        ("spelling", "ratio", "form"),
        [
            ("hoat", 0.750, "one wrong character - under the cutoff, so the ratio alone drops it"),
            ("hots", 0.750, "two adjacent characters swapped - the same arithmetic"),
            ("hos", 0.857, "one dropped character - the ratio already sees this"),
            ("hosst", 0.889, "one doubled character - the ratio already sees this"),
        ],
    )
    def test_every_typo_of_a_four_letter_parameter_is_refused(self, spelling: str, ratio: float, form: str) -> None:
        assert round(difflib.SequenceMatcher(None, spelling, "host").ratio(), 3) == ratio, form
        with pytest.raises(TypeError) as info:
            create_policy("tolerant_test", **{spelling: "10.0.0.2"})
        assert f"{spelling!r} (did you mean 'host'?)" in str(info.value), form

    @pytest.mark.parametrize(
        ("spelling", "why"),
        [
            ("hostname", "a name of its own, not a misspelling of `host`"),
            ("hoab", "two wrong characters is two typos, not one"),
        ],
    )
    def test_a_name_further_than_one_typo_from_a_parameter_passes_through(self, spelling: str, why: str) -> None:
        # The boundary: a sink's pass-through is what it is for, and widening
        # the screen past one typo would start eating the names it forwards.
        policy = create_policy("tolerant_test", **{spelling: "gr00t.local"})
        assert isinstance(policy, _Tolerant)
        assert policy.extra == {spelling: "gr00t.local"}, why
        assert policy.host == "localhost"

    def test_a_name_in_the_ratio_s_grey_band_is_forwarded_rather_than_renamed(self) -> None:
        """The 0.80 cutoff is what keeps a shared ``policy_config`` working.

        Twenty-five of the parameter names the registered providers declare sit
        between 0.70 and 0.80 of a *different* provider's parameter - ``api_key``
        against ``api_token``, ``strict`` against ``strict_keys``, ``device``
        against ``device_cfg``, ``config`` against ``data_config``. Loosening the
        cutoff would answer each of them with the other provider's name instead
        of forwarding it, and forwarding is what a ``**kwargs`` sink is for.
        """
        ratio = difflib.SequenceMatcher(None, "actions", "action_space").ratio()
        assert 0.70 < ratio < _MISSPELLING_RATIO
        policy = create_policy("tolerant_test", actions=[1, 2])
        assert isinstance(policy, _Tolerant)
        assert policy.extra == {"actions": [1, 2]}
        assert policy.action_space == "joint"

    def test_no_declared_parameter_of_any_provider_is_left_unscreened(self) -> None:
        """Every registered provider, every parameter, every single-typo spelling.

        The pin is the whole domain rather than the names that happened to be
        reported, because the hole was a property of *length*: nothing about
        ``host`` made it special except being short, so the next four-letter
        parameter added to any provider would have inherited the same silence.
        """
        unscreened: list[tuple[str, str, str]] = []
        for provider in sorted(set(list_policy_providers())):
            _canonical, PolicyClass, _kwargs = _resolve_policy_class(provider)
            accepted, _sink = _constructor_keywords(PolicyClass)
            for declared in accepted:
                for index in range(len(declared)):
                    for form in (
                        declared[:index] + "q" + declared[index + 1 :],  # substituted
                        declared[:index] + declared[index + 1 :],  # dropped
                        declared[:index] + declared[index] * 2 + declared[index + 1 :],  # doubled
                    ):
                        if form in accepted or _misspelling_of(form, accepted) is not None:
                            continue
                        unscreened.append((provider, declared, form))
        assert unscreened == []


class TestTheHelperItself:
    def test_a_class_without_an_introspectable_signature_is_a_no_op(self):
        assert policy_kwargs_error("x", int, {"anything": 1}) is None

    def test_nothing_to_say_when_every_name_is_bound(self):
        assert policy_kwargs_error("strict_test", _Strict, {"host": "h", "port": 1}) is None


@pytest.mark.skipif(not _HAS_ZMQ, reason="groot extra")
def test_the_groot_reproduction_from_the_lab():
    # PC2-009 verbatim: pre-fix this returned a Gr00tPolicy dialling localhost.
    with pytest.raises(TypeError) as info:
        create_policy("groot", hots="10.0.0.2", data_config="so101")
    assert "Gr00tPolicy (policy provider 'groot') does not accept 'hots' (did you mean 'host'?)." in str(info.value)
