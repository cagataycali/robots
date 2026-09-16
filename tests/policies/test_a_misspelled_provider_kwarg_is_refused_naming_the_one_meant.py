"""One rule for every provider: a keyword that misspells a constructor parameter is refused.

Pre-fix each provider had its own behaviour. A constructor with ``**kwargs``
dropped ``create_policy("groot", hots="x")`` silently, so the client dialled
the default host under ``status="success"``; ``remote`` and ``lerobot_async``
logged "ignoring unexpected constructor kwarg(s)" where no agent reads it; a
constructor without a sink raised CPython's ``__init__() got an unexpected
keyword argument 'acton_space'``, naming neither the provider nor the
parameter meant. ``create_policy`` now screens the resolved kwargs against the
constructor's own signature before constructing anything.
"""

from __future__ import annotations

import pytest

from strands_robots.policies import Policy, create_policy, register_policy
from strands_robots.policies.factory import policy_kwargs_error


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


class TestTheHelperItself:
    def test_a_class_without_an_introspectable_signature_is_a_no_op(self):
        assert policy_kwargs_error("x", int, {"anything": 1}) is None

    def test_nothing_to_say_when_every_name_is_bound(self):
        assert policy_kwargs_error("strict_test", _Strict, {"host": "h", "port": 1}) is None


@pytest.mark.skipif(pytest.importorskip("zmq", reason="groot extra") is None, reason="groot extra")
def test_the_groot_reproduction_from_the_lab():
    # PC2-009 verbatim: pre-fix this returned a Gr00tPolicy dialling localhost.
    with pytest.raises(TypeError) as info:
        create_policy("groot", hots="10.0.0.2", data_config="so101")
    assert "Gr00tPolicy (policy provider 'groot') does not accept 'hots' (did you mean 'host'?)." in str(info.value)
