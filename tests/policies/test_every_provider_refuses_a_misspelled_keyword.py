"""Every registered provider refuses a misspelled ``policy_config`` keyword before it runs.

docs/learn/policies/index.md: "A misspelled keyword is a TypeError before any
download". ``create_policy`` screens each constructor's declared keywords with a
did-you-mean, but the mock declared none and swallowed ``**kwargs``, so
``policy_config={"amplitud": 0.5}`` ran the sinusoid on its default without a
word (#4165). The mock now declares ``amplitude`` and ``seed``; its sink stays for the server
address the hardware drivers hand every provider.
"""

from __future__ import annotations

import inspect

import pytest

from strands_robots.policies import create_policy
from strands_robots.policies.mock import MockPolicy
from strands_robots.registry.policies import list_policy_providers


def _providers_with_a_keyword() -> list[tuple[str, str]]:
    """(provider, one keyword its constructor declares) for every registry provider."""
    from strands_robots.policies.base import provider_policy_class

    out = []
    for name in sorted(list_policy_providers()):
        try:
            cls = provider_policy_class(name)
        except Exception:  # noqa: BLE001 - a provider whose extra is absent cannot be screened here
            continue
        if cls is None:
            continue
        params = [
            p.name
            for p in inspect.signature(cls).parameters.values()
            if p.name != "self" and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY) and len(p.name) >= 5
        ]
        if params:
            out.append((name, params[0]))
    return out


class TestTheMockHasASignature:
    def test_a_misspelled_amplitude_is_refused_with_a_did_you_mean(self) -> None:
        with pytest.raises(TypeError, match=r"does not accept 'amplitud' \(did you mean 'amplitude'\?\)"):
            create_policy("mock", amplitud=0.5)

    def test_the_drivers_server_address_is_still_forwarded_to_the_sink(self) -> None:
        # ``UR.start_task`` and its siblings hand every provider ``host`` (and ``port``); a mock
        # on a real arm must keep building, so those land in the sink rather than a refusal.
        p = create_policy("mock", host="localhost", port=5555)
        assert isinstance(p, MockPolicy)

    def test_the_declared_keywords_bind(self) -> None:
        p = create_policy("mock", amplitude=0.25, seed=3)
        assert isinstance(p, MockPolicy) and p.amplitude == 0.25 and p.seed == 3
        declared = [q.name for q in inspect.signature(MockPolicy).parameters.values() if q.kind is not q.VAR_KEYWORD]
        assert declared == ["amplitude", "seed"]

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), "0.5", None, -0.1])
    def test_the_amplitude_domain_is_checked(self, bad: object) -> None:
        with pytest.raises(ValueError, match="amplitude"):
            MockPolicy(amplitude=bad)  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_the_amplitude_shapes_the_sinusoid(self) -> None:
        loud = MockPolicy(amplitude=1.0)
        loud.set_robot_state_keys(["j0"])
        quiet = MockPolicy(amplitude=0.0)
        quiet.set_robot_state_keys(["j0"])
        loud_actions = await loud.get_actions({"j0": 0.0}, "")
        quiet_actions = await quiet.get_actions({"j0": 0.0}, "")
        assert max(abs(a["j0"]) for a in loud_actions) > 0.0
        assert all(a["j0"] == 0.0 for a in quiet_actions)


@pytest.mark.parametrize(("provider", "keyword"), _providers_with_a_keyword())
def test_every_buildable_provider_refuses_one_misspelled_keyword(
    provider: str, keyword: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The screen sits behind the trust gate for the Hub-loading providers; the gate is not
    # what this grades, and nothing is downloaded because the refusal precedes the build.
    monkeypatch.setenv("STRANDS_TRUST_REMOTE_CODE", "1")
    # Drop the last character: a length-changing typo every provider's screen sees.
    typo = keyword[:-1]
    with pytest.raises(TypeError, match=rf"does not accept '{typo}' \(did you mean '{keyword}'\?\)"):
        create_policy(provider, **{typo: object()})
