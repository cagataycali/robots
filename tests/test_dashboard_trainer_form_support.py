"""Q48: the provider dropdown must not offer a provider this form cannot submit.

`ppo` and `fast_sac` require an RLTrainSpec, which the dashboard's training form does not build and
cannot build - it is a dataset form. Offering them cost the operator a dataset choice, a click and
the message "ppo requires an RLTrainSpec, got TrainSpec", which names internal classes and describes
a path that can never succeed.
"""

from __future__ import annotations

from strands_robots.dashboard import training


def test_rl_providers_are_marked_unsupported_for_the_form():
    out = training.form_unsupported()
    for provider in ("ppo", "fast_sac"):
        assert provider in out, f"{provider} needs an RLTrainSpec this form never builds"
        reason = out[provider]
        # The reason is for an operator, not a stack trace: it must say what the trainer
        # needs instead, in words that suggest what to do.
        assert "reinforcement-learning" in reason
        assert "live environment" in reason
        assert "script" in reason
        # Plural, article-free: the form prints this after a list of names, and the singular
        # wording produced "fast_sac and ppo are a reinforcement-learning trainer".
        assert not reason.startswith("a ")
        assert "they are" in reason


def test_supervised_providers_stay_offered():
    """cosmos3 is excluded on purpose: Q49 refuses it for a different reason (a recipe TOML the
    form has no field for). groot stays because `embodiment` IS expressible."""
    out = training.form_unsupported()
    for provider in ("lerobot_local", "mock", "groot"):
        assert provider not in out, (
            f"{provider} trains from a dataset - refusing it here would hide a working backend"
        )


def test_every_unsupported_provider_is_a_real_provider():
    """A stale name in this map would grey out an option that no longer exists, or worse,
    fail to grey out the one that does."""
    known = set(training.list_trainers())
    assert set(training.form_unsupported()) <= known


def test_unknown_provider_shape_is_not_guessed(monkeypatch):
    """A registry entry we cannot classify is offered as usual - guessing 'unsupported'
    would hide a backend that works."""
    monkeypatch.setattr(training, "list_trainers", lambda: ["mystery"])
    assert training.form_unsupported() == {}


def test_a_malformed_registry_entry_cannot_blank_the_form(monkeypatch):
    import strands_robots.registry.policies as reg

    monkeypatch.setattr(training, "list_trainers", lambda: ["ppo", "boom"])

    real = reg.get_policy_provider

    def explode(name: str):
        if name == "boom":
            raise RuntimeError("malformed entry")
        return real(name)

    monkeypatch.setattr(reg, "get_policy_provider", explode)
    monkeypatch.setattr("strands_robots.registry.policies.get_policy_provider", explode, raising=False)
    # ppo is still classified; the broken entry is skipped rather than taking the route down.
    assert training.form_unsupported() == {"ppo": training._RL_REASON}


# --- Q49: the mirror must not rot -------------------------------------------------------
# _FORM_CANNOT_EXPRESS mirrors a requirement the SDK declares only inside a trainer's
# validate(). These tests ASK validate, so the day cosmos3 grows a default recipe (or
# lerobot starts demanding one) the mirror fails loudly instead of lying quietly.

_MINIMAL = {"dataset_root": "/tmp/does-not-exist", "output_dir": "/tmp/out", "steps": 10}


def _problems(provider: str) -> str:
    return str(training.validate({"provider": provider, **_MINIMAL}).get("text", ""))


def test_cosmos3_really_demands_something_the_form_cannot_send():
    text = _problems("cosmos3")
    assert "sft_toml" in text or "recipe TOML" in text, text
    # And the field it wants is genuinely absent from the form's vocabulary - that is WHY
    # this one is refused up front instead of getting a new input.
    assert not any("toml" in key.lower() for key in training.SPEC_KEYS)
    assert training.form_unsupported()["cosmos3"] == training._FORM_CANNOT_EXPRESS["cosmos3"]


def test_a_dataset_trainer_demands_nothing_the_form_lacks():
    """lerobot_local's complaints must all be about things the operator can supply here."""
    text = _problems("lerobot_local")
    assert "sft_toml" not in text
    assert "RLTrainSpec" not in text
    assert "lerobot_local" not in training.form_unsupported()


def test_groot_wants_an_embodiment_and_the_form_can_express_it():
    """GR00T is NOT refused: `embodiment` is a real spec key, so the form grows a field
    instead (providerFields.ts). This pins the requirement the field exists for."""
    text = _problems("groot")
    assert "embodiment" in text
    assert "embodiment" in training.SPEC_KEYS
    assert "groot" not in training.form_unsupported()
