"""``embodiment`` and ``tune`` reach the lerobot policy config that declares them.

:class:`~strands_robots.training.lerobot.LerobotTrainer` lists ``groot`` among
the policy types it trains, and lerobot's own ``GrootConfig`` declares both an
``embodiment_tag`` (which state/action projector head the run trains) and four
per-component toggles (``tune_llm`` / ``tune_visual`` / ``tune_projector`` /
``tune_diffusion_model``). Two :class:`~strands_robots.training.base.TrainSpec`
fields are documented as exactly those requests, and the trainer read neither:
``spec.embodiment`` was read nowhere in the module, and ``spec.tune`` only for
the lora/expert-only mutual-exclusion check. So a run asking to unfreeze the
language backbone on a named embodiment trained the config defaults - the
inverse request on all four toggles - and ``validate()`` reported no problem.

The rule these cells pin is the one two sibling fields in the same method
already follow: ``learning_rate`` and ``relative_actions`` are applied when the
policy's config declares the field and REFUSED when it does not, rather than
dropped on a ``hasattr`` guard that leaves the run reporting success for a
recipe it did not run.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from strands_robots.training.base import TrainSpec
from strands_robots.training.lerobot import (
    _EMBODIMENT_TAG_POLICY_TYPES_FALLBACK,
    _TUNE_COMPONENT_FIELDS,
    _TUNE_COMPONENT_POLICY_TYPES_FALLBACK,
    LerobotTrainer,
    _lerobot_policy_types,
    _policy_supports_embodiment_tag,
    _policy_tune_components,
)

# The one lerobot policy family with an embodiment tag and per-component
# toggles. Every other policy takes its state/action shape from the dataset
# features and tunes as a whole.
TUNABLE = "groot"
WHOLE = "act"


@pytest.fixture
def spec(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(json.dumps({"total_episodes": 10}))
    return TrainSpec(
        dataset_root=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        steps=10,
        global_batch_size=2,
        save_freq=5,
    )


def _policy(spec: TrainSpec, ptype: str):
    """The ``cfg.policy`` a spec builds for *ptype* (device pinned off-GPU)."""
    pytest.importorskip("lerobot")
    spec.extra["policy_type"] = ptype
    return LerobotTrainer(device="cpu").build_config(spec).policy


class TestTheRequestReachesThePolicy:
    """A toggle or tag the policy declares is set to what was asked."""

    @pytest.mark.parametrize("component,field", sorted(_TUNE_COMPONENT_FIELDS.items()))
    @pytest.mark.parametrize("requested", [True, False])
    def test_each_component_lands_on_its_field(self, spec, component, field, requested):
        spec.tune = {component: requested}
        assert getattr(_policy(spec, TUNABLE), field) is requested

    def test_a_whole_recipe_lands_verbatim(self, spec):
        """The four toggles together - the shape a GR00T finetune is specified in.

        This request is the inverse of ``GrootConfig``'s defaults on every
        component, so a config that ignores it is indistinguishable from one
        that was never given it.
        """
        spec.tune = {"llm": True, "visual": True, "projector": False, "diffusion": False}
        policy = _policy(spec, TUNABLE)
        assert (policy.tune_llm, policy.tune_visual, policy.tune_projector, policy.tune_diffusion_model) == (
            True,
            True,
            False,
            False,
        )

    def test_embodiment_lands_on_the_tag(self, spec):
        spec.embodiment = "so101_follower"
        assert _policy(spec, TUNABLE).embodiment_tag == "so101_follower"

    def test_an_unrequested_field_keeps_the_policy_default(self, spec):
        """Not asking is not a request: the config's own defaults stand."""
        from lerobot.policies.factory import make_policy_config

        pytest.importorskip("lerobot")
        default = make_policy_config(TUNABLE)
        spec.tune = {"llm": True}
        policy = _policy(spec, TUNABLE)
        assert policy.tune_llm is True
        assert policy.tune_projector is default.tune_projector
        assert policy.embodiment_tag == default.embodiment_tag


class TestAPolicyWithoutTheFieldRefuses:
    """The ``learning_rate`` / ``relative_actions`` rule, for the two new fields."""

    def test_validate_reports_the_embodiment(self, spec):
        spec.embodiment = "so101_follower"
        spec.extra["policy_type"] = WHOLE
        problems = LerobotTrainer().validate(spec)
        assert any("embodiment" in p and "embodiment_tag" in p and TUNABLE in p for p in problems)

    def test_build_refuses_the_embodiment(self, spec):
        spec.embodiment = "so101_follower"
        with pytest.raises(ValueError, match="embodiment_tag"):
            _policy(spec, WHOLE)

    def test_validate_reports_the_component(self, spec):
        spec.tune = {"llm": True}
        spec.extra["policy_type"] = WHOLE
        problems = LerobotTrainer().validate(spec)
        assert any("llm" in p and "tune" in p and TUNABLE in p for p in problems)

    def test_build_refuses_the_component(self, spec):
        spec.tune = {"llm": True}
        with pytest.raises(ValueError, match="tune_llm"):
            _policy(spec, WHOLE)

    @pytest.mark.parametrize("typo", ["vision", "language", "action_head", ""])
    def test_a_key_naming_no_component_is_refused(self, spec, typo):
        """``vision`` for ``visual`` matched nothing to forward and trained the default."""
        spec.tune = {typo: True}
        spec.extra["policy_type"] = TUNABLE
        problems = LerobotTrainer().validate(spec)
        assert any("no tunable component" in p and repr(typo) in p for p in problems)

    def test_expert_only_stays_the_methods_own_key(self, spec):
        """``tune['expert_only']`` is the lora mutual-exclusion check's input, not a component."""
        spec.tune = {"expert_only": True}
        spec.extra["policy_type"] = TUNABLE
        assert LerobotTrainer().validate(spec) == []
        assert "expert_only" not in _TUNE_COMPONENT_FIELDS


# The spellings a YAML- or JSON-sourced config carries for "do not train this
# component". Every one is truthy, so a truthiness read trains it.
TRUTHY_SPELLINGS_OF_OFF: list[object] = ["false", "no", "off", "0"]
# The rest of the non-boolean domain boolean_flag_error refuses: a number that
# would pass as a silent 1 or 0, a sentinel, and a stringly true.
OTHER_NON_BOOLEANS: list[object] = [1, 0, None, "true"]


class TestATuneValueIsCheckedNotRead:
    """A component toggle is a posture flag: ``"false"`` must not train the component.

    The key axis is graded above; this is the value axis. ``bool("false")`` is
    ``True``, so a coerced read builds ``tune_llm=True`` for a caller who asked
    to freeze the language backbone, and nothing raises - the inverted recipe
    this PR's headline fix names, reached through the value instead of the key.
    """

    @pytest.mark.parametrize("value", TRUTHY_SPELLINGS_OF_OFF + OTHER_NON_BOOLEANS)
    def test_validate_refuses_it_by_the_flags_own_name(self, spec, value):
        spec.tune = {"llm": value}
        spec.extra["policy_type"] = TUNABLE
        problems = LerobotTrainer().validate(spec)
        assert any("tune['llm']" in p and "must be a boolean" in p for p in problems), problems

    @pytest.mark.parametrize("value", TRUTHY_SPELLINGS_OF_OFF)
    def test_argv_never_carries_a_truthy_spelling_of_off_as_true(self, spec, value):
        spec.tune = {"llm": value}
        spec.extra["policy_type"] = TUNABLE
        with pytest.raises(ValueError, match=r"tune\['llm'\] must be a boolean"):
            LerobotTrainer(device="cpu").build_command(spec)

    @pytest.mark.parametrize("value", TRUTHY_SPELLINGS_OF_OFF)
    def test_the_typed_config_refuses_the_same_value(self, spec, value):
        spec.tune = {"llm": value}
        with pytest.raises(ValueError, match=r"tune\['llm'\] must be a boolean"):
            _policy(spec, TUNABLE)

    def test_every_component_is_graded_not_only_the_first(self, spec):
        """The grade walks every requested component, in canonical order."""
        spec.tune = {component: "false" for component in _TUNE_COMPONENT_FIELDS}
        spec.extra["policy_type"] = TUNABLE
        problems = LerobotTrainer().validate(spec)
        named = [c for c in _TUNE_COMPONENT_FIELDS if any(f"tune['{c}']" in p for p in problems)]
        assert sorted(named) == sorted(_TUNE_COMPONENT_FIELDS)

    @pytest.mark.parametrize("value", [True, False])
    def test_a_boolean_is_honoured_as_given(self, spec, value):
        """The control: a checked flag is a gate, not a mute."""
        spec.tune = {"llm": value}
        spec.extra["policy_type"] = TUNABLE
        assert not [p for p in LerobotTrainer().validate(spec) if "tune['llm']" in p]
        argv = LerobotTrainer(device="cpu").build_command(spec)
        assert f"--policy.tune_llm={'true' if value else 'false'}" in argv

    def test_a_numpy_boolean_is_a_boolean(self, spec):
        """``boolean_flag_error``'s domain includes ``numpy.bool_``; so does this one."""
        np = pytest.importorskip("numpy")
        spec.tune = {"llm": np.bool_(False)}
        spec.extra["policy_type"] = TUNABLE
        assert not [p for p in LerobotTrainer().validate(spec) if "tune['llm']" in p]
        assert "--policy.tune_llm=false" in LerobotTrainer(device="cpu").build_command(spec)

    def test_the_unknown_key_refusal_is_not_lost_to_the_value_grade(self, spec):
        """Both axes are reported when both are wrong."""
        spec.tune = {"vision": "false", "llm": "false"}
        spec.extra["policy_type"] = TUNABLE
        problems = LerobotTrainer().validate(spec)
        assert any("no tunable component" in p for p in problems)
        assert any("tune['llm']" in p for p in problems)


class TestTheDocumentedCliSaysTheSameThing:
    """``build_command`` is the argv the typed config maps to, so it carries them too."""

    def test_argv_carries_the_tag_and_the_toggles(self, spec):
        spec.embodiment = "so101_follower"
        spec.tune = {"llm": True, "diffusion": False}
        spec.extra["policy_type"] = TUNABLE
        argv = LerobotTrainer(device="cpu").build_command(spec)
        assert "--policy.embodiment_tag=so101_follower" in argv
        assert "--policy.tune_llm=true" in argv
        assert "--policy.tune_diffusion_model=false" in argv

    def test_argv_order_is_the_canonical_one(self, spec):
        """A caller's dict order must not change the documented command."""
        spec.extra["policy_type"] = TUNABLE
        trainer = LerobotTrainer(device="cpu")
        spec.tune = {"diffusion": True, "llm": True}
        one = [a for a in trainer.build_command(spec) if a.startswith("--policy.tune_")]
        spec.tune = {"llm": True, "diffusion": True}
        two = [a for a in trainer.build_command(spec) if a.startswith("--policy.tune_")]
        assert one == two == ["--policy.tune_llm=true", "--policy.tune_diffusion_model=true"]

    def test_argv_omits_what_was_not_requested(self, spec):
        spec.extra["policy_type"] = TUNABLE
        argv = LerobotTrainer(device="cpu").build_command(spec)
        assert not [a for a in argv if a.startswith("--policy.tune_") or "embodiment_tag" in a]


class TestTheCapabilityProbesTrackLerobot:
    """The probes are derived, so a policy lerobot adds is graded on arrival."""

    def test_probes_agree_with_the_installed_configs(self):
        pytest.importorskip("lerobot")
        from lerobot.policies.factory import make_policy_config

        for ptype in sorted(_lerobot_policy_types()):
            try:
                names = {f.name for f in dataclasses.fields(make_policy_config(ptype))}
            except Exception:  # noqa: BLE001 - a policy whose config needs args is not gradeable here
                continue
            assert _policy_supports_embodiment_tag(ptype) == ("embodiment_tag" in names), ptype
            assert _policy_tune_components(ptype) == {
                c for c, field in _TUNE_COMPONENT_FIELDS.items() if field in names
            }, ptype

    @pytest.mark.parametrize(
        "fallback,probe",
        [
            (_EMBODIMENT_TAG_POLICY_TYPES_FALLBACK, _policy_supports_embodiment_tag),
            (_TUNE_COMPONENT_POLICY_TYPES_FALLBACK, lambda t: bool(_policy_tune_components(t))),
        ],
    )
    def test_the_offline_fallback_matches_the_live_answer(self, fallback, probe):
        """The documented static set is what the live probe says, or it lies offline."""
        pytest.importorskip("lerobot")
        assert set(fallback) == {t for t in _lerobot_policy_types() if probe(t)}
