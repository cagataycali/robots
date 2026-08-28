"""``rtc_max_guidance_weight`` reaches the field lerobot reads, or is refused.

lerobot's RTC denoiser takes the guidance ceiling from
``self.rtc_config.max_guidance_weight`` - it clamps with
``torch.minimum(guidance_weight, max_guidance_weight)`` and feeds the same value
to ``nan_to_num(..., posinf=max_guidance_weight)``. It is *not* part of the RTC
kwarg contract: ``ActionSelectKwargs`` carries ``inference_delay``,
``prev_chunk_left_over`` and ``execution_horizon`` and nothing else.

So a ceiling kept only on the policy object cannot take effect. Pre-fix
``_init_rtc`` read the checkpoint's value as a *default* into
``self._rtc_max_guidance_weight`` and never wrote a caller's override back, so
the override reached one ``logger.info`` line and nothing else: a caller asking
for ``2.0`` ran the model's own ``10.0``. Its adjacent twin
``rtc_execution_horizon`` did take effect, because that one IS a kwarg and
``_predict_with_rtc`` forwards it per call.

The value also had no domain, while every numeric sibling in the same
constructor has one - so ``0``, ``-5.0``, ``nan``, ``inf``, ``True`` and
``"abc"`` were all accepted. ``0`` and a negative are values lerobot's own
``RTCConfig.__post_init__`` refuses outright ("max_guidance_weight must be
positive"); that constructor never sees this one, because the override is
written onto an already-built config, so the domain is asked at the seam the
caller names instead.

Why the suite was green over it: ``test_rtc_user_overrides_config_values`` is
named for both knobs overriding the config and asserts only that
``policy._rtc_max_guidance_weight`` keeps the value it was given - an attribute
round-trip, which held either way. ``TestRTCConfigSchemaContract`` pins that
lerobot still *exposes* the field, i.e. it grades the read and never the
application.
"""

from __future__ import annotations

import ast
import inspect
import math
import textwrap
from typing import Any
from unittest.mock import MagicMock

import pytest

from strands_robots.policies.lerobot_local import policy as policy_module
from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy

# The ceiling the stand-in checkpoint declares. Distinct from every value a cell
# asks for, so "the override landed" and "the model's own value survived" can
# never be the same reading.
MODEL_CEILING = 10.0

# Values that cannot serve as a clamp. ``0`` and ``-3.5`` are the two lerobot's
# own config constructor refuses; the rest are the spellings a bare comparison
# would let through (``nan`` compares False against everything, ``inf`` clamps
# nothing, ``bool`` is an ``int`` subclass, and the last three are not numbers).
UNUSABLE: tuple[Any, ...] = (0, 0.0, -3.5, math.nan, math.inf, True, False, "2.0", [2.0], {"w": 2.0})

# Accepted spellings: a float, an int, and a value below the model's own.
USABLE: tuple[Any, ...] = (2.0, 0.5, 1, 25.0)


class _StubRtcConfig:
    """The RTC block a flow-matching checkpoint (Pi0, SmolVLA) carries.

    A plain class rather than a ``MagicMock``: a mock auto-provides every
    attribute and accepts every assignment, so it cannot distinguish "the
    override was written onto the config" from "nothing happened". Field names
    are pinned against lerobot's real dataclass in
    :class:`TestThePremisesThisRestsOn`.
    """

    def __init__(self, *, enabled: bool = True, horizon: int = 15, ceiling: float = MODEL_CEILING) -> None:
        self.enabled = enabled
        self.execution_horizon = horizon
        self.max_guidance_weight = ceiling


def _policy(**kwargs: Any) -> LerobotLocalPolicy:
    """Build the policy with model loading disabled, as the sibling suite does."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(LerobotLocalPolicy, "_load_model", lambda self: None)
        return LerobotLocalPolicy(model_path="stub", **kwargs)


def _enable_rtc(pol: LerobotLocalPolicy, cfg: _StubRtcConfig) -> _StubRtcConfig:
    """Drive ``_init_rtc`` against ``cfg`` and hand the same object back.

    Returns the config so a cell reads the ceiling off the object lerobot would
    read it from, rather than off the policy's own copy.
    """
    loaded = MagicMock()
    loaded.predict_action_chunk = MagicMock()
    loaded.config.rtc_config = cfg
    pol._policy = loaded
    pol._loaded = True
    pol._init_rtc()
    return cfg


class TestTheOverrideReachesTheFieldLerobotReads:
    """A supplied ceiling lands on ``rtc_config``, not only on the policy."""

    @pytest.mark.parametrize("weight", USABLE)
    def test_a_supplied_ceiling_is_written_onto_the_checkpoint_config(self, weight: Any) -> None:
        pol = _policy(rtc_enabled=True, rtc_max_guidance_weight=weight)
        cfg = _enable_rtc(pol, _StubRtcConfig())

        assert cfg.max_guidance_weight == weight, (
            f"asked for a {weight} ceiling and rtc_config still carries "
            f"{cfg.max_guidance_weight}; lerobot reads the clamp from this field, "
            "so a ceiling kept only on the policy never reaches the denoiser"
        )

    def test_the_policy_and_the_config_agree_after_init(self) -> None:
        """The two copies cannot disagree: one is written from the other."""
        pol = _policy(rtc_enabled=True, rtc_max_guidance_weight=2.0)
        cfg = _enable_rtc(pol, _StubRtcConfig())

        assert pol._rtc_max_guidance_weight == cfg.max_guidance_weight == 2.0

    def test_a_real_config_takes_the_override(self) -> None:
        """The whole path, against lerobot's genuine dataclass.

        The stand-in config is a plain class; this is the same assertion against
        the real ``RTCConfig``, so a rename of the field would fail here too.
        """
        config = pytest.importorskip("lerobot.policies.rtc.configuration_rtc")
        cfg = config.RTCConfig(enabled=True, execution_horizon=15, max_guidance_weight=MODEL_CEILING)

        pol = _policy(rtc_enabled=True, rtc_max_guidance_weight=2.0)
        loaded = MagicMock()
        loaded.predict_action_chunk = MagicMock()
        loaded.config.rtc_config = cfg
        pol._policy = loaded
        pol._loaded = True
        pol._init_rtc()

        assert cfg.max_guidance_weight == 2.0

    def test_the_model_ceiling_is_replaced_not_merely_shadowed(self) -> None:
        """The pre-fix reading is gone: the model's own value must not survive."""
        pol = _policy(rtc_enabled=True, rtc_max_guidance_weight=2.0)
        cfg = _enable_rtc(pol, _StubRtcConfig(ceiling=MODEL_CEILING))

        assert cfg.max_guidance_weight != MODEL_CEILING


class TestADomainThatMatchesTheClamp:
    """A ceiling that cannot clamp is refused where the caller names it."""

    @pytest.mark.parametrize("weight", UNUSABLE)
    def test_the_constructor_refuses_an_unusable_ceiling(self, weight: Any) -> None:
        with pytest.raises(ValueError, match=r"rtc_max_guidance_weight must be > 0"):
            _policy(rtc_enabled=True, rtc_max_guidance_weight=weight)

    @pytest.mark.parametrize("weight", UNUSABLE)
    def test_the_preflight_refuses_an_unusable_ceiling(self, weight: Any) -> None:
        """The dict entry point refuses it too, beside its four siblings.

        :meth:`LerobotLocalPolicy.preflight` already refuses ``actions_per_step``,
        ``image_keys``, ``rtc_execution_horizon`` and ``tokenizer_max_length``
        there, so that a rollout gets a structured error before the weight
        download rather than a raise from ``__init__`` after it.
        """
        with pytest.raises(ValueError, match=r"rtc_max_guidance_weight must be > 0"):
            LerobotLocalPolicy.preflight(set(), rtc_max_guidance_weight=weight)

    def test_the_refusal_names_the_parameter_and_the_provider(self) -> None:
        with pytest.raises(ValueError) as caught:
            _policy(rtc_enabled=True, rtc_max_guidance_weight=-1.0)

        text = str(caught.value)
        assert "rtc_max_guidance_weight" in text
        assert "lerobot_local" in text

    def test_the_module_consults_the_shared_domain(self) -> None:
        """A local re-implementation would drift from the sibling knobs."""
        calls = [
            node
            for node in ast.walk(ast.parse(inspect.getsource(policy_module)))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "positive_finite_number_error"
        ]
        assert len(calls) == 2, f"expected the constructor and the pre-flight to share the domain, found {len(calls)}"

    def test_the_guard_precedes_the_write_onto_the_config(self) -> None:
        """A refused value must never be assigned onto the checkpoint's config.

        Structural, because the behavioural cells cannot see it: the constructor
        refuses first, so ``_init_rtc`` is unreachable with a bad value. If the
        order were reversed a direct ``_init_rtc`` caller would install a ceiling
        that clamps nothing.
        """
        source = textwrap.dedent(inspect.getsource(LerobotLocalPolicy.__init__))
        guard = source.index("positive_finite_number_error(")
        store = source.index("self._rtc_max_guidance_weight = rtc_max_guidance_weight")
        assert guard < store, "the domain must be consulted before the value is stored"


class TestWhatIsUnchanged:
    """The over-reach controls: what the fix must not disturb."""

    def test_none_still_adopts_the_checkpoint_ceiling(self) -> None:
        """``None`` is the documented "use the model's value" request."""
        pol = _policy(rtc_enabled=True, rtc_max_guidance_weight=None)
        cfg = _enable_rtc(pol, _StubRtcConfig(ceiling=8.0))

        assert pol._rtc_max_guidance_weight == 8.0
        assert cfg.max_guidance_weight == 8.0, "defaulting must not rewrite the config it read from"

    def test_the_default_construction_supplies_no_ceiling(self) -> None:
        """Omitting the parameter is the same request as passing ``None``."""
        assert _policy(rtc_enabled=True)._rtc_max_guidance_weight is None

    def test_a_config_without_the_field_still_falls_back_to_ten(self) -> None:
        pol = _policy(rtc_enabled=True)
        loaded = MagicMock()
        loaded.predict_action_chunk = MagicMock()
        loaded.config.rtc_config = MagicMock(spec=["enabled"])
        loaded.config.rtc_config.enabled = True
        pol._policy = loaded
        pol._loaded = True
        pol._init_rtc()

        assert pol._rtc_max_guidance_weight == 10.0

    def test_the_horizon_twin_is_untouched(self) -> None:
        """``execution_horizon`` is a kwarg, so it needs no write onto the config.

        Recorded as a control because it is the shape a reader would reach for to
        argue the ceiling needed forwarding rather than assigning, and it is not:
        the horizon reaches the model through ``rtc_kwargs``, the ceiling cannot.
        """
        pol = _policy(rtc_enabled=True, rtc_execution_horizon=20)
        cfg = _enable_rtc(pol, _StubRtcConfig(horizon=15))

        assert pol._rtc_execution_horizon == 20
        assert pol.execution_horizon == 20
        assert cfg.execution_horizon == 15, "the horizon is forwarded per call, not written onto the config"

    def test_a_usable_ceiling_leaves_the_rest_of_the_config_alone(self) -> None:
        pol = _policy(rtc_enabled=True, rtc_max_guidance_weight=2.0)
        cfg = _enable_rtc(pol, _StubRtcConfig(enabled=True, horizon=15))

        assert cfg.enabled is True
        assert cfg.execution_horizon == 15

    def test_rtc_stays_off_when_the_checkpoint_carries_no_rtc_config(self) -> None:
        """A supplied ceiling must not enable RTC on a policy that cannot run it."""
        pol = _policy(rtc_enabled=True, rtc_max_guidance_weight=2.0)
        loaded = MagicMock()
        loaded.predict_action_chunk = MagicMock()
        loaded.config = MagicMock(spec=[])
        pol._policy = loaded
        pol._loaded = True
        pol._init_rtc()

        assert pol._rtc_enabled is False


class TestThePremisesThisRestsOn:
    """The lerobot-side facts the fix is built on, asserted against lerobot."""

    def test_the_rtc_kwarg_contract_does_not_carry_the_ceiling(self) -> None:
        """If it ever becomes a kwarg, forwarding beats assigning - fail here.

        The ceiling is written onto the config precisely *because* it is absent
        from this contract. Were it added upstream, this cell fires and the
        author reconsiders the write.
        """
        smolvla = pytest.importorskip("lerobot.policies.smolvla.modeling_smolvla")
        keys = set(smolvla.ActionSelectKwargs.__annotations__)

        assert "execution_horizon" in keys, "premise: the horizon IS a kwarg"
        assert "max_guidance_weight" not in keys, (
            "lerobot now accepts the ceiling as an RTC kwarg; forward it in "
            "_predict_with_rtc instead of writing it onto rtc_config"
        )

    def test_lerobot_reads_the_ceiling_off_its_config(self) -> None:
        """The field written to is the field the denoiser reads."""
        rtc = pytest.importorskip("lerobot.policies.rtc.modeling_rtc")
        source = inspect.getsource(rtc)

        assert "self.rtc_config.max_guidance_weight" in source

    def test_lerobot_refuses_the_ceilings_this_domain_refuses(self) -> None:
        """The two agree on ``<= 0``, which is why the domain is not stricter."""
        config = pytest.importorskip("lerobot.policies.rtc.configuration_rtc")

        for weight in (0, -3.5):
            with pytest.raises(ValueError, match="max_guidance_weight must be positive"):
                config.RTCConfig(enabled=True, max_guidance_weight=weight)


class TestTheProbeSetIsHonest:
    """Non-vacuity: the swept values are what the cells claim they are."""

    def test_no_probe_value_collides_with_the_model_ceiling(self) -> None:
        """Otherwise "the override landed" and "the default survived" coincide."""
        assert MODEL_CEILING not in USABLE

    def test_the_unusable_set_covers_both_failure_kinds(self) -> None:
        """A domain that only refused non-numbers would miss the ``0`` case."""
        numeric = [v for v in UNUSABLE if isinstance(v, (int, float)) and not isinstance(v, bool)]
        assert numeric, "premise: some refused values are numbers"
        assert [v for v in UNUSABLE if not isinstance(v, (int, float))], "and some are not"
