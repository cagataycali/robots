"""``VeraConfig``'s two video-planner sampler knobs take the domains their siblings do.

:class:`~strands_robots.policies.vera.VeraConfig` declares seven numeric fields
and held five of them to a shared domain on the *effective* value - both ports,
``render_width``, ``motion_plan_scale`` and ``server_ready_timeout``. The two it
did not look at were ``sample_steps`` and ``teacache_thresh``, the WAN planner's
denoise-step count and its teacache rel_l1 threshold, and neither is read
anywhere else: their only consumer is the launch command, which carries them as
TEXT - ``str(cfg.sample_steps)`` and ``str(cfg.teacache_thresh)`` in
``VeraServerRunner._build_command``, and ``f"VERA_SAMPLE_STEPS={...}"`` in
``DockerServerRunner``'s ``-e`` overlay.

Measured on ``88ef2e1``, one config per row, then the argv
``VeraServerRunner._build_command`` composes for it. Nothing is launched:

| value | constructed | argv token | flag's own type parses it |
| --- | --- | --- | --- |
| ``teacache_thresh=0.10`` (default) | accepted | ``0.1`` | yes |
| ``teacache_thresh=0.25`` | accepted | ``0.25`` | yes |
| ``teacache_thresh=0`` | accepted | ``0`` | yes |
| ``teacache_thresh=-1.0`` | accepted | ``-1.0`` | yes |
| ``teacache_thresh=nan`` | accepted | ``nan`` | yes |
| ``teacache_thresh=inf`` | accepted | ``inf`` | yes |
| ``teacache_thresh=True`` | accepted | ``True`` | **no** |
| ``teacache_thresh="0.1"`` | accepted | ``0.1`` | yes |
| ``teacache_thresh=None`` | accepted | ``None`` | **no** |
| ``sample_steps=10`` | accepted | ``10`` | yes |
| ``sample_steps=20 / 2`` | accepted | ``10.0`` | **no** |
| ``sample_steps=0`` | accepted | ``0`` | yes |
| ``sample_steps=-5`` | accepted | ``-5`` | yes |
| ``sample_steps=2.7`` | accepted | ``2.7`` | **no** |
| ``sample_steps=nan`` | accepted | ``nan`` | **no** |
| ``sample_steps=inf`` | accepted | ``inf`` | **no** |
| ``sample_steps=True`` | accepted | ``True`` | **no** |
| ``sample_steps="ten"`` | accepted | ``ten`` | **no** |

Every row was constructed, so the config refused nothing and the server was left
to report all of it. It has two ways to, and neither names the field. A token the
flag's own type cannot parse makes the server exit before it opens its port, and
``_wait_until_ready`` answers ``VERA server exited early (code N) ... common
causes are missing checkpoints (set VERA_CKPT_ROOT / ckpt_root) or CUDA OOM`` -
two causes that are not the cause. A token it *can* parse starts a server
configured by a value nobody asked for: ``0`` or ``-5`` denoise steps, a
threshold of ``nan`` (below nothing) or ``inf`` (below everything). Which of the
two happens is not a property of the value being usable, it is a property of how
``str()`` happens to spell it.

``start()`` already takes this position two statements above the launch:
``_require_vera_installed`` exists because, in its own words, without it "a
missing install surfaces only as an opaque 'server exited early (code 1)'
RuntimeError several seconds later".

``sample_steps=20 / 2`` is the row that makes this more than hygiene. It is not a
malformed value, it is a computed count: a positive whole number that the shared
count domain accepts, whose ``str()`` is ``'10.0'``, which ``--sample-steps``
cannot parse. Converting after the domain accepts it is what puts ``10`` on the
command line.

The two spellings of each knob also disagreed. ``_env_int`` and ``_env_float``
return ``None`` for anything ``int()``/``float()`` refuses, so
``VERA_SAMPLE_STEPS=ten`` is absorbed and the planner yaml decides - deliberate,
and pinned by ``test_vera_unit``. The keyword spelling of the same knob was
checked nowhere, so one knob was guarded from the environment and unguarded from
the API.

``0`` is not an opt-out for the threshold: ``teacache=False`` is, and it emits
``--no-teacache`` in place of the flag entirely. The documented quality cliff
above ``0.15`` is guidance rather than a bound, so ``0.25`` stays a legitimate
request and is a control here.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from strands_robots.policies.vera import VeraConfig
from strands_robots.policies.vera.server_runner import make_server_runner
from strands_robots.utils import positive_finite_number_error, positive_whole_number_error

#: The threshold a config that names none resolves to, stated here rather than
#: imported: the documented default is the contract, and a test that read it back
#: out of the module could not notice the module changing it.
DOCUMENTED_DEFAULT_THRESH = 0.10

# Counts no number of denoise steps can be. ``0``/``-5`` because a step count
# below one denoises nothing; ``2.7``/``nan``/``inf`` because no whole number
# stands for them; ``True`` because an ``int`` subclass silently meant one step;
# ``"ten"`` because the keyword spelling carried the very text the environment
# spelling is careful to absorb.
UNUSABLE_STEPS: tuple[tuple[str, Any], ...] = (
    ("zero", 0),
    ("negative", -5),
    ("fractional", 2.7),
    ("nan", math.nan),
    ("inf", math.inf),
    ("bool", True),
    ("str", "ten"),
    ("none-numeric str", "10"),
    ("list", [10]),
)

# Thresholds no rel_l1 threshold can be. ``0`` because ``teacache=False`` is the
# opt-out; a negative because a relative difference is never below zero; ``nan``
# because it is below nothing and ``inf`` because it is below everything, so each
# turns the comparison the flag exists for into a constant; ``True`` because it
# is the one spelling ``str()`` renders as a token no float parses; ``None``
# because the field is declared ``float`` and ``str(None)`` is the literal token
# ``'None'``.
UNUSABLE_THRESH: tuple[tuple[str, Any], ...] = (
    ("zero", 0),
    ("negative", -1.0),
    ("nan", math.nan),
    ("inf", math.inf),
    ("bool", True),
    ("numeric str", "0.1"),
    ("none", None),
    ("dict", {}),
)


def _config(**kwargs: Any) -> VeraConfig:
    """Build a config through the funnel, splatted so an off-type value reaches it.

    mypy does not narrow a ``**dict[str, Any]`` splat, which is what lets a test
    hand the dataclass a value its annotation forbids - the same idiom
    ``test_vera_n_action_steps_removed`` uses. It is not only a test convenience:
    the annotation is not a runtime check, and this config is built from
    environment strings, from dicts and from computed expressions in untyped
    code, so every value below is one a real caller can deliver.
    """
    return VeraConfig(**kwargs)


def _argv(**kwargs: Any) -> list[str]:
    """Argv a local-subprocess launch would use for this config. Nothing starts."""
    return list(make_server_runner(VeraConfig(embodiment="mimicgen", **kwargs))._build_command())


def _token(argv: list[str], flag: str) -> str:
    """The value ``flag`` carries in ``argv``."""
    assert flag in argv, f"premise: {flag} absent from {argv}"
    return argv[argv.index(flag) + 1]


class TestASamplerKnobIsRefusedRatherThanForwardedAsText:
    """Neither knob may reach the launch command as a value nothing can use."""

    @pytest.mark.parametrize(("label", "value"), UNUSABLE_STEPS, ids=[k for k, _ in UNUSABLE_STEPS])
    def test_an_unusable_step_count_is_refused(self, label: str, value: Any) -> None:
        with pytest.raises(ValueError, match="sample_steps"):
            _config(embodiment="mimicgen", sample_steps=value)

    @pytest.mark.parametrize(("label", "value"), UNUSABLE_THRESH, ids=[k for k, _ in UNUSABLE_THRESH])
    def test_an_unusable_threshold_is_refused(self, label: str, value: Any) -> None:
        with pytest.raises(ValueError, match="teacache_thresh"):
            _config(embodiment="mimicgen", teacache_thresh=value)

    def test_the_refusal_is_the_shared_count_domain_verbatim(self) -> None:
        """Graded through the answer, so a hand-rolled copy of the rule cannot drift in."""
        with pytest.raises(ValueError) as caught:
            _config(embodiment="mimicgen", sample_steps=2.7)
        assert str(caught.value) == positive_whole_number_error(2.7, "sample_steps", "VeraConfig")

    def test_the_refusal_is_the_shared_continuous_domain_verbatim(self) -> None:
        with pytest.raises(ValueError) as caught:
            _config(embodiment="mimicgen", teacache_thresh=math.inf)
        assert str(caught.value) == positive_finite_number_error(math.inf, "teacache_thresh", "VeraConfig")


class TestTheThresholdIsCheckedWhateverTheCacheFlagSays:
    """The check is not scoped to ``teacache``, because that flag can move after it."""

    def test_a_bad_threshold_is_refused_with_the_cache_off(self) -> None:
        with pytest.raises(ValueError, match="teacache_thresh"):
            _config(embodiment="mimicgen", teacache=False, teacache_thresh=math.nan)

    def test_turning_the_cache_on_afterwards_cannot_reach_an_unchecked_value(self) -> None:
        """A plain dataclass, so the flag is mutable and the value must already be sound."""
        cfg = _config(embodiment="mimicgen", teacache=False, teacache_thresh=0.2)
        cfg.teacache = True
        assert _token(list(make_server_runner(cfg)._build_command()), "--teacache-thresh") == "0.2"


class TestACountIsNormalizedBeforeItBecomesText:
    """The conversion is what keeps a computed count off the command line as a float."""

    def test_a_computed_count_reaches_the_flag_as_a_whole_number(self) -> None:
        cfg = _config(embodiment="mimicgen", sample_steps=20 / 2)
        assert cfg.sample_steps == 10
        assert not isinstance(cfg.sample_steps, float)
        token = _token(list(make_server_runner(cfg)._build_command()), "--sample-steps")
        assert token == "10"
        # The flag is an int flag, so the token has to survive int() to be usable.
        assert int(token) == 10

    def test_a_numpy_count_reaches_the_flag_as_a_whole_number(self) -> None:
        """The shared domain admits any real scalar, so the field is declared ``int``."""
        cfg = _config(embodiment="mimicgen", sample_steps=np.int64(12))
        assert _token(list(make_server_runner(cfg)._build_command()), "--sample-steps") == "12"

    def test_a_numpy_threshold_reaches_the_flag_as_a_plain_float(self) -> None:
        cfg = _config(embodiment="mimicgen", teacache_thresh=np.float32(0.25))
        assert isinstance(cfg.teacache_thresh, float)
        assert float(_token(list(make_server_runner(cfg)._build_command()), "--teacache-thresh")) == pytest.approx(0.25)


class TestTheValuesTheServerCanUseStillReachIt:
    """Controls. Each holds on both sides of the fix, so a refusal that swallowed
    the whole knob would fail here rather than pass quietly."""

    def test_the_documented_default_is_unchanged(self) -> None:
        assert VeraConfig(embodiment="mimicgen").teacache_thresh == DOCUMENTED_DEFAULT_THRESH

    def test_a_threshold_past_the_documented_cliff_is_still_a_legitimate_request(self) -> None:
        """``>0.15`` is guidance about quality, not a bound this domain enforces."""
        assert _token(_argv(teacache_thresh=0.25), "--teacache-thresh") == "0.25"

    def test_the_deploy_step_count_still_reaches_the_flag(self) -> None:
        assert _token(_argv(sample_steps=10), "--sample-steps") == "10"

    def test_an_unset_count_still_leaves_the_flag_off_the_command(self) -> None:
        """``None`` is the documented opt-out - the planner yaml decides."""
        cfg = VeraConfig(embodiment="mimicgen")
        assert cfg.sample_steps is None
        assert "--sample-steps" not in _argv()

    def test_switching_the_cache_off_still_replaces_the_flag(self) -> None:
        argv = _argv(teacache=False)
        assert "--no-teacache" in argv
        assert "--teacache-thresh" not in argv


class TestTheEnvironmentSpellingIsUntouched:
    """The env parsers absorb a malformed value by design; that is not changed here."""

    def test_a_malformed_step_count_in_the_environment_still_falls_back(self, monkeypatch: Any) -> None:
        monkeypatch.setenv("VERA_SAMPLE_STEPS", "ten")
        assert VeraConfig(embodiment="mimicgen").sample_steps is None

    def test_a_usable_step_count_from_the_environment_reaches_the_flag(self, monkeypatch: Any) -> None:
        monkeypatch.setenv("VERA_SAMPLE_STEPS", "12")
        assert _token(_argv(), "--sample-steps") == "12"

    def test_an_unusable_step_count_from_the_environment_is_refused(self, monkeypatch: Any) -> None:
        """``_env_int`` accepts every integer, so a zero reaches the field from there too."""
        monkeypatch.setenv("VERA_SAMPLE_STEPS", "0")
        with pytest.raises(ValueError, match="sample_steps"):
            VeraConfig(embodiment="mimicgen")
