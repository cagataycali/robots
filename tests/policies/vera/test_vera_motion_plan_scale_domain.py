"""``VeraConfig.motion_plan_scale`` is a multiplier, so it takes the scale domain.

:class:`~strands_robots.policies.vera.VeraConfig` validates three of its fields
on the *effective* value - both ports and ``render_width`` - and its own comment
says why: ``__post_init__`` is "the one funnel every caller passes through - the
``VeraPolicy`` keywords, a pre-built config handed to it, and the ``VERA_*_PORT``
environment overrides above". ``motion_plan_scale`` arrives through that same
funnel, twenty lines further down, and was checked nowhere.

It is the third scale in this package. The other two are already held to a
domain - ``translation_scale`` to
:func:`strands_robots.utils.positive_finite_number_error` and ``ik_smoothing`` to
a local ``[0, 1)`` rule - and that guard's docstring names "a dimensionless
multiplier" as one of the families it exists for.

Measured on ``ad8696b``, one ``VeraConfig(embodiment="mimicgen",
motion_plan_scale=X)`` per row, then ``_ensure_started()`` against a recording
stub client. No ``vera`` package, no server, no socket:

| ``motion_plan_scale=`` | stored | what reached the server |
| --- | --- | --- |
| ``0.8`` (usable)  | ``0.8``    | ``{'motion_plan_scale': 0.8}`` |
| ``0`` / ``0.0``   | ``0.0``    | ``{'motion_plan_scale': 0.0}`` - the plan scaled to nothing |
| ``-1.5``          | ``-1.5``   | ``{'motion_plan_scale': -1.5}`` - the plan inverted |
| ``nan``           | ``nan``    | ``{'motion_plan_scale': nan}`` |
| ``inf``           | ``inf``    | ``{'motion_plan_scale': inf}`` |
| ``True``          | ``True``   | ``{'motion_plan_scale': 1.0}`` - a silent scale of one |
| ``"0.8"``         | ``'0.8'``  | ``{'motion_plan_scale': 0.8}`` - silently coerced |
| ``[0.8]``         | ``[0.8]``  | **nothing** - see below |

The environment path is wider still, because ``_env_float`` returns whatever
``float()`` accepts and only falls back on ``ValueError``. Every one of
``VERA_MOTION_PLAN_SCALE`` in ``0``, ``-1.5``, ``nan``, ``inf``, ``1e999`` and
``Infinity`` landed on the field (the last two as ``inf``); only a non-numeric
spelling such as ``abc`` fell back to ``None``.

The ``[0.8]`` row is the one that makes this worth refusing at the config rather
than downstream. ``_ensure_started`` applies the scale with::

    if self.config.motion_plan_scale is not None:
        try:
            self._client.configure({"motion_plan_scale": float(...)})
        except Exception as e:
            logger.info("VeraPolicy live configure(motion_plan_scale) skipped: %s", e)

so a value ``float()`` cannot convert is neither applied nor reported: nothing is
sent, the ``TypeError`` is logged at INFO, ``_started`` is set anyway, and the
rollout proceeds at whatever scale the server already had while the config says
otherwise. The swallow is correct for a genuine transport failure - live tuning
is best-effort - which is precisely why the *value* has to be refused before it
gets there. That is the same argument ``render_width`` on this dataclass is
already validated on, one step worse: an unusable width at least raises.

``None`` stays valid throughout. It is the documented opt-out and it gates the
``configure`` call away entirely, so "leave the server's scale alone" and "scale
the plan to nothing" remain different requests - which is why ``0`` is refused
rather than treated as the off switch.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from strands_robots.policies.vera import VeraConfig, VeraPolicy
from strands_robots.policies.vera import provider as provider_mod
from strands_robots.policies.vera.config import _env_float

# Scales no motion plan can be multiplied by. ``0`` and ``False`` are here
# because scaling a plan to nothing is not the opt-out (``None`` is); ``True``
# and ``"0.8"`` because they were honored as a silent 1.0 and 0.8; ``[0.8]``
# because it reached the swallow and was dropped without a report.
UNUSABLE_SCALES: list[Any] = [
    0,
    0.0,
    False,
    -1.5,
    True,
    float("nan"),
    float("inf"),
    float("-inf"),
    "0.8",
    [0.8],
]

# Environment spellings of the same unusable values. ``1e999`` and ``Infinity``
# are included because ``float()`` accepts both and yields ``inf``, so neither
# reaches the field as the string an operator typed.
UNUSABLE_ENV_SCALES: list[str] = ["0", "-1.5", "nan", "inf", "1e999", "Infinity"]

# Values a plan can be scaled by. The domain admits any real scalar, so a NumPy
# float and a plain ``int`` are usable and must survive normalization.
USABLE_SCALES: list[Any] = [0.8, 1, 2, 0.05, np.float64(0.8)]


def _config(**kwargs: Any) -> VeraConfig:
    """Build a config, funnelling the deliberately off-type values through ``Any``.

    The rows above pass values the field's ``float | None`` annotation does not
    describe, which is the point of the test; one funnel states that once
    instead of a suppression at each call.
    """
    return VeraConfig(**kwargs)


class _RecordingClient:
    """Records what ``_ensure_started`` sends, standing in for the server."""

    def __init__(self) -> None:
        self.configured: list[dict[str, Any]] = []

    def get_server_metadata(self) -> dict[str, Any]:
        return {"view_keys": ["image"], "action_horizon": 4}

    def configure(self, params: dict[str, Any]) -> dict[str, Any]:
        self.configured.append(dict(params))
        return {"applied": dict(params)}

    def reset(self, reset_info: dict[str, Any] | None = None) -> None:
        return None

    def close(self) -> None:
        return None


def _policy(client: Any, **kwargs: Any) -> VeraPolicy:
    """A policy wired to a stub client, with no server launch."""
    return VeraPolicy(embodiment="mimicgen", client=client, auto_launch_server=False, **kwargs)


class TestTheConfigRefusesAnUnusableScale:
    """A scale no motion plan can be multiplied by is refused at construction."""

    @pytest.mark.parametrize("value", UNUSABLE_SCALES)
    def test_an_unusable_scale_is_refused(self, value: Any) -> None:
        with pytest.raises(ValueError) as excinfo:
            _config(embodiment="mimicgen", motion_plan_scale=value)
        assert "motion_plan_scale" in str(excinfo.value)

    @pytest.mark.parametrize("value", UNUSABLE_SCALES)
    def test_the_message_names_the_field_and_the_surface(self, value: Any) -> None:
        with pytest.raises(ValueError) as excinfo:
            _config(embodiment="mimicgen", motion_plan_scale=value)
        text = str(excinfo.value)
        assert text.startswith("VeraConfig: motion_plan_scale "), text
        assert repr(value) in text or str(value) in text, text


class TestTheEnvironmentOverrideTakesTheSameDomain:
    """``VERA_MOTION_PLAN_SCALE`` goes through the same check as a keyword."""

    @pytest.mark.parametrize("raw", UNUSABLE_ENV_SCALES)
    def test_an_unusable_environment_scale_is_refused(self, raw: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VERA_MOTION_PLAN_SCALE", raw)
        with pytest.raises(ValueError) as excinfo:
            _config(embodiment="mimicgen")
        assert "motion_plan_scale" in str(excinfo.value)

    @pytest.mark.parametrize("raw", UNUSABLE_ENV_SCALES)
    def test_the_resolver_really_does_return_those_values(self, raw: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """Premise: the field guard is what closes the environment path.

        ``_env_float`` is deliberately left permissive - it falls back only on
        ``ValueError``, matching ``_env_int`` beside it - so every row above
        genuinely reaches the field and the check on the effective value is
        what refuses it.
        """
        monkeypatch.setenv("VERA_MOTION_PLAN_SCALE", raw)
        resolved = _env_float("VERA_MOTION_PLAN_SCALE")
        assert resolved is not None
        assert resolved <= 0.0 or not math.isfinite(resolved)

    def test_a_usable_environment_scale_is_applied(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VERA_MOTION_PLAN_SCALE", "0.75")
        assert _config(embodiment="mimicgen").motion_plan_scale == pytest.approx(0.75)

    def test_a_non_numeric_environment_scale_still_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An unparsable spelling resolves to ``None``, as it does for the ports.

        This is the established convention for every ``VERA_*`` override on this
        dataclass, and ``None`` is a usable answer here rather than a silent
        substitution: it leaves the server's own scale alone.
        """
        monkeypatch.setenv("VERA_MOTION_PLAN_SCALE", "abc")
        assert _config(embodiment="mimicgen").motion_plan_scale is None


class TestNoneIsTheOptOut:
    """``None`` is not merely accepted - it is the documented way to opt out."""

    def test_none_constructs(self) -> None:
        assert _config(embodiment="mimicgen", motion_plan_scale=None).motion_plan_scale is None

    def test_unset_resolves_to_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("VERA_MOTION_PLAN_SCALE", raising=False)
        assert _config(embodiment="mimicgen").motion_plan_scale is None

    def test_none_sends_no_configure_call_at_all(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("VERA_MOTION_PLAN_SCALE", raising=False)
        client: Any = _RecordingClient()
        policy = _policy(client, motion_plan_scale=None)
        policy._ensure_started()
        assert client.configured == []


class TestAUsableScaleIsHonoredUnchanged:
    """Every value the domain admits still reaches the server verbatim."""

    @pytest.mark.parametrize("value", USABLE_SCALES)
    def test_a_usable_scale_is_stored_as_a_float(self, value: Any) -> None:
        stored = _config(embodiment="mimicgen", motion_plan_scale=value).motion_plan_scale
        assert type(stored) is float
        assert stored == pytest.approx(float(value))

    @pytest.mark.parametrize("value", USABLE_SCALES)
    def test_a_usable_scale_reaches_the_server(self, value: Any) -> None:
        client: Any = _RecordingClient()
        policy = _policy(client, motion_plan_scale=value)
        policy._ensure_started()
        assert client.configured == [{"motion_plan_scale": pytest.approx(float(value))}]


class TestTheRefusalPrecedesTheServerLaunch:
    """A refused scale leaves nothing half-configured behind."""

    @pytest.mark.parametrize("value", UNUSABLE_SCALES)
    def test_no_server_runner_is_built(self, value: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        def fatal(_config_arg: Any) -> Any:
            raise AssertionError("the refused scale reached the server runner factory")

        monkeypatch.setattr(provider_mod, "make_server_runner", fatal)
        with pytest.raises(ValueError) as excinfo:
            VeraPolicy(embodiment="mimicgen", motion_plan_scale=value)
        assert "motion_plan_scale" in str(excinfo.value)


class TestTheBestEffortSwallowIsUnchanged:
    """Live tuning stays best-effort for a genuine transport failure.

    The check above refuses unusable *values*; it must not turn a server that is
    momentarily unreachable into a failed rollout, which is what the ``except``
    in ``_ensure_started`` exists for.
    """

    def test_a_configure_that_raises_still_starts_the_policy(self) -> None:
        class RefusingClient(_RecordingClient):
            def configure(self, params: dict[str, Any]) -> dict[str, Any]:
                raise ConnectionError("server went away")

        client: Any = RefusingClient()
        policy = _policy(client, motion_plan_scale=0.8)
        policy._ensure_started()
        assert policy._started is True


class TestTheScalesInThisPackageShareOneDomain:
    """Neither scale may accept what the other refuses.

    ``translation_scale`` on ``set_ik_target`` is held to the same shared guard,
    so the two answer the same question about the same kind of value. Its
    check runs before any state is mutated, so a bogus model object never
    reaches MuJoCo.
    """

    @pytest.mark.parametrize("value", UNUSABLE_SCALES)
    def test_translation_scale_refuses_it_too(self, value: Any) -> None:
        client: Any = _RecordingClient()
        policy = _policy(client)
        with pytest.raises(ValueError) as excinfo:
            policy.set_ik_target(object(), "hand", translation_scale=value)  # type: ignore[arg-type]
        assert "translation_scale" in str(excinfo.value)

    @pytest.mark.parametrize("value", USABLE_SCALES)
    def test_both_accept_the_same_usable_values(self, value: Any) -> None:
        assert _config(embodiment="mimicgen", motion_plan_scale=value).motion_plan_scale is not None
        from strands_robots.utils import positive_finite_number_error

        assert positive_finite_number_error(value, "translation_scale", "set_ik_target") is None
