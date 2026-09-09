"""One setting, two sources: the ``allow_insecure`` resolver must not disagree with itself.

:func:`~strands_robots.device_connect.resolve_allow_insecure` resolves a single
security posture from two sources, and its docstring documents the argument as
outranking the ``DEVICE_CONNECT_ALLOW_INSECURE`` environment variable. The two
sources carry that setting in different shapes, so each is held to its own
declared type: an environment variable is a string by construction and is
*parsed*, while the argument is declared ``bool | None`` and is *checked*.

Returning the argument as given made the two sources disagree about the same
value. A non-empty string is truthy, so ``"false"`` supplied as the argument
enabled insecure transport while ``DEVICE_CONNECT_ALLOW_INSECURE=false`` disabled
it - the inversion landing only on the path documented as the higher precedence,
and only for the spellings that mean *off*.

The existing hardening pins are what made that invisible: every case there that
passes a string passes it as ``env_value``, and every case that passes the
argument passes a genuine ``bool``. The string vocabulary was enumerated for one
source and never for the other.

``TestParsingTheArgumentWasRejectedForAMeasuredReason`` records why the argument
is refused rather than parsed with the environment vocabulary, so that choice is
measured here rather than restated as a preference.

Every reference to the package goes through :func:`_device_connect`, which
imports it inside the call as the sibling suites do. Importing it at module scope
would bind the real ``device_connect_edge`` submodules during collection, before
the modules that substitute them for their own use can - and this module sorts
ahead of those, so a module-scope import breaks them.
"""

import asyncio
from typing import Any

import numpy as np
import pytest

#: String spellings of a boolean that an operator, a config file or a CLI could
#: hand to either source, split by what the environment vocabulary makes them
#: mean - because that is the posture the argument path has to agree with.
OPT_IN_SPELLINGS = ("true", "1", "yes")
OFF_SPELLINGS = ("false", "False", "FALSE", "no", "0", "off")
UNRECOGNISED_SPELLINGS = ("on", "enabled", "y", "banana", "")

#: Every non-boolean the argument path can receive. ``None`` is excluded: it is
#: the documented "fall through to the environment" sentinel, not a bad value.
NOT_A_BOOLEAN: tuple[Any, ...] = (
    *OPT_IN_SPELLINGS,
    *OFF_SPELLINGS,
    *UNRECOGNISED_SPELLINGS,
    0,
    1,
    0.0,
    1.0,
    [],
    [0],
    {},
    {"insecure": True},
    object(),
)


def _device_connect() -> Any:
    """Import the integration package inside the call rather than at module scope.

    See this module's docstring: a module-scope import binds the real transport
    submodules during collection and breaks the suites that substitute them.
    """
    import strands_robots.device_connect as module

    return module


def _vocabulary() -> tuple[str, ...]:
    """The opt-in spellings, read from the module that owns them.

    They moved out of ``_impl`` so the stdlib-only authorization module can share
    them rather than keep a second copy - see
    ``tests/test_insecure_transport_posture_has_one_owner``. Imported inside the
    call for the same reason as :func:`_device_connect`.
    """
    from strands_robots.device_connect._authz import INSECURE_TRUE

    return INSECURE_TRUE


def _resolve(explicit: Any = None, env_value: Any = None) -> Any:
    """Call the resolver with values outside its declared parameter types.

    One funnel so the deliberately off-type arguments below need no per-call type
    suppression: the point of every case here is what the runtime does with a
    value a caller really can supply, not what a checker accepts.
    """
    return _device_connect().resolve_allow_insecure(explicit, env_value)


class TestTheTwoSourcesNeverDisagreeAboutOneValue:
    """The headline invariant, stated over the spellings rather than the types.

    For any spelling of the setting, the argument path must not resolve to a
    posture the environment path would not give it. It may refuse - a string is
    not a declared boolean - but it must never silently invert.
    """

    @pytest.mark.parametrize("spelling", OFF_SPELLINGS + OPT_IN_SPELLINGS + UNRECOGNISED_SPELLINGS)
    def test_no_spelling_resolves_to_opposite_postures(self, spelling: str) -> None:
        env_posture = _resolve(None, spelling)
        assert isinstance(env_posture, bool)
        try:
            arg_posture = _resolve(spelling)
        except ValueError:
            return  # refused, so there is no second posture to disagree with
        assert bool(arg_posture) == env_posture, (
            f"{spelling!r} resolves to {'insecure' if arg_posture else 'secure'} as the argument "
            f"and {'insecure' if env_posture else 'secure'} as the environment variable"
        )

    @pytest.mark.parametrize("spelling", OFF_SPELLINGS)
    def test_an_off_spelling_never_enables_insecure_transport(self, spelling: str) -> None:
        """The severe half: a caller writing *off* must not get *on*."""
        assert _resolve(None, spelling) is False
        with pytest.raises(ValueError, match="allow_insecure must be a bool or None"):
            _resolve(spelling)


class TestTheArgumentIsCheckedAgainstItsDeclaredType:
    """A value outside ``bool | None`` is refused, and the refusal is actionable."""

    @pytest.mark.parametrize("value", NOT_A_BOOLEAN)
    def test_a_non_boolean_argument_is_refused(self, value: Any) -> None:
        with pytest.raises(ValueError) as excinfo:
            _resolve(value)
        assert "allow_insecure" in str(excinfo.value)

    def test_the_refusal_names_the_value_and_the_source_that_does_parse_strings(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            _resolve("false")
        message = str(excinfo.value)
        assert "bool or None" in message
        assert "'false'" in message
        assert "DEVICE_CONNECT_ALLOW_INSECURE" in message
        for spelling in OPT_IN_SPELLINGS:
            assert spelling in message


class TestABooleanIsHonoredAndNormalized:
    """Every boolean is accepted, and the declared ``bool`` return is a real one."""

    def test_the_documented_arguments_are_unchanged(self) -> None:
        assert _resolve(None, None) is False
        assert _resolve(True, "false") is True
        assert _resolve(False, "true") is False

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(np.True_, True), (np.False_, False), (np.array(True), True), (np.array(False), False)],
        ids=["np.True_", "np.False_", "np.array(True)", "np.array(False)"],
    )
    def test_a_numpy_boolean_is_normalized_to_a_real_bool(self, value: Any, expected: bool) -> None:
        """A comparison result is a boolean the annotation should accept.

        It is returned as a real ``bool`` so the identity assertions the runtime's
        setting is pinned with keep holding for it.
        """
        assert _resolve(value) is expected


class TestTheEnvironmentPathStillParses:
    """The parsed source is unchanged, and its own declared type is enforced too."""

    @pytest.mark.parametrize("spelling", OPT_IN_SPELLINGS)
    def test_an_opt_in_spelling_enables_insecure_transport(self, spelling: str) -> None:
        assert _resolve(None, spelling) is True

    @pytest.mark.parametrize("spelling", OFF_SPELLINGS + UNRECOGNISED_SPELLINGS)
    def test_every_other_spelling_is_secure(self, spelling: str) -> None:
        assert _resolve(None, spelling) is False

    @pytest.mark.parametrize("value", [123, 1.0, True, False, ["true"], {}], ids=repr)
    def test_a_non_string_env_value_is_refused(self, value: Any) -> None:
        """It carries a raw environment value, which is a string by construction.

        Refused rather than left to raise ``AttributeError`` from ``.lower()``, so
        the whole declared ``-> bool`` return is total.
        """
        with pytest.raises(ValueError, match="env_value must be a str or None"):
            _resolve(None, value)


class TestParsingTheArgumentWasRejectedForAMeasuredReason:
    """Why the argument is checked rather than parsed with the same vocabulary.

    Parsing would move which spellings invert rather than remove the inversion:
    the environment vocabulary has no entry for several natural opt-in spellings,
    so each would resolve to secure while reading as an opt-in.
    """

    @pytest.mark.parametrize("spelling", ["on", "enabled", "y"])
    def test_an_opt_in_spelling_outside_the_vocabulary_would_read_as_secure(self, spelling: str) -> None:
        assert spelling not in _vocabulary()
        assert _resolve(None, spelling) is False

    def test_the_vocabulary_is_the_one_the_refusal_quotes(self) -> None:
        # Spelled once, in the stdlib-only ``_authz``, and imported here - the
        # authorizer's own fallback needs the same answer and cannot import this
        # module. See ``tests/test_insecure_transport_posture_has_one_owner``.
        assert _vocabulary() == OPT_IN_SPELLINGS


class TestTheRefusalPrecedesTheRuntime:
    """A refused setting must not reach the transport, or warn about a posture."""

    def test_a_string_argument_never_constructs_a_runtime(self, monkeypatch, caplog) -> None:
        import logging
        from unittest.mock import patch

        monkeypatch.delenv("DEVICE_CONNECT_ALLOW_INSECURE", raising=False)
        module = _device_connect()
        built: list[dict[str, Any]] = []

        class _FatalRuntime:
            def __init__(self, **kwargs: Any) -> None:
                built.append(kwargs)
                raise AssertionError("the refused setting reached the transport")

        class _Robot:
            tool_name_str = "so100"

        async def _go() -> None:
            with patch.object(module, "DeviceRuntime", _FatalRuntime):
                await module.init_device_connect(_Robot(), peer_id="p1", allow_insecure="false")

        with caplog.at_level(logging.WARNING, logger="strands_robots.device_connect"):
            with pytest.raises(ValueError, match="allow_insecure must be a bool or None"):
                asyncio.run(_go())

        assert built == []
        assert [r for r in caplog.records if "INSECURE mode" in r.getMessage()] == []
