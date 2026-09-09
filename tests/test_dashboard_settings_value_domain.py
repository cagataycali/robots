"""A settings value that is not usable is refused or degraded, never published.

``settings`` coerces on two paths. The strict one (UI / API writes, via
:func:`~strands_robots.dashboard.settings.update_strict`) REPORTS an unusable
value. The lenient one (the settings file, the environment, the CLI, via
:func:`~strands_robots.dashboard.settings.update` and
:func:`~strands_robots.dashboard.settings.load`) cannot report to anyone, so it
degrades - and the module states the rule that makes degrading safe: it must
degrade to *the key's own shape*, because "a list key that fell back to a scalar
poisons every comma-split consumer, which is worse than the empty default".

That rule was applied to the list keys and omitted for the one boolean key,
``runtime.trust_remote_code`` - whose consumer is the remote-code execution
gate. The lenient path returned the value that had failed
to be a boolean, so a non-boolean sat in a boolean slot, and
:func:`~strands_robots.dashboard.settings.apply_mesh_env` publishes that key by
TRUTHINESS. A spelling like ``"maybe"`` was therefore published as the literal
``"1"`` - which is exactly what
``policies.factory._check_trust_remote_code`` accepts, since its allowlist is
``("1", "true", "yes")``. The same spelling reaching that gate directly is
refused. So the settings layer converted a value the gate rejects into the one
it accepts, off a typo in ``settings.json``, while the API path refused the very
same value by name.

``test_an_unreadable_spelling_does_not_open_the_remote_code_gate`` is the cell
that discriminates: it drives the real gate and fails against the passthrough.
The rest pin the two paths' answers across the whole value domain as one table,
so a key added to the schema without a domain is visible, and the spellings that
legitimately mean true or false are pinned as controls - they pass either way,
which is what keeps the fix from reading as "refuse more things".

And it reached two of the four numeric keys, not four.
``TestANonFiniteNumberIsReportedNotRaised`` covers the half it missed: the two
whose shape is a float go through ``_finite_float``, which refuses a non-finite
by name, while the two whose shape is an integer converted with a bare ``int()``
whose ``except (TypeError, ValueError)`` does not catch the ``OverflowError``
that ``int(float("inf"))`` raises. So neither path answered - the strict one
raised instead of reporting the key, and the lenient one raised out of ``load``
itself, taking the whole tree with it rather than degrading one value.

Those rows are a class of their own rather than four more ``_UNUSABLE`` entries
because the lenient column there is resolved through the settings FILE, and a
file cannot carry a non-finite at all: ``json.dumps`` writes it as the bare
``Infinity`` token, which ``_read_file`` already refuses as corrupt (see
``test_a_value_the_file_cannot_hold_heals_to_the_default``), so those rows would
pass through the healing path without reaching the coercion under test. The
lenient vector that does reach it is ``override`` - a process-scoped value,
which is also what made the failure permanent for the life of the process.
The four keys are one roster in the module now, rather than a literal tuple per
path, and one cell here holds this table to it - so a numeric key added to the
schema is held to this domain rather than quietly escaping it.
"""

from __future__ import annotations

import decimal
import json

import pytest

from strands_robots.dashboard import settings

# (section, key, value, substring the strict path must report, lenient result).
# One row per unusable value; the lenient column is the key's own shape.
# Annotated because the shapes differ per row: the value column spans str / float
# / dict and the lenient column spans None / [] / False, so the bare ``[]`` leaves
# mypy with a partial type it cannot close.
_UNUSABLE: list[tuple[str, str, object, str, object]] = [
    ("agent", "temperature", "abc", "is not a number", None),
    ("agent", "temperature", 5.0, "outside 0..2", None),
    ("agent", "temperature", float("inf"), "is not a finite number", None),
    ("agent", "max_tokens", "x", "is not an integer", None),
    ("agent", "max_tokens", 0, "must be at least 1", None),
    ("mesh", "port", 70000, "outside 1..65535", None),
    ("mesh", "camera_hz", 0.0, "outside (0, 240]", None),
    ("mesh", "camera_hz", 500.0, "outside (0, 240]", None),
    ("mesh", "connect", 5, "expected a list or comma-separated string", []),
    ("security", "cors_origins", {"a": 1}, "expected a list", []),
    ("runtime", "trust_remote_code", "maybe", "is not a boolean", False),
    ("runtime", "trust_remote_code", {"a": 1}, "is not a boolean", False),
]

# Spellings that DO mean something. Controls: green on both trees.
_USABLE_BOOL = [("1", True), ("true", True), (1, True), ("false", False), ("off", False), ("", False), (0, False)]

# Every spelling of "not a finite number" that ``json.loads`` produces: it accepts
# the bare ``Infinity``/``-Infinity``/``NaN`` tokens, so a request body is one of
# the ways one of these arrives in a numeric slot.
_NON_FINITE = [float("inf"), float("-inf"), float("nan")]

# The numeric keys, by the shape their value takes. Spelled here rather than read
# from the module so the table collects against a tree that has not got the
# roster yet, and held to the module's own by
# ``test_the_modules_numeric_roster_is_the_one_this_table_covers`` - so a numeric
# key added to the schema fails that cell rather than quietly escaping the domain.
_INT = ("max_tokens", "port")
_NUMERIC = ("temperature", "camera_hz") + _INT

# Numbers each integer key must still take, so the refusal above cannot read as
# "refuse more things". Controls: green on both trees.
_USABLE_INT = [("max_tokens", 1, 1), ("max_tokens", "8", 8), ("port", 65535, 65535), ("port", 1.0, 1)]


def _section_of(key: str) -> str:
    """The schema section holding *key* - both coercion paths match numeric keys by name."""
    return next(section for section, keys in settings._SCHEMA.items() if key in keys)


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """Point the module at a scratch settings file, resolved from built-in defaults.

    Every environment variable the schema reads is dropped, so a value under test
    comes from the file and nothing else. ``apply_mesh_env`` writes to
    ``os.environ`` by design, and a leftover ``ZENOH_CONNECT`` would otherwise
    reach the next test as that key's default.
    """
    path = tmp_path / "settings.json"
    path.write_text("{}")
    monkeypatch.setattr(settings, "SETTINGS_FILE", path)
    for keys in settings._SCHEMA.values():
        for env_name, _default in keys.values():
            if env_name:
                monkeypatch.delenv(env_name, raising=False)
    settings.clear_overrides()
    settings.load(refresh=True)
    yield path
    settings.clear_overrides()
    settings.load(refresh=True)


def _from_file(path, section: str, key: str, value):
    """Write one raw value to the file and return what ``load()`` resolves."""
    path.write_text(json.dumps({section: {key: value}}))
    settings.clear_overrides()
    return settings.load(refresh=True)[section][key]


class TestTheRemoteCodeGateIsNotOpenedByAValueItWouldRefuse:
    def test_an_unreadable_spelling_does_not_open_the_remote_code_gate(self, store, monkeypatch):
        """The end-to-end consequence, driven through the real gate."""
        from strands_robots.policies.factory import UntrustedRemoteCodeError, _check_trust_remote_code

        gated = "kimodo"
        # Premise: with nothing set, the gate refuses. If this ever stops being
        # true the rest of the cell proves nothing.
        with pytest.raises(UntrustedRemoteCodeError):
            _check_trust_remote_code(gated)

        store.write_text(json.dumps({"runtime": {"trust_remote_code": "maybe"}}))
        settings.clear_overrides()
        settings.load(refresh=True)
        settings.apply_mesh_env()

        # Pre-fix this published "1" - the gate's own opt-in spelling - so the
        # gate opened off a value it refuses when handed it directly.
        with pytest.raises(UntrustedRemoteCodeError):
            _check_trust_remote_code(gated)

    def test_the_spelling_is_not_republished_as_the_gates_opt_in_token(self, store, monkeypatch):
        monkeypatch.setenv("STRANDS_TRUST_REMOTE_CODE", "maybe")
        settings.clear_overrides()
        settings.load(refresh=True)

        applied = settings.apply_mesh_env()

        assert "STRANDS_TRUST_REMOTE_CODE" not in applied, (
            f"an unreadable spelling was republished as {applied.get('STRANDS_TRUST_REMOTE_CODE')!r}"
        )

    @pytest.mark.parametrize(("spelled", "expected"), _USABLE_BOOL)
    def test_a_spelling_that_does_mean_something_still_means_it(self, store, spelled, expected):
        """Control: the readable spellings are unchanged in both directions."""
        assert _from_file(store, "runtime", "trust_remote_code", spelled) is expected

    def test_an_opted_in_setting_still_reaches_the_environment(self, store):
        _from_file(store, "runtime", "trust_remote_code", "true")
        assert settings.apply_mesh_env().get("STRANDS_TRUST_REMOTE_CODE") == "1"


class TestAnUnusableValueIsReportedOrDegradedToTheKeysShape:
    @pytest.mark.parametrize(("section", "key", "value", "reported", "_degraded"), _UNUSABLE)
    def test_the_strict_path_reports_it_and_stores_nothing(self, store, section, key, value, reported, _degraded):
        changed, errors = settings.update_strict({section: {key: value}})

        assert changed == []
        assert len(errors) == 1, errors
        assert errors[0].startswith(f"{section}.{key}:")
        assert reported in errors[0]
        assert json.loads(store.read_text()) == {}, "a value that was reported was also stored"

    @pytest.mark.parametrize(("section", "key", "value", "_reported", "degraded"), _UNUSABLE)
    def test_the_lenient_path_degrades_to_the_keys_own_shape(self, store, section, key, value, _reported, degraded):
        got = _from_file(store, section, key, value)

        assert got == degraded
        assert type(got) is type(degraded), f"{section}.{key} degraded to a {type(got).__name__}"


class TestTheStoreStillAnswersTheRestOfItsSurface:
    def test_a_value_the_file_cannot_hold_heals_to_the_default(self, store):
        """A non-finite literal is not JSON, so the file counts as corrupt."""
        store.write_text('{"agent": {"temperature": NaN}}')
        settings.clear_overrides()

        assert settings.load(refresh=True)["agent"]["temperature"] is None

    def test_unknown_keys_names_what_the_schema_does_not_know(self, store):
        assert settings.unknown_keys({"agent": {"model_id": "m", "nope": 1}, "bogus": {"x": 1}}) == [
            "agent.nope",
            "bogus.*",
        ]

    def test_get_answers_a_section_a_key_and_a_default(self, store):
        _from_file(store, "voice", "voice_name", "alloy")

        assert settings.get("voice", "voice_name") == "alloy"
        assert settings.get("voice")["voice_name"] == "alloy"
        assert settings.get("voice", "provider") == "openai"
        assert settings.get("nosuchsection", "k", "fallback") == "fallback"
        # An unset key is the default, not an empty string.
        assert settings.get("agent", "model_id", "fallback") == "fallback"

    def test_a_list_setting_reaches_the_environment_comma_joined(self, store):
        _from_file(store, "mesh", "connect", ["tcp/a:7447", "tcp/b:7447"])

        assert settings.apply_mesh_env()["ZENOH_CONNECT"] == "tcp/a:7447,tcp/b:7447"

    def test_a_section_the_file_spells_as_a_scalar_is_ignored(self, store):
        store.write_text(json.dumps({"mesh": "not-a-section", "voice": {"provider": "kept"}}))
        settings.clear_overrides()

        tree = settings.load(refresh=True)

        assert tree["voice"]["provider"] == "kept"
        assert tree["mesh"]["connect"] == []


class TestANonFiniteNumberIsReportedNotRaised:
    """A non-finite number in any numeric slot is one of the two paths' answers, not an escape."""

    def test_the_modules_numeric_roster_is_the_one_this_table_covers(self):
        """One owner for which keys are numbers: both coercion paths read this roster."""
        assert tuple(settings._NUMERIC_KEYS) == _NUMERIC
        assert tuple(settings._INT_KEYS) == _INT

    @pytest.mark.parametrize("key", _NUMERIC)
    @pytest.mark.parametrize("value", _NON_FINITE)
    def test_the_strict_path_reports_the_key_and_stores_nothing(self, store, key, value):
        section = _section_of(key)

        changed, errors = settings.update_strict({section: {key: value}})

        assert changed == []
        assert errors == [f"{section}.{key}: {value!r} is not a finite number"]
        assert json.loads(store.read_text()) == {}, "a value that was reported was also stored"

    @pytest.mark.parametrize("key", _NUMERIC)
    @pytest.mark.parametrize("value", _NON_FINITE)
    def test_the_lenient_path_degrades_that_one_key_and_leaves_the_tree_readable(self, store, key, value):
        """``override`` is the lenient vector: the file cannot carry a non-finite at all."""
        section = _section_of(key)
        settings.override(section, key, value)

        assert settings.load(refresh=True)[section][key] is None
        # The consequence of raising from `load` rather than degrading: one bad
        # value took every OTHER setting with it, for as long as it was set.
        assert settings.get("voice", "provider") == "openai"
        assert isinstance(settings.apply_mesh_env(), dict)

    @pytest.mark.parametrize("key", _INT)
    def test_an_infinity_that_is_not_a_float_instance_is_reported_too(self, store, key):
        """``Decimal("Infinity")`` is not a ``float``, and converts with the same OverflowError."""
        section = _section_of(key)
        value = decimal.Decimal("Infinity")

        _changed, errors = settings.update_strict({section: {key: value}})

        assert len(errors) == 1, errors
        assert "is not a finite number" in errors[0]

    @pytest.mark.parametrize(("key", "value", "expected"), _USABLE_INT)
    def test_a_number_the_key_can_hold_is_still_held(self, store, key, value, expected):
        """Control: the guard refuses non-finite values and nothing else."""
        section = _section_of(key)

        changed, errors = settings.update_strict({section: {key: value}})

        assert errors == []
        assert changed == [f"{section}.{key}"]
        assert settings.load(refresh=True)[section][key] == expected
