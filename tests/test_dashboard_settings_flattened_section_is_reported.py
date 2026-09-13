"""Every part of a settings patch that cannot be applied is reported to the caller.

``settings`` offers a caller two doors for grading a patch before or while it is
applied: :func:`~strands_robots.dashboard.settings.unknown_keys` reports NAMES
the schema does not know, and
:func:`~strands_robots.dashboard.settings.update_strict` reports VALUES it cannot
use. Between them they are meant to account for every part of a patch that does
not land, because the only other thing the caller can report is the changed list
- and an empty changed list with nothing else to say reads as a success.

One patch shape escaped both: a section the schema DOES know, given something
other than a mapping of its keys. ``{"security": "<token>"}`` instead of
``{"security": {"auth_token": "<token>"}}`` is the flattening a hand-written
request body or a config file makes, and it returned no unknown name, no error
and no changed key, and stored nothing - while ``{"bogus": "x"}``, an UNKNOWN
section flattened the same way, was already reported by name. So the store
distinguished the two cases and dropped the one whose section it recognised.

``security`` is the section that carries ``auth_token``, the bearer every
``/api`` and ``/ws`` request must present, so on that section the silence has a
direction: the operator is told the write went through, and the door stays open.

``TestASectionIsAccountedForByExactlyOneDoor`` holds the whole patch domain to
the invariant as one table, with the shapes that were already reported kept as
controls, so the fix cannot read as "refuse more things". The lenient path
(:func:`~strands_robots.dashboard.settings.update`) has no error channel to
report through, so it keeps skipping - but at WARNING, because there the report
the caller can make is a success.
"""

from __future__ import annotations

import logging

import pytest

from strands_robots.dashboard import settings

# (label, patch, unknown_keys(), substrings update_strict() must report, changed).
# One row per patch shape. The first four are known sections flattened to
# something that is not a mapping of their keys; the rest are the shapes that
# were already accounted for, kept as controls.
_PATCHES: list[tuple[str, dict, list[str], list[str], list[str]]] = [
    ("a known section flattened to a string", {"security": "tok"}, [], ["security: expected a mapping", "got str"], []),
    (
        "a known section flattened to a list",
        {"mesh": ["tcp/127.0.0.1:7447"]},
        [],
        ["mesh: expected a mapping", "got list"],
        [],
    ),
    ("a known section given null", {"agent": None}, [], ["agent: expected a mapping", "got NoneType"], []),
    ("a known section given a number", {"runtime": 1}, [], ["runtime: expected a mapping", "got int"], []),
    ("an unknown section", {"bogus": {"x": 1}}, ["bogus.*"], [], []),
    ("an unknown section, flattened", {"bogus": "x"}, ["bogus"], [], []),
    ("a known section, unknown key", {"security": {"nope": 1}}, ["security.nope"], [], []),
    (
        "a known key, unusable value",
        {"agent": {"temperature": "abc"}},
        [],
        ["agent.temperature", "is not a number"],
        [],
    ),
    ("a well-formed patch", {"security": {"auth_token": "T"}}, [], [], ["security.auth_token"]),
]


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """Point the module at a scratch settings file, resolved from built-in defaults.

    Every environment variable the schema reads is dropped, so what a patch does
    - or does not do - is visible against the defaults and nothing else.
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


class TestASectionIsAccountedForByExactlyOneDoor:
    @pytest.mark.parametrize("label,patch,unknown,reported,expect_changed", _PATCHES, ids=[r[0] for r in _PATCHES])
    def test_the_two_doors_account_for_the_patch(self, store, label, patch, unknown, reported, expect_changed):
        assert settings.unknown_keys(patch) == unknown
        changed, errors = settings.update_strict(patch)
        assert changed == expect_changed
        assert len(errors) == len(reported and [1] or [])
        for fragment in reported:
            assert fragment in errors[0], f"{label}: {errors[0]!r} does not name {fragment!r}"
        # The invariant: a patch entry that did not land is named by one of the
        # doors. Nothing may be dropped in silence.
        assert unknown or reported or expect_changed, f"{label}: accounted for by nothing"

    def test_a_flattened_section_stores_nothing(self, store):
        settings.update_strict({"security": "tok"})
        assert store.read_text() == "{}"


class TestTheBearerIsNotReportedAsStoredWhenItIsNot:
    def test_a_flattened_security_section_is_refused_not_silently_accepted(self, store):
        """The section carrying the ``/api`` bearer cannot go unset behind a success."""
        changed, errors = settings.update_strict({"security": "hunter2"})
        assert changed == []
        assert errors, "the token did not land and the caller was told nothing"
        assert settings.load()["security"]["auth_token"] is None


class TestTheLenientPathSaysSoRatherThanReportingASuccess:
    def test_update_logs_a_warning_naming_the_section(self, store, caplog):
        with caplog.at_level(logging.WARNING, logger="strands_robots.dashboard.settings"):
            assert settings.update({"security": "tok"}) == []
        said = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("security: expected a mapping" in line for line in said), said


class TestTheDoorsDoNotBothClaimIt:
    def test_unknown_keys_does_not_report_a_name_the_schema_knows(self, store):
        """The split is by name-versus-value, so neither door has to guess the other's answer."""
        assert settings.unknown_keys({"security": "tok"}) == []
        assert settings.unknown_keys({"bogus": "tok"}) == ["bogus"]
