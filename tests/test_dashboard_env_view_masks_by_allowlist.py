"""The env view shows a value in full only for a key on a closed allowlist, and masks fully.

``env_view()`` decided by NAME: a key matching ``SECRET_RX`` (KEY, TOKEN, PASSWORD, ...)
was masked, every other key was returned verbatim, and the mask itself kept the first
three and last two characters. ``STRANDS_MESH_AUDIT_PSK``, the HMAC key that makes the
safety audit log tamper evident, matches none of the words, so ``GET /api/config`` handed
it back in clear text to any admitted session, ``editable: false`` beside it.

The read path now mirrors the write path: ``SHOWN_ENV_KEYS`` is the closed set whose
values are safe to display, everything else reports only whether it is set, and a masked
value carries no character of the secret. ``is_secret`` stays as the name based backstop
behind the allowlist, never as the sole decision.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from strands_robots.dashboard import config_api

PSK = "audit-psk-4f9c2b7e1d"


@pytest.fixture
def env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / ".env"
    monkeypatch.setattr(config_api, "ENV_FILE", path)
    for key in ("STRANDS_MESH_AUDIT_PSK", "OPENAI_API_KEY", "AWS_REGION", "MY_UNNAMED_THING"):
        monkeypatch.delenv(key, raising=False)
    return path


def _rows(env_file: Path, text: str) -> dict[str, dict]:
    env_file.write_text(text, encoding="utf-8")
    return {row["key"]: row for row in config_api.env_view()}


def test_a_secret_under_an_unrecognised_name_is_not_disclosed(env_file):
    rows = _rows(env_file, f"STRANDS_MESH_AUDIT_PSK={PSK}\n")
    row = rows["STRANDS_MESH_AUDIT_PSK"]
    assert row["set"] is True
    assert row["secret"] is True
    assert PSK not in row["value"]
    assert config_api.looks_masked(row["value"]), "the UI recognises an untouched mask"


def test_an_unknown_operator_key_defaults_to_hidden(env_file):
    rows = _rows(env_file, "MY_UNNAMED_THING=hunter2-but-longer\n")
    assert "hunter2" not in rows["MY_UNNAMED_THING"]["value"]
    assert rows["MY_UNNAMED_THING"]["secret"] is True


def test_a_masked_value_carries_no_character_of_the_secret(env_file):
    secret = "sk-live-0123456789abcdef"
    rows = _rows(env_file, f"OPENAI_API_KEY={secret}\n")
    shown = rows["OPENAI_API_KEY"]["value"]
    assert shown
    assert not any(ch in shown for ch in set(secret)), shown
    assert len(shown) != len(secret), "the length is not the label either"


def test_mask_reveals_neither_prefix_nor_suffix():
    assert "sk-" not in config_api.mask("sk-abcdefghijklmnop")
    assert not config_api.mask("sk-abcdefghijklmnop").endswith("op")
    assert config_api.mask("") == ""
    assert config_api.mask("abc") == config_api.mask("a much longer value with spaces")


def test_a_key_on_the_allowlist_is_shown_in_full(env_file):
    rows = _rows(env_file, "AWS_REGION=eu-west-1\n")
    assert rows["AWS_REGION"]["value"] == "eu-west-1"
    assert rows["AWS_REGION"]["secret"] is False


def test_the_allowlist_is_a_subset_of_the_keys_the_page_knows():
    known = set(config_api.ALLOWED_ENV_KEYS) | set(config_api.INTERESTING_ENV)
    assert set(config_api.SHOWN_ENV_KEYS) <= known, sorted(set(config_api.SHOWN_ENV_KEYS) - known)


def test_the_name_based_backstop_still_wins_over_the_allowlist():
    """A key that names a credential is masked even if someone adds it to the shown set."""
    for key in config_api.SHOWN_ENV_KEYS:
        assert not config_api.is_secret(key), f"{key} names a credential and cannot be shown"
    assert not config_api.is_displayable("OPENAI_API_KEY")
    assert not config_api.is_displayable("HF_TOKEN")
    assert not config_api.is_displayable("STRANDS_MESH_AUDIT_PSK")


def test_every_page_writable_key_has_a_decided_display(env_file):
    """Nothing the page can write is left to the name heuristic alone."""
    for key in sorted(config_api.ALLOWED_ENV_KEYS):
        assert key in config_api.SHOWN_ENV_KEYS or config_api.is_secret(key), key
