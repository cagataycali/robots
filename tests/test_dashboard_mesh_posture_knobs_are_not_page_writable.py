"""Every mesh wire knob is refused on the page's env write, not two of them by name.

``STRANDS_MESH_LOCAL_DEV`` alone turns wire auth off: ``resolve_auth_mode`` defaults to
``none`` under it and skips the ``STRANDS_MESH_I_KNOW_THIS_IS_INSECURE`` second factor.
``STRANDS_MESH_MULTICAST=true`` reopens LAN scouting. Both sat in ``INTERESTING_ENV``,
which is splatted into ``ALLOWED_ENV_KEYS``, and neither matched the gate-bearing fence,
whose mesh prefixes (``STRANDS_MESH_AUTH``, ``STRANDS_MESH_MTLS``, ``STRANDS_MESH_INSECURE``)
named one real variable between them. So one ``POST /api/config`` wrote the downgrade to the
env file and to the live process, and the next mesh session came up with no mTLS and no ACL.

The fence now covers the whole ``STRANDS_MESH_`` vocabulary: every knob the wire layer reads
is set on the host, never from the page. The roster is derived here from the package's own
source rather than kept by hand, so the next knob cannot repeat the defect.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from strands_robots.dashboard import config_api

_PACKAGE = Path(config_api.__file__).resolve().parents[1]
_MESH_ENV_RX = re.compile(r"\bSTRANDS_MESH_[A-Z][A-Z0-9_]*[A-Z0-9]\b")

#: The two the finding named, plus the knobs whose value IS the wire posture.
POSTURE_KNOBS = (
    "STRANDS_MESH_LOCAL_DEV",
    "STRANDS_MESH_MULTICAST",
    "STRANDS_MESH_AUTH_MODE",
    "STRANDS_MESH_I_KNOW_THIS_IS_INSECURE",
    "STRANDS_MESH_ACCEPT_PERMISSIVE_ACL",
    "STRANDS_MESH_ACL_FILE",
    "STRANDS_MESH_DISABLE_CA_PIN",
    "STRANDS_MESH_CA_PINS",
    "STRANDS_MESH_OVERRIDE_CODE",
    "STRANDS_MESH_AUDIT_PSK",
    "STRANDS_MESH_TLS_KEY",
    "STRANDS_MESH_POLICY_TYPE_ALLOW",
    "STRANDS_MESH_POLICY_HOST_ALLOW",
    "STRANDS_MESH_HF_REPO_ALLOW",
)


def _mesh_env_names_the_package_reads() -> set[str]:
    names: set[str] = set()
    for path in _PACKAGE.rglob("*.py"):
        if "frontend" in path.parts:
            continue
        names.update(_MESH_ENV_RX.findall(path.read_text(encoding="utf-8")))
    return names


def test_the_scan_finds_the_knobs_known_to_exist():
    found = _mesh_env_names_the_package_reads()
    assert set(POSTURE_KNOBS) <= found, sorted(set(POSTURE_KNOBS) - found)


@pytest.mark.parametrize("key", POSTURE_KNOBS)
def test_a_posture_knob_is_gate_bearing(key: str):
    assert config_api.env_key_gate_bearing(key), key
    assert not config_api.env_key_allowed(key), key


@pytest.mark.parametrize("key", sorted(_mesh_env_names_the_package_reads()))
def test_every_mesh_env_name_in_the_package_is_refused_on_the_env_write(key: str):
    """Derived from the source: a knob the wire layer reads is not page-writable."""
    problem = config_api.env_entry_error(key, "1")
    assert problem is not None, key
    assert "not dashboard-managed" in problem, problem


@pytest.mark.parametrize("key", ("STRANDS_MESH_LOCAL_DEV", "STRANDS_MESH_MULTICAST"))
def test_the_downgrade_is_refused_before_anything_is_written(tmp_path, monkeypatch, key: str):
    monkeypatch.setattr(config_api, "ENV_FILE", tmp_path / ".env")
    monkeypatch.delenv(key, raising=False)
    result = config_api.apply({"env": {key: "1"}})
    assert result["env_written"] == []
    assert any(key in e for e in result["errors"]), result["errors"]
    assert not (tmp_path / ".env").exists(), "a refused write must not touch the file"
    assert key not in __import__("os").environ, "and must not reach the live process"


@pytest.mark.parametrize("key", ("STRANDS_MESH_LOCAL_DEV", "STRANDS_MESH_MULTICAST"))
def test_the_knob_stays_visible_but_read_only_in_the_env_view(tmp_path, monkeypatch, key: str):
    """The same treatment STRANDS_DASH_TASK_REQUIRES_CONFIRM already gets."""
    monkeypatch.setattr(config_api, "ENV_FILE", tmp_path / ".env")
    rows = {row["key"]: row for row in config_api.env_view()}
    assert key in rows, "the operator can still discover what the process reads"
    assert rows[key]["editable"] is False


def test_the_page_writable_keys_are_untouched():
    for key in ("OPENAI_API_KEY", "HF_TOKEN", "AWS_REGION", "VOICE_MODEL", "STRANDS_ROBOTS_VIDEO_ROOT"):
        assert config_api.env_entry_error(key, "x") is None, key
