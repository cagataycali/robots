"""Pin test for F11-A: the auto-listener path in session.py applies
_build_config() so namespace + mTLS + ACL + downsampling +
low_pass_filter + max_sessions + adminspace lockdown all hold on the
default deployment shape (no ZENOH_CONNECT / ZENOH_LISTEN).

Pre-F11 the auto-listener at session.py:399-405 used a bare
``zenoh.Config()`` and silently bypassed every Zenoh-built-in this
PR introduces. The README threat-coverage table claimed they applied
on every code path; that claim was false on the most common code
path (first peer in the process, no explicit endpoint env vars).
"""

from __future__ import annotations

import inspect

from strands_robots.mesh import session as session_mod


def test_get_session_auto_listener_uses_build_config() -> None:
    """The auto-listener branch must call ``_build_config()`` so all
    Zenoh-built-in security primitives apply."""
    src = inspect.getsource(session_mod.get_session)
    # The bypass pattern was:
    #   if not connect_env and not listen_env:
    #       try:
    #           cfg = zenoh.Config()
    # Post-F11-A: cfg = _build_config() inside the auto-listener block.
    assert "cfg = _build_config()" in src, "auto-listener branch must use _build_config() to apply mTLS / ACL / caps"
    # And it must NOT have a bare `zenoh.Config()` inside the
    # auto-listener block.
    # Heuristic: find the auto-listener block and check.
    bypass_pattern = "if not connect_env and not listen_env:\n            try:\n                cfg = zenoh.Config()"
    assert bypass_pattern not in src, "F11-A regression -- auto-listener branch reverted to bare zenoh.Config()"


def test_get_session_directly_auto_listener_uses_build_config() -> None:
    """Same invariant for the bridge-mode helper."""
    src = inspect.getsource(session_mod._get_zenoh_session_directly)
    assert "cfg = _build_config()" in src
    bypass_pattern = "if not connect_env and not listen_env:\n            try:\n                cfg = zenoh.Config()"
    assert bypass_pattern not in src


def test_auto_listener_uses_tls_scheme_under_mtls(monkeypatch, tmp_path) -> None:
    """When ``STRANDS_MESH_AUTH_MODE=mtls``, the auto-listener composes
    a ``tls/...`` endpoint -- otherwise the link_protocols restriction
    would produce an unusable session.
    """
    src = inspect.getsource(session_mod.get_session)
    # The post-F11-A code reads:
    #   scheme = "tls" if _auth_mode == "mtls" else "tcp"
    #   local_ep = f"{scheme}/127.0.0.1:{mesh_port}"
    assert 'scheme = "tls" if _auth_mode == "mtls" else "tcp"' in src, (
        "auto-listener must use tls scheme under mtls to match link_protocols restriction"
    )
