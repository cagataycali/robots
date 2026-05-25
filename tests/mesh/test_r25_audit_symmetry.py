"""the prior audit-symmetry pins for this PR.

This file pins the four sites that the prior fix review (2026-05-25 06:42) flagged
as asymmetric with established discipline elsewhere in the mesh module.
Each test fails on pre-the prior fix code and passes on the prior commit.

1. ``security.py``: ``teleop_receive.source_peer_id`` lacked the charset /
   length contract that ``model_path`` already enforces (prior thread on
   ``security.py:551``). An authenticated peer could publish a
   ``teleop_receive`` cmd with arbitrary unicode / control characters /
   NULs / shell metacharacters in ``source_peer_id``.

2. ``core.py``: ``Mesh.start`` declare-subscribers cleanup used a bare
   ``except Exception`` (and a nested ``except Exception: pass``) on the
   partial-failure path, masking programmer errors against the prior
   wire-handler tuple discipline (prior thread on ``core.py:380``).

3. ``audit.py``: the seq lockfile open at line 366 lacked ``O_NOFOLLOW``,
   while the audit log itself, the seq sidecar, and the ACL loader all
   set it (prior thread on ``audit.py:366``). An attacker with write access
   to the audit dir could pre-create the lockfile as a symlink.

4. ``_zenoh_config.py`` docstring: the ``STRANDS_MESH_TLS_KEY`` env-var
   description promised ``mode 0o600`` without the Windows caveat. The
   loader silently skips the mode check on non-POSIX (prior thread on
   ``_zenoh_config.py:508``).
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path

import pytest

from strands_robots.mesh import _zenoh_config as zc
from strands_robots.mesh import audit as audit_mod
from strands_robots.mesh import core as core_mod
from strands_robots.mesh import security as sec

# === FIX 1: security.py source_peer_id charset / length =====================


class TestSourcePeerIdValidation:
    """teleop_receive.source_peer_id must enforce model_path-style contract."""

    def test_constants_exported(self):
        """The new contract surfaces ``MAX_PEER_ID_LEN`` for downstream callers."""
        assert hasattr(sec, "MAX_PEER_ID_LEN")
        assert isinstance(sec.MAX_PEER_ID_LEN, int)
        assert sec.MAX_PEER_ID_LEN >= 64
        # The peer_id charset is intentionally tighter than _MODEL_PATH_RE
        # (no '/'). Confirm the regex exists and is restrictive.
        assert sec._PEER_ID_RE.fullmatch("operator-7") is not None
        assert sec._PEER_ID_RE.fullmatch("robot.fleet-1_main") is not None

    def test_rejects_control_characters(self):
        """ASCII control characters (NUL, newline, tab) are rejected."""
        for bad in ("op\x00rator", "op\nrator", "op\trator", "op\rator"):
            with pytest.raises(sec.ValidationError, match="source_peer_id"):
                sec.validate_command({"action": "teleop_receive", "source_peer_id": bad})

    def test_rejects_shell_metacharacters(self):
        """Shell metacharacters that would be hostile in a downstream subprocess interpolation."""
        for bad in (
            "op;rm -rf /",
            "op|cat",
            "op`whoami`",
            "op$(id)",
            "op&bg",
            "op>file",
            "op<file",
            "op*",
        ):
            with pytest.raises(sec.ValidationError, match="source_peer_id"):
                sec.validate_command({"action": "teleop_receive", "source_peer_id": bad})

    def test_rejects_whitespace(self):
        """Internal whitespace (spaces, unicode whitespace) is rejected."""
        for bad in ("op rator", "op\u00a0rator", "op\u2003rator"):
            with pytest.raises(sec.ValidationError, match="source_peer_id"):
                sec.validate_command({"action": "teleop_receive", "source_peer_id": bad})

    def test_rejects_unicode_outside_ascii(self):
        """Non-ASCII unicode (e.g. RTL override, zero-width, emoji) is rejected."""
        for bad in ("op\u202erator", "op\u200brator", "op\U0001f600rator"):
            with pytest.raises(sec.ValidationError, match="source_peer_id"):
                sec.validate_command({"action": "teleop_receive", "source_peer_id": bad})

    def test_rejects_path_separators(self):
        """``/`` is forbidden in peer_ids -- they are not paths and a '/' is a wire red flag."""
        for bad in ("../etc/passwd", "robot/fleet", "/abs/path"):
            with pytest.raises(sec.ValidationError, match="source_peer_id"):
                sec.validate_command({"action": "teleop_receive", "source_peer_id": bad})

    def test_rejects_oversized(self):
        """Length cap mirrors model_path's DoS-bound discipline."""
        bad = "a" * (sec.MAX_PEER_ID_LEN + 1)
        with pytest.raises(sec.ValidationError, match="MAX_PEER_ID_LEN"):
            sec.validate_command({"action": "teleop_receive", "source_peer_id": bad})

    def test_accepts_valid_peer_ids(self):
        """The legitimate peer_id shape that ``Mesh.peer_id`` produces still passes."""
        for good in (
            "operator-7",
            "robot.fleet-1_main",
            "host-1234-abc",
            "a",
            "A1",
            "_underscore_only_",
            "a" * sec.MAX_PEER_ID_LEN,  # at the boundary
        ):
            out = sec.validate_command({"action": "teleop_receive", "source_peer_id": good})
            assert out["source_peer_id"] == good


# === FIX 2: core.py Mesh.start lifecycle bare except =======================


class TestLifecycleNarrowExcept:
    """Pin: ``Mesh.start`` cleanup uses (RuntimeError, OSError), not bare ``Exception``."""

    def test_no_bare_except_in_start_subscriber_cleanup(self):
        """Source-level pin: bare ``except Exception`` MUST NOT appear in ``Mesh.start``.

        We scan the AST of ``core.Mesh.start`` for any ``ExceptHandler``
        whose ``type`` is a single ``Name(id='Exception')``. The the prior wire-handler narrowing established the project standard; the
        lifecycle path now matches.
        """
        import ast

        source_path = Path(inspect.getfile(core_mod))
        tree = ast.parse(source_path.read_text(encoding="utf-8"))

        offenders: list[tuple[str, int]] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.AsyncFunctionDef) and not isinstance(node, ast.FunctionDef):
                continue
            if node.name != "start":
                continue
            for sub in ast.walk(node):
                if not isinstance(sub, ast.ExceptHandler):
                    continue
                etype = sub.type
                if isinstance(etype, ast.Name) and etype.id == "Exception":
                    offenders.append((node.name, sub.lineno))

        assert not offenders, (
            f"core.py::Mesh.start contains bare 'except Exception' at {offenders}; "
            "narrow to (RuntimeError, OSError) per R24-A wire-handler precedent"
        )

    def test_subscriber_cleanup_uses_runtime_or_os_error_tuple(self):
        """The cleanup tuple includes RuntimeError + OSError (Zenoh ZError + transport).

        We grep the source for the exact tuple shape used in the
        cleanup branch. If a future refactor reverts to bare Exception
        OR drops one of the two members, this pin fires.
        """
        source = Path(inspect.getfile(core_mod)).read_text(encoding="utf-8")
        # The cleanup handler must catch (RuntimeError, OSError).
        # We look for the canonical form that the prior fix lands.
        assert "except (RuntimeError, OSError) as exc:" in source, (
            "Mesh.start cleanup branch must catch (RuntimeError, OSError) -- "
            "ZError is a RuntimeError subclass; transport faults surface as OSError"
        )


# === FIX 3: audit.py seq lockfile O_NOFOLLOW ==============================


@pytest.mark.skipif(
    not hasattr(os, "symlink") or os.name == "nt",
    reason="symlink semantics differ on Windows; O_NOFOLLOW is 0 there",
)
class TestSeqLockfileNofollow:
    """Pin: ``_seq_flock`` opens the lockfile with O_NOFOLLOW.

    Pre-the prior fix the open used ``os.O_RDWR | os.O_CREAT`` only. A symlink
    swap at ``mesh_audit.seq.lock`` would have ``flock`` land on the
    target inode rather than fail closed.
    """

    def test_seq_flock_refuses_symlinked_lockfile(self, tmp_path, monkeypatch, caplog):
        """A symlink at the lockfile path must NOT be followed.

        Pre-fix: ``os.open`` returns the fd of the link target (e.g.
        a co-tenant file the attacker cannot otherwise touch).
        Post-fix: ``O_NOFOLLOW`` causes ELOOP and the helper logs a
        WARNING, then yields the unlocked branch (best-effort).
        """
        import logging

        monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
        # Pre-create the lockfile path as a symlink. Target is a file we
        # do NOT want flocked (a co-tenant marker).
        cotenant = tmp_path / "cotenant.txt"
        cotenant.write_text("not the lockfile\n")
        lockfile = audit_mod._seq_lockfile_path()
        lockfile.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(cotenant, lockfile)
        assert lockfile.is_symlink()

        with caplog.at_level(logging.WARNING, logger="strands_robots.mesh.audit"):
            with audit_mod._seq_flock():
                # The helper still yields (best-effort fallback to
                # in-process locking). The CONTRACT being pinned is that
                # it logs a WARNING and does NOT flock the symlink target.
                pass

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any(
            "symlink" in (r.message or "").lower() and "lockfile" in (r.message or "").lower() for r in warnings
        ), f"expected a WARNING line about the symlinked seq lockfile; got: {[r.message for r in warnings]}"

    def test_seq_flock_normal_path_still_locks(self, tmp_path, monkeypatch):
        """The non-symlink happy path still acquires the flock."""
        monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
        # No pre-existing lockfile -- normal flow. The context manager
        # creates one with O_RDWR | O_CREAT | O_NOFOLLOW and flocks it.
        with audit_mod._seq_flock():
            lockfile = audit_mod._seq_lockfile_path()
            assert lockfile.exists()
            assert not lockfile.is_symlink()


def test_seq_flock_source_uses_o_nofollow():
    """Source-level pin: the open call must include O_NOFOLLOW.

    Cheap and platform-agnostic. The constant evaluates to 0 on
    Windows so the runtime behaviour degrades gracefully there;
    what we pin here is the *source-level intent*, which the prior
    review specifically called out as missing (asymmetric with
    every other inode-write site in the same module).
    """
    source = Path(inspect.getfile(audit_mod)).read_text(encoding="utf-8")
    # Find the _seq_flock function block.
    assert "def _seq_flock" in source
    seq_block_start = source.index("def _seq_flock")
    # The next def or class is the end of the block.
    rest = source[seq_block_start:]
    next_def = min(
        (rest.find("\ndef ", 1) if rest.find("\ndef ", 1) > 0 else len(rest)),
        (rest.find("\nclass ", 1) if rest.find("\nclass ", 1) > 0 else len(rest)),
    )
    seq_block = rest[:next_def]
    assert "O_NOFOLLOW" in seq_block, (
        "_seq_flock body must reference O_NOFOLLOW symmetric with the audit log / sidecar / ACL loader open paths"
    )
    # And the open call itself must use it.
    assert "os.O_RDWR | os.O_CREAT | nofollow" in seq_block or "os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW" in seq_block, (
        "the os.open call in _seq_flock must include the O_NOFOLLOW flag"
    )


# === FIX 4: _zenoh_config.py docstring covers Windows mode-skip ============


def test_tls_key_docstring_documents_windows_caveat():
    """The STRANDS_MESH_TLS_KEY description must call out the Windows skip.

    Pre-the prior fix the docstring said only ``mode 0o600`` and the README env-var
    matrix had no caveat -- a Windows operator who reads the docs and
    ships a key thinks the mode check is enforcing something. It isn't.
    """
    doc = inspect.getmodule(zc).__doc__ or ""
    # The fix is one paragraph in the env-var section. We pin its
    # essential phrase rather than the literal wording so a future
    # rewrite can refine the docs without breaking the pin.
    assert "STRANDS_MESH_TLS_KEY" in doc
    # Windows / non-POSIX skip is documented.
    lower = doc.lower()
    assert "non-posix" in lower or "windows" in lower, (
        "_zenoh_config module docstring must document that the mode 0o600 check is skipped on non-POSIX hosts"
    )
    # And the recommended mitigation (NTFS ACLs) is named so the operator
    # is not left wondering what to do.
    assert "ntfs" in lower or "filesystem acl" in lower, (
        "_zenoh_config docstring must point Windows operators at NTFS ACLs (or equivalent) as the mode-gate replacement"
    )
