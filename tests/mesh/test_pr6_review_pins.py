"""R7 review-feedback regression pins (timing oracle).

Each test here pins an invariant flagged in PR-6 review threads. Keep
this file thin and citation-heavy: each test docstring should name the
exact thread it pins so a future reader can trace the fix to the
review without spelunking PR history.
"""

from __future__ import annotations

import pytest

from strands_robots.mesh import core


# ---------------------------------------------------------------------------
# Thread 4: _resume_lockout HMAC compare is constant-time independent of
# the byte length of the provided override code.
# ---------------------------------------------------------------------------


class TestResumeLockoutTimingOracleClosed:
    """The ``hmac.compare_digest`` call in ``_resume_lockout`` must run
    over fixed-length 32-byte sha256 digests so an attacker probing with
    varying-length override codes cannot use response time to learn
    ``len(STRANDS_MESH_OVERRIDE_CODE)``.

    Pre-fix: ``compare_digest(expected.encode() or b"\x00" * len(provided),
    provided.encode())`` left a residual ``len(expected) ==
    len(provided)`` oracle when ``expected`` was configured. CPython's
    ``hmac.compare_digest`` is documented constant-time only when both
    inputs share a length; on length mismatch it shortcircuits to
    ``False`` quickly.

    Post-fix: both inputs are sha256-hashed before the compare, so the
    compare always operates on 32-byte buffers regardless of input
    length or whether ``expected`` is configured at all.
    """

    @pytest.fixture
    def stub_mesh(self):
        import threading

        m = core.Mesh.__new__(core.Mesh)
        m.peer_id = "test-peer"
        m._estop_lockout = threading.Event()
        m._last_estop_ts = 0.0
        m.publish_safety_event = lambda **kw: None
        return m

    def test_compare_target_is_fixed_length_when_expected_unset(
        self, stub_mesh, monkeypatch
    ):
        """When ``STRANDS_MESH_OVERRIDE_CODE`` is unset the placeholder
        digest must still be 32 bytes (the sha256 of a fixed sentinel),
        matching the byte length of any real digest. This collapses the
        ``configured-vs-unconfigured`` oracle that the prior fix only
        partially closed.
        """
        monkeypatch.delenv("STRANDS_MESH_OVERRIDE_CODE", raising=False)
        stub_mesh._estop_lockout.set()

        # We do not expose the internal _EXPECTED_HASH; instead, assert
        # the source-level invariant: any call path through
        # _resume_lockout must always reject when expected is unset, and
        # the rejection must NOT depend on the provided length.
        result_short = stub_mesh._resume_lockout("ab")
        result_long = stub_mesh._resume_lockout("a" * 4096)
        assert result_short == {"status": "error", "error": "resume rejected"}
        assert result_long == {"status": "error", "error": "resume rejected"}
        # Lockout must not have cleared either way.
        assert stub_mesh._estop_lockout.is_set()

    def test_compare_runs_on_fixed_digest_length_regardless_of_provided_length(
        self, stub_mesh, monkeypatch
    ):
        """The compare must operate on 32-byte sha256 digests so probes
        of varying length cannot leak ``len(expected)``.

        We cannot directly observe ``hmac.compare_digest``'s internal
        length-mismatch shortcut, but we can verify that probes of
        wildly varying lengths against a configured but wrong code are
        all rejected uniformly, AND that the source code calls
        ``hashlib.sha256(...).digest()`` on both compare inputs (pinned
        in ``test_source_level_pre_hash_invariant``).
        """
        monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", "secret-32char-hex-1234567890abcd")
        stub_mesh._estop_lockout.set()

        # Length-varied probes -- all must reject (none equals the
        # configured code; the sha256 pre-hash ensures the compare runs
        # over 32 bytes regardless).
        for probe in ["", "a", "ab", "a" * 16, "a" * 32, "a" * 1024, "a" * 65536]:
            stub_mesh._estop_lockout.set()  # re-arm in case any path cleared it
            result = stub_mesh._resume_lockout(probe)
            assert result == {"status": "error", "error": "resume rejected"}, (
                f"length-{len(probe)} probe leaked through compare (expected reject)"
            )
            assert stub_mesh._estop_lockout.is_set(), (
                f"length-{len(probe)} probe cleared lockout (compare oracle leak)"
            )

    def test_correct_override_code_clears_lockout(self, stub_mesh, monkeypatch):
        """The pre-hash refactor must not break the happy path: a
        matching override code still clears the lockout. Without this
        the security-hardening could silently lock the resume path."""
        secret = "correct-override-code-32-chars-x"
        monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", secret)
        stub_mesh._estop_lockout.set()

        result = stub_mesh._resume_lockout(secret)
        assert result == {"status": "ok"}
        assert not stub_mesh._estop_lockout.is_set()

    def test_source_level_pre_hash_invariant(self):
        """Structural pin: the resume-compare path must hash both
        inputs to a fixed digest before calling
        ``hmac.compare_digest``. Pin via source-text inspection so an
        accidental revert (e.g. someone restoring the
        ``b"\x00" * len(provided)`` placeholder) trips this test
        regardless of runtime path coverage.
        """
        import inspect

        source = inspect.getsource(core.Mesh._resume_lockout)
        assert "hashlib.sha256(provided.encode()).digest()" in source, (
            "_resume_lockout must hash provided to a fixed-length digest "
            "before compare_digest -- the prior placeholder approach "
            "left a len(expected)-vs-len(provided) timing oracle."
        )
        assert "hashlib.sha256(expected.encode()).digest()" in source, (
            "_resume_lockout must hash expected when configured."
        )
        # The prior placeholder pattern must NOT reappear -- a
        # readback of the form ``b"\x00" * max(1, len(provided))``
        # would re-introduce the length oracle.
        assert 'b"\\x00" * max(1, len(provided))' not in source, (
            "the variable-length placeholder must not re-appear -- it "
            "leaks len(provided) via compare_digest's length-mismatch "
            "shortcut."
        )


# ---------------------------------------------------------------------------
# Thread 5: Mesh.publish must shadow SensorLoopsMixin.publish via MRO.
# ---------------------------------------------------------------------------


def test_mesh_publish_shadows_sensor_loops_mixin():
    """``Mesh.publish`` must shadow the ``SensorLoopsMixin.publish`` stub.

    Review feedback: the mixin's ``publish`` body raises
    ``NotImplementedError`` -- a deliberate replacement for the prior
    ``...`` no-effect statement (CodeQL #226). The contract is that
    ``Mesh`` itself defines a real ``publish`` so the stub is never
    reached at runtime.

    The contract chain ('``Mesh.publish`` shadows this stub via MRO')
    depends on every host class declaring ``class Mesh(SensorLoopsMixin)``
    AND defining its own ``publish``. A future refactor that inserts
    another mixin between them (e.g. one that also implements ``publish``
    but forwards differently), or removes ``Mesh.publish`` entirely,
    would silently fall through to ``NotImplementedError`` only at
    runtime when a sensor loop fires (POSE_HZ tick, IMU tick, etc.) --
    a latent fault that escapes import-time checks and unit tests of
    other paths.

    This test surfaces such a regression at collection time, so a
    subclass authoring error trips CI before any sensor loop runs in
    production. Per AGENTS.md > "Pin regression tests for reviewed
    fixes".
    """
    from strands_robots.mesh import sensors

    mesh_publish = core.Mesh.publish
    mixin_publish = sensors.SensorLoopsMixin.publish

    assert mesh_publish is not mixin_publish, (
        "Mesh.publish must override SensorLoopsMixin.publish; the mixin "
        "stub raises NotImplementedError and is never meant to execute. "
        "If this fires, either Mesh lost its publish definition or a "
        "mixin was reordered -- check the MRO."
    )

    # Belt-and-braces: Mesh's own ``__dict__`` must carry ``publish``,
    # not just inherit it from somewhere on the MRO. This catches the
    # subtler regression where someone deletes ``Mesh.publish`` and
    # accidentally relies on a different mixin's implementation.
    assert "publish" in core.Mesh.__dict__, (
        "Mesh.publish must be defined on Mesh itself, not inherited "
        "from the mixin."
    )
