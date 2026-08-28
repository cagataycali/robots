"""Regression tests for ``Robot(mesh_backend=...)`` / ``init_mesh(mesh_backend=...)``.

The ``STRANDS_MESH_BACKEND`` env var is the historical single-owner switch;
this test module locks in the new constructor-arg override contract added to
:mod:`strands_robots.mesh._backend_select`, :func:`strands_robots.mesh.init_mesh`
and :func:`strands_robots.robot.Robot` so that a caller can swap Zenoh/IoT/bridge
at the call site without touching process env.

Design notes -- what these tests are actually asserting:

* ``select_backend()`` reads the override first, env var second. An override
  survives an env-var typo (env-var typos fall back to zenoh with a report;
  overrides do not).
* ``push_backend_override(None)`` is a documented no-op -- passing ``None``
  from a caller who did not opt in must not overwrite an existing override.
* An unknown value passed to ``push_backend_override`` raises ``ValueError``
  at push time. Env-var typos fall back silently to zenoh by policy; a
  caller-side typo we can name at the call site is a caller mistake we
  refuse instead.
* The override is a ``ContextVar`` so concurrent constructions with
  different backends on different threads do not stamp over each other.
* ``init_mesh(..., mesh_backend=...)`` pushes the override for the duration
  of ``Mesh.start()`` (verified via a fake that captures ``select_backend()``
  inside its start hook) and clears it on both success and failure paths.
* ``Robot(..., mesh_backend=...)`` forwards to ``init_mesh`` -- verified via
  a monkey-patched ``init_mesh`` that captures its kwargs.
"""

from __future__ import annotations

import threading

import pytest

from strands_robots.mesh import _backend_select
from strands_robots.mesh._backend_select import (
    BACKEND_ENV_VAR,
    BACKENDS,
    DEFAULT_BACKEND,
    current_backend_override,
    pop_backend_override,
    push_backend_override,
    select_backend,
)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset env vars so tests start from the DEFAULT_BACKEND.

    The repo-wide ``tests/conftest.py`` sets ``STRANDS_MESH=false`` as a safety
    default so unit tests never accidentally open a real Zenoh session. This
    module explicitly monkeypatches Mesh to a test fake and needs mesh
    construction to run, so we lift that kill switch for the duration of each
    test here.
    """
    monkeypatch.delenv(BACKEND_ENV_VAR, raising=False)
    monkeypatch.delenv("STRANDS_MESH", raising=False)


class TestPushPopContract:
    """The ``push_backend_override`` / ``pop_backend_override`` primitive."""

    def test_push_valid_value_makes_select_return_it(self) -> None:
        token = push_backend_override("iot")
        try:
            assert select_backend() == "iot"
        finally:
            pop_backend_override(token)

    def test_pop_restores_previous_value(self) -> None:
        assert current_backend_override() is None
        token = push_backend_override("bridge")
        pop_backend_override(token)
        assert current_backend_override() is None

    def test_none_pushes_a_no_op_scope(self) -> None:
        # Passing ``None`` explicitly is a documented no-op: it opens a scope
        # in which the override is None. This must not clobber an outer
        # override -- pop from the inner scope restores the outer one.
        outer = push_backend_override("iot")
        try:
            inner = push_backend_override(None)
            try:
                # Inner scope has None override; falls back to env-var default.
                assert current_backend_override() is None
                assert select_backend() == DEFAULT_BACKEND
            finally:
                pop_backend_override(inner)
            # Outer override is back.
            assert current_backend_override() == "iot"
            assert select_backend() == "iot"
        finally:
            pop_backend_override(outer)

    def test_normalizes_case_and_whitespace(self) -> None:
        token = push_backend_override("  IoT  ")
        try:
            assert select_backend() == "iot"
        finally:
            pop_backend_override(token)

    @pytest.mark.parametrize("bad", ["iott", "kafka", "", "  "])
    def test_unknown_value_raises_immediately(self, bad: str) -> None:
        # Unlike env-var typos (which fall back to zenoh silently), a caller
        # mistake at push time is named -- ValueError with the invalid value.
        with pytest.raises(ValueError, match="Unknown mesh_backend"):
            push_backend_override(bad)


class TestSelectResolutionOrder:
    """Override wins over env; unset override falls through to env-var default."""

    @pytest.mark.parametrize("value", BACKENDS)
    def test_override_wins_over_env(
        self, value: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Env asks for a *different* value; the override wins for every valid
        # backend in the vocabulary.
        env_value = "zenoh" if value != "zenoh" else "iot"
        monkeypatch.setenv(BACKEND_ENV_VAR, env_value)
        token = push_backend_override(value)
        try:
            assert select_backend() == value
        finally:
            pop_backend_override(token)

    def test_override_wins_over_env_var_typo(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Env-var typos are silently normalized to DEFAULT_BACKEND; an
        # explicit override sails past that fallback so a caller who
        # specifies iot never gets accidentally-zenoh because the env is
        # misconfigured on the host.
        monkeypatch.setenv(BACKEND_ENV_VAR, "iott")
        token = push_backend_override("iot")
        try:
            assert select_backend() == "iot"
        finally:
            pop_backend_override(token)

    def test_no_override_falls_through_to_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(BACKEND_ENV_VAR, "iot")
        assert current_backend_override() is None
        assert select_backend() == "iot"

    def test_no_override_no_env_returns_default(self) -> None:
        assert current_backend_override() is None
        assert select_backend() == DEFAULT_BACKEND


class TestContextVarIsolation:
    """Concurrent overrides on different threads do not stamp over each other."""

    def test_threads_see_independent_overrides(self) -> None:
        seen: dict[str, str] = {}
        barrier = threading.Barrier(2)

        def worker(name: str, backend: str) -> None:
            token = push_backend_override(backend)
            try:
                # Sync so both threads have their override installed before
                # either reads back -- guarantees the assertion catches
                # leakage rather than serial happens-before ordering.
                barrier.wait(timeout=5.0)
                seen[name] = select_backend()
            finally:
                pop_backend_override(token)

        t1 = threading.Thread(target=worker, args=("t1", "iot"))
        t2 = threading.Thread(target=worker, args=("t2", "bridge"))
        t1.start()
        t2.start()
        t1.join(timeout=10.0)
        t2.join(timeout=10.0)

        assert seen == {"t1": "iot", "t2": "bridge"}
        # Main thread untouched.
        assert current_backend_override() is None


class TestInitMeshForwardsBackend:
    """``init_mesh(mesh_backend=...)`` installs the override across Mesh.start()."""

    def test_mesh_backend_visible_inside_mesh_start(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The override must be installed by the time ``Mesh.start`` runs."""
        captured: dict[str, str] = {}

        class _FakeMesh:
            def __init__(self, robot: object, peer_id: str, peer_type: str) -> None:
                # Constructor also runs under the override; capture here too
                # to prove installation is not deferred until start().
                captured["ctor"] = select_backend()
                self.robot = robot
                self.peer_id = peer_id
                self.peer_type = peer_type
                self.alive = False

            def start(self) -> None:
                captured["start"] = select_backend()

        # Patch the Mesh class the way ``init_mesh`` reaches it.
        from strands_robots.mesh import core as mesh_core

        monkeypatch.setattr(mesh_core, "Mesh", _FakeMesh)

        # Sanity: make sure the mesh kill switch is not tripped by
        # inherited env from the test process (STRANDS_MESH=false would
        # short-circuit init_mesh before our fake ever runs, and hide
        # the point of the assertion behind a mistaken "no override".)
        monkeypatch.delenv("STRANDS_MESH", raising=False)

        result = mesh_core.init_mesh(
            robot=object(),
            peer_id="test-peer",
            peer_type="sim",
            mesh=True,
            mesh_backend="iot",
        )

        assert captured == {"ctor": "iot", "start": "iot"}
        # Override cleared after the call regardless of success.
        assert current_backend_override() is None
        assert isinstance(result, _FakeMesh)

    def test_override_cleared_after_start_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Even if ``Mesh.start`` raises, the override must be popped."""

        class _ExplodingMesh:
            def __init__(self, robot: object, peer_id: str, peer_type: str) -> None:
                self.peer_id = peer_id
                self.alive = False

            def start(self) -> None:
                raise RuntimeError("boom")

        from strands_robots.mesh import core as mesh_core

        monkeypatch.setattr(mesh_core, "Mesh", _ExplodingMesh)

        with pytest.raises(RuntimeError, match="boom"):
            mesh_core.init_mesh(
                robot=object(),
                peer_id="test-peer",
                peer_type="sim",
                mesh=True,
                mesh_backend="bridge",
            )

        # The finally clause in init_mesh ran even though start() blew up.
        assert current_backend_override() is None

    def test_unset_mesh_backend_is_a_no_op(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``mesh_backend=None`` (the default) must not touch resolution."""
        seen: dict[str, str] = {}

        class _CaptureMesh:
            def __init__(self, robot: object, peer_id: str, peer_type: str) -> None:
                self.peer_id = peer_id
                self.alive = False

            def start(self) -> None:
                seen["backend"] = select_backend()

        from strands_robots.mesh import core as mesh_core

        monkeypatch.setattr(mesh_core, "Mesh", _CaptureMesh)
        monkeypatch.setenv(BACKEND_ENV_VAR, "iot")

        mesh_core.init_mesh(
            robot=object(),
            peer_id="test-peer",
            peer_type="sim",
            mesh=True,
            # mesh_backend not passed -- default is None -- env var wins.
        )

        assert seen["backend"] == "iot"

    def test_mesh_disabled_short_circuits_before_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When ``mesh=False`` there is nothing to override -- just return None."""
        from strands_robots.mesh import core as mesh_core

        # If mesh=False caused an override push/pop, an invalid value would
        # raise. Since mesh=False short-circuits, the invalid value never
        # even reaches push_backend_override.
        result = mesh_core.init_mesh(
            robot=object(),
            peer_id="test-peer",
            peer_type="sim",
            mesh=False,
            mesh_backend="not-a-valid-backend",
        )
        assert result is None
        assert current_backend_override() is None


class TestRobotForwardsBackend:
    """``Robot(mesh_backend=...)`` reaches ``init_mesh`` untouched."""

    def test_sim_path_forwards_mesh_backend(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, object] = {}

        def _fake_init_mesh(*args: object, **kwargs: object) -> None:
            captured.update(kwargs)
            return None

        # Patch where robot.py imports it (locally inside the sim branch),
        # so the ``from strands_robots.mesh import init_mesh`` reads our fake.
        import strands_robots.mesh as mesh_pkg

        monkeypatch.setattr(mesh_pkg, "init_mesh", _fake_init_mesh)

        # ``_attach_mesh`` is the seam we're testing -- it wraps ``init_mesh``
        # for the hardware branch and is also directly callable, so we exercise
        # it without needing to construct a real hardware driver.
        from strands_robots.robot import _attach_mesh

        _attach_mesh(
            instance=object(),
            canonical="so100",
            peer_id="test-peer",
            mesh=True,
            mesh_backend="iot",
        )

        assert captured["mesh_backend"] == "iot"
        assert captured["mesh"] is True
        assert captured["peer_id"] == "test-peer"
        assert captured["peer_type"] == "robot"

    def test_attach_mesh_default_arg_is_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Existing callers that don't pass ``mesh_backend`` keep working."""
        captured: dict[str, object] = {}

        def _fake_init_mesh(*args: object, **kwargs: object) -> None:
            captured.update(kwargs)
            return None

        import strands_robots.mesh as mesh_pkg

        monkeypatch.setattr(mesh_pkg, "init_mesh", _fake_init_mesh)

        from strands_robots.robot import _attach_mesh

        # Call the old four-arg positional shape to prove back-compat --
        # ``mesh_backend`` defaults to ``None`` and forwards as such.
        _attach_mesh(object(), "so100", "test-peer", True)

        assert captured["mesh_backend"] is None
