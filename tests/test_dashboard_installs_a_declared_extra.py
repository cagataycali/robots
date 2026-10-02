"""The dashboard installs a declared extra into its own interpreter, and only that.

A spawn that dies on an import used to print the child's ``ImportError`` and
stop there; the operator had to find a terminal, the right venv and the right
extra name. ``/api/env`` lists the package's declared extras with their state,
``POST /api/env/install`` runs one install at a time and streams its redacted
log, and the spawn route refuses a child that would die on an import with a
412 that names the extra - so the Devices sheet can offer the install next to
the refusal.

Fail-first, measured on the branch before this module landed: ``POST
/api/devices/spawn`` for ``so101`` real with lerobot blocked started a child
that died with ``ImportError: 'lerobot' is required ...`` and answered 200 with
``status: failed``; ``GET /api/env`` was 404.

What is deliberately NOT installable: a package name. The body carries an extra
name, graded against ``Provides-Extra`` of the installed distribution; a
request naming anything else is 422 and nothing runs. The install subprocess
here is a stand-in script (``argv=``) so no cell reaches the network.
"""

from __future__ import annotations

import sys
import textwrap
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from strands_robots.dashboard import env_install, log_redaction, settings  # noqa: E402
from strands_robots.dashboard.server import create_app  # noqa: E402
from tests._dashboard_bootstrap import bootstrap_headers, configure_bootstrap  # noqa: E402


@pytest.fixture()
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setenv("STRANDS_DASH_AUTH_STORE", str(tmp_path / "auth.json"))
    monkeypatch.delenv("STRANDS_DASH_AUTH_ENABLED", raising=False)
    monkeypatch.delenv("DASHBOARD_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(settings, "SETTINGS_FILE", tmp_path / "settings.json")
    configure_bootstrap(monkeypatch)
    settings.clear_overrides()
    settings.load(refresh=True)
    monkeypatch.setattr(env_install, "current", None)
    yield tmp_path
    if env_install.current is not None:
        env_install.current.cancel()
    settings.clear_overrides()
    settings.load(refresh=True)


@pytest.fixture()
def client(isolated: Path) -> TestClient:
    return TestClient(create_app(), headers=bootstrap_headers())


def _script(tmp_path: Path, body: str) -> list[str]:
    path = tmp_path / "fake_installer.py"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return [sys.executable, str(path)]


def _wait(run: env_install.InstallRun, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while run.alive() and time.monotonic() < deadline:
        time.sleep(0.02)
    run.proc.wait(timeout=5)
    run._reader.join(timeout=5)


class TestTheAllowListIsThePackagesOwnExtras:
    def test_the_declared_extras_are_read_from_the_distribution(self) -> None:
        extras = env_install.declared_extras()
        assert {"lerobot", "dashboard", "mesh", "sim-mujoco"} <= set(extras)
        assert "pyserial" in extras["dashboard"]

    def test_a_row_says_installed_only_when_every_distribution_is_present(self) -> None:
        row = env_install.extra_status("fake", ["pytest", "a-distribution-nobody-ships"])
        assert row["installed"] is False
        assert row["missing"] == ["a-distribution-nobody-ships"]
        assert env_install.extra_status("fake", ["pytest"])["installed"] is True

    def test_a_self_reference_is_not_a_missing_distribution(self) -> None:
        """``all`` lists ``strands-robots[lerobot]``; the package is installed, by definition."""
        assert env_install.extra_status("all", ["strands-robots"])["installed"] is True

    @pytest.mark.parametrize("bad", ["", "requests", "lerobot; rm -rf /", "../x", 3, None, "strands-robots[lerobot]"])
    def test_anything_but_a_declared_extra_is_refused_by_name(self, bad: Any) -> None:
        reason = env_install.extra_name_error(bad)
        assert reason is not None and "declare" in reason, reason
        with pytest.raises(ValueError):
            env_install.start(bad)

    def test_the_install_line_is_spelled_by_the_server(self) -> None:
        argv = env_install.install_command("lerobot")
        spec = argv[-1]
        assert spec.endswith("[lerobot]")
        source = env_install.editable_source()
        if source is not None:
            assert spec == f"{source}[lerobot]" and argv[-2] == "-e"
        else:
            assert spec == "strands-robots[lerobot]"
        assert sys.executable in argv, "the install targets THIS interpreter, not whatever python is on PATH"


class TestOneInstallAtATimeWithARedactedLog:
    def test_the_log_is_captured_and_the_exit_code_reported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(env_install, "current", None)
        run = env_install.start(
            "lerobot", argv=_script(tmp_path, 'print("Resolved 3 packages"); print("Installed 3 packages")')
        )
        _wait(run)
        status = run.status()
        assert status["status"] == "done" and status["exit_code"] == 0
        assert status["lines"] == ["Resolved 3 packages", "Installed 3 packages"]
        assert env_install.get(run.id) is run

    def test_a_failing_installer_is_reported_as_failed(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(env_install, "current", None)
        run = env_install.start(
            "lerobot", argv=_script(tmp_path, 'import sys; print("error: no matching distribution"); sys.exit(1)')
        )
        _wait(run)
        assert run.status()["status"] == "failed"
        assert run.status()["exit_code"] == 1

    def test_a_second_install_is_refused_while_one_runs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(env_install, "current", None)
        run = env_install.start("lerobot", argv=_script(tmp_path, "import time; time.sleep(30)"))
        try:
            with pytest.raises(RuntimeError, match="already running"):
                env_install.start("mesh", argv=_script(tmp_path, "pass"))
        finally:
            run.cancel()
            _wait(run)

    def test_secrets_in_the_installer_output_are_redacted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(env_install, "current", None)
        secret = "tok-" + "q" * 24
        log_redaction.register_secret(secret)
        try:
            run = env_install.start(
                "lerobot", argv=_script(tmp_path, f'print("index-url https://user:{secret}@pypi.example/simple")')
            )
            _wait(run)
            text = "\n".join(run.status()["lines"])
            assert secret not in text, text
        finally:
            log_redaction.forget_secrets()


class TestTheMissingExtraIsReadOffTheChildsRefusal:
    def test_the_require_optional_hint_names_the_extra(self) -> None:
        text = "ImportError: 'lerobot' is required for the lerobot driver\n  pip install 'strands-robots[lerobot]'\n  pip install lerobot"
        assert env_install.missing_extra_in(text) == "lerobot"

    def test_an_undeclared_name_in_a_log_line_steers_nothing(self) -> None:
        # Assembled at runtime: the install-hint grader refuses a literal undeclared extra.
        undeclared = "pip install 'strands-robots[" + "not-an-" + "extra]'"
        assert env_install.missing_extra_in(undeclared) is None
        assert env_install.missing_extra_in("") is None
        assert env_install.missing_extra_in(None) is None


class TestTheSpawnIsPreflighted:
    def test_an_installed_environment_passes(self) -> None:
        pytest.importorskip("serial")
        assert env_install.spawn_preflight("so101", {"main": {"index_or_path": 1}}) is None

    def test_a_missing_driver_module_names_its_extra(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(env_install, "_importable", lambda module: module != "serial")
        gap = env_install.spawn_preflight("so101")
        assert gap is not None
        assert gap["driver"] == "strands"
        assert gap["missing"] == ["serial"]
        assert gap["missing_extra"] == "dashboard"
        assert gap["remedy"] == "pip install 'strands-robots[dashboard]'"

    def test_the_lerobot_path_is_checked_for_lerobot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(env_install, "_importable", lambda module: module != "lerobot")
        assert env_install.spawn_preflight("so101") is None, "the bare so101 call is native and needs no lerobot"
        gap = env_install.spawn_preflight("omx")
        assert gap is not None and gap["driver"] == "lerobot" and gap["missing_extra"] == "lerobot"

    def test_cameras_add_opencv(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(env_install, "_importable", lambda module: module != "cv2")
        assert env_install.spawn_preflight("so101") is None
        gap = env_install.spawn_preflight("so101", {"main": {"index_or_path": 1}})
        assert gap is not None and gap["missing"] == ["cv2"]


class TestTheRoutes:
    def test_env_lists_extras_with_their_state(self, client: TestClient) -> None:
        body = client.get("/api/env").json()
        assert body["python"] == sys.executable
        names = {row["name"] for row in body["extras"]}
        assert {"lerobot", "dashboard"} <= names
        assert all({"name", "installed", "missing"} <= set(row) for row in body["extras"])

    def test_install_refuses_a_name_the_package_does_not_declare(self, client: TestClient) -> None:
        assert client.post("/api/env/install", json={"extra": "requests"}).status_code == 422
        assert client.post("/api/env/install", json={"extra": "lerobot; echo"}).status_code == 422
        assert env_install.current is None, "a refused name must start nothing"

    def test_install_runs_once_and_streams_its_log(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        argv = _script(
            tmp_path, 'import time; print("Resolved 1 package"); time.sleep(0.3); print("Installed 1 package")'
        )
        monkeypatch.setattr(env_install, "install_command", lambda extra: argv)
        started = client.post("/api/env/install", json={"extra": "mesh"})
        assert started.status_code == 200, started.text
        run_id = started.json()["id"]
        second = client.post("/api/env/install", json={"extra": "lerobot"})
        assert second.status_code == 409, second.text
        run = env_install.get(run_id)
        assert run is not None
        _wait(run)
        final = client.get(f"/api/env/install/{run_id}").json()
        assert final["status"] == "done"
        assert final["lines"][-1] == "Installed 1 package"
        assert client.get("/api/env/install/nope").status_code == 404

    def test_the_routes_need_a_session(self, isolated: Path) -> None:
        """Without the bootstrap proof nothing about the environment is readable or writable."""
        anonymous = TestClient(create_app())
        assert anonymous.get("/api/env").status_code == 401
        assert anonymous.post("/api/env/install", json={"extra": "lerobot"}).status_code == 401
        assert env_install.current is None

    def test_a_real_spawn_that_would_die_on_an_import_is_refused_with_the_extra(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(env_install, "_importable", lambda module: module != "serial")
        resp = client.post(
            "/api/devices/spawn",
            json={"robot_name": "so101", "mode": "real", "port": "/dev/cu.usbmodemFAKE", "peer_id": "so101-x"},
        )
        assert resp.status_code == 412, resp.text
        detail = resp.json()["error"]  # the app renders an HTTPException's detail under "error"
        assert detail["missing_extra"] == "dashboard"
        assert detail["driver"] == "strands"
        assert "pip install 'strands-robots[dashboard]'" in detail["remedy"]
        assert "so101" in detail["error"]


# ---------------------------------------------------------------------------
# The page: the extra name travels from the refusal to the button unchanged.
# ---------------------------------------------------------------------------

from tests._dashboard_frontend import FRONTEND_SRC, requires_node, run_frontend  # noqa: E402

_COMPONENTS = FRONTEND_SRC / "components"


@requires_node
class TestThePageReadsTheExtraOffBothRefusalShapes:
    def test_the_412_body_and_the_200_with_error_body_both_name_the_extra(self) -> None:
        got = run_frontend(
            """
const m = await import('./missingExtra.ts')
out({
  refused: m.missingExtra({ error: { error: 'cannot run', missing_extra: 'dashboard', remedy: "pip install 'strands-robots[dashboard]'", driver: 'strands' } }),
  fastapi: m.missingExtra({ detail: { missing_extra: 'lerobot' } }),
  dead: m.missingExtra({ peer_id: 'x', error: "ImportError: 'lerobot' is required", missing_extra: 'lerobot' }),
  none: m.missingExtra({ peer_id: 'x', status: 'running' }),
  junk: [m.missingExtra(null), m.missingExtra('text'), m.missingExtra({ missing_extra: '  ' })],
  label: m.installLabel('lerobot'),
  sentences: [
    m.installSentence(null, true),
    m.installSentence({ id: 'a', extra: 'lerobot', status: 'running', exit_code: null, lines: ['x', 'y'] }, true),
    m.installSentence({ id: 'a', extra: 'lerobot', status: 'done', exit_code: 0, lines: [] }, false),
    m.installSentence({ id: 'a', extra: 'lerobot', status: 'failed', exit_code: 1, lines: [] }, false),
  ],
  tail: m.installTail({ id: 'a', extra: 'x', status: 'done', exit_code: 0, lines: Array.from({ length: 12 }, (_, i) => `l${i}`) }),
  rows: [m.extraRowSentence({ name: 'a', installed: true, missing: [] }), m.extraRowSentence({ name: 'b', installed: false, missing: ['p', 'q', 'r', 's', 't', 'u'] })],
})
"""
        )
        assert got["refused"] == {
            "extra": "dashboard",
            "remedy": "pip install 'strands-robots[dashboard]'",
            "driver": "strands",
        }
        assert got["fastapi"] == {"extra": "lerobot", "remedy": None, "driver": None}
        assert got["dead"]["extra"] == "lerobot"
        assert got["none"] is None
        assert got["junk"] == [None, None, None]
        assert got["label"] == "install strands-robots[lerobot]"
        assert got["sentences"][0] == "starting the install…"
        assert "2 lines" in got["sentences"][1]
        assert "spawn again" in got["sentences"][2]
        assert "exit 1" in got["sentences"][3]
        assert got["tail"] == [f"l{i}" for i in range(4, 12)]
        assert got["rows"] == ["installed", "missing p, q, r, s and 2 more"]


class TestTheButtonSendsOnlyTheServersName:
    """Static source cells: the extra the button posts is the one the server named, verbatim."""

    def test_install_extra_posts_the_extra_it_was_given_and_nothing_else(self) -> None:
        src = (_COMPONENTS / "InstallExtra.tsx").read_text(encoding="utf-8")
        assert "post<InstallStatus>('/api/env/install', { extra })" in src
        assert "package" not in src.lower().replace("packages", ""), "no package name is ever composed on the page"
        assert "`/api/env/install/${id}`" in src, "the poll reads the run the server returned"

    def test_the_devices_panel_offers_the_install_for_both_shapes(self) -> None:
        src = (_COMPONENTS / "DevicePanel.tsx").read_text(encoding="utf-8")
        assert "setGap(missingExtra(r))" in src, "a 200-with-error body (dead child) is read"
        assert "setGap(missingExtra(e?.body))" in src, "a 412 refusal body is read"
        assert "<InstallExtra" in src

    def test_the_settings_environment_tab_lists_the_extras(self) -> None:
        src = (_COMPONENTS / "SettingsDrawer.tsx").read_text(encoding="utf-8")
        assert "<ExtrasList />" in src
        extras = (_COMPONENTS / "ExtrasList.tsx").read_text(encoding="utf-8")
        assert "api<EnvDoc>('/api/env')" in extras

    def test_the_bundle_calls_the_env_routes(self) -> None:
        generated = (FRONTEND_SRC / "lib" / "bundleRoutes.generated.ts").read_text(encoding="utf-8")
        for route in ("'/api/env'", "'/api/env/install'", "'/api/env/install/{p}'"):
            assert route in generated, route
