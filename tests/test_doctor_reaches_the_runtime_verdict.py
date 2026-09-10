"""``strands-robots doctor`` must reach the verdict the runtime reaches.

Three rows were measured passing while the runtime refused or blew up on the
same configuration (deep-dive D-079):

* ``import lerobot`` passed with a torchcodec that could not load, so the first
  ``LeRobotDataset`` printed ~100 traceback lines and fell back to pyav.
* no row said what ``Robot(...).run()`` would do about TLS on device-connect.
* ``PASS zenoh available`` printed while ``Robot(mesh=True)`` logged
  "Mesh did NOT start" and handed back a session that never opened.

Each probe here is graded against the runtime's own decision function or text,
so a PASS cannot outlive a refusal. The table itself is graded too: every row
must name a probe that exists, and replacing a probe by name must change the
run - the existing ``run_doctor`` tests rely on exactly that.
"""

from __future__ import annotations

import sys
import types
from importlib.metadata import PackageNotFoundError

import pytest

from strands_robots import doctor


@pytest.fixture(autouse=True)
def _plain_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor, "_NO_COLOR", True)


class TestTheTable:
    def test_every_row_names_a_probe_that_exists(self) -> None:
        for label, probe in doctor.CHECKS:
            assert callable(getattr(doctor, probe)), f"{label!r} names {probe!r}, which the module does not define"

    def test_the_three_measured_false_passes_have_rows(self) -> None:
        probes = {probe for _, probe in doctor.CHECKS}
        assert {"check_torchcodec_abi", "check_device_connect", "check_mesh"} <= probes

    def test_a_probe_replaced_by_name_is_the_one_that_runs(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The table resolves names at run time, so tests and operators can swap a row."""
        for _, probe in doctor.CHECKS:
            monkeypatch.setattr(doctor, probe, lambda p=probe: doctor._pass(f"stub {p}"))
        assert doctor.run_doctor() == 0
        out = capsys.readouterr().out
        assert all(f"stub {probe}" in out for _, probe in doctor.CHECKS)


class TestTorchcodecAbi:
    @staticmethod
    def _versions(monkeypatch: pytest.MonkeyPatch, present: dict[str, str]) -> None:
        import importlib.metadata as md

        def version(dist: str) -> str:
            if dist in present:
                return present[dist]
            raise PackageNotFoundError(dist)

        monkeypatch.setattr(md, "version", version)

    @staticmethod
    def _native_import(monkeypatch: pytest.MonkeyPatch, outcome: BaseException | None) -> None:
        real = doctor.importlib.import_module

        def import_module(name: str, package: str | None = None):
            if name == "torchcodec._core.ops":
                if outcome is not None:
                    raise outcome
                return types.ModuleType(name)
            return real(name, package)

        monkeypatch.setattr(doctor.importlib, "import_module", import_module)

    def test_skips_naming_the_absent_wheel(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._versions(monkeypatch, {"torch": "2.11.0"})
        result = doctor.check_torchcodec_abi()
        assert "  SKIP  " in result
        assert "torchcodec not installed" in result

    def test_passes_when_the_native_ops_load(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._versions(monkeypatch, {"torch": "2.11.0", "torchcodec": "0.11.1"})
        self._native_import(monkeypatch, None)
        result = doctor.check_torchcodec_abi()
        assert "  PASS  " in result
        assert "torchcodec 0.11.1 / torch 2.11.0" in result

    def test_an_abi_mismatch_fails_with_the_loader_line_and_the_pair(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The measured failure: uv.lock's torchcodec 0.10.0 next to torch 2.11.0."""
        self._versions(monkeypatch, {"torch": "2.11.0", "torchcodec": "0.10.0"})
        text = (
            "Could not load libtorchcodec. Likely causes:\n"
            "          1. FFmpeg is not properly installed ...\n"
            "OSError: dlopen(/venv/site-packages/torchcodec/libtorchcodec_core7.dylib, 0x0006): "
            "Symbol not found: __ZN3c1013MessageLoggerC1EPKciib\n"
        )
        self._native_import(monkeypatch, RuntimeError(text))
        result = doctor.check_torchcodec_abi()
        assert "  FAIL  " in result
        assert "torchcodec 0.10.0 / torch 2.11.0" in result
        assert "Symbol not found: __ZN3c1013MessageLoggerC1EPKciib" in result
        assert "Likely causes" not in result, "the row prints the loader's reason, not torchcodec's preamble"
        assert "different torch" in result

    def test_missing_ffmpeg_gets_the_ffmpeg_remedy_not_the_abi_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._versions(monkeypatch, {"torch": "2.11.0", "torchcodec": "0.11.1"})
        text = "Could not load libtorchcodec.\nOSError: dlopen(x.dylib, 6): Library not loaded: @rpath/libavutil.60.dylib\n"
        self._native_import(monkeypatch, RuntimeError(text))
        monkeypatch.setattr(sys, "platform", "linux")
        result = doctor.check_torchcodec_abi()
        assert "  FAIL  " in result
        assert "ffmpeg shared libraries not found" in result
        assert "Library not loaded: @rpath/libavutil.60.dylib" in result
        assert "different torch" not in result
        assert "ffmpeg" in result.split("Fix:")[1]

    def test_on_macos_with_homebrew_ffmpeg_the_remedy_is_the_dyld_export(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from strands_robots import _dyld

        self._versions(monkeypatch, {"torch": "2.11.0", "torchcodec": "0.11.1"})
        self._native_import(monkeypatch, OSError("Library not loaded: @rpath/libavutil.60.dylib"))
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(_dyld, "_find_ffmpeg_lib_dir", lambda: "/opt/homebrew/lib")
        result = doctor.check_torchcodec_abi()
        assert f"export {_dyld._DYLD_VAR}=/opt/homebrew/lib" in result

    def test_the_probe_never_decodes_video(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Only the native-ops import is attempted: no file, no fixture, no network."""
        self._versions(monkeypatch, {"torch": "2.11.0", "torchcodec": "0.11.1"})
        imported: list[str] = []
        real = doctor.importlib.import_module

        def import_module(name: str, package: str | None = None):
            imported.append(name)
            return types.ModuleType(name) if name.startswith("torchcodec") else real(name, package)

        monkeypatch.setattr(doctor.importlib, "import_module", import_module)
        doctor.check_torchcodec_abi()
        assert imported == ["torchcodec._core.ops"]


class TestDeviceConnectPosture:
    """Each branch mirrors one outcome of ``Robot(...).run()``; the row calls the
    runtime's own decision functions rather than re-deriving them."""

    @staticmethod
    def _impl(monkeypatch: pytest.MonkeyPatch, *, authenticated: bool) -> types.ModuleType:
        from strands_robots.device_connect._authz import insecure_env_opts_in

        mod = types.ModuleType("strands_robots.device_connect._impl")
        mod.resolve_allow_insecure = lambda explicit, env_value: insecure_env_opts_in(env_value)  # type: ignore[attr-defined]
        mod._TLS_ENV = ("MESSAGING_CREDENTIALS_FILE",)  # type: ignore[attr-defined]
        calls: list[tuple[str, object]] = []
        mod.calls = calls  # type: ignore[attr-defined]

        def transport_is_authenticated(backend: str, urls, env=None) -> bool:
            calls.append((backend, urls))
            return authenticated

        mod.transport_is_authenticated = transport_is_authenticated  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "strands_robots.device_connect._impl", mod)
        return mod

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in (
            "DEVICE_CONNECT_ALLOW_INSECURE",
            "DEVICE_CONNECT_RPC_ALLOW",
            "MESSAGING_BACKEND",
            "MESSAGING_CREDENTIALS_FILE",
        ):
            monkeypatch.delenv(name, raising=False)

    def test_skips_with_the_install_line_when_the_extra_is_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "strands_robots.device_connect._impl", None)
        result = doctor.check_device_connect()
        assert "  SKIP  " in result
        assert "strands-robots[device-connect]" in result

    def test_the_names_it_reads_are_the_ones_the_runtime_defines(self) -> None:
        """No hedged ``getattr`` default: the row calls the shipped functions by name."""
        from strands_robots.device_connect import _impl

        for name in ("resolve_allow_insecure", "transport_is_authenticated", "_TLS_ENV"):
            assert hasattr(_impl, name), name

    def test_authenticated_transport_passes_naming_the_credentials_variable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mod = self._impl(monkeypatch, authenticated=True)
        monkeypatch.setenv("MESSAGING_CREDENTIALS_FILE", "/etc/dc/bundle.creds.json")
        result = doctor.check_device_connect()
        assert "  PASS  " in result
        assert "MESSAGING_CREDENTIALS_FILE" in result
        assert "/etc/dc/bundle.creds.json" not in result, "the row never prints credential contents or paths"
        assert mod.calls == [("zenoh", None)], "the runtime's decision function is what decided"

    def test_no_tls_and_no_opt_in_warns_that_run_refuses(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._impl(monkeypatch, authenticated=False)
        result = doctor.check_device_connect()
        assert "  WARN  " in result
        assert "run() will refuse" in result
        assert "DEVICE_CONNECT_ALLOW_INSECURE" in result
        assert "MESSAGING_CREDENTIALS_FILE" in result

    def test_insecure_opt_in_without_an_allowlist_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Plaintext AND any caller may execute/stop: the one posture worse than refusing."""
        self._impl(monkeypatch, authenticated=False)
        monkeypatch.setenv("DEVICE_CONNECT_ALLOW_INSECURE", "true")
        result = doctor.check_device_connect()
        assert "  FAIL  " in result
        assert "UNENCRYPTED" in result
        assert "DEVICE_CONNECT_RPC_ALLOW=<caller-id,...>" in result

    def test_insecure_opt_in_with_an_allowlist_warns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._impl(monkeypatch, authenticated=False)
        monkeypatch.setenv("DEVICE_CONNECT_ALLOW_INSECURE", "1")
        monkeypatch.setenv("DEVICE_CONNECT_RPC_ALLOW", "operator-laptop")
        result = doctor.check_device_connect()
        assert "  WARN  " in result
        assert "UNENCRYPTED" in result
        assert "DEVICE_CONNECT_RPC_ALLOW" in result

    def test_the_backend_in_the_row_is_the_one_run_reads(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mod = self._impl(monkeypatch, authenticated=True)
        monkeypatch.setenv("MESSAGING_BACKEND", "nats")
        result = doctor.check_device_connect()
        assert "(nats)" in result
        assert mod.calls == [("nats", None)]


class TestMeshPosture:
    @pytest.fixture(autouse=True)
    def _offline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No sockets: the hub port reads free and every endpoint reads unreachable."""
        pytest.importorskip("zenoh")
        for name in (
            "STRANDS_MESH",
            "STRANDS_MESH_LOCAL_DEV",
            "STRANDS_MESH_AUTH_MODE",
            "STRANDS_MESH_I_KNOW_THIS_IS_INSECURE",
            "STRANDS_MESH_ACCEPT_PERMISSIVE_ACL",
            "STRANDS_MESH_ACL_FILE",
            "STRANDS_MESH_TLS_CA",
            "STRANDS_MESH_TLS_CERT",
            "STRANDS_MESH_TLS_KEY",
            "STRANDS_MESH_PORT",
            "STRANDS_MESH_MULTICAST",
            "ZENOH_CONNECT",
        ):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr(doctor, "_tcp_reachable", lambda host, port, timeout_s: False)
        monkeypatch.setattr(doctor, "_listener_owner", lambda port: "")

    def test_the_default_posture_is_the_refusal_the_runtime_logs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Measured: PASS printed while ``Robot(mesh=True)`` logged "Mesh did NOT start"."""
        from strands_robots.mesh.core import PERMISSIVE_ACL_REFUSAL

        result = doctor.check_mesh()
        assert "  PASS  " not in result
        assert "mesh=True would not start" in result
        for env_name in (
            "STRANDS_MESH_LOCAL_DEV",
            "STRANDS_MESH_ACCEPT_PERMISSIVE_ACL",
            "STRANDS_MESH_ACL_FILE",
            "STRANDS_MESH=false",
        ):
            assert env_name in PERMISSIVE_ACL_REFUSAL and env_name in result, env_name

    def test_the_refusal_is_a_warning_until_the_operator_turns_the_mesh_on(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert "  WARN  " in doctor.check_mesh()
        monkeypatch.setenv("STRANDS_MESH", "true")
        assert "  FAIL  " in doctor.check_mesh()

    def test_the_gate_is_the_runtimes_gate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A non-permissive ACL from ``snapshot_acl`` is believed; the row does not re-read the file."""
        from strands_robots.mesh import _acl_config

        monkeypatch.setattr(_acl_config, "snapshot_acl", lambda namespace="strands": (False, {"rules": ["x"]}))
        monkeypatch.setenv("STRANDS_MESH_ACL_FILE", "/etc/mesh/acl.json5")
        for name in ("STRANDS_MESH_TLS_CA", "STRANDS_MESH_TLS_CERT", "STRANDS_MESH_TLS_KEY"):
            monkeypatch.setenv(name, f"/etc/mesh/{name}.pem")
        result = doctor.check_mesh()
        assert "  PASS  " in result
        assert "mtls + ACL /etc/mesh/acl.json5" in result

    def test_mtls_without_certificates_names_the_missing_variables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STRANDS_MESH_ACCEPT_PERMISSIVE_ACL", "1")
        monkeypatch.setenv("STRANDS_MESH_TLS_CA", "/etc/mesh/ca.pem")
        result = doctor.check_mesh()
        assert "  WARN  " in result
        assert "STRANDS_MESH_TLS_CERT, STRANDS_MESH_TLS_KEY" in result
        assert "STRANDS_MESH_TLS_CA," not in result

    def test_local_dev_passes_as_plaintext_localhost(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STRANDS_MESH_LOCAL_DEV", "true")
        result = doctor.check_mesh()
        assert "  PASS  " in result
        assert "local-dev (plaintext, localhost only)" in result
        assert "hub 127.0.0.1:7447 free" in result
        assert "gossip-only" in result

    def test_a_bad_auth_mode_is_the_runtimes_value_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STRANDS_MESH_AUTH_MODE", "mtsl")
        result = doctor.check_mesh()
        assert "  WARN  " in result
        assert "STRANDS_MESH_AUTH_MODE='mtsl' not supported" in result

    def test_a_bound_hub_port_names_its_owner(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STRANDS_MESH_LOCAL_DEV", "1")
        monkeypatch.setenv("STRANDS_MESH_PORT", "7450")
        monkeypatch.setattr(doctor, "_tcp_reachable", lambda host, port, timeout_s: (host, port) == ("127.0.0.1", 7450))
        monkeypatch.setattr(doctor, "_listener_owner", lambda port: "4242/zenohd")
        result = doctor.check_mesh()
        assert "hub 127.0.0.1:7450 owned by 4242/zenohd" in result

    def test_unreachable_connect_endpoints_are_listed_with_a_deadline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STRANDS_MESH_LOCAL_DEV", "1")
        monkeypatch.setenv("ZENOH_CONNECT", "tcp/10.0.0.9:7447, tls/hub.lab:7448")
        seen: list[tuple[str, int, float]] = []

        def reachable(host: str, port: int, timeout_s: float) -> bool:
            seen.append((host, port, timeout_s))
            return host == "hub.lab"

        monkeypatch.setattr(doctor, "_tcp_reachable", reachable)
        result = doctor.check_mesh()
        assert "  WARN  " in result
        assert "ZENOH_CONNECT unreachable: tcp/10.0.0.9:7447" in result
        assert "hub.lab" not in result.split("unreachable:")[1]
        assert [s for s in seen if s[0] != "127.0.0.1"] == [("10.0.0.9", 7447, 1.0), ("hub.lab", 7448, 1.0)]

    @pytest.mark.parametrize("raw", ["", "abc", "0", "99999", "-1", "74 47"])
    def test_a_port_the_runtime_would_reject_is_not_printed_as_the_hub(
        self, monkeypatch: pytest.MonkeyPatch, raw: str
    ) -> None:
        """``open_session`` range-checks the value and warns before using 7447."""
        monkeypatch.setenv("STRANDS_MESH_LOCAL_DEV", "1")
        monkeypatch.setenv("STRANDS_MESH_PORT", raw)
        result = doctor.check_mesh()
        assert "  WARN  " in result
        assert "hub 127.0.0.1:7447 free" in result
        assert f"STRANDS_MESH_PORT={raw!r}" in result
        assert "falls back to 7447" in result

    def test_a_port_the_runtime_accepts_is_the_hub_with_no_note(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STRANDS_MESH_LOCAL_DEV", "1")
        monkeypatch.setenv("STRANDS_MESH_PORT", "65535")
        result = doctor.check_mesh()
        assert "  PASS  " in result
        assert "hub 127.0.0.1:65535 free" in result
        assert "STRANDS_MESH_PORT" not in result

    def test_every_exposure_is_reported_not_just_the_first(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Three notes at once: an early return would have hidden two of them."""
        monkeypatch.setenv("STRANDS_MESH_LOCAL_DEV", "1")
        monkeypatch.setenv("STRANDS_MESH_PORT", "abc")
        monkeypatch.setenv("STRANDS_MESH_MULTICAST", "true")
        monkeypatch.setenv("ZENOH_CONNECT", "tcp/10.0.0.9:7447")
        result = doctor.check_mesh()
        assert "  WARN  " in result
        assert "STRANDS_MESH_PORT='abc'" in result
        assert "ZENOH_CONNECT unreachable: tcp/10.0.0.9:7447" in result
        assert "STRANDS_MESH_MULTICAST=true" in result

    def test_multicast_scouting_warns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STRANDS_MESH_LOCAL_DEV", "1")
        monkeypatch.setenv("STRANDS_MESH_MULTICAST", "true")
        result = doctor.check_mesh()
        assert "  WARN  " in result
        assert "224.0.0.224:7446" in result
        assert "STRANDS_MESH_MULTICAST=true" in result

    def test_the_probe_opens_no_zenoh_session(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import zenoh

        def boom(*args, **kwargs):
            raise AssertionError("doctor must not open a zenoh session")

        monkeypatch.setattr(zenoh, "open", boom)
        monkeypatch.setenv("STRANDS_MESH_LOCAL_DEV", "1")
        assert "  PASS  " in doctor.check_mesh()


class TestTheRuntimeUsesTheSameText:
    def test_mesh_start_logs_the_constant_the_doctor_prints(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """One string, two readers: the runtime's ERROR and the doctor row cannot name different env vars."""
        import inspect

        from strands_robots.mesh import core

        source = inspect.getsource(core.Mesh._refuse_under_permissive_default_acl)
        assert "PERMISSIVE_ACL_REFUSAL" in source
        assert "Pick one" not in source, "the text lives in the constant, not retyped in the method"
