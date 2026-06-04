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
