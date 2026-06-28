"""Behavioural tests for the ``gr00t_inference`` service-start and dispatch paths.

These exercise the structured-error contract an LLM relies on when driving the
tool without docker actually present:

* ``action="start"`` with ``container_name=None`` must resolve a running GR00T
  container first and degrade to a structured error (never crash) when none is
  up or when container discovery itself errored.
* A failed ``docker exec`` launch surfaces as a structured error, not a raised
  ``CalledProcessError``.
* ``_is_service_running`` is a best-effort TCP probe that returns ``False`` on
  any socket failure rather than propagating.
* The container-build actions (``build_image`` / ``start_container`` /
  ``lifecycle``) gate the operator-resolved image through the allowlist at the
  dispatch boundary, so an off-allowlist ``STRANDS_GR00T_IMAGE`` fails closed
  before any docker call.
* ``action="download_checkpoint"`` forwards its resolved arguments to the
  download helper once ``hf_repo`` is supplied.

Nothing here touches docker: ``subprocess.run``, container discovery, the
socket, and the download helper are all mocked.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from strands_robots.tools.gr00t_inference import (
    _is_service_running,
    gr00t_inference,
)

_MODULE = "strands_robots.tools.gr00t_inference"


class TestStartServiceContainerResolution:
    """``action="start"`` must locate a running container before exec."""

    def test_no_running_container_returns_structured_error(self):
        discovered = {
            "status": "success",
            "containers": [{"name": "gr00t-old", "status": "Exited (0) 1 hour ago"}],
        }
        with patch(f"{_MODULE}._find_gr00t_containers", return_value=discovered):
            result = gr00t_inference(action="start", checkpoint_path="/data/ckpt")
        assert result["status"] == "error"
        assert "No running GR00T containers" in result["message"]

    def test_container_discovery_error_is_propagated(self):
        discovery_error = {"status": "error", "message": "docker binary not found"}
        with patch(f"{_MODULE}._find_gr00t_containers", return_value=discovery_error):
            result = gr00t_inference(action="start", checkpoint_path="/data/ckpt")
        assert result["status"] == "error"
        assert result["message"] == "docker binary not found"

    def test_failed_docker_launch_returns_structured_error(self):
        running = {
            "status": "success",
            "containers": [{"name": "gr00t", "status": "Up 2 minutes"}],
        }
        launch_failure = subprocess.CalledProcessError(1, "docker", stderr="exec format error")
        with (
            patch(f"{_MODULE}._find_gr00t_containers", return_value=running),
            patch(f"{_MODULE}.subprocess.run", side_effect=launch_failure),
        ):
            result = gr00t_inference(action="start", checkpoint_path="/data/ckpt")
        assert result["status"] == "error"
        assert "Failed to start service" in result["message"]
        assert "exec format error" in result["message"]


class TestServiceProbe:
    """``_is_service_running`` is a best-effort TCP probe."""

    def test_true_when_port_accepts_connection(self):
        sock = MagicMock()
        sock.connect_ex.return_value = 0
        with patch(f"{_MODULE}.socket.socket", return_value=sock):
            assert _is_service_running(5555) is True

    def test_false_when_connection_refused(self):
        sock = MagicMock()
        sock.connect_ex.return_value = 111
        with patch(f"{_MODULE}.socket.socket", return_value=sock):
            assert _is_service_running(5555) is False

    def test_false_on_socket_failure(self):
        with patch(f"{_MODULE}.socket.socket", side_effect=OSError("network unreachable")):
            assert _is_service_running(5555) is False


class TestDispatchImageAllowlistGate:
    """Build/start/lifecycle reject an off-allowlist operator image up front."""

    @pytest.mark.parametrize(
        ("action", "extra"),
        [
            ("build_image", {}),
            ("start_container", {}),
            ("lifecycle", {"lifecycle": "full"}),
        ],
    )
    def test_off_allowlist_image_fails_closed(self, monkeypatch, action, extra):
        monkeypatch.setenv("STRANDS_GR00T_IMAGE", "evil/image:tag")
        result = gr00t_inference(action=action, **extra)
        assert result["status"] == "error"
        assert "allowlist" in result["message"].lower()


class TestDownloadCheckpointDispatch:
    """``action="download_checkpoint"`` forwards to the download helper."""

    def test_forwards_resolved_arguments(self):
        sentinel = {"status": "success", "skipped": False, "message": "downloaded"}
        with patch(f"{_MODULE}._download_checkpoint", return_value=sentinel) as mock_dl:
            result = gr00t_inference(
                action="download_checkpoint",
                hf_repo="nvidia/GR00T-N1.5-3B",
                hf_subfolder="ckpt",
            )
        assert result is sentinel
        mock_dl.assert_called_once()
        assert mock_dl.call_args.kwargs["hf_repo"] == "nvidia/GR00T-N1.5-3B"
        assert mock_dl.call_args.kwargs["hf_subfolder"] == "ckpt"
