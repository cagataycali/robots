"""Pin: gr00t_inference does not let an agent mount host paths or pick images.

The agent-facing tool must not expose ``volumes``, ``image_name``, or
``container_command`` -- those let a prompt-injected agent mount the host
filesystem, the docker socket, or run an arbitrary container command
(host RCE). Container topology is operator-config-driven.

Defence in depth: the private ``_start_container`` entry point (reachable
by operators/tests) also rejects dangerous mounts and off-allowlist images.

These tests fail on pre-fix code, where ``_start_container`` appended any
caller-supplied ``-v host:container`` mount straight into the docker argv.
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

import pytest

import strands_robots.tools.gr00t_inference as gi

# --- agent surface: dangerous params are gone --------------------------


def _tool_params() -> set[str]:
    fn = getattr(gi.gr00t_inference, "__wrapped__", None) or gi.gr00t_inference
    return set(inspect.signature(fn).parameters)


def test_tool_signature_drops_volumes():
    assert "volumes" not in _tool_params()


def test_tool_signature_drops_image_name():
    assert "image_name" not in _tool_params()


def test_tool_signature_drops_container_command():
    assert "container_command" not in _tool_params()


def test_tool_rejects_volumes_kwarg():
    # Passing the removed kwarg must raise (TypeError) rather than silently
    # mount anything.
    with pytest.raises(TypeError):
        gi.gr00t_inference(action="start_container", volumes={"/": "/host"})


# --- image allowlist ---------------------------------------------------


def test_is_allowed_image_accepts_canonical():
    assert gi._is_allowed_image("gr00t:latest") is True
    assert gi._is_allowed_image("gr00t:n1.7") is True
    assert gi._is_allowed_image("nvcr.io/nvidia/isaac-gr00t:n1.7") is True


def test_is_allowed_image_rejects_arbitrary():
    assert gi._is_allowed_image("alpine:latest") is False
    assert gi._is_allowed_image("evil/image:tag") is False
    assert gi._is_allowed_image("") is False


def test_image_allowlist_env_extends(monkeypatch):
    monkeypatch.setenv("STRANDS_GR00T_IMAGE_ALLOW", "myreg/gr00t:*")
    assert gi._is_allowed_image("myreg/gr00t:v1") is True


# --- _start_container guards (defence in depth) ------------------------


def _no_run_patches():
    """Patch subprocess.run + _container_state so a failing guard is the
    only reason docker run would be skipped."""
    return (
        patch.object(gi, "_container_state", return_value="absent"),
        patch.object(gi.subprocess, "run", side_effect=AssertionError("docker run must NOT be called")),
    )


def test_start_container_rejects_host_root_mount():
    state_p, run_p = _no_run_patches()
    with state_p, run_p:
        result = gi._start_container(
            image_name="alpine:latest",
            container_name="x",
            port=5555,
            volumes={"/": "/host"},
            hf_token=None,
            container_command="sh -c 'id'",
            hf_local_dir=None,
            force=True,
        )
    assert result["status"] == "error"


def test_start_container_rejects_docker_socket_mount():
    state_p, run_p = _no_run_patches()
    with state_p, run_p:
        result = gi._start_container(
            image_name="gr00t:latest",
            container_name="x",
            port=5555,
            volumes={"/var/run/docker.sock": "/var/run/docker.sock"},
            hf_token=None,
            container_command="docker ps",
            hf_local_dir=None,
            force=True,
        )
    assert result["status"] == "error"
    assert "docker" in result["message"].lower() or "socket" in result["message"].lower()


def test_start_container_rejects_etc_mount():
    state_p, run_p = _no_run_patches()
    with state_p, run_p:
        result = gi._start_container(
            image_name="gr00t:latest",
            container_name="x",
            port=5555,
            volumes={"/etc": "/host_etc"},
            hf_token=None,
            container_command="tail -f /dev/null",
            hf_local_dir=None,
            force=True,
        )
    assert result["status"] == "error"


# --- _check_volume_safety: child-of-protected-dir prefix coverage ------
# Regression for the exact-match gap (PR #372 review): mounting a *child*
# of a protected dir (e.g. /etc/shadow, /root/.ssh) must be rejected, not
# just the bare dir. These fail on the pre-fix `norm in blocked_dirs` code.


@pytest.mark.parametrize(
    "host_path",
    [
        "/etc/shadow",
        "/root/.ssh",
        "/root/.ssh/id_rsa",
        "/home/ubuntu/.aws/credentials",
        "/proc/1/environ",
        "/sys/kernel",
        "/var/run/docker.sock.bak",
        "/etc",
        "/",
    ],
)
def test_check_volume_safety_rejects_protected_paths(host_path):
    assert gi._check_volume_safety({host_path: "/x"}) is not None


@pytest.mark.parametrize(
    "host_path",
    ["/mnt/models", "/data/checkpoints", "/opt/gr00t", "/srv/data"],
)
def test_check_volume_safety_allows_legit_mounts(host_path):
    # A prefix check must not over-block: legitimate non-protected mounts
    # (and especially anything that merely starts with "/") still pass.
    assert gi._check_volume_safety({host_path: "/x"}) is None


def test_check_volume_safety_expands_user_home():
    # ~ expansion lands under /home or the real HOME -> must be rejected when
    # it resolves under a protected dir.
    import os

    reason = gi._check_volume_safety({"~/.ssh/id_rsa": "/x"})
    home = os.path.expanduser("~")
    if home.startswith(("/home", "/root", "/Users")):
        # /Users is not in the Linux blocklist; only assert when protected.
        if home.startswith(("/home", "/root")):
            assert reason is not None


def test_start_container_rejects_off_allowlist_image():
    state_p, run_p = _no_run_patches()
    with state_p, run_p:
        result = gi._start_container(
            image_name="alpine:latest",
            container_name="x",
            port=5555,
            volumes=None,
            hf_token=None,
            container_command="tail -f /dev/null",
            hf_local_dir=None,
            force=True,
        )
    assert result["status"] == "error"
    assert "allowlist" in result["message"]


def test_start_container_allows_safe_defaults():
    """The legitimate path (allowlisted image, default checkpoint volumes)
    still reaches docker run."""
    runs: list[list[str]] = []

    def fake_run(cmd, *a, **kw):
        runs.append(list(cmd))
        return MagicMock(stdout="", stderr="", returncode=0)

    with (
        patch.object(gi, "_container_state", return_value="absent"),
        patch.object(gi.subprocess, "run", side_effect=fake_run),
    ):
        result = gi._start_container(
            image_name="gr00t:latest",
            container_name="gr00t",
            port=5555,
            volumes=None,
            hf_token=None,
            container_command="tail -f /dev/null",
            hf_local_dir="/data/cp",
            force=True,
        )
    assert result["status"] == "success"
    argv = next(c for c in runs if c[:2] == ["docker", "run"])
    joined = " ".join(argv)
    # No host root / etc / socket mount in the emitted argv.
    assert "-v /:/host" not in joined
    assert "/var/run/docker.sock" not in joined
    assert "/data/cp:/data/checkpoints" in joined
