"""``hf_local_dir`` under the caller's own home, or the system temp dir, is admitted; hidden home entries and other homes are not.

The bind-mount blocklist names ``/home``, ``/root`` and ``/var``. Read as a
prefix rule those cover the only places an ordinary user can write: on Linux
``hf_local_dir="~/checkpoints"`` was refused as "under protected host path
'/home'" - and so was the tool's own default ``~/.strands_robots/checkpoints``
when spelled out - while ``/opt``, ``/mnt`` and ``/srv`` are root-owned. On
macOS the system temp dir is ``/var/folders/.../T``, so every ``$TMPDIR``
path was refused and this file's own ``tmp_path`` fixtures with it.

These tests build a Linux-shaped host under ``tmp_path`` - a ``home`` tree
with two users - and point the blocklist and the "current user" at it, so
the verdicts do not depend on the machine running the suite.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest

gi = importlib.import_module("strands_robots.tools.gr00t_inference")


@pytest.fixture
def host(tmp_path, monkeypatch):
    """A fake host: ``<root>/home/{me,other}``, ``<root>/etc``, the blocklist naming them."""
    root = tmp_path / "host"
    me = root / "home" / "me"
    for d in (
        me / "checkpoints",
        me / ".ssh",
        me / ".aws",
        me / ".strands_robots" / "checkpoints",
        me / ".cache" / "huggingface",
        root / "home" / "other" / "ckpt",
        root / "etc",
        root / "tmp",
    ):
        d.mkdir(parents=True)
    (me / ".aws" / "credentials").write_text("secret")
    (me / "to-etc").symlink_to(root / "etc")
    (me / "to-ssh").symlink_to(me / ".ssh")
    blocked = tuple(str(root / p.lstrip("/")) for p in ("/etc", "/root", "/home", "/var", "/var/run", "/run"))
    monkeypatch.setattr(gi, "_BLOCKED_VOLUME_HOST_PATHS", blocked)
    monkeypatch.setattr(gi, "_BLOCKED_VOLUME_EXACT", (str(root / "var/run/docker.sock"),))
    monkeypatch.setattr(gi, "_user_home", lambda: os.path.realpath(me))
    monkeypatch.setattr(gi, "_temp_root", lambda: os.path.realpath(root / "tmp"))
    monkeypatch.setattr(gi, "_checkpoints_dir", lambda: me / ".strands_robots" / "checkpoints")
    monkeypatch.setenv("HF_HOME", str(me / ".cache" / "huggingface"))
    return root


def _verdict(path: Path | str) -> str | None:
    return gi._check_hf_local_dir_safety(str(path))


class TestTheCallersOwnVisibleDirectoriesAreAdmitted:
    def test_a_visible_directory_in_the_home(self, host):
        assert _verdict(host / "home/me/checkpoints") is None

    def test_a_directory_that_does_not_exist_yet(self, host):
        assert _verdict(host / "home/me/gr00t-n1") is None

    def test_the_tools_own_default_spelled_out(self, host):
        assert _verdict(host / "home/me/.strands_robots/checkpoints") is None

    def test_a_sub_checkpoint_under_the_default(self, host):
        assert _verdict(host / "home/me/.strands_robots/checkpoints/nvidia__GR00T") is None

    def test_the_hugging_face_cache(self, host):
        assert _verdict(host / "home/me/.cache/huggingface/hub") is None

    def test_the_system_temp_dir(self, host):
        assert _verdict(host / "tmp/pytest-of-me/ckpt") is None


class TestWhatStaysRefused:
    def test_the_home_directory_itself(self, host):
        reason = _verdict(host / "home/me")
        assert reason is not None and "protected host path" in reason

    def test_another_users_home(self, host):
        reason = _verdict(host / "home/other/ckpt")
        assert reason is not None and "protected host path" in reason

    @pytest.mark.parametrize("hidden", [".ssh", ".aws/credentials", ".docker/config.json", ".gnupg"])
    def test_a_hidden_entry_of_the_own_home_by_name(self, host, hidden):
        reason = _verdict(host / "home/me" / hidden)
        assert reason is not None
        assert "hidden entry of your home directory" in reason
        assert repr(hidden.split("/")[0]) in reason
        assert "'~/checkpoints'" in reason  # the remedy

    def test_a_hidden_entry_reached_through_a_visible_path(self, host):
        reason = _verdict(host / "home/me/checkpoints/../.ssh")
        assert reason is not None and "'.ssh'" in reason

    def test_a_symlink_in_the_home_pointing_at_a_protected_dir(self, host):
        reason = _verdict(host / "home/me/to-etc")
        assert reason is not None and "protected host path" in reason

    def test_a_symlink_in_the_home_pointing_at_a_hidden_entry(self, host):
        reason = _verdict(host / "home/me/to-ssh")
        assert reason is not None and "'.ssh'" in reason

    def test_etc_and_the_docker_socket(self, host):
        assert "protected host path" in (_verdict(host / "etc/shadow") or "")
        assert "docker socket" in (_verdict(host / "var/run/docker.sock") or "")

    def test_var_outside_the_temp_dir(self, host):
        assert "protected host path" in (_verdict(host / "var/log") or "")


def test_the_download_probe_accepts_a_temp_dir_on_this_host(tmp_path):
    """The failure that led here: ``tmp_path`` refused on macOS as under ``/var``."""
    local = tmp_path / "ckpt"
    local.mkdir()
    (local / "config.json").write_text("{}")
    result = gi._download_checkpoint(
        hf_repo="nvidia/foo", hf_subfolder=None, hf_local_dir=str(local), force=False, hf_token=None
    )
    assert result["status"] == "success", result


def test_this_hosts_own_home_is_consistent_with_the_rule():
    """On the machine running the suite: a visible home dir passes, ``~/.ssh`` does not."""
    assert gi._check_hf_local_dir_safety("~/strands-checkpoints") is None
    reason = gi._check_hf_local_dir_safety("~/.ssh")
    assert reason is not None and "'.ssh'" in reason


def test_a_protected_directory_inside_the_temp_dir_is_still_refused(tmp_path, monkeypatch):
    """A blocklist entry inside the allowance zone is more specific and wins."""
    protected = tmp_path / "etc"
    protected.mkdir()
    monkeypatch.setattr(gi, "_BLOCKED_VOLUME_HOST_PATHS", (str(protected),))
    monkeypatch.setattr(gi, "_temp_root", lambda: os.path.realpath(tmp_path))
    reason = gi._check_hf_local_dir_safety(str(protected / "shadow"))
    assert reason is not None and "protected host path" in reason
    assert gi._check_hf_local_dir_safety(str(tmp_path / "ckpt")) is None
