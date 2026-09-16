"""``hf_local_dir`` at the tool's own checkpoints dir, the Hugging Face cache or the system temp dir is admitted; the rest of the home is not.

The bind-mount blocklist names ``/home``, ``/root`` and ``/var``. Read as a
prefix rule those cover the two trees the tool mounts on its own: spelled out,
its default ``~/.strands_robots/checkpoints`` was refused as "under protected
host path '/home'", and so was the Hugging Face cache. On macOS the system
temp dir is ``/var/folders/.../T``, so every ``$TMPDIR`` path was refused and
this file's own ``tmp_path`` fixtures with it.

Exactly those three places are admitted. An arbitrary visible directory of the
caller's home (``~/checkpoints``) is not: ``hf_local_dir`` is agent-supplied,
the mount is read-write and ``hf_repo`` is any repository, so admitting it
would let a prompt-injected ``download_checkpoint`` write that repository into
a directory the user's shell or build reads. It is refused naming the places
that are admitted, so the caller learns where a checkpoint may go.

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
    monkeypatch.setattr(gi, "_checkpoints_dir", lambda **_: me / ".strands_robots" / "checkpoints")
    monkeypatch.setenv("HF_HOME", str(me / ".cache" / "huggingface"))
    return root


def _verdict(path: Path | str) -> str | None:
    return gi._check_hf_local_dir_safety(str(path))


_ADMITTED = "only the tool's own checkpoints dir"


class TestTheToolsOwnTreesAndTheTempDirAreAdmitted:
    def test_the_tools_own_default_spelled_out(self, host):
        assert _verdict(host / "home/me/.strands_robots/checkpoints") is None

    def test_a_sub_checkpoint_under_the_default(self, host):
        assert _verdict(host / "home/me/.strands_robots/checkpoints/nvidia__GR00T") is None

    def test_a_sub_checkpoint_that_does_not_exist_yet(self, host):
        assert _verdict(host / "home/me/.strands_robots/checkpoints/not-downloaded-yet") is None

    def test_the_hugging_face_cache(self, host):
        assert _verdict(host / "home/me/.cache/huggingface/hub") is None

    def test_the_system_temp_dir(self, host):
        assert _verdict(host / "tmp/pytest-of-me/ckpt") is None


class TestTheRestOfTheOwnHomeIsRefusedNamingWhereACheckpointMayGo:
    """``hf_local_dir`` is agent-supplied and the mount is read-write, so ``~/<anything>`` is not a mount."""

    @pytest.mark.parametrize("visible", ["checkpoints", "gr00t-n1", "workspace/proj", "bin"])
    def test_a_visible_directory_of_the_own_home(self, host, visible):
        reason = _verdict(host / "home/me" / visible)
        assert reason is not None
        assert _ADMITTED in reason
        assert repr(str(host / "home/me/.strands_robots/checkpoints")) in reason
        assert repr(str(host / "home/me/.cache/huggingface")) in reason
        assert repr(str(host / "tmp")) in reason
        assert "leave hf_local_dir unset" in reason

    @pytest.mark.parametrize("hidden", [".ssh", ".aws/credentials", ".docker/config.json", ".gnupg"])
    def test_a_hidden_entry_of_the_own_home(self, host, hidden):
        reason = _verdict(host / "home/me" / hidden)
        assert reason is not None and _ADMITTED in reason

    def test_a_hidden_entry_reached_through_a_visible_path(self, host):
        reason = _verdict(host / "home/me/checkpoints/../.ssh")
        assert reason is not None and _ADMITTED in reason

    def test_a_symlink_in_the_home_pointing_at_a_hidden_entry(self, host):
        reason = _verdict(host / "home/me/to-ssh")
        assert reason is not None and _ADMITTED in reason

    def test_a_sibling_of_the_checkpoints_dir_is_not_under_it(self, host):
        """Prefix, not substring: ``checkpoints-evil`` shares the spelling and none of the trust."""
        reason = _verdict(host / "home/me/.strands_robots/checkpoints-evil")
        assert reason is not None and _ADMITTED in reason


class TestWhatTheBlocklistStillDecides:
    def test_the_home_directory_itself(self, host):
        reason = _verdict(host / "home/me")
        assert reason is not None and "protected host path" in reason

    def test_another_users_home(self, host):
        reason = _verdict(host / "home/other/ckpt")
        assert reason is not None and "protected host path" in reason

    def test_a_symlink_in_the_home_pointing_at_a_protected_dir(self, host):
        reason = _verdict(host / "home/me/to-etc")
        assert reason is not None and "protected host path" in reason

    def test_etc_and_the_docker_socket(self, host):
        assert "protected host path" in (_verdict(host / "etc/shadow") or "")
        assert "docker socket" in (_verdict(host / "var/run/docker.sock") or "")

    def test_var_outside_the_temp_dir(self, host):
        assert "protected host path" in (_verdict(host / "var/log") or "")

    def test_a_home_the_blocklist_names_is_not_an_allowance_zone(self, host, monkeypatch):
        """Run the tool as root and ``~`` *is* the protected ``/root``.

        Both zones are read from the environment, so "the caller's own home"
        can name a directory the blocklist protects in its own right. It stays
        refused - even at the tool's own checkpoints dir under it - which is
        what keeps the promise that ``/root`` is unchanged.
        """
        (host / "root" / ".strands_robots" / "checkpoints").mkdir(parents=True)
        monkeypatch.setattr(gi, "_user_home", lambda: os.path.realpath(host / "root"))
        monkeypatch.setattr(gi, "_checkpoints_dir", lambda **_: host / "root" / ".strands_robots" / "checkpoints")
        reason = _verdict(host / "root/.strands_robots/checkpoints")
        assert reason is not None and "protected host path" in reason

    def test_a_temp_dir_the_blocklist_names_is_not_an_allowance_zone(self, host, monkeypatch):
        """``TMPDIR`` is an ordinary env var; pointed at ``/etc`` it admits nothing."""
        monkeypatch.setattr(gi, "_temp_root", lambda: os.path.realpath(host / "etc"))
        reason = _verdict(host / "etc/shadow")
        assert reason is not None and "protected host path" in reason


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
    """On the machine running the suite: the tool's default passes, ``~/checkpoints`` and ``~/.ssh`` do not."""
    if os.path.realpath(gi._user_home()) in gi._with_resolved(gi._BLOCKED_VOLUME_HOST_PATHS):
        pytest.skip("the suite runs as a user whose home the blocklist names (root); nothing under it is admitted")
    assert gi._check_hf_local_dir_safety(str(gi._checkpoints_dir(create=False))) is None
    for elsewhere in ("~/strands-checkpoints", "~/.ssh"):
        reason = gi._check_hf_local_dir_safety(elsewhere)
        assert reason is not None and _ADMITTED in reason, elsewhere


def test_a_protected_directory_inside_the_temp_dir_is_still_refused(tmp_path, monkeypatch):
    """A blocklist entry inside the allowance zone is more specific and wins."""
    protected = tmp_path / "etc"
    protected.mkdir()
    monkeypatch.setattr(gi, "_BLOCKED_VOLUME_HOST_PATHS", (str(protected),))
    monkeypatch.setattr(gi, "_temp_root", lambda: os.path.realpath(tmp_path))
    reason = gi._check_hf_local_dir_safety(str(protected / "shadow"))
    assert reason is not None and "protected host path" in reason
    assert gi._check_hf_local_dir_safety(str(tmp_path / "ckpt")) is None


def test_asking_neither_writes_to_the_host_nor_raises(tmp_path, monkeypatch):
    """Deciding a mount consults the tool's checkpoints dir - as a path, not a directory.

    ``_checkpoints_dir()`` resolves through ``get_base_dir()``, which *creates*
    what it returns. Consulting it from this validator therefore made asking
    "is this mount safe?" write to the host filesystem, and raise when the
    write was impossible - an unwritable or occupied home turned a mount
    refusal into an unhandled exception at the tool boundary, where every
    caller expects a reason string. A file where the base dir would go is
    uncreatable for any user, root included.
    """
    home = tmp_path / "home" / "me"
    (home / "checkpoints").mkdir(parents=True)
    monkeypatch.setattr(gi, "_user_home", lambda: os.path.realpath(home))
    monkeypatch.setattr(gi, "_temp_root", lambda: os.path.realpath(tmp_path / "not-the-temp-dir"))
    occupied = tmp_path / "base"
    occupied.write_text("not a directory")
    monkeypatch.setenv("STRANDS_BASE_DIR", str(occupied))

    reason = _verdict(home / "checkpoints")
    assert reason is not None and _ADMITTED in reason  # answered, with a reason, not an exception
    assert occupied.read_text() == "not a directory"  # answering changed nothing
