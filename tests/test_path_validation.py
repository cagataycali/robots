"""Tests for strands_robots.tools._path_validation module."""

import os
from unittest.mock import patch

import pytest

from strands_robots.tools._path_validation import BLOCKED_PREFIXES, validate_save_path


class TestValidateSavePath:
    """Tests for the validate_save_path helper."""

    # ── Happy-path tests ──────────────────────────────────────────────

    def test_returns_resolved_absolute_path(self, tmp_path):
        """A relative path should be resolved to an absolute path."""
        target = str(tmp_path / "output" / "file.txt")
        result = validate_save_path(target)
        assert os.path.isabs(result)

    def test_accepts_tmp_directory(self, tmp_path):
        """Paths under /tmp should be allowed."""
        target = str(tmp_path / "robot_data" / "capture.png")
        result = validate_save_path(target)
        assert result == target

    def test_accepts_home_directory(self, tmp_path):
        """Paths under the user's home directory should be allowed."""
        target = str(tmp_path / "my_data")
        result = validate_save_path(target)
        assert result == target

    def test_tilde_expansion(self, tmp_path):
        """~ should expand to the user home directory."""
        with patch.dict(os.environ, {"HOME": str(tmp_path)}):
            result = validate_save_path("~/robot_data")
            assert result.startswith(str(tmp_path))
            assert "~" not in result

    def test_custom_label_in_success(self, tmp_path):
        """Custom label should not affect a successful result."""
        target = str(tmp_path / "data.json")
        result = validate_save_path(target, label="save_path")
        assert result == target

    # ── Empty / null-byte rejection ───────────────────────────────────

    def test_rejects_empty_string(self):
        with pytest.raises(ValueError, match="must not be empty"):
            validate_save_path("")

    def test_rejects_null_byte(self):
        with pytest.raises(ValueError, match="contains invalid characters"):
            validate_save_path("/tmp/evil\x00file.txt")

    def test_rejects_null_byte_in_middle(self):
        with pytest.raises(ValueError, match="contains invalid characters"):
            validate_save_path("/tmp/foo\x00/bar")

    # ── Directory-traversal rejection ─────────────────────────────────

    def test_rejects_double_dot_component(self):
        with pytest.raises(ValueError, match="path traversal"):
            validate_save_path("/tmp/data/../../../etc/passwd")

    def test_rejects_leading_double_dot(self):
        with pytest.raises(ValueError, match="path traversal"):
            validate_save_path("../../etc/shadow")

    def test_rejects_backslash_traversal(self):
        """Backslash-separated '..' should also be caught (Windows-style)."""
        with pytest.raises(ValueError, match="path traversal"):
            validate_save_path("data\\..\\..\\etc\\passwd")

    def test_allows_dots_in_filenames(self, tmp_path):
        """A file named '..hidden' or 'file..bak' is NOT traversal."""
        result = validate_save_path(str(tmp_path / "..hidden"))
        assert "..hidden" in result

    def test_allows_single_dot(self, tmp_path):
        """A single '.' component (current dir) is benign."""
        target = str(tmp_path / "." / "output.txt")
        result = validate_save_path(target)
        assert os.path.isabs(result)

    # ── Blocked prefix rejection ──────────────────────────────────────

    @pytest.mark.parametrize("prefix", BLOCKED_PREFIXES)
    def test_rejects_all_blocked_prefixes(self, prefix):
        """Every entry in BLOCKED_PREFIXES must be rejected."""
        dangerous_path = prefix + "evil_file.txt"
        with pytest.raises(ValueError, match="protected system directory"):
            validate_save_path(dangerous_path)

    def test_rejects_etc_passwd(self):
        with pytest.raises(ValueError, match="protected system directory"):
            validate_save_path("/etc/passwd")

    def test_rejects_usr_bin(self):
        with pytest.raises(ValueError, match="protected system directory"):
            validate_save_path("/usr/bin/python3")

    def test_rejects_proc_self(self):
        with pytest.raises(ValueError, match="protected system directory"):
            validate_save_path("/proc/self/environ")

    def test_rejects_dev_null_write(self):
        with pytest.raises(ValueError, match="protected system directory"):
            validate_save_path("/dev/sda1")

    def test_rejects_var_spool_cron(self):
        """The /var/spool/cron/ prefix must be blocked."""
        with pytest.raises(ValueError, match="protected system directory"):
            validate_save_path("/var/spool/cron/root")

    def test_rejects_var_spool_at(self):
        """The /var/spool/at/ prefix must be blocked."""
        with pytest.raises(ValueError, match="protected system directory"):
            validate_save_path("/var/spool/at/job.001")

    # ── Trailing-slash correctness (the review comment) ───────────────

    def test_blocked_prefix_trailing_slash_precision(self):
        """Paths that merely share a common prefix but are NOT inside
        the blocked directory should be allowed.

        For example, ``/var/spool/crondata`` shares the ``/var/spool/cron``
        prefix but is a *sibling* directory, not a child.  The trailing-
        slash on the blocked prefix ensures this is handled correctly.
        """
        # This should NOT raise — it's /var/spool/crondata, not /var/spool/cron/
        # We need to mock realpath since /var/spool/crondata won't exist
        with patch("os.path.realpath", return_value="/var/spool/crondata/myfile"):
            result = validate_save_path("/var/spool/crondata/myfile")
            assert result == "/var/spool/crondata/myfile"

    def test_blocked_prefix_exact_dir_match(self):
        """The exact blocked directory itself (e.g. /var/spool/cron)
        should also be rejected — it's the container directory."""
        with patch("os.path.realpath", return_value="/var/spool/cron"):
            with pytest.raises(ValueError, match="protected system directory"):
                validate_save_path("/var/spool/cron")

    def test_all_blocked_prefixes_end_with_slash(self):
        """Invariant: every entry in BLOCKED_PREFIXES must end with '/'
        for correct startswith matching."""
        for prefix in BLOCKED_PREFIXES:
            assert prefix.endswith("/"), f"BLOCKED_PREFIXES entry missing trailing slash: {prefix!r}"

    # ── Custom label tests ────────────────────────────────────────────

    def test_custom_label_in_empty_error(self):
        with pytest.raises(ValueError, match="save_path must not be empty"):
            validate_save_path("", label="save_path")

    def test_custom_label_in_traversal_error(self):
        with pytest.raises(ValueError, match="output_dir must not contain"):
            validate_save_path("/../etc", label="output_dir")

    def test_custom_label_in_blocked_error(self):
        with pytest.raises(ValueError, match="storage_dir resolves to"):
            validate_save_path("/etc/crontab", label="storage_dir")

    # ── Symlink resolution ────────────────────────────────────────────

    def test_symlink_to_blocked_dir_is_rejected(self, tmp_path):
        """A symlink pointing into a blocked directory should be caught."""
        link = tmp_path / "innocent_link"
        link.symlink_to("/etc/cron.d")
        with pytest.raises(ValueError, match="protected system directory"):
            validate_save_path(str(link / "evil_job"))

    def test_symlink_to_safe_dir_is_allowed(self, tmp_path):
        """A symlink pointing to a safe location should pass."""
        target_dir = tmp_path / "real_data"
        target_dir.mkdir()
        link = tmp_path / "link_data"
        link.symlink_to(target_dir)
        result = validate_save_path(str(link / "file.txt"))
        assert str(target_dir) in result
