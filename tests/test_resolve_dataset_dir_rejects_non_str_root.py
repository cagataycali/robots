"""Pin: resolve_dataset_dir rejects a non-str root with ValueError.

Mirrors the sibling repo_id type guard. Without this guard,
start_recording(root=<non-str>) raised raw TypeError out of the agent-tool
envelope while the other kwargs (fps, push_to_hub, overwrite, cameras,
repo_id) all return structured status=error.

See also:
    strands_robots/simulation/recording.py:1335-1339 (except ValueError catches
      the error we raise, converts to envelope).
    strands_robots/dataset_source.py:192-203 (repo_id + root guards, side by
      side).
"""
from __future__ import annotations

import pytest

from strands_robots.dataset_source import resolve_dataset_dir


class TestResolveDatasetDirRootType:
    """resolve_dataset_dir(root=<non-str>) must raise ValueError, not TypeError."""

    @pytest.mark.parametrize(
        "bad_root",
        [
            42,
            True,
            3.14,
            b"/tmp/x",
            ["/tmp/x"],
            {"path": "/tmp/x"},
        ],
    )
    def test_non_str_root_raises_valueerror(self, bad_root):
        with pytest.raises(ValueError, match="dataset root must be a string path"):
            resolve_dataset_dir("local/x", bad_root)

    def test_none_root_is_accepted(self, tmp_path, monkeypatch):
        # None root falls through to local_dataset_dir / hub_dataset_dir.
        monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path))
        resolved = resolve_dataset_dir("owner/name", None)
        # Just confirm no exception; concrete path depends on the env.
        assert resolved is not None

    def test_str_root_is_accepted(self, tmp_path):
        # A valid str root is used verbatim.
        resolved = resolve_dataset_dir("local/x", str(tmp_path / "data"))
        assert resolved == (tmp_path / "data")

    def test_repo_id_guard_still_fires(self):
        # The sibling guard this one mirrors must still work.
        with pytest.raises(ValueError, match="dataset id must be a non-empty string"):
            resolve_dataset_dir(42, None)
