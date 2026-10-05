"""verify_dataset distinguishes the three root-shaped user mistakes.

Four physical shapes reach the "no episode parquet" fallback with
indistinguishable `No meta/episodes parquet under <root>. The dataset is empty
or was never finalized...` output; three of them (typo, broken symlink,
file-as-root) are user-path mistakes that `stop_recording` cannot fix. This
test pins the three-way split so a regression that collapses them back onto
one message fails loudly.
"""

from __future__ import annotations

import pathlib

import pytest

from strands_robots.dataset_metadata import read_dataset_episode_indices
from strands_robots.verify_dataset import verify_dataset


def test_nonexistent_root_names_the_typo_shape(tmp_path: pathlib.Path) -> None:
    missing = tmp_path / "typo_prefix" / "so101_reach"

    with pytest.raises(FileNotFoundError, match=r"does not exist"):
        read_dataset_episode_indices(missing)

    report = verify_dataset(missing, expected=5)
    assert report["status"] == "error"
    [problem] = report["problems"]
    assert "does not exist" in problem
    assert "never finalized" not in problem


def test_broken_symlink_root_names_the_typo_shape(tmp_path: pathlib.Path) -> None:
    link = tmp_path / "link_to_nowhere"
    link.symlink_to(tmp_path / "nope")

    with pytest.raises(FileNotFoundError, match=r"does not exist"):
        read_dataset_episode_indices(link)


def test_file_root_names_the_not_a_directory_shape(tmp_path: pathlib.Path) -> None:
    f = tmp_path / "accidentally_a_file"
    f.write_text("")

    with pytest.raises(NotADirectoryError, match=r"is not a directory"):
        read_dataset_episode_indices(f)

    report = verify_dataset(f, expected=5)
    assert report["status"] == "error"
    [problem] = report["problems"]
    assert "is not a directory" in problem
    assert "never finalized" not in problem


def test_empty_existing_dir_still_names_the_finalize_shape(tmp_path: pathlib.Path) -> None:
    # The one case where the "stop_recording/finalize" advice is correct must
    # keep that message - the split only disambiguates the three mistakes
    # the advice does not fit.
    with pytest.raises(FileNotFoundError, match=r"never finalized"):
        read_dataset_episode_indices(tmp_path)

    report = verify_dataset(tmp_path, expected=5)
    assert report["status"] == "error"
    [problem] = report["problems"]
    assert "never finalized" in problem
