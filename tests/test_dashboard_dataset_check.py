"""A directory that only LOOKS like a dataset is named as such before it costs a run (Q37).

Grounded in two real datasets on this machine:
  * local/sim_recording        - info.json: total_episodes 0, NO data/ directory at all
  * cagataydev/scout-sim-...   - info.json: 30 episodes, 14595 frames, and exactly ONE parquet file

The second one is why this check never compares file count to episode count: v3.0 packs many
episodes into one file (data/chunk-000/file-000.parquet), so a count comparison would condemn
every correct v3 dataset on disk.
"""

from __future__ import annotations

from strands_robots.dashboard.dataset_check import dataset_verdict


class TestTheAbandonedSession:
    def test_zero_episodes_is_not_a_dataset(self) -> None:
        # The exact metadata of local/sim_recording, live on this machine.
        v = dataset_verdict(
            {"total_episodes": 0, "total_frames": 0, "fps": 30, "robot_type": "so101"},
            has_data_files=False,
        )
        assert v["usable"] is False
        assert v["reason"] == "no_episodes"
        assert "OPENS" in v["problem"], "the operator needs the mechanism, not 'empty dataset'"
        assert "delete it" in v["problem"], "and something to do about it"

    def test_zero_episodes_wins_over_the_data_probe(self) -> None:
        # Even if data files exist, 0 episodes is the more specific truth to report.
        v = dataset_verdict({"total_episodes": 0, "fps": 30}, has_data_files=True)
        assert v["reason"] == "no_episodes"


class TestMetadataWithoutFrames:
    def test_episodes_claimed_but_no_data_files(self) -> None:
        v = dataset_verdict({"total_episodes": 12, "total_frames": 4000, "fps": 30}, has_data_files=False)
        assert v["usable"] is False
        assert v["reason"] == "missing_data"
        assert "12 episodes" in v["problem"]
        assert "after the setup it charges you for" in v["problem"]

    def test_unreadable_meta_says_what_it_could_be(self) -> None:
        for empty in (None, {}):
            v = dataset_verdict(empty, has_data_files=True)
            assert v["usable"] is False and v["reason"] == "unreadable_meta"
            # A recording IN PROGRESS is the innocent explanation and must be offered first.
            assert "running right now" in v["problem"]


class TestTheHealthyOnes:
    def test_a_real_v3_dataset_with_one_parquet_file_passes(self) -> None:
        # cagataydev/scout-sim-apartment-v0: 30 episodes, ONE file. The count comparison this
        # check refuses to make would have called it broken.
        v = dataset_verdict(
            {"total_episodes": 30, "total_frames": 14595, "fps": 10, "robot_type": "scout"},
            has_data_files=True,
        )
        assert v["usable"] is True
        assert v["episodes"] == 30
        assert "opens a shard" in v["note"], "a pass must not read as a load guarantee"

    def test_a_single_episode_is_a_dataset(self) -> None:
        # One demonstration is a legitimate thing to replay or overfit on; refusing it would be
        # a taste judgment wearing a safety check's clothes.
        assert dataset_verdict({"total_episodes": 1, "total_frames": 300, "fps": 30}, has_data_files=True)["usable"] is True

    def test_not_probing_data_is_reported_not_assumed(self) -> None:
        v = dataset_verdict({"total_episodes": 5, "total_frames": 900, "fps": 30})
        assert v["usable"] is True
        assert "not checked" in v["note"], "silence about data/ must not read as a pass on data/"

    def test_stale_frame_count_warns_but_does_not_refuse(self) -> None:
        v = dataset_verdict({"total_episodes": 4, "total_frames": 0, "fps": 30}, has_data_files=True)
        assert v["usable"] is True
        assert "stale" in v["warning"]

    def test_odd_metadata_types_do_not_crash_the_listing(self) -> None:
        for bad in ({"total_episodes": "many", "fps": 30}, {"total_episodes": None, "fps": 30},
                    {"total_episodes": True, "fps": 30}, {"total_episodes": 3.0, "fps": 30}):
            v = dataset_verdict(bad, has_data_files=True)
            assert isinstance(v["usable"], bool)
        # A float that IS a count is honoured; a bool is not a count.
        assert dataset_verdict({"total_episodes": 3.0, "fps": 30}, has_data_files=True)["usable"] is True
        assert dataset_verdict({"total_episodes": True, "fps": 30}, has_data_files=True)["usable"] is True


class TestTheListingCarriesIt:
    """The verdict must ride on the ROW, or the tab has to re-derive it and they can disagree."""

    def _dataset(self, root, repo, episodes, *, with_data):
        d = root / repo
        (d / "meta").mkdir(parents=True)
        (d / "meta" / "info.json").write_text(
            '{"codebase_version": "v3.0", "total_episodes": %d, "total_frames": %d, "fps": 30}'
            % (episodes, episodes * 100)
        )
        if with_data:
            chunk = d / "data" / "chunk-000"
            chunk.mkdir(parents=True)
            (chunk / "file-000.parquet").write_bytes(b"PAR1")
        return d

    def test_an_abandoned_session_is_listed_but_marked(self, tmp_path, monkeypatch) -> None:
        # It stays LISTED on purpose: hiding it would leave the operator wondering where the
        # recording they just made went, and they are the one who can delete it.
        self._dataset(tmp_path, "local/sim_recording", 0, with_data=False)
        self._dataset(tmp_path, "cagataydev/good-one", 30, with_data=True)
        monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path))
        monkeypatch.delenv("STRANDS_ROBOTS_DATA_DIRS", raising=False)

        from strands_robots.dashboard import training

        rows = {r["repo_id"]: r for r in training.local_datasets()}
        assert set(rows) == {"local/sim_recording", "cagataydev/good-one"}
        assert rows["local/sim_recording"]["usable"] is False
        assert rows["local/sim_recording"]["reason"] == "no_episodes"
        assert rows["cagataydev/good-one"]["usable"] is True
        # The metadata the tab already renders must survive the merge.
        assert rows["cagataydev/good-one"]["total_episodes"] == 30
        assert rows["cagataydev/good-one"]["fps"] == 30

    def test_meta_without_any_data_dir_reports_missing_data(self, tmp_path, monkeypatch) -> None:
        self._dataset(tmp_path, "org/frames-never-landed", 12, with_data=False)
        monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path))
        monkeypatch.delenv("STRANDS_ROBOTS_DATA_DIRS", raising=False)

        from strands_robots.dashboard import training

        row = training.local_datasets()[0]
        assert row["usable"] is False and row["reason"] == "missing_data"
        assert "12 episodes" in row["problem"]
