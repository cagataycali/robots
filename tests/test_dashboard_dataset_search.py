"""R6: the train screen's dataset picker searches the Hub, not just this disk.

Training always accepted a Hub ``dataset_repo_id`` -- nothing ever OFFERED one,
so a machine with no local recording showed an empty picker and a dead end
(JOURNEYS.md:344 measured it; the user asked for it in as many words).

These tests pin the parts that are easy to get wrong and impossible to notice:
the local/hub distinction that decides WHICH field training gets filled, the
limit promise, and the refusal to render an outage and "no matches" the same.
"""

from __future__ import annotations

from typing import Any

import pytest

from strands_robots.dashboard import training


class _Ds:
    def __init__(self, rid: str, downloads: int = 0, tags: list[str] | None = None) -> None:
        self.id = rid
        self.downloads = downloads
        self.tags = tags or []


@pytest.fixture(autouse=True)
def _clear_cache() -> Any:
    training._HUB_DS_CACHE.clear()
    yield
    training._HUB_DS_CACHE.clear()


def _fake_hub(monkeypatch: pytest.MonkeyPatch, rows: list[_Ds], *, boom: Exception | None = None,
              seen: dict[str, Any] | None = None) -> None:
    class _Api:
        def list_datasets(self, **kw: Any) -> list[_Ds]:
            if seen is not None:
                seen.update(kw)
                seen["calls"] = seen.get("calls", 0) + 1
            if boom is not None:
                raise boom
            return rows

    mod = type("m", (), {"HfApi": _Api})
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", mod)


class TestHubDatasets:
    def test_it_searches_the_lerobot_tag_not_all_of_hf(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, Any] = {}
        _fake_hub(monkeypatch, [_Ds("lerobot/pusht", 1200)], seen=seen)
        rows, problem = training.hub_datasets("pusht", 5)
        assert problem is None
        assert seen["filter"] == "LeRobot", "an unfiltered search offers text corpora train_policy cannot open"
        assert seen["search"] == "pusht"
        assert rows[0]["repo_id"] == "lerobot/pusht"

    def test_a_hub_row_has_no_root_because_that_is_how_callers_tell_them_apart(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _fake_hub(monkeypatch, [_Ds("lerobot/pusht")])
        rows, _ = training.hub_datasets("", 5)
        assert "root" not in rows[0], "a Hub dataset trains from dataset_repo_id, a local one from dataset_root"
        assert rows[0]["local"] is False

    def test_an_outage_is_a_sentence_not_an_empty_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _fake_hub(monkeypatch, [], boom=OSError("no route to host"))
        rows, problem = training.hub_datasets("x", 5)
        assert rows == []
        assert problem and "OSError" in problem, "outage, no network and no-matches must not look identical"

    def test_a_failure_is_never_cached_so_the_next_keystroke_retries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, Any] = {}
        _fake_hub(monkeypatch, [], boom=OSError("flap"), seen=seen)
        training.hub_datasets("q", 5)
        training.hub_datasets("q", 5)
        assert seen["calls"] == 2, "caching a failure would freeze the picker for 5 minutes"

    def test_a_success_is_cached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, Any] = {}
        _fake_hub(monkeypatch, [_Ds("a/b")], seen=seen)
        training.hub_datasets("q", 5)
        training.hub_datasets("q", 5)
        assert seen["calls"] == 1

    def test_a_hub_client_version_bump_cannot_kill_search(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[dict[str, Any]] = []

        class _Api:
            def list_datasets(self, **kw: Any) -> list[_Ds]:
                calls.append(kw)
                if "sort" in kw:
                    raise TypeError("unexpected keyword argument 'sort'")
                return [_Ds("a/b")]

        monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", type("m", (), {"HfApi": _Api}))
        rows, problem = training.hub_datasets("", 5)
        assert problem is None and rows[0]["repo_id"] == "a/b"
        assert len(calls) == 2 and "sort" not in calls[1]

    def test_noisy_hub_tags_are_dropped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _fake_hub(monkeypatch, [_Ds("a/b", tags=["LeRobot", "region:us", "license:mit", "so101"])])
        rows, _ = training.hub_datasets("", 5)
        assert rows[0]["tags"] == ["LeRobot", "so101"]


class TestSearchDatasets:
    def test_local_comes_first_and_keeps_its_root(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(training, "local_datasets",
                            lambda q="": [{"root": "/data/mine", "repo_id": "mine", "total_episodes": 3}])
        _fake_hub(monkeypatch, [_Ds("lerobot/pusht", 999)])
        out = training.search_datasets("", 10)
        assert out["datasets"][0]["repo_id"] == "mine", "what you just recorded is what the train screen offers first"
        assert out["datasets"][0]["root"] == "/data/mine"
        assert out["datasets"][0]["local"] is True
        assert out["local_count"] == 1 and out["hub_count"] == 1

    def test_a_dataset_on_disk_is_not_offered_as_a_download(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(training, "local_datasets",
                            lambda q="": [{"root": "/c/lerobot/pusht", "repo_id": "lerobot/pusht"}])
        _fake_hub(monkeypatch, [_Ds("lerobot/pusht", 999), _Ds("other/ds")])
        out = training.search_datasets("", 10)
        ids = [r["repo_id"] for r in out["datasets"]]
        assert ids.count("lerobot/pusht") == 1, "the local copy trains offline; the Hub duplicate is noise"
        assert out["datasets"][0]["local"] is True

    def test_the_limit_is_a_promise_even_when_local_rows_are_many(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # THE BUG I NEARLY SHIPPED: `[: max(limit, len(local))]` (which
        # checkpoints.clamp_limit's docstring exists to warn about) means a
        # type-ahead asking for 1 row gets every local row instead.
        monkeypatch.setattr(training, "local_datasets",
                            lambda q="": [{"root": f"/d/{i}", "repo_id": f"d{i}"} for i in range(9)])
        _fake_hub(monkeypatch, [_Ds("hub/one")])
        out = training.search_datasets("", 2)
        assert len(out["datasets"]) == 2
        assert out["total_matched"] == 10, "the count before the cut is still reported"
        assert all(r["local"] for r in out["datasets"]), "truncation drops hub rows before local ones"

    def test_no_matches_is_not_reported_as_a_problem(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(training, "local_datasets", lambda q="": [])
        _fake_hub(monkeypatch, [])
        out = training.search_datasets("zzz", 5)
        assert out["datasets"] == [] and out["problem"] is None

    def test_the_hub_half_failing_still_returns_local_rows_with_the_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(training, "local_datasets", lambda q="": [{"root": "/d", "repo_id": "mine"}])
        _fake_hub(monkeypatch, [], boom=OSError("offline"))
        out = training.search_datasets("", 5)
        assert [r["repo_id"] for r in out["datasets"]] == ["mine"]
        assert out["problem"] and "local datasets only" in out["problem"]

    def test_an_auth_probe_failure_cannot_break_search(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(training, "local_datasets", lambda q="": [])
        _fake_hub(monkeypatch, [_Ds("a/b")])
        from strands_robots.dashboard import checkpoints

        def _boom() -> dict[str, Any]:
            raise RuntimeError("hub client exploded")

        monkeypatch.setattr(checkpoints, "hf_auth_state", _boom)
        out = training.search_datasets("", 5)
        assert out["datasets"][0]["repo_id"] == "a/b"
        assert out["hf_auth"]["authenticated"] is False
        assert "unavailable" in out["hf_auth"]["detail"]
