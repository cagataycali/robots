"""``replay_episode`` reads back what this session recorded, and a Hub miss is a refusal with a remedy.

Measured on ``Robot("so101", mode="sim")``: ``start_recording(root=<dir>,
repo_id="lab/wave")`` → one episode → ``stop_recording`` →
``replay_episode(repo_id="lab/wave", episode=0)``. The reply was the raw
HuggingFace 404 - "Repository Not Found ... Request ID ... make sure you are
authenticated" - for a dataset the same sim had written a moment earlier. An
``owner/name`` id with no root is read as ``$HF_LEROBOT_HOME/{repo_id}`` and,
on the miss, fetched from the Hub; the directory the recording went to was
never tried, and nothing in the reply named it or the ``root=`` remedy.

``start_recording`` now stashes the id beside the root it already stashed, and
``PolicyRunner.replay`` resolves an id THIS sim recorded to a non-default
directory back to that directory (saying so in the reply; the default location
wins when it exists). A Hub 404 that still happens is translated into the
directory that was tried, the ``root=`` remedy, and where this session last
recorded. Explicit roots, path-shaped ids and every other load error are
untouched.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import requests
from huggingface_hub.errors import OfflineModeIsEnabled

from strands_robots.dataset_source import resolve_dataset_dir
from strands_robots.simulation.policy_runner import PolicyRunner
from strands_robots.simulation.recording import DatasetRecordingMixin
from tests.simulation.test_policy_runner import FakeSim


class RecordedSim(DatasetRecordingMixin, FakeSim):
    """A sim whose recording state is a mapping the test owns.

    Mixes in the production :class:`DatasetRecordingMixin` so the seams the
    runner reads (``_active_dataset_repo_id`` / ``_active_dataset_root``) are the
    real ones, and overrides only ``_recording_state`` - which is what the Isaac
    backend overrides too, its ``_world`` being the Isaac Sim ``World`` handle
    with no ``_backend_state`` mapping on it. So this fake is also the shape that
    catches a reader reaching into ``_world._backend_state`` directly.
    """

    def __init__(self, state: dict | None) -> None:
        super().__init__()
        self._state = state

    def _recording_state(self) -> dict | None:
        return self._state


def _runner(state: dict | None) -> PolicyRunner:
    return PolicyRunner(RecordedSim(state))


class _RepositoryNotFoundError(Exception):
    pass


HUB_404 = (
    "404 Client Error. (Request ID: Root=1-abc;def)\n\nRepository Not Found for url: "
    "https://huggingface.co/api/datasets/lab/wave/refs.\nPlease make sure you specified the correct "
    "`repo_id` and `repo_type`.\nIf you are trying to access a private or gated repo, make sure you are authenticated"
)


class TestTheRootIsResolvedFromThisSessionsRecording:
    def test_an_id_this_sim_recorded_elsewhere_reads_back_from_there(self, tmp_path) -> None:
        root, note = _runner({"last_dataset_repo_id": "lab/wave", "last_dataset_root": str(tmp_path)})._replay_root(
            "lab/wave", None
        )
        assert root == str(tmp_path)
        assert "where this session recorded lab/wave" in note and str(tmp_path) in note

    def test_an_explicit_root_is_never_overridden(self, tmp_path) -> None:
        root, note = _runner({"last_dataset_repo_id": "lab/wave", "last_dataset_root": str(tmp_path)})._replay_root(
            "lab/wave", "/elsewhere"
        )
        assert (root, note) == ("/elsewhere", "")

    def test_a_path_shaped_id_is_left_to_the_loader(self, tmp_path) -> None:
        state = {"last_dataset_repo_id": str(tmp_path), "last_dataset_root": str(tmp_path)}
        assert _runner(state)._replay_root(str(tmp_path), None) == (None, "")

    def test_another_id_keeps_the_absent_root(self, tmp_path) -> None:
        state = {"last_dataset_repo_id": "lab/wave", "last_dataset_root": str(tmp_path)}
        assert _runner(state)._replay_root("lab/other", None) == (None, "")

    def test_a_recording_at_the_default_location_keeps_the_absent_root(self, tmp_path, monkeypatch) -> None:
        # lerobot reads $HF_LEROBOT_HOME once at import, so redirect the home
        # this repo resolves through rather than the environment.
        import strands_robots.dataset_source as dataset_source

        monkeypatch.setattr(dataset_source, "_lerobot_home", lambda: tmp_path)
        default = Path(resolve_dataset_dir("lab/wave", None))
        default.mkdir(parents=True)
        state = {"last_dataset_repo_id": "lab/wave", "last_dataset_root": str(default)}
        assert _runner(state)._replay_root("lab/wave", None) == (None, "")

    def test_a_directory_that_is_gone_keeps_the_absent_root(self, tmp_path) -> None:
        state = {"last_dataset_repo_id": "lab/wave", "last_dataset_root": str(tmp_path / "deleted")}
        assert _runner(state)._replay_root("lab/wave", None) == (None, "")

    def test_no_world_no_resolution(self) -> None:
        assert _runner(None)._replay_root("lab/wave", None) == (None, "")

    def test_the_state_is_read_through_the_engine_seam_not_the_world(self, tmp_path) -> None:
        # The Isaac backend keeps its recording state behind
        # ``_recording_state()`` because its ``_world`` is the engine's own World
        # handle. A reader that reaches into ``_world._backend_state`` resolves
        # nothing there, and this runner serves every backend.
        sim = RecordedSim({"last_dataset_repo_id": "lab/wave", "last_dataset_root": str(tmp_path)})
        assert not hasattr(sim._world, "_backend_state")
        assert PolicyRunner(sim)._replay_root("lab/wave", None)[0] == str(tmp_path)

    def test_a_dataset_at_the_default_location_is_not_shadowed(self, tmp_path, monkeypatch) -> None:
        # An absent root has always read the default location, so a recording
        # elsewhere must not move the directory under a call that worked.
        import strands_robots.dataset_source as dataset_source

        monkeypatch.setattr(dataset_source, "_lerobot_home", lambda: tmp_path / "home")
        (Path(resolve_dataset_dir("lab/wave", None)) / "meta").mkdir(parents=True)
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        state = {"last_dataset_repo_id": "lab/wave", "last_dataset_root": str(elsewhere)}
        assert _runner(state)._replay_root("lab/wave", None) == (None, "")


class TestEveryBackendRecordsTheIdWithTheRoot:
    """One resolve-and-stash, so no backend can record a root without its id."""

    def test_the_resolved_dir_is_stashed_with_the_id(self, tmp_path) -> None:
        sim = RecordedSim({})
        assert sim._stash_dataset_target("lab/wave", str(tmp_path)) == tmp_path
        assert sim._recording_state() == {"last_dataset_root": str(tmp_path), "last_dataset_repo_id": "lab/wave"}

    def test_the_root_is_stashed_in_exactly_one_module(self) -> None:
        # The id was added to one backend's longhand copy of resolve-and-stash,
        # which left the other two recording datasets no reader could locate.
        import strands_robots

        pkg = Path(strands_robots.__file__).parent
        writers = sorted(
            str(f.relative_to(pkg)) for f in pkg.rglob("*.py") if '"last_dataset_root"] =' in f.read_text()
        )
        assert writers == ["simulation/recording.py"]

    def test_the_id_and_the_root_are_read_from_the_same_source(self) -> None:
        # A live recorder's root paired with a stale stashed id would compare
        # unequal to the id the caller asked for, and resolve nothing.
        sim = RecordedSim(
            {
                "dataset_recorder": SimpleNamespace(repo_id="lab/live", root="/live"),
                "last_dataset_repo_id": "lab/stale",
                "last_dataset_root": "/stale",
            }
        )
        assert (sim._active_dataset_repo_id(), sim._active_dataset_root()) == ("lab/live", "/live")


class TestAHubMissIsTranslated:
    def test_names_the_default_tried_and_the_remedy(self) -> None:
        text = _runner({})._replay_load_failure("lab/wave", None, _RepositoryNotFoundError(HUB_404))
        assert text.startswith("No dataset 'lab/wave' at the local default ")
        assert str(resolve_dataset_dir("lab/wave", None)) in text
        assert "no Hub repository by that name" in text
        assert "pass root=" in text
        assert "Request ID" not in text and "authenticated" not in text

    def test_names_where_this_session_last_recorded(self, tmp_path) -> None:
        state = {"last_dataset_repo_id": "lab/other", "last_dataset_root": str(tmp_path)}
        text = _runner(state)._replay_load_failure("lab/wave", None, Exception(HUB_404))
        assert f"This session last recorded lab/other to {tmp_path}" in text

    def test_the_hint_is_absent_when_nothing_was_recorded(self) -> None:
        assert "This session last recorded" not in _runner({})._replay_load_failure(
            "lab/wave", None, Exception(HUB_404)
        )

    def test_an_explicit_root_names_the_directory_that_was_read(self, tmp_path) -> None:
        # Was pinned verbatim ("explicit roots ... are untouched"), so the one
        # caller who DID name a directory got the raw 404 - request id,
        # repo_type advice, a gated-repo paragraph - and the directory they
        # chose appeared nowhere in the reply.
        text = _runner({})._replay_load_failure("lab/wave", str(tmp_path), Exception(HUB_404))
        assert text.startswith(f"No dataset 'lab/wave' in the root= directory {tmp_path} ")
        assert "no Hub repository by that name" in text
        assert "the one holding meta/" in text
        assert "Request ID" not in text and "authenticated" not in text

    def test_other_errors_are_verbatim(self) -> None:
        err = ValueError("Episode 5 out of range (0-0)")
        assert _runner({})._replay_load_failure("lab/wave", None, err) == str(err)


class TestAnUnreachableHubIsNotAMissingDataset:
    """A Hub that could not be reached said only ``[Errno 111] Connection refused``.

    Measured on the real door with ``HF_ENDPOINT`` closed, an unresolvable host
    and ``HF_HUB_OFFLINE=1``: the reply named neither the dataset, the
    directory that was read, nor the Hub. The classes are the real ones, so a
    Hub client whose exception hierarchy moves again is caught here - lerobot
    swapped ``requests`` for ``httpx``, which is why matching the message
    ("Max retries exceeded") does not hold.
    """

    @pytest.mark.parametrize(
        "error",
        [
            pytest.param(httpx.ConnectError("[Errno 111] Connection refused"), id="connection-refused"),
            pytest.param(httpx.ConnectError("[Errno -2] Name or service not known"), id="no-such-host"),
            pytest.param(httpx.ConnectTimeout("timed out"), id="timeout"),
            pytest.param(
                OfflineModeIsEnabled(
                    "Cannot reach https://huggingface.co/api/datasets/lab/wave/refs: offline mode is "
                    "enabled. To disable it, please unset the `HF_HUB_OFFLINE` environment variable."
                ),
                id="offline-mode",
            ),
            pytest.param(requests.exceptions.ConnectionError("Max retries exceeded"), id="requests-client"),
        ],
    )
    def test_it_names_the_dataset_the_directory_and_the_hub(self, error, tmp_path) -> None:
        text = _runner({})._replay_load_failure("lab/wave", str(tmp_path), error)
        assert text.startswith(
            f"No local copy of 'lab/wave' at {tmp_path} and the Hugging Face Hub could not be reached "
        )
        # The library's own text is kept here (it names the endpoint, and
        # offline mode names the variable to unset) - unlike the 404's.
        assert f"({error})" in text
        assert "pass root=" in text.lower()

    def test_a_404_is_still_read_as_a_missing_repository(self) -> None:
        assert "could not be reached" not in _runner({})._replay_load_failure(
            "lab/wave", None, _RepositoryNotFoundError(HUB_404)
        )


class TestTheDatasetsOnDiskAreNamed:
    """The answer to a typo, or to a ``root=`` aimed one directory too high."""

    @staticmethod
    def _make(parent: Path, names, decoys=()) -> None:
        for name in names:
            (parent / name / "meta").mkdir(parents=True)
        for name in decoys:
            (parent / name).mkdir(parents=True)

    def test_a_typo_is_answered_with_the_datasets_beside_it(self, tmp_path, monkeypatch) -> None:
        from strands_robots import dataset_source

        self._make(tmp_path / "lab", ["wave"], decoys=["notes"])
        monkeypatch.setattr(dataset_source, "_lerobot_home", lambda: tmp_path)
        text = _runner({})._replay_load_failure("lab/waev", None, _RepositoryNotFoundError(HUB_404))
        assert f"Datasets on disk in {tmp_path / 'lab'}: wave." in text
        assert "notes" not in text

    def test_a_root_one_level_too_high_is_answered_with_what_is_inside_it(self, tmp_path) -> None:
        self._make(tmp_path, ["wave"])
        text = _runner({})._replay_load_failure("lab/wave", str(tmp_path), _RepositoryNotFoundError(HUB_404))
        assert f"Datasets on disk in {tmp_path}: wave." in text

    def test_a_long_list_is_capped(self, tmp_path) -> None:
        self._make(tmp_path, [f"ds{i}" for i in range(10)])
        text = _runner({})._replay_load_failure("lab/wave", str(tmp_path), _RepositoryNotFoundError(HUB_404))
        assert f"Datasets on disk in {tmp_path}: ds0, ds1, ds2, ds3, ds4, ds5, ds6, ds7, ...." in text

    def test_nothing_is_offered_when_no_dataset_is_there(self, tmp_path) -> None:
        assert "Datasets on disk" not in _runner({})._replay_load_failure(
            "lab/wave", str(tmp_path), _RepositoryNotFoundError(HUB_404)
        )


class TestOnTheRealSim:
    @staticmethod
    def _text(result) -> str:
        return " ".join(c.get("text", "") for c in result["content"] if isinstance(c, dict))

    def test_record_to_a_custom_root_then_replay_by_id(self, tmp_path) -> None:
        pytest.importorskip("lerobot")
        from strands_robots import Robot

        sim = Robot("so101", mode="sim")
        try:
            assert sim.start_recording(root=str(tmp_path), task="t", fps=30, repo_id="lab/wave")["status"] == "success"
            sim.set_joint_positions(robot_name="so101", positions=[0.3, 0, 0, 0, 0, 0], hold=True)
            sim.step(n_steps=300)
            assert sim.stop_recording()["status"] == "success"

            result = sim.replay_episode(repo_id="lab/wave", episode=0)
            assert result["status"] == "success", self._text(result)
            assert f"Root: {tmp_path}" in self._text(result)
            json = next(c["json"] for c in result["content"] if isinstance(c, dict) and "json" in c)
            assert json["root"] == str(tmp_path)
            assert json["frames_with_action"] > 0
        finally:
            sim.cleanup()
