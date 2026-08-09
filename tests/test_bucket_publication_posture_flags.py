"""The publication-posture flags of the dataset-publication surface are checked.

``sync_dataset_to_bucket`` allowlist-validates ``bucket`` and ``run_id`` before
any subprocess, and its own docstring says so. The three flags in that same
signature - ``create``, ``private`` and ``delete`` - were read by truthiness,
and so was ``private`` on :meth:`DatasetRecorder.push_to_hub` beside it. Each
selects a *posture* on a remote store rather than scaling a quantity, which is
:func:`~strands_robots.utils.boolean_flag_error`'s documented family.

Read by truthiness these fail toward the permissive posture in *both*
directions. Every non-empty string is truthy, so ``delete="false"`` - the
spelling an operator reaches for when opting out - appended ``--delete`` to
``hf sync`` and mirror-deleted remote files absent locally. Every falsy
non-boolean takes the other branch, so ``private=0`` dropped ``--private`` and
created the bucket public. Both returned ``status="success"``.

The module is imported under an alias so a test can reach the private
``_hf_executable`` / ``_huggingface_hub_version_error`` probes it monkeypatches.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
from typing import Any

import pytest

from strands_robots import dataset_recorder as recorder_mod
from strands_robots.utils import boolean_flag_error

#: Flags on this module's publication surface. Each selects a posture on a
#: remote store: whether a bucket is created, whether it is private, and
#: whether the sync mirror-deletes remote files that are absent locally.
PUBLICATION_FLAGS = ("create", "private", "delete")

#: Truthy non-booleans. Every one of these selected the *permissive* posture:
#: ``delete`` mirror-deleted, ``create`` created a bucket.
TRUTHY_NON_BOOLEANS: list[Any] = ["false", "no", "off", "0", "true", 1, 2.5, float("nan"), [0], {"a": 1}]

#: Falsy non-booleans. Every one of these selected the *other* posture without
#: ever being a declared spelling of it: ``private=0`` created a public bucket.
FALSY_NON_BOOLEANS: list[Any] = [0, 0.0, "", None, [], {}]

UNUSABLE: list[Any] = [*TRUTHY_NON_BOOLEANS, *FALSY_NON_BOOLEANS]


def _label(value: Any) -> str:
    """Stable, address-free id for a probe value."""
    return f"{type(value).__name__}-{value!r}"


IDS = [_label(v) for v in UNUSABLE]


class _RecordingSubprocess:
    """Records every argv ``sync_dataset_to_bucket`` would run, and runs none."""

    def __init__(self) -> None:
        self.argv: list[list[str]] = []
        self.returncode = 0
        self.stdout = ""
        self.stderr = ""

    def run(self, cmd: Any, **_kw: Any) -> _RecordingSubprocess:
        self.argv.append(list(cmd))
        return self

    @property
    def mirror_deleted(self) -> bool:
        """Whether any recorded argv carries ``hf sync --delete``."""
        return any("--delete" in c for c in self.argv)

    @property
    def created_private(self) -> bool:
        """Whether any recorded argv creates the bucket with ``--private``."""
        return any("--private" in c for c in self.argv)

    @property
    def created_bucket(self) -> bool:
        """Whether any recorded argv runs ``hf buckets create``."""
        return any(len(c) > 1 and c[1] == "buckets" for c in self.argv)


@pytest.fixture
def wire(monkeypatch: pytest.MonkeyPatch) -> _RecordingSubprocess:
    """A recorded ``hf`` CLI: the flags reach an argv, never a real process."""
    rec = _RecordingSubprocess()
    import subprocess

    monkeypatch.setattr(subprocess, "run", rec.run)
    monkeypatch.setattr(recorder_mod, "_hf_executable", lambda: "/usr/bin/hf")
    monkeypatch.setattr(recorder_mod, "_huggingface_hub_version_error", lambda: None)
    return rec


@pytest.fixture
def finalized(tmp_path: pathlib.Path) -> pathlib.Path:
    """A dataset root that passes the ``meta/`` finalization check."""
    (tmp_path / "meta").mkdir()
    return tmp_path


def _sync(root: pathlib.Path, **kwargs: Any) -> dict[str, Any]:
    """Funnel so deliberately off-type flags need no per-call suppression."""
    return recorder_mod.sync_dataset_to_bucket(root, "acme/robotdata", run_id="run1", **kwargs)


class _FakeHubDataset:
    """Records what reaches LeRobot's ``push_to_hub``, and reaches no Hub."""

    repo_id = "acme/robotdata"

    def __init__(self, root: str = "/nonexistent") -> None:
        self.pushed: list[Any] = []
        # ``sync_to_bucket`` reads this to build the local root it forwards.
        self.root = root

    def push_to_hub(self, tags: Any = None, private: Any = None) -> None:
        self.pushed.append(private)


def _recorder(dataset: _FakeHubDataset, *, frames: int = 10, episodes: int = 1) -> recorder_mod.DatasetRecorder:
    """A recorder with a chosen frame/episode count, built without LeRobot.

    ``DatasetRecorder.__init__`` only records the dataset handed to it, so the
    fake above constructs the shipped class fully; the counts are then set to
    the state under test. Constructing the real class rather than subclassing it
    keeps the methods exercised here the shipped ones.
    """
    recorder = recorder_mod.DatasetRecorder(dataset)
    recorder.frame_count = frames
    recorder.episode_count = episodes
    return recorder


def _push(recorder: recorder_mod.DatasetRecorder, **kwargs: Any) -> dict[str, Any]:
    """Funnel so deliberately off-type flags need no per-call suppression."""
    return recorder.push_to_hub(**kwargs)


def _sync_via_recorder(recorder: recorder_mod.DatasetRecorder, **kwargs: Any) -> dict[str, Any]:
    """Funnel for the delegate, for the same reason as :func:`_push`."""
    return recorder.sync_to_bucket("acme/robotdata", run_id="run1", **kwargs)


class TestTheDoubleIsARealRecorder:
    """The surface under test is driven on the production class, not a stand-in.

    A hand-written ``__init__`` sets whichever attributes the methods under test
    happen to read today - three of the thirteen
    :meth:`~strands_robots.dataset_recorder.DatasetRecorder.__init__` sets - so
    ``push_to_hub`` and ``sync_to_bucket`` would be read off a recorder
    production cannot build, and the omission would be silent because neither
    method reads the other ten. Comparing the double's attributes against a
    reference recorder's is what keeps the stand-in from coming back.
    """

    def test_the_double_carries_every_attribute_production_sets(self) -> None:
        reference = recorder_mod.DatasetRecorder(_FakeHubDataset())
        assert set(vars(_recorder(_FakeHubDataset()))) == set(vars(reference))

    def test_the_double_reports_the_counts_it_was_given(self) -> None:
        recorder = _recorder(_FakeHubDataset(), frames=7, episodes=3)
        assert (recorder.frame_count, recorder.episode_count) == (7, 3)


class TestTheDomainIsTheSharedOne:
    """The values below are refused by the shared posture domain, not locally."""

    @pytest.mark.parametrize("value", UNUSABLE, ids=IDS)
    def test_the_shared_domain_refuses_every_probe_value(self, value: Any) -> None:
        assert boolean_flag_error(value, "delete", "sync_dataset_to_bucket") is not None

    @pytest.mark.parametrize("flag", PUBLICATION_FLAGS)
    def test_the_bucket_sync_message_is_the_shared_one_verbatim(
        self, wire: _RecordingSubprocess, finalized: pathlib.Path, flag: str
    ) -> None:
        result = _sync(finalized, **{flag: "false"})
        assert result["message"] == boolean_flag_error("false", flag, "sync_dataset_to_bucket")

    def test_the_push_message_is_the_shared_one_verbatim(self) -> None:
        result = _push(_recorder(_FakeHubDataset()), private="false")
        assert result["message"] == boolean_flag_error("false", "private", "push_to_hub")


class TestTheBucketSyncRefusesANonBooleanPosture:
    """Every flag, every unusable value, on the function that owns the rule."""

    @pytest.mark.parametrize("flag", PUBLICATION_FLAGS)
    @pytest.mark.parametrize("value", UNUSABLE, ids=IDS)
    def test_an_unusable_flag_is_refused_and_runs_no_cli(
        self, wire: _RecordingSubprocess, finalized: pathlib.Path, flag: str, value: Any
    ) -> None:
        result = _sync(finalized, **{flag: value})
        assert result["status"] == "error"
        assert flag in result["message"]
        assert wire.argv == [], "the refused call reached the hf CLI"


class TestTheTruthinessReadFailedTowardThePermissivePosture:
    """The two directions the old read inverted, pinned as the reason it changed."""

    @pytest.mark.parametrize("value", ["false", "no", "off", "0"], ids=["false", "no", "off", "zero"])
    def test_a_truthy_spelling_of_off_no_longer_mirror_deletes(
        self, wire: _RecordingSubprocess, finalized: pathlib.Path, value: str
    ) -> None:
        result = _sync(finalized, delete=value)
        assert result["status"] == "error"
        assert not wire.mirror_deleted, "hf sync --delete ran for a spelling that reads as off"

    @pytest.mark.parametrize("value", [0, "", None, []], ids=["zero", "empty-str", "None", "empty-list"])
    def test_a_falsy_non_boolean_no_longer_creates_a_public_bucket(
        self, wire: _RecordingSubprocess, finalized: pathlib.Path, value: Any
    ) -> None:
        result = _sync(finalized, private=value)
        assert result["status"] == "error"
        assert not wire.created_bucket, "a bucket was created for a value that is not a posture"


class TestTheRefusalPrecedesEveryProbeAndSideEffect:
    """Guard placement: nothing is located, run or published for a refused flag."""

    @pytest.mark.parametrize("flag", PUBLICATION_FLAGS)
    def test_the_bucket_sync_refusal_precedes_the_cli_probe(
        self, monkeypatch: pytest.MonkeyPatch, finalized: pathlib.Path, flag: str
    ) -> None:
        def fatal() -> str:
            raise AssertionError("the refused call probed for the hf CLI")

        monkeypatch.setattr(recorder_mod, "_hf_executable", fatal)
        result = _sync(finalized, **{flag: "false"})
        assert result["status"] == "error"
        assert flag in result["message"]

    def test_the_bucket_sync_reports_the_flag_with_no_cli_installed(
        self, monkeypatch: pytest.MonkeyPatch, finalized: pathlib.Path
    ) -> None:
        """The same mistake reports identically whether or not ``hf`` exists."""
        monkeypatch.setattr(recorder_mod, "_hf_executable", lambda: None)
        result = _sync(finalized, delete="false")
        assert "delete" in result["message"]
        assert "hf` CLI not found" not in result["message"]

    def test_the_push_refusal_reaches_no_hub_call(self) -> None:
        dataset = _FakeHubDataset()
        result = _push(_recorder(dataset), private="false")
        assert result["status"] == "error"
        assert dataset.pushed == [], "the refused call published to the Hub"

    def test_the_push_refusal_does_not_depend_on_the_recorder_being_non_empty(self) -> None:
        """An empty recorder still reports the flag, not its own state."""
        empty = _recorder(_FakeHubDataset(), frames=0, episodes=0)
        result = _push(empty, private="false")
        assert "private" in result["message"]
        assert "empty dataset" not in result["message"]


class TestThePushVisibilityFlagIsChecked:
    """``private`` decides whether a published dataset is world-readable."""

    @pytest.mark.parametrize("value", UNUSABLE, ids=IDS)
    def test_an_unusable_visibility_is_refused(self, value: Any) -> None:
        dataset = _FakeHubDataset()
        result = _push(_recorder(dataset), private=value)
        assert result["status"] == "error"
        assert "private" in result["message"]
        assert dataset.pushed == []

    @pytest.mark.parametrize("value", [True, False], ids=["private", "public"])
    def test_a_boolean_visibility_is_forwarded_verbatim(self, value: bool) -> None:
        dataset = _FakeHubDataset()
        result = _push(_recorder(dataset), private=value)
        assert result["status"] == "success"
        assert dataset.pushed == [value]


class TestTheDelegateInheritsTheRule:
    """``sync_to_bucket`` forwards the flags, so it inherits the refusal."""

    @pytest.mark.parametrize("flag", PUBLICATION_FLAGS)
    def test_the_recorder_method_refuses_the_same_value(
        self, wire: _RecordingSubprocess, finalized: pathlib.Path, flag: str
    ) -> None:
        recorder = _recorder(_FakeHubDataset(root=str(finalized)))
        result = _sync_via_recorder(recorder, **{flag: "false"})
        assert result["status"] == "error"
        assert flag in result["message"]
        assert wire.argv == []

    def test_the_recorder_method_still_syncs_with_a_usable_posture(
        self, wire: _RecordingSubprocess, finalized: pathlib.Path
    ) -> None:
        """The delegation clause is not satisfied by a delegate that never runs."""
        recorder = _recorder(_FakeHubDataset(root=str(finalized)))
        result = _sync_via_recorder(recorder, delete=True)
        assert result["status"] == "success"
        assert wire.mirror_deleted


class TestAUsablePostureIsUnchanged:
    """The accepted domain is untouched: both booleans still reach the same argv."""

    def test_the_default_posture_builds_the_same_two_commands(
        self, wire: _RecordingSubprocess, finalized: pathlib.Path
    ) -> None:
        result = _sync(finalized)
        assert result["status"] == "success"
        assert wire.created_bucket and wire.created_private
        assert not wire.mirror_deleted

    def test_an_explicit_mirror_delete_still_reaches_the_cli(
        self, wire: _RecordingSubprocess, finalized: pathlib.Path
    ) -> None:
        result = _sync(finalized, delete=True)
        assert result["status"] == "success"
        assert wire.mirror_deleted

    def test_an_explicit_public_bucket_still_omits_private(
        self, wire: _RecordingSubprocess, finalized: pathlib.Path
    ) -> None:
        result = _sync(finalized, private=False)
        assert result["status"] == "success"
        assert wire.created_bucket and not wire.created_private

    def test_skipping_creation_still_syncs(self, wire: _RecordingSubprocess, finalized: pathlib.Path) -> None:
        result = _sync(finalized, create=False)
        assert result["status"] == "success"
        assert not wire.created_bucket
        assert any(len(c) > 1 and c[1] == "sync" for c in wire.argv)

    @pytest.mark.parametrize("value", [True, False], ids=["true", "false"])
    def test_numpy_booleans_are_honoured_like_python_ones(
        self, wire: _RecordingSubprocess, finalized: pathlib.Path, value: bool
    ) -> None:
        """The shared domain accepts ``np.bool_``, so this surface must too."""
        np = pytest.importorskip("numpy")
        result = _sync(finalized, delete=np.bool_(value))
        assert result["status"] == "success"
        assert wire.mirror_deleted is value


# ---------------------------------------------------------------------------
# Structural sweep: no publication-posture flag may reach the wire unchecked.
# ---------------------------------------------------------------------------

_MODULE_PATH = pathlib.Path(inspect.getfile(recorder_mod))
_DOMAIN = "boolean_flag_error"


def _module_tree(source: str | None = None) -> ast.Module:
    return ast.parse(source if source is not None else _MODULE_PATH.read_text())


def _boolean_flags(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Publication-posture flags this function declares, ``bool``-annotated."""
    out = []
    for arg in fn.args.args + fn.args.kwonlyargs:
        if arg.arg not in PUBLICATION_FLAGS or arg.annotation is None:
            continue
        if ast.unparse(arg.annotation).strip() == "bool":
            out.append(arg.arg)
    return out


def _guarded(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Flags handed to the shared domain, directly or through a guard loop."""
    guarded: set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
            if name == _DOMAIN:
                for arg in list(node.args) + [kw.value for kw in node.keywords]:
                    if isinstance(arg, ast.Name):
                        guarded.add(arg.id)
        # for flag_name, flag_value in (("create", create), ...):
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Tuple | ast.List):
            for element in node.iter.elts:
                if isinstance(element, ast.Tuple):
                    for item in element.elts:
                        if isinstance(item, ast.Name):
                            guarded.add(item.id)
    return guarded


def _forwarded(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Flags passed on by keyword under their own name, so the callee decides."""
    forwarded: set[str] = set()
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg and isinstance(kw.value, ast.Name) and kw.value.id == kw.arg:
                forwarded.add(kw.arg)
    return forwarded


def _surfaces(source: str | None = None) -> dict[str, tuple[list[str], set[str], set[str]]]:
    """Public surfaces declaring a publication flag, with their verdicts."""
    found: dict[str, tuple[list[str], set[str], set[str]]] = {}
    for node in ast.walk(_module_tree(source)):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if node.name.startswith("_"):
            continue
        flags = _boolean_flags(node)
        if flags:
            found[node.name] = (flags, _guarded(node), _forwarded(node))
    return found


class TestEveryPublicationFlagIsCheckedOrForwarded:
    """A flag reaching the wire unchecked cannot ship from this module."""

    def test_the_expected_surfaces_are_the_ones_found(self) -> None:
        """Non-vacuity: a mis-rooted scan cannot report a clean sweep."""
        assert set(_surfaces()) == {"sync_dataset_to_bucket", "sync_to_bucket", "push_to_hub"}

    def test_the_owner_checks_all_three_flags(self) -> None:
        flags, guarded, _forward = _surfaces()["sync_dataset_to_bucket"]
        assert sorted(flags) == sorted(PUBLICATION_FLAGS)
        assert set(flags) <= guarded

    def test_no_surface_leaves_a_flag_unchecked_and_unforwarded(self) -> None:
        adrift = {
            name: sorted(set(flags) - guarded - forwarded)
            for name, (flags, guarded, forwarded) in _surfaces().items()
            if set(flags) - guarded - forwarded
        }
        assert not adrift, f"publication flags reaching the wire unchecked: {adrift}"

    def test_the_sweep_detects_a_planted_unchecked_surface(self) -> None:
        """A scanner that silently matched nothing would look like a clean module."""
        planted = _MODULE_PATH.read_text() + (
            "\n\ndef publish_somewhere(target: str, *, delete: bool = False) -> None:\n"
            '    """Planted surface that reads a posture flag with no domain."""\n'
            "    if delete:\n"
            "        _run_mirror_delete(target)\n"
        )
        surfaces = _surfaces(planted)
        assert "publish_somewhere" in surfaces
        flags, guarded, forwarded = surfaces["publish_somewhere"]
        assert set(flags) - guarded - forwarded == {"delete"}
