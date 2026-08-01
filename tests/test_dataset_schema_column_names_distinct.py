"""``DatasetRecorder.create`` refuses schema column names it cannot honor.

``camera_keys``, ``joint_names`` and ``action_names`` declare the recorded
dataset's column names. Each has to be a list of distinct, non-blank names, and
neither way of getting that wrong could be honored as written:

* A single name passed as a bare string is iterable per character, so
  ``joint_names="gripper"`` declared seven columns (``g``, ``r``, ``i``, ``p``,
  ``p``, ``e``, ``r``). ``add_frame`` reads each declared name out of the
  observation - see the zero-fill behaviour pinned by
  ``test_add_frame_fills_missing_keys_with_zero`` in the sibling recorder unit
  tests - and none of those names is in it, so every column recorded 0.0 while
  ``create``, ``add_frame``, ``save_episode`` and ``finalize`` all succeeded.
* A repeated name collapses where it keys a dict and doubles where it indexes a
  position: ``camera_keys=["front", "front"]`` declared ONE camera column for
  the two the caller asked for, and ``joint_names=["j1", "j2", "j2"]`` recorded
  ``j2`` twice and the joint the caller meant not at all.

The rule is the shared name-list domain already applied by the three backends'
``start_recording`` (its ``cameras`` subset), the plain-MP4 recorders and every
provider's ``set_robot_state_keys``. The recorder those recording facades all
flush through - and the documented direct API - was the last consumer of that
vocabulary reaching it unvalidated.

The refusal is placed ahead of both side effects ``create`` has, which is what
these tests pin: ahead of the lazy lerobot import, so the same caller mistake
reports identically whether or not the dataset extra is installed (which is also
why every refusal test here runs without it), and ahead of the on-disk target,
which ``overwrite=True`` removes.
"""

import ast
import inspect
from pathlib import Path
from typing import Any

import pytest

from strands_robots import dataset_recorder as recorder_mod
from strands_robots.dataset_recorder import DatasetRecorder
from strands_robots.utils import name_list_error

# One list-of-names parameter per schema vocabulary that ``create`` declares.
NAME_LIST_PARAMS = ("camera_keys", "joint_names", "action_names")

# Values that cannot be honored as a list of distinct names, with the substring
# each one's message must carry so a caller is told which mistake they made.
UNUSABLE: list[tuple[str, Any, str]] = [
    ("bare_string", "gripper", "not a single string"),
    ("bare_bytes", b"gripper", "not a single string"),
    ("mapping", {"j1": 0, "j2": 0}, "not a mapping"),
    ("generator", (n for n in ("j1", "j2")), "must be a list of names"),
    ("non_str_entry", ["j1", 2, "j3"], "must be a name (str)"),
    ("blank_entry", ["j1", "", "j3"], "must be a non-blank name"),
    ("repeated", ["j1", "j2", "j2"], "must not repeat a name"),
]

# Values that mean "not supplied" (the schema is derived instead) or are a
# usable list of distinct names. None may be refused.
USABLE: list[tuple[str, Any]] = [
    ("none", None),
    ("empty_list", []),
    ("empty_tuple", ()),
    ("one_name", ["gripper"]),
    ("several_names", ["j1", "j2", "j3"]),
    ("tuple_of_names", ("j1", "j2")),
]


class _FakeLeRobotDataset:
    """Stand-in for ``LeRobotDataset``, recording whether ``create`` was reached."""

    calls: list[dict[str, Any]] = []

    def __init__(self, features: dict[str, Any]) -> None:
        self.repo_id = "local/fake"
        self.features = features
        self.meta = None

    @classmethod
    def create(cls, **kwargs: Any) -> "_FakeLeRobotDataset":
        cls.calls.append(kwargs)
        return cls(kwargs.get("features", {}))


def _create(**kwargs: Any) -> DatasetRecorder:
    """Call ``DatasetRecorder.create`` with keywords the signature disallows.

    Several tests here pass a value the parameter's annotation forbids (a bare
    string where ``list[str] | None`` is declared) because that is precisely the
    caller mistake the runtime guard exists for. Routing them through one
    ``**kwargs: Any`` funnel states that intent once instead of scattering
    per-call type suppressions.
    """
    return DatasetRecorder.create(**kwargs)


@pytest.fixture
def fake_lerobot(monkeypatch: pytest.MonkeyPatch) -> type[_FakeLeRobotDataset]:
    """Route ``create`` onto a fake dataset class, so no lerobot extra is needed."""
    _FakeLeRobotDataset.calls = []
    monkeypatch.setattr(recorder_mod, "_get_lerobot_dataset_class", lambda: _FakeLeRobotDataset)
    return _FakeLeRobotDataset


@pytest.fixture
def no_lerobot(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make reaching the lazy lerobot import fatal.

    A refusal must be decided before ``create`` probes the dataset extra, so the
    same caller mistake is reported the same way on a minimal install.
    """

    def _fatal() -> Any:
        raise AssertionError("the lerobot extra was probed before the name lists were checked")

    monkeypatch.setattr(recorder_mod, "_get_lerobot_dataset_class", _fatal)


class TestUnusableNameListsAreRefused:
    """Every schema name list is refused on the shared domain, before any work."""

    @pytest.mark.parametrize("param", NAME_LIST_PARAMS)
    @pytest.mark.parametrize(("label", "value", "expected"), UNUSABLE, ids=[c[0] for c in UNUSABLE])
    def test_refused_with_the_mistake_named(
        self, no_lerobot: None, param: str, label: str, value: Any, expected: str
    ) -> None:
        with pytest.raises(ValueError) as excinfo:
            _create(repo_id="local/probe", **{param: value})
        text = str(excinfo.value)
        assert expected in text, text
        assert param in text, text
        assert "DatasetRecorder.create" in text, text

    @pytest.mark.parametrize("param", NAME_LIST_PARAMS)
    @pytest.mark.parametrize(("label", "value"), USABLE, ids=[c[0] for c in USABLE])
    def test_usable_name_lists_still_reach_the_dataset(
        self, fake_lerobot: type[_FakeLeRobotDataset], param: str, label: str, value: Any
    ) -> None:
        """``None`` / ``[]`` still mean "not supplied"; distinct names are accepted."""
        _create(repo_id="local/probe", **{param: value})
        assert len(fake_lerobot.calls) == 1


class TestARefusedCreateLeavesTheTargetAlone:
    """The refusal precedes the on-disk work, so nothing is destroyed by it."""

    def test_overwrite_does_not_remove_an_existing_dataset(
        self, tmp_path: Path, fake_lerobot: type[_FakeLeRobotDataset]
    ) -> None:
        """``overwrite=True`` removes the target directory - a refusal must not."""
        root = tmp_path / "ds"
        (root / "meta").mkdir(parents=True)
        (root / "meta" / "info.json").write_text('{"fps": 30}')

        raised: Exception | None = None
        try:
            _create(repo_id="local/probe", root=str(root), joint_names="gripper", overwrite=True)
        except ValueError as exc:
            raised = exc

        # Assert the surviving target first: that is the consequence that matters,
        # and it is what a refusal arriving after ``_prepare_create_target`` would
        # already have destroyed.
        assert (root / "meta" / "info.json").is_file(), "the refused call deleted the dataset"
        assert fake_lerobot.calls == [], "the refused call still built a dataset"
        assert raised is not None, "the unusable joint_names was not refused"
        assert "not a single string" in str(raised), str(raised)

    def test_a_usable_call_does_reach_the_dataset_target(
        self, tmp_path: Path, fake_lerobot: type[_FakeLeRobotDataset]
    ) -> None:
        """The control: the same call with a real name list is not refused."""
        _create(repo_id="local/probe", root=str(tmp_path / "fresh"), joint_names=["gripper"])
        assert len(fake_lerobot.calls) == 1


class TestTheDomainCannotDriftFromTheSharedRule:
    """``create`` delegates to the shared domain rather than restating it."""

    @pytest.mark.parametrize("param", NAME_LIST_PARAMS)
    @pytest.mark.parametrize(
        ("label", "value"),
        [(lbl, val) for lbl, val, _ in UNUSABLE if lbl != "generator"] + USABLE,
        ids=[lbl for lbl, _, _ in UNUSABLE if lbl != "generator"] + [lbl for lbl, _ in USABLE],
    )
    def test_verdict_matches_name_list_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_lerobot: type[_FakeLeRobotDataset],
        param: str,
        label: str,
        value: Any,
    ) -> None:
        """A value the shared domain refuses is refused here, and vice versa."""
        shared_refuses = bool(value) and name_list_error(value, param, "x") is not None
        try:
            _create(repo_id="local/probe", **{param: value})
            create_refuses = False
        except ValueError:
            create_refuses = True
        assert create_refuses is shared_refuses, f"verdicts differ for {param}={value!r}"

    def test_every_name_list_parameter_of_create_is_guarded(self) -> None:
        """A new list-of-names parameter cannot join ``create`` unguarded."""
        source = inspect.getsource(DatasetRecorder.create)
        tree = ast.parse(inspect.cleandoc(source).replace("@classmethod\n", "", 1))
        func = tree.body[0]
        assert isinstance(func, ast.FunctionDef)

        declared = {
            arg.arg
            for arg in func.args.args + func.args.kwonlyargs
            if arg.annotation is not None and ast.unparse(arg.annotation) == "list[str] | None"
        }
        assert declared == set(NAME_LIST_PARAMS), declared

        # The guard iterates ``(value, "param_name")`` pairs; collect the value
        # each pair names so a parameter added to the signature but not to the
        # loop is reported by name.
        guarded: set[str] = set()
        for node in ast.walk(func):
            if not isinstance(node, ast.Tuple) or len(node.elts) != 2:
                continue
            value_node, name_node = node.elts
            if not isinstance(value_node, ast.Name) or not isinstance(name_node, ast.Constant):
                continue
            if name_node.value == value_node.id:
                guarded.add(value_node.id)
        assert declared <= guarded, f"unguarded name-list parameter(s): {sorted(declared - guarded)}"
