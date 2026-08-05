"""``StreamingDatasetReader.open`` refuses a numeric knob it cannot honor.

``open`` forwards ``tolerance_s`` / ``buffer_size`` / ``max_num_shards`` /
``seed`` into ``StreamingLeRobotDataset``, whose constructor validates only
``repo_type`` and stores the rest verbatim. Every consumer of the four is
downstream of the call that returned successfully, so an unusable value used to
surface - when it surfaced at all - as a NumPy error part-way through iteration,
a shard count that streamed nothing, or a grid check that answered the same way
for every input.

The premise tests here measure each consumer directly rather than asserting it
in prose, so the reason a value is refused stays true against the installed
lerobot/NumPy rather than against this module's description of them. They skip
when lerobot is absent; every other test in this file runs without it, which is
the point of validating ahead of the import.
"""

from __future__ import annotations

import math
import types
from typing import Any

import numpy as np
import pytest

import strands_robots.streaming_dataset as sd

# Values no consumer of these knobs can honor, grouped by why.
UNUSABLE_TOLERANCES = [-1.0, -1e-9, float("nan"), float("inf"), float("-inf"), True, "1e-4", None, [1e-4]]
UNUSABLE_COUNTS = [0, -1, -16, 2.7, float("nan"), float("inf"), True, False, "8", None, [8]]
UNUSABLE_SEEDS = [-1, -42, 2.7, float("nan"), float("inf"), True, False, "42", None, [7]]

# Accepted values, one per knob, that must keep reaching the constructor. Annotated
# because the lists are deliberately heterogeneous - a NumPy scalar tolerance has to
# be accepted alongside a Python float.
USABLE: dict[str, list[Any]] = {
    "tolerance_s": [0.0, 1e-4, 0.5, np.float32(1e-4)],
    "buffer_size": [1, 256, 1000],
    "max_num_shards": [1, 8, 16],
    "seed": [0, 42],
}


class _FakeStreaming:
    """Constructor-shaped stand-in that records the kwargs it was handed."""

    def __init__(self, repo_id: str, **kw: Any) -> None:
        self.repo_id = repo_id
        self.kw = kw
        self.num_frames = 1000
        self.num_episodes = 10
        self.fps = 30

    def __iter__(self) -> Any:
        yield {"observation.state": [0.0]}


@pytest.fixture
def fake_lerobot(monkeypatch: pytest.MonkeyPatch) -> type[_FakeStreaming]:
    """Inject the constructor stand-in ``_get_streaming_cls`` honors."""
    monkeypatch.setattr(sd, "StreamingLeRobotDataset", _FakeStreaming, raising=False)
    return _FakeStreaming


@pytest.fixture
def lerobot_import_is_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make reaching the lerobot import a test failure.

    Every refusal must be decided before it, so a refusal test that trips this
    is a guard-placement regression rather than a domain one.
    """

    def _explode() -> Any:
        raise AssertionError("the lerobot import was reached: the refusal must precede it")

    monkeypatch.setattr(sd, "_get_streaming_cls", _explode)


def _open(**kwargs: Any) -> Any:
    """Call ``open`` with a splat so a deliberately off-type value type-checks."""
    return sd.StreamingDatasetReader.open("org/ds", validate_deltas=False, **kwargs)


class TestTheNumericKnobsAreRefusedBeforeTheImport:
    """An unusable knob is a caller mistake, so it needs no lerobot installed."""

    @pytest.mark.parametrize("value", UNUSABLE_TOLERANCES, ids=repr)
    def test_an_unusable_tolerance_is_refused(self, value: Any, lerobot_import_is_fatal: None) -> None:
        with pytest.raises(ValueError, match="tolerance_s"):
            _open(tolerance_s=value)

    @pytest.mark.parametrize("value", UNUSABLE_COUNTS, ids=repr)
    def test_an_unusable_buffer_size_is_refused(self, value: Any, lerobot_import_is_fatal: None) -> None:
        with pytest.raises(ValueError, match="buffer_size"):
            _open(buffer_size=value)

    @pytest.mark.parametrize("value", UNUSABLE_COUNTS, ids=repr)
    def test_an_unusable_shard_count_is_refused(self, value: Any, lerobot_import_is_fatal: None) -> None:
        with pytest.raises(ValueError, match="max_num_shards"):
            _open(max_num_shards=value)

    @pytest.mark.parametrize("value", UNUSABLE_SEEDS, ids=repr)
    def test_an_unusable_seed_is_refused(self, value: Any, lerobot_import_is_fatal: None) -> None:
        with pytest.raises(ValueError, match="seed"):
            _open(seed=value)

    def test_the_message_names_the_surface_the_parameter_and_the_value(self, lerobot_import_is_fatal: None) -> None:
        with pytest.raises(ValueError) as excinfo:
            _open(max_num_shards=0)
        text = str(excinfo.value)
        assert text.startswith("open: "), text
        assert "max_num_shards" in text
        assert "0" in text


class TestAUsableKnobStillReachesTheConstructor:
    """The guard is additive: every honorable value is forwarded unchanged."""

    @pytest.mark.parametrize(
        ("param", "value"),
        [(p, v) for p, values in USABLE.items() for v in values],
        ids=[f"{p}={v!r}" for p, values in USABLE.items() for v in values],
    )
    def test_a_usable_value_is_forwarded_verbatim(
        self, param: str, value: Any, fake_lerobot: type[_FakeStreaming]
    ) -> None:
        reader = _open(**{param: value})
        assert reader.dataset.kw[param] == value

    def test_the_defaults_are_inside_every_domain(self, fake_lerobot: type[_FakeStreaming]) -> None:
        """A call passing none of the four must not be refused by its own defaults."""
        reader = _open()
        assert reader.dataset.kw["buffer_size"] == 1000
        assert reader.dataset.kw["max_num_shards"] == 16
        assert reader.dataset.kw["seed"] == 42
        assert reader.dataset.kw["tolerance_s"] == pytest.approx(1e-4)


class TestZeroIsFirstClassForTheToleranceOnly:
    """``0`` is the strictest grid match, and a degenerate size/shard count."""

    def test_a_zero_tolerance_is_accepted(self, fake_lerobot: type[_FakeStreaming]) -> None:
        assert _open(tolerance_s=0.0).dataset.kw["tolerance_s"] == 0.0

    def test_a_zero_tolerance_is_the_only_difference_from_the_shared_signed_domain(self) -> None:
        """The floor is the whole of ``_tolerance_error``'s own contribution."""
        from strands_robots.utils import finite_number_error

        for value in [0.0, 1e-4, float("nan"), True, "x", None, [1.0]]:
            shared = finite_number_error(value, "tolerance_s", "open")
            local = sd._tolerance_error(value)
            assert (local is None) == (shared is None), value
        assert sd._tolerance_error(-1.0) is not None
        assert finite_number_error(-1.0, "tolerance_s", "open") is None

    @pytest.mark.parametrize("param", ["buffer_size", "max_num_shards"])
    def test_a_zero_count_is_refused(self, param: str, lerobot_import_is_fatal: None) -> None:
        with pytest.raises(ValueError, match=param):
            _open(**{param: 0})

    def test_a_zero_seed_is_accepted(self, fake_lerobot: type[_FakeStreaming]) -> None:
        assert _open(seed=0).dataset.kw["seed"] == 0


class TestTheRefusalPrecedesEveryEffect:
    """Nothing is imported, constructed or fetched for a refused call."""

    def test_no_constructor_is_called(self, monkeypatch: pytest.MonkeyPatch) -> None:
        built: list[str] = []

        class _Recording(_FakeStreaming):
            def __init__(self, repo_id: str, **kw: Any) -> None:
                built.append(repo_id)
                super().__init__(repo_id, **kw)

        monkeypatch.setattr(sd, "StreamingLeRobotDataset", _Recording, raising=False)
        with pytest.raises(ValueError):
            _open(buffer_size=-1)
        assert built == []

    def test_a_usable_call_does_reach_the_constructor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-vacuity for the test above: the recorder does fire when it should."""
        built: list[str] = []

        class _Recording(_FakeStreaming):
            def __init__(self, repo_id: str, **kw: Any) -> None:
                built.append(repo_id)
                super().__init__(repo_id, **kw)

        monkeypatch.setattr(sd, "StreamingLeRobotDataset", _Recording, raising=False)
        _open(buffer_size=1)
        assert built == ["org/ds"]


class TestEveryNumericParameterOfOpenHasADomain:
    """The signature, the domain table and the values passed to it must agree.

    A knob added to ``open`` without an entry here is a knob forwarded raw into
    a constructor that validates only ``repo_type`` - which is how these four
    came to be unguarded - so the pairing is asserted rather than left to
    review.
    """

    @staticmethod
    def _open_node() -> Any:
        import ast
        import inspect
        import pathlib

        tree = ast.parse(pathlib.Path(inspect.getfile(sd)).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "open":
                return node
        raise AssertionError("StreamingDatasetReader.open not found")

    @staticmethod
    def _numeric_params(node: Any) -> set[str]:
        import ast

        found = set()
        for arg in list(node.args.args) + list(node.args.kwonlyargs):
            if arg.annotation is not None and ast.unparse(arg.annotation) in {"int", "float"}:
                found.add(arg.arg)
        return found

    def test_the_signature_declares_exactly_the_four_numeric_knobs(self) -> None:
        """Non-vacuity: a scanner finding nothing would pass every test below."""
        assert self._numeric_params(self._open_node()) == {
            "tolerance_s",
            "buffer_size",
            "max_num_shards",
            "seed",
        }

    def test_the_domain_table_covers_every_numeric_parameter(self) -> None:
        assert set(sd._NUMERIC_DOMAINS) == self._numeric_params(self._open_node())

    def test_the_checked_values_are_the_parameters_themselves(self) -> None:
        """The mapping ``open`` builds must read each knob, not a stale copy."""
        import ast

        node = self._open_node()
        for stmt in ast.walk(node):
            if (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and stmt.targets[0].id == "supplied"
                and isinstance(stmt.value, ast.Dict)
            ):
                keys = {k.value for k in stmt.value.keys if isinstance(k, ast.Constant)}
                values = {v.id for v in stmt.value.values if isinstance(v, ast.Name)}
                assert keys == set(sd._NUMERIC_DOMAINS)
                assert values == keys
                return
        raise AssertionError("open() does not build a `supplied` mapping of its numeric knobs")

    def test_a_planted_knob_without_a_domain_is_detected(self) -> None:
        """The scanner must fail for the shape it exists to catch."""
        import ast

        node = self._open_node()
        planted = ast.arg(arg="prefetch_depth", annotation=ast.Name(id="int"))
        node.args.kwonlyargs.append(planted)
        assert self._numeric_params(node) - set(sd._NUMERIC_DOMAINS) == {"prefetch_depth"}


class TestWhyEachValueCannotBeHonored:
    """Measure each consumer, so the reasons stay true against the installed deps."""

    def test_an_infinite_tolerance_accepts_an_off_grid_delta(self) -> None:
        """The grid check ``open`` replicates answers the same way for every input."""
        feature_utils = pytest.importorskip("lerobot.datasets.feature_utils")
        off_grid = {"observation.state": [0.0, -0.0167]}
        with pytest.raises(ValueError):
            feature_utils.check_delta_timestamps(off_grid, 30, 1e-4, raise_value_error=True)
        assert feature_utils.check_delta_timestamps(off_grid, 30, float("inf"), raise_value_error=True)

    def test_a_nan_tolerance_refuses_an_on_grid_delta(self) -> None:
        feature_utils = pytest.importorskip("lerobot.datasets.feature_utils")
        on_grid = {"observation.state": [0.0, -1 / 30, -2 / 30]}
        assert feature_utils.check_delta_timestamps(on_grid, 30, 1e-4, raise_value_error=True)
        with pytest.raises(ValueError):
            feature_utils.check_delta_timestamps(on_grid, 30, float("nan"), raise_value_error=True)

    @pytest.mark.parametrize("value", [0, -5])
    def test_a_non_positive_shard_count_iterates_no_shards(self, value: int) -> None:
        """``min(hf_shards, v)`` then ``range(num_shards)`` - so nothing streams."""
        assert list(range(min(16, value))) == []

    @pytest.mark.parametrize("value", [0, -5])
    def test_a_non_positive_buffer_size_has_no_reservoir_index(self, value: int) -> None:
        with pytest.raises(ValueError, match="high <= 0"):
            np.random.default_rng(0).integers(0, value, size=1)

    def test_a_fractional_shard_count_is_not_a_range_bound(self) -> None:
        # Bound through Any: the point is what the runtime does with the value the
        # clamp lets through, not what a type checker would have said about it.
        bound: Any = min(16, 2.7)
        with pytest.raises(TypeError, match="cannot be interpreted as an integer"):
            range(bound)

    @pytest.mark.parametrize("value", [-1, 2.7, float("nan")])
    def test_an_unusable_seed_has_no_generator(self, value: Any) -> None:
        with pytest.raises((ValueError, TypeError)):
            np.random.default_rng(value)

    def test_a_non_finite_shard_count_is_discarded_by_the_clamp(self) -> None:
        """``min`` keeps the left operand, so nan/inf silently mean "the default"."""
        assert min(16, float("nan")) == 16
        assert min(16, float("inf")) == 16


class TestTheDataloaderKnobsStayOutOfScope:
    """``dataloader`` hands its knobs to torch, which refuses them at construction."""

    def test_torch_refuses_an_unusable_batch_size_itself(self) -> None:
        torch = pytest.importorskip("torch")
        for value in [0, -8, 2.7, True, math.nan, "32"]:
            with pytest.raises(ValueError, match="batch_size"):
                torch.utils.data.DataLoader([1, 2, 3], batch_size=value)

    def test_dataloader_is_not_guarded_here(self) -> None:
        """Its knobs are absent from the table, and that is deliberate."""
        assert "batch_size" not in sd._NUMERIC_DOMAINS
        assert "num_workers" not in sd._NUMERIC_DOMAINS


class TestTheGuardDoesNotDisturbTheNeighbouringContracts:
    """The three non-numeric decisions ``open`` already made are unchanged."""

    def test_a_bucket_repo_type_still_raises_on_an_older_constructor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _Narrow:
            def __init__(self, repo_id: str) -> None:
                self.repo_id = repo_id
                self.num_frames = self.num_episodes = self.fps = 0

        monkeypatch.setattr(sd, "StreamingLeRobotDataset", _Narrow, raising=False)
        with pytest.raises(RuntimeError, match="repo_type"):
            _open(repo_type="bucket")

    def test_drop_videos_without_proprio_deltas_still_raises(self, fake_lerobot: type[_FakeStreaming]) -> None:
        with pytest.raises(ValueError, match="drop_videos"):
            _open(drop_videos=True)

    def test_a_missing_lerobot_still_reports_the_install_remedy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A usable call on an install without lerobot keeps its own message."""
        monkeypatch.setattr(sd, "StreamingLeRobotDataset", None, raising=False)
        monkeypatch.setitem(__import__("sys").modules, "lerobot.datasets", None)
        with pytest.raises(ImportError, match="strands-robots\\[lerobot\\]"):
            _open(buffer_size=256)


def test_the_module_needs_no_simulation_import(monkeypatch: pytest.MonkeyPatch) -> None:
    """The domains come from ``utils``, so the dataset layer stays independent."""
    import ast
    import inspect
    import pathlib

    tree = ast.parse(pathlib.Path(inspect.getfile(sd)).read_text(encoding="utf-8"))
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module is not None}
    assert not any(m.startswith("strands_robots.simulation") for m in imported), imported
    assert "strands_robots.utils" in imported


def test_the_domain_table_entries_are_callables_returning_a_reason() -> None:
    """Every entry answers with a string or None - never raises, never a bool."""
    for param, domain in sd._NUMERIC_DOMAINS.items():
        assert isinstance(domain, types.FunctionType), param
        assert domain(-1) is not None, param
        reason = domain(-1)
        assert isinstance(reason, str) and param in reason, (param, reason)
