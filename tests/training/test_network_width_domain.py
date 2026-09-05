"""A hidden layer of width zero severs the policy from the observation.

``hidden_dims`` is the one spec field that decides the *shape* of the networks a
from-scratch RL run trains. All three RL backends expand it identically, once
per network they build - the on-policy actor and critic, and off-policy the
actor, its Polyak target and all four Q heads::

    for h in spec.hidden_dims:
        layers += [nn.Linear(last, h), nn.ReLU()]
        last = h
    layers.append(nn.Linear(last, out_dim))

That loop judges nothing, and ``nn.Linear`` does not either: a width of zero is
a legal layer. ``torch`` only warns ("Initializing zero-element tensors is a
no-op"), the layer emits a ``(batch, 0)`` activation, and the layer after it
therefore emits its bias alone - so the network's output stops being a function
of its input. Measured over a full ``train()`` on each backend, the trained
actor returned a bit-identical action for an all-zero observation and an
all-``50`` one, while ``train()`` reported ``status="success"`` with a real
``actor_loss`` and exported ``policy.pt`` + ``policy_meta.json``: a deployable
checkpoint whose actor commands one fixed action in every state the robot can
reach.

The empty sequence is deliberately NOT in that class and stays accepted - it is
the honest spelling of a linear policy, and its action still varies with the
observation - so the domain is per element rather than on the length, and the
message names the offending index.

Every test reaches the real ``validate`` entry point, so the wiring into each
backend is covered as well as the domain.
"""

from __future__ import annotations

import ast
import inspect
import json
import pathlib
from typing import Any

import numpy as np
import pytest

from strands_robots.training import create_trainer
from strands_robots.training.base import Trainer
from strands_robots.training.rl import RLTrainSpec
from strands_robots.utils import positive_count_error
from tests.training._spec_field_reads import reads_spec_field

# The one field this domain owns.
WIDTH_FIELD = "hidden_dims"

# Every from-scratch RL backend builds its actor and critics from the field.
RL_BACKENDS = ("ppo", "fast_sac", "fast_td3")

# A backend whose architecture comes from a pretrained checkpoint, not the spec.
NO_WIDTH_BACKENDS = ("mock",)

# Widths no reading makes usable. Zero is the silent one - the network stops
# depending on its input - negatives and non-ints raise inside ``setup`` after
# the env is built, ``True`` would pass a bare ``< 1`` test as a width of one,
# and ``np.int64`` builds but cannot be serialized into ``policy_meta.json``.
UNUSABLE_WIDTHS: list[Any] = [
    0,
    -1,
    -16,
    True,
    False,
    16.0,
    0.5,
    float("nan"),
    float("inf"),
    "16",
    None,
    [16],
    np.int64(16),
]

# Widths the expansion loop honors.
USABLE_WIDTHS: list[Any] = [1, 8, 16, 128, 512]

# Values that are not a sequence of widths at all, so no index can be named.
NOT_A_SEQUENCE: list[Any] = [16, None, "16", {"a": 16}, 16.0]


@pytest.fixture
def spec() -> RLTrainSpec:
    """An otherwise-valid RL spec, so only the field under test is exercised."""
    return RLTrainSpec(
        output_dir="/tmp/network_width_domain",
        env_factory=lambda: None,  # type: ignore[arg-type,return-value]
    )


def _width_reports(provider: str, spec: RLTrainSpec) -> list[str]:
    """Problems the real ``validate`` entry point reports about ``hidden_dims``.

    Filtered on the ``"{context}: hidden_dims"`` prefix rather than a bare
    substring, so an unrelated problem can neither mask a missing refusal nor be
    mistaken for one. The prefix carries no trailing space because a per-element
    problem names its index (``hidden_dims[1] ...``) while a whole-container one
    names the field (``hidden_dims must be a sequence ...``).
    """
    return [p for p in create_trainer(provider).validate(spec) if p.startswith(f"{provider}: {WIDTH_FIELD}")]


class TestEveryRlBackendRefusesAnUnusableWidth:
    """A width the expansion loop cannot honor is refused by all three."""

    @pytest.mark.parametrize("provider", RL_BACKENDS)
    @pytest.mark.parametrize("value", UNUSABLE_WIDTHS, ids=repr)
    def test_it_is_reported_as_a_problem(self, spec: RLTrainSpec, provider: str, value: Any) -> None:
        spec.hidden_dims = (value,)  # type: ignore[assignment]
        assert _width_reports(provider, spec), f"{provider} accepted hidden_dims=({value!r},)"

    @pytest.mark.parametrize("provider", RL_BACKENDS)
    @pytest.mark.parametrize("value", UNUSABLE_WIDTHS, ids=repr)
    def test_the_problem_names_the_index_and_the_value(self, spec: RLTrainSpec, provider: str, value: Any) -> None:
        """A sequence field must say WHICH entry was refused."""
        spec.hidden_dims = (128, value)  # type: ignore[assignment]
        (problem,) = _width_reports(provider, spec)
        assert problem.startswith(f"{provider}: hidden_dims[1] "), problem
        assert repr(value) in problem, problem

    @pytest.mark.parametrize("provider", RL_BACKENDS)
    def test_every_unusable_width_is_reported_at_once(self, spec: RLTrainSpec, provider: str) -> None:
        """Two bad widths come back as two problems, not one per round."""
        spec.hidden_dims = (0, 128, -4)  # type: ignore[assignment]
        problems = _width_reports(provider, spec)
        assert len(problems) == 2, problems
        assert problems[0].startswith(f"{provider}: hidden_dims[0] ")
        assert problems[1].startswith(f"{provider}: hidden_dims[2] ")

    @pytest.mark.parametrize("provider", RL_BACKENDS)
    @pytest.mark.parametrize("value", NOT_A_SEQUENCE, ids=repr)
    def test_a_field_that_is_not_a_sequence_of_widths_is_refused(
        self, spec: RLTrainSpec, provider: str, value: Any
    ) -> None:
        spec.hidden_dims = value  # type: ignore[assignment]
        (problem,) = _width_reports(provider, spec)
        assert problem == (f"{provider}: hidden_dims must be a sequence of positive int layer widths, got {value!r}"), (
            problem
        )

    @pytest.mark.parametrize("provider", RL_BACKENDS)
    def test_a_one_shot_iterator_is_refused(self, spec: RLTrainSpec, provider: str) -> None:
        """A generator is consumed by the first network built, so the rest get none."""
        spec.hidden_dims = (w for w in (128, 128))  # type: ignore[assignment]
        assert _width_reports(provider, spec), "a generator was accepted as hidden_dims"


class TestTheUsableDomainIsUntouched:
    """A width the loop honors is not newly refused."""

    @pytest.mark.parametrize("provider", RL_BACKENDS)
    @pytest.mark.parametrize("value", USABLE_WIDTHS, ids=repr)
    def test_a_usable_width_reports_nothing(self, spec: RLTrainSpec, provider: str, value: Any) -> None:
        spec.hidden_dims = (value, value)  # type: ignore[assignment]
        assert _width_reports(provider, spec) == []

    @pytest.mark.parametrize("provider", RL_BACKENDS)
    def test_the_shipped_default_reports_nothing(self, spec: RLTrainSpec, provider: str) -> None:
        assert spec.hidden_dims == (128, 128)
        assert _width_reports(provider, spec) == []

    @pytest.mark.parametrize("provider", RL_BACKENDS)
    @pytest.mark.parametrize("container", [(), []], ids=["tuple", "list"])
    def test_no_hidden_layer_is_a_linear_policy_and_stays_accepted(
        self, spec: RLTrainSpec, provider: str, container: Any
    ) -> None:
        """The one boundary value that IS a real configuration.

        An empty sequence means "no hidden layer": the input feeds the output
        layer directly. That policy still varies with the observation, so it is
        not the severed-network class this gate exists for and must not be
        swept up by a length check.
        """
        spec.hidden_dims = container
        assert _width_reports(provider, spec) == []

    @pytest.mark.parametrize("provider", RL_BACKENDS)
    def test_a_list_of_widths_is_accepted(self, spec: RLTrainSpec, provider: str) -> None:
        """A config-loaded spec carries a list, not a tuple."""
        spec.hidden_dims = [64, 64]  # type: ignore[assignment]
        assert _width_reports(provider, spec) == []


class TestABackendWithNoSpecArchitectureStaysQuiet:
    """A backend whose shape comes from a checkpoint must not report on the field."""

    @pytest.mark.parametrize("provider", NO_WIDTH_BACKENDS)
    def test_it_reports_nothing_about_the_widths(self, spec: RLTrainSpec, provider: str) -> None:
        spec.hidden_dims = (0,)  # type: ignore[assignment]
        assert _width_reports(provider, spec) == []

    @pytest.mark.parametrize("provider", NO_WIDTH_BACKENDS)
    def test_silence_is_scoping_rather_than_an_empty_preflight(self, spec: RLTrainSpec, provider: str) -> None:
        """The same spec's *own* learning rate is still refused by that backend."""
        spec.learning_rate = 0.0
        problems = create_trainer(provider).validate(spec)
        assert any(p.startswith(f"{provider}: learning_rate ") for p in problems), problems


class TestTheGateAddsNothingToTheSharedDomain:
    """The per-element verdict is the shared rule's, so the two cannot drift."""

    @pytest.mark.parametrize("value", UNUSABLE_WIDTHS + USABLE_WIDTHS, ids=repr)
    def test_the_verdict_matches_the_shared_domain(self, spec: RLTrainSpec, value: Any) -> None:
        spec.hidden_dims = (value,)  # type: ignore[assignment]
        shared = positive_count_error(value, "hidden_dims[0]", "fast_td3")
        assert _width_reports("fast_td3", spec) == ([shared] if shared is not None else [])


class TestTheSilentReadingIsReal:
    """The measured premises behind the domain."""

    @staticmethod
    def _mlp(hidden: Any) -> Any:
        from strands_robots.training.rl.fast_td3 import _mlp

        return _mlp(6, hidden, 4)

    def test_a_zero_width_makes_the_output_independent_of_the_input(self) -> None:
        """The premise: the layer after an empty activation emits its bias alone."""
        torch = pytest.importorskip("torch")
        net = self._mlp((16, 0))
        with torch.no_grad():
            quiet = net(torch.zeros(1, 6))
            loud = net(torch.full((1, 6), 50.0))
        assert torch.equal(quiet, loud), "premise: a zero width severs the output from the input"

    def test_no_hidden_layer_keeps_the_output_dependent_on_the_input(self) -> None:
        """The exemption's premise: a linear policy still responds to the state."""
        torch = pytest.importorskip("torch")
        net = self._mlp(())
        with torch.no_grad():
            quiet = net(torch.zeros(1, 6))
            loud = net(torch.full((1, 6), 50.0))
        assert not torch.equal(quiet, loud), "premise: an empty sequence is a usable architecture"

    def test_a_one_shot_iterator_builds_a_different_shape_each_time(self) -> None:
        """The premise for refusing the container: the critics would not match the actor."""
        pytest.importorskip("torch")
        widths = (w for w in (16, 16))
        first = sum(p.numel() for p in self._mlp(widths).parameters())
        second = sum(p.numel() for p in self._mlp(widths).parameters())
        assert first != second, "premise: a generator is exhausted by the first network built"

    def test_a_numpy_width_cannot_be_recorded_in_the_checkpoint(self) -> None:
        """The premise for refusing ``np.int64``: ``save_checkpoint`` JSON-encodes the field."""
        with pytest.raises(TypeError, match="not JSON serializable"):
            json.dumps({"hidden_dims": list((np.int64(16),))})


class TestTheRefusalReachesTheRunEntryPoint:
    """A refused width stops the run rather than training a severed policy."""

    def test_train_is_fail_closed_on_an_unusable_width(self, tmp_path: pathlib.Path) -> None:
        pytest.importorskip("torch")
        spec = RLTrainSpec(
            output_dir=str(tmp_path),
            env_factory=lambda: None,  # type: ignore[arg-type,return-value]
            hidden_dims=(16, 0),
        )
        result = create_trainer("fast_td3").train(spec)
        assert result.status == "error"
        assert "hidden_dims[1]" in result.message
        assert not list(tmp_path.rglob("policy.pt")), "a refused run must not export a checkpoint"


# --- one owner for the domain ------------------------------------------------


def _training_modules() -> list[pathlib.Path]:
    """Every training module except the one that owns the gate."""
    root = pathlib.Path(inspect.getfile(Trainer)).parent
    owner = (root / "_validate.py").resolve()
    return sorted(p for p in root.rglob("*.py") if p.name != "__init__.py" and p.resolve() != owner)


def _reads_the_widths(source: str) -> bool:
    """Does *source* read ``hidden_dims`` off a spec, by name or through a table?"""
    return reads_spec_field(source, (WIDTH_FIELD,))


def _calls_the_gate(source: str) -> bool:
    """Does *source* route through the shared gate?"""
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_network_width_problems"
        for node in ast.walk(ast.parse(source))
    )


class TestOneOwnerForTheNetworkWidthDomain:
    """No backend may skip the domain, and none may re-implement it.

    The set of backends in scope is derived from the tree rather than listed, so
    a fourth from-scratch RL backend fails this test until it routes through the
    shared gate.
    """

    def test_every_module_that_reads_them_routes_through_the_shared_gate(self) -> None:
        adrift = [
            p.name
            for p in _training_modules()
            if _reads_the_widths(p.read_text()) and not _calls_the_gate(p.read_text())
        ]
        assert adrift == [], f"modules build networks from hidden_dims without the shared gate: {adrift}"

    def test_the_reader_set_is_the_expected_one(self) -> None:
        """Non-vacuity: a mis-rooted scan cannot report a clean sweep over nothing."""
        readers = {p.name for p in _training_modules() if _reads_the_widths(p.read_text())}
        assert readers == {"ppo.py", "fast_sac.py", "fast_td3.py"}, readers

    def test_the_scanner_detects_a_planted_reader(self) -> None:
        """A module reading the field without the gate is really reported."""
        planted = "def build(self):\n    return spec.hidden_dims\n"
        assert _reads_the_widths(planted)
        assert not _calls_the_gate(planted)

    def test_no_backend_re_implements_the_domain(self) -> None:
        """A local comparison would drift from the shared rule."""
        offenders = [
            p.name
            for p in _training_modules()
            if f"len(spec.{WIDTH_FIELD})" in p.read_text() or f"spec.{WIDTH_FIELD} ==" in p.read_text()
        ]
        assert offenders == [], f"modules judge hidden_dims locally: {offenders}"
