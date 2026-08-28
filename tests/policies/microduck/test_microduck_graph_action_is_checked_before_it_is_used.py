"""The raw action the ONNX graph returns is checked before the policy uses it.

``MicroduckPolicy`` holds three width contracts. Two were checked: ``default_pose``
against ``len(joint_names)`` in ``_ensure_config``, and a ``command`` override
against the width ``command_names`` declares in ``_apply_command_kwargs``. The
third - the width of the action the graph itself returns - was not, and it is the
one that differs from the other two by being fed BACK: ``last_action`` *is* that
array, so it sets the width of the observation the graph is handed on the next
tick.

Measured on the pre-fix tree, with a stub session returning a width other than the
joint count:

===================  ==============================  =========================
graph action width   observation widths fed to graph  outcome
===================  ==============================  =========================
14 (the contract)    ``[61, 61, 61]``                 3 x full action dict
1                    ``[61, 48, 48]``                 3 x full action dict
15 / 13 / 7          ``[61]``                         ``ValueError`` from numpy
===================  ==============================  =========================

A width of 1 broadcasts against ``default_pose`` in ``decode_action``, so it
decoded silently, gave every joint the same target, and from tick 2 onward handed
the graph a 48-wide vector where that graph's own ``observation_names`` metadata
declares 61 - reporting a full 14-key action dict throughout. Any other width
raised ``operands could not be broadcast together with shapes (14,) (15,)`` from
inside numpy, naming neither this policy nor the graph.

A non-finite component had the same two consequences at once: it was written to a
joint target *and* fed back into the next observation, with the rollout reporting
success. ``EmpiricalNormalization`` is fused into the graph, so nothing downstream
sanitises it - the same reason the sibling scene-construction guards refuse one.

Why nothing caught it: ``_StubSession`` in the sibling suite already takes an
``n_joints`` parameter, so the knob for this existed. Every one of its 14
instantiations there leaves it at the default 14, which is measured below as a
premise - a stub parametrised for a width it is never driven at cannot see a
width defect.
"""

from __future__ import annotations

import ast
import inspect
import re
import textwrap
from typing import Any

import numpy as np
import pytest

from strands_robots.policies.microduck import MICRODUCK_JOINT_NAMES, MicroduckPolicy
from strands_robots.policies.microduck import policy as policy_mod
from strands_robots.policies.microduck.observation import build_observation

#: Widths the graph must not be allowed to return. ``1`` is the broadcastable one
#: (silent pre-fix); the rest raised from inside numpy naming nothing.
_WRONG_WIDTHS = (1, 7, 13, 15, 28)

#: The command width the shipped alpha policies declare (twist 3 + head 4 + body 6).
_ALPHA_COMMAND = ("twist", "head_pose", "body_pose")
_ALPHA_COMMAND_WIDTH = 13

#: Fixed part of the observation: ang_vel(3) + gravity(3) + pos(14) + vel(14) + last(14).
_FIXED_OBS_WIDTH = 48


class _WidthStub:
    """Duck-typed session returning an action of a chosen width, recording obs widths.

    The sibling suite's ``_StubSession`` self-describes through ONNX metadata; this
    one takes its config explicitly so a cell can hand the policy a width the
    metadata would never advertise - which is the whole point.
    """

    def __init__(self, action: np.ndarray) -> None:
        self.action = action
        self.obs_widths: list[int] = []

    def get_inputs(self) -> list[Any]:
        class _Input:
            name = "obs"

        return [_Input()]

    def run(self, _output_names: Any, input_feed: dict[str, Any]) -> list[np.ndarray]:
        vector = np.asarray(next(iter(input_feed.values()))).reshape(-1)
        self.obs_widths.append(int(vector.shape[0]))
        return [np.asarray(self.action, dtype=np.float32).reshape(1, -1)]


def _joints(n: int = 14) -> list[str]:
    return [f"j{i}" for i in range(n)]


def _obs_dict(names: list[str]) -> dict[str, Any]:
    d: dict[str, Any] = {"base_ang_vel": [0.0, 0.0, 0.0], "base_quat": [1.0, 0.0, 0.0, 0.0]}
    for name in names:
        d[name] = 0.1
        d[f"{name}.vel"] = 0.2
    return d


def _policy(stub: _WidthStub, *, names: list[str] | None = None, command_names: tuple[str, ...] = _ALPHA_COMMAND):
    names = names or _joints()
    return MicroduckPolicy(
        session=stub,  # type: ignore[arg-type]
        joint_names=names,
        default_pose=[0.0] * len(names),
        action_scale=0.25,
        command_names=list(command_names),
    )


def _tick(policy: MicroduckPolicy, names: list[str]) -> dict[str, float]:
    return policy.get_actions_sync(_obs_dict(names), "")[0]


class TestAWrongWidthActionIsRefused:
    """The regression: the graph's action width is held to the joint count."""

    @pytest.mark.parametrize("width", _WRONG_WIDTHS)
    def test_a_width_other_than_the_joint_count_is_refused(self, width: int) -> None:
        names = _joints()
        stub = _WidthStub(np.zeros(width, dtype=np.float32))
        with pytest.raises(ValueError, match="the ONNX graph returned"):
            _tick(_policy(stub), names)

    @pytest.mark.parametrize("width", _WRONG_WIDTHS)
    def test_the_refusal_names_both_widths(self, width: int) -> None:
        names = _joints()
        stub = _WidthStub(np.zeros(width, dtype=np.float32))
        with pytest.raises(ValueError) as excinfo:
            _tick(_policy(stub), names)
        text = str(excinfo.value)
        assert f"returned {width} action" in text, text
        assert f"{len(names)} joints" in text, text

    def test_the_refusal_names_the_policy(self) -> None:
        stub = _WidthStub(np.zeros(15, dtype=np.float32))
        with pytest.raises(ValueError, match=r"^MicroduckPolicy:"):
            _tick(_policy(stub), _joints())

    def test_the_refusal_names_where_the_expected_width_comes_from(self) -> None:
        stub = _WidthStub(np.zeros(15, dtype=np.float32))
        with pytest.raises(ValueError, match="joint_names"):
            _tick(_policy(stub), _joints())

    def test_a_broadcastable_width_no_longer_narrows_the_next_observation(self) -> None:
        """Width 1 was the silent one: it decoded, then changed the obs width."""
        names = _joints()
        stub = _WidthStub(np.zeros(1, dtype=np.float32))
        with pytest.raises(ValueError, match="the ONNX graph returned 1 action"):
            for _ in range(3):
                _tick(_policy(stub), names)
        # It is refused on the FIRST tick, so the graph is never handed a
        # narrowed vector at all.
        assert stub.obs_widths == [_FIXED_OBS_WIDTH + _ALPHA_COMMAND_WIDTH], stub.obs_widths

    def test_the_refusal_arrives_before_any_joint_target_is_produced(self) -> None:
        stub = _WidthStub(np.zeros(1, dtype=np.float32))
        policy = _policy(stub)
        with pytest.raises(ValueError):
            _tick(policy, _joints())
        # last_action is still the seeded zeros of the contract width, not the
        # graph's 1-element output.
        assert policy._last_action is not None
        assert policy._last_action.shape[0] == len(_joints())


class TestANonFiniteActionIsRefused:
    """A nan/inf action is written to every joint AND fed back; refuse it."""

    @pytest.mark.parametrize(
        "action",
        [
            pytest.param(np.array([np.nan] + [0.0] * 13, dtype=np.float32), id="one-nan"),
            pytest.param(np.full(14, np.inf, dtype=np.float32), id="all-inf"),
            pytest.param(np.array([0.0] * 13 + [-np.inf], dtype=np.float32), id="trailing-neg-inf"),
        ],
    )
    def test_a_non_finite_component_is_refused(self, action: np.ndarray) -> None:
        stub = _WidthStub(action)
        with pytest.raises(ValueError, match="must contain finite numbers"):
            _tick(_policy(stub), _joints())

    def test_the_non_finite_refusal_names_the_policy_and_the_action(self) -> None:
        stub = _WidthStub(np.array([np.nan] + [0.0] * 13, dtype=np.float32))
        with pytest.raises(ValueError) as excinfo:
            _tick(_policy(stub), _joints())
        text = str(excinfo.value)
        assert "MicroduckPolicy.get_actions" in text, text
        assert "the ONNX action" in text, text

    def test_no_joint_receives_a_non_finite_target(self) -> None:
        stub = _WidthStub(np.full(14, np.nan, dtype=np.float32))
        with pytest.raises(ValueError):
            _tick(_policy(stub), _joints())


class TestTheGraphOutputIsCheckedAtTheRightPoint:
    """Structural: the check precedes both consumers of the raw action."""

    #: The width comparison, spelled so a literal joint count fails by name
    #: rather than raising ``ValueError`` out of ``str.index``.
    _WIDTH_GUARD = "raw_action.shape[0] != len(self._joint_names)"

    def _get_actions_source(self) -> str:
        return inspect.getsource(MicroduckPolicy.get_actions)

    def _guard_offset(self, source: str) -> int:
        assert self._WIDTH_GUARD in source, (
            f"expected the width comparison {self._WIDTH_GUARD!r}, so the expected "
            "width stays derived from joint_names rather than a literal"
        )
        return source.index(self._WIDTH_GUARD)

    def test_the_width_check_precedes_the_last_action_record(self) -> None:
        source = self._get_actions_source()
        record = source.index("self._last_action = raw_action")
        assert self._guard_offset(source) < record, "the width check must run before the action is fed back"

    def test_the_width_check_precedes_the_decode(self) -> None:
        source = self._get_actions_source()
        decode = source.index("decode_action")
        assert self._guard_offset(source) < decode, "the width check must run before the action becomes joint targets"

    def test_the_finiteness_check_consults_the_shared_vector_domain(self) -> None:
        """Not a local isfinite loop: the shared domain owns the component rule."""
        calls = [
            node
            for node in ast.walk(ast.parse(textwrap.dedent(self._get_actions_source())))
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "finite_vector_error"
        ]
        assert len(calls) == 1, f"expected exactly one shared-domain call, found {len(calls)}"

    def test_the_shared_domain_is_a_real_export(self) -> None:
        """Non-vacuity for the cell above: the name it looks for is importable."""
        from strands_robots import utils

        assert callable(utils.finite_vector_error)
        assert policy_mod.finite_vector_error is utils.finite_vector_error


class TestWhatTheContractWidthStillDoes:
    """Over-reach controls: everything the pre-fix code accepted correctly still works."""

    def test_the_contract_width_reaches_every_joint_for_many_ticks(self) -> None:
        names = _joints()
        stub = _WidthStub(np.arange(14, dtype=np.float32) * 0.01)
        policy = _policy(stub)
        for _ in range(3):
            action = _tick(policy, names)
            assert len(action) == len(names)
            assert all(np.isfinite(v) for v in action.values())
        assert stub.obs_widths == [_FIXED_OBS_WIDTH + _ALPHA_COMMAND_WIDTH] * 3, stub.obs_widths

    def test_a_row_shaped_action_is_still_accepted(self) -> None:
        """The graph returns ``(1, n)``; ``infer_raw`` squeezes it."""
        stub = _WidthStub(np.zeros((1, 14), dtype=np.float32))
        assert len(_tick(_policy(stub), _joints())) == 14

    def test_the_legacy_twist_only_command_width_still_works(self) -> None:
        stub = _WidthStub(np.zeros(14, dtype=np.float32))
        policy = _policy(stub, command_names=("twist",))
        _tick(policy, _joints())
        assert stub.obs_widths == [_FIXED_OBS_WIDTH + 3], stub.obs_widths

    def test_a_large_but_finite_action_is_accepted(self) -> None:
        """Only non-finite is refused - the guard is not a magnitude bound."""
        stub = _WidthStub(np.full(14, -1.0e6, dtype=np.float32))
        action = _tick(_policy(stub), _joints())
        assert all(np.isfinite(v) for v in action.values())

    def test_a_robot_with_a_different_joint_count_still_works(self) -> None:
        """The expected width is derived, not the literal 14."""
        names = _joints(9)
        stub = _WidthStub(np.zeros(9, dtype=np.float32))
        assert len(_tick(_policy(stub, names=names), names)) == 9

    def test_the_two_pre_existing_width_guards_are_untouched(self) -> None:
        stub = _WidthStub(np.zeros(14, dtype=np.float32))
        with pytest.raises(ValueError, match="command override has width"):
            policy = _policy(stub)
            policy.get_actions_sync(_obs_dict(_joints()), "", command=np.zeros(5, dtype=np.float32))


class TestThePremisesTheDefectRestedOn:
    """Facts that hold on both trees and make the measurement above legible."""

    def test_numpy_broadcasts_a_one_element_action_against_the_pose(self) -> None:
        """Why width 1 was silent rather than loud."""
        assert (np.zeros(14, dtype=np.float32) + np.zeros(1, dtype=np.float32)).shape == (14,)
        with pytest.raises(ValueError, match="could not be broadcast"):
            _ = np.zeros(14, dtype=np.float32) + np.zeros(15, dtype=np.float32)

    def test_the_observation_width_is_the_fixed_block_plus_the_command(self) -> None:
        names = _joints()
        vector = build_observation(
            _obs_dict(names),
            joint_names=names,
            default_pose=np.zeros(14, dtype=np.float32),
            last_action=np.zeros(14, dtype=np.float32),
            command=np.zeros(_ALPHA_COMMAND_WIDTH, dtype=np.float32),
        )
        assert vector.shape[0] == _FIXED_OBS_WIDTH + _ALPHA_COMMAND_WIDTH == 61

    def test_the_builder_takes_the_last_action_width_on_trust(self) -> None:
        """The builder cannot keep its documented ``48 + len(command)`` promise alone.

        This is why the guard belongs at the seam where the graph's output enters
        the policy rather than inside the builder: a wrong-width ``last_action``
        silently changes the width the builder returns.
        """
        names = _joints()
        vector = build_observation(
            _obs_dict(names),
            joint_names=names,
            default_pose=np.zeros(14, dtype=np.float32),
            last_action=np.zeros(1, dtype=np.float32),
            command=np.zeros(_ALPHA_COMMAND_WIDTH, dtype=np.float32),
        )
        assert vector.shape[0] == _FIXED_OBS_WIDTH - 14 + 1 + _ALPHA_COMMAND_WIDTH == 48

    def test_the_sibling_stub_parametrises_a_width_it_is_never_driven_at(self) -> None:
        """Why no existing cell could see this."""
        from pathlib import Path

        suite = Path(__file__).parent
        sources = [p.read_text() for p in suite.glob("test_*.py") if p.name != Path(__file__).name]
        joined = "\n".join(sources)
        assert "n_joints" in joined, "the sibling stub is expected to carry a width knob"
        assert not re.search(r"n_joints\s*=\s*(?!14\b)\d+", joined), (
            "a sibling cell now drives the stub off the contract width - fold this "
            "file's coverage into it rather than keeping two accounts"
        )

    def test_the_contract_joint_count_is_fourteen(self) -> None:
        assert len(MICRODUCK_JOINT_NAMES) == 14
