"""``build_observation`` refuses a non-finite component instead of feeding it to the graph.

#2887 held the two floating-base blocks to a width, and a width was the only
thing the builder held anything to. Every value it reads - the two base blocks,
the ``len(joint_names)`` position and velocity scalars, the previous action and
the command - reached the vector the ONNX graph consumes without anyone asking
whether it was a number.

Measured on the pre-fix tree with the shipped alpha command width (C=13, so the
documented return width is ``48 + 13 == 61``):

============================  =========  ==========================================
caller input                  outcome    what reached the graph
============================  =========  ==========================================
contract (control)            success    61 finite components
``base_quat`` has a ``nan``   success    3 non-finite (the whole gravity block)
``base_ang_vel`` has a ``nan``  success  1 non-finite
a joint position is ``nan``   success    1 non-finite
a joint velocity is ``nan``   success    1 non-finite
``last_action`` has a ``nan``   success  1 non-finite
``command`` has a ``nan``     success    1 non-finite
============================  =========  ==========================================

Every row reported success at the documented width, which is what makes the
failure hard to attribute: a ``nan`` observation is shaped exactly like a healthy
one. What happens next depends on the checkpoint. A graph that tolerates the
value answers with a number nobody can trace, and a graph that propagates it -
which is what any real one does - hands back a ``nan`` action that
``get_actions``' own finiteness guard (#2882) then refuses **as** ``'the ONNX
action'``:

    MicroduckPolicy.get_actions: 'the ONNX action' must contain finite numbers
    (no nan/inf), got array([nan, ...], dtype=float32)

So the caller was told the checkpoint's graph produced a bad action, for a
``nan`` the caller's own observation dict supplied. That is the same
misattribution shape as an IK solve refusing ``target_pose`` for a seed the
caller never named (#2879).

The guard runs on the ASSEMBLED vector, once, because that is the single place
every input path meets. It is a plain :func:`numpy.isfinite` rather than
:func:`~strands_robots.utils.finite_vector_error` for a measured reason: at that
point the value is a ``float32`` 1-D array the builder itself made, so none of
the spellings the shared domain exists to judge can reach it, the two agree on
everything that can, and the shared domain costs more than the whole build
(40.68 us against a 37.92 us build, where the ``isfinite`` pass costs 1.27 us).
``TestTheSharedDomainWouldAgreeHere`` pins the agreement so the equivalence is
measured rather than assumed.

The sibling locomotion policy holds its caller-supplied vectors to both width and
finiteness and says so in its own module docstring
(``strands_robots.policies.motionbricks.observation``: "Every component must be a
finite number; a count outside two or three, a ``nan``/``inf`` component or a
``bool`` is refused by name"). This package refused a non-finite vector on the
graph's OUTPUT and not on the caller's INPUT.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Any

import numpy as np
import pytest

from strands_robots.policies.microduck import build_observation
from strands_robots.policies.microduck import observation as obs_mod
from strands_robots.utils import finite_vector_error

#: The shipped alpha command width, matching the width-refusal sibling.
_ALPHA_COMMAND_WIDTH = 13

#: Fixed part of the observation: 3 + 3 + 14 + 14 + 14.
_FIXED_OBS_WIDTH = 48

_N_JOINTS = 14

#: The two non-finite spellings. Both are float32-representable, so both survive
#: the builder's own ``astype`` and reach the graph.
_NON_FINITE = (float("nan"), float("inf"))


def _joints() -> list[str]:
    return [f"j{i}" for i in range(_N_JOINTS)]


def _obs_dict(**over: Any) -> dict[str, Any]:
    """A contract-shaped observation dict, with named keys overridden."""
    d: dict[str, Any] = {}
    for name in _joints():
        d[name] = 0.1
        d[f"{name}.vel"] = 0.0
    d["base_ang_vel"] = [0.0, 0.0, 0.0]
    d["base_quat"] = [1.0, 0.0, 0.0, 0.0]
    d.update(over)
    return d


def _build(obs_over: dict[str, Any] | None = None, **kw: Any) -> np.ndarray:
    joints = _joints()
    args: dict[str, Any] = {
        "joint_names": joints,
        "default_pose": np.zeros(_N_JOINTS, np.float32),
        "last_action": np.zeros(_N_JOINTS, np.float32),
        "command": np.zeros(_ALPHA_COMMAND_WIDTH, np.float32),
    }
    args.update(kw)
    return build_observation(_obs_dict(**(obs_over or {})), **args)


def _poisoned_vector(index: int, value: float) -> np.ndarray:
    v = np.zeros(_N_JOINTS, np.float32)
    v[index] = np.float32(value)
    return v


class TestEveryInputPathIsRefused:
    """A non-finite component is refused wherever the builder reads it from.

    Six paths reach the assembled vector. The two base blocks and the per-joint
    scalars come from the caller's observation dict; ``last_action`` and
    ``command`` are passed in. All six were unchecked.
    """

    @pytest.mark.parametrize("bad", _NON_FINITE, ids=["nan", "inf"])
    def test_a_base_quat_component_is_refused(self, bad: float) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            _build({"base_quat": [1.0, bad, 0.0, 0.0]})

    @pytest.mark.parametrize("bad", _NON_FINITE, ids=["nan", "inf"])
    def test_a_base_ang_vel_component_is_refused(self, bad: float) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            _build({"base_ang_vel": [bad, 0.0, 0.0]})

    @pytest.mark.parametrize("bad", _NON_FINITE, ids=["nan", "inf"])
    def test_a_joint_position_is_refused(self, bad: float) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            _build({_joints()[0]: bad})

    @pytest.mark.parametrize("bad", _NON_FINITE, ids=["nan", "inf"])
    def test_a_joint_velocity_is_refused(self, bad: float) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            _build({f"{_joints()[3]}.vel": bad})

    @pytest.mark.parametrize("bad", _NON_FINITE, ids=["nan", "inf"])
    def test_a_last_action_component_is_refused(self, bad: float) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            _build(last_action=_poisoned_vector(2, bad))

    @pytest.mark.parametrize("bad", _NON_FINITE, ids=["nan", "inf"])
    def test_a_command_component_is_refused(self, bad: float) -> None:
        cmd = np.zeros(_ALPHA_COMMAND_WIDTH, np.float32)
        cmd[1] = np.float32(bad)
        with pytest.raises(ValueError, match="non-finite"):
            _build(command=cmd)


class TestTheRefusalNamesTheBlockAReaderMustFix:
    """The message names the offending block, and a joint block names the joint.

    An observation is assembled from six blocks and the caller controls each one
    separately, so "the observation has a nan" leaves them to bisect it. The
    layout is known here, so the block can be named for free on the refusal path.
    """

    def test_a_base_ang_vel_offender_is_named(self) -> None:
        with pytest.raises(ValueError, match=r"base_ang_vel at \[0\]"):
            _build({"base_ang_vel": [float("nan"), 0.0, 0.0]})

    def test_a_base_quat_offender_is_named_through_the_block_it_becomes(self) -> None:
        # base_quat is reduced to projected_gravity before it reaches the vector,
        # so the assembled block carries the fault and the message says which
        # input it came from.
        with pytest.raises(ValueError, match=r"projected_gravity \(from base_quat\)"):
            _build({"base_quat": [1.0, float("nan"), 0.0, 0.0]})

    def test_a_joint_position_offender_names_the_joint(self) -> None:
        joint = _joints()[5]
        with pytest.raises(ValueError, match=rf"joint_pos \({joint}\)"):
            _build({joint: float("nan")})

    def test_a_joint_velocity_offender_names_the_joint(self) -> None:
        joint = _joints()[9]
        with pytest.raises(ValueError, match=rf"joint_vel \({joint}\)"):
            _build({f"{joint}.vel": float("nan")})

    def test_a_last_action_offender_is_named(self) -> None:
        with pytest.raises(ValueError, match=r"last_action at \[2\]"):
            _build(last_action=_poisoned_vector(2, float("nan")))

    def test_the_block_names_are_derived_from_the_joints_they_were_read_for(self) -> None:
        # The layout the message walks is built from len(joint_names), not from
        # the 14 the shipped checkpoint happens to use. A robot with a different
        # joint count would otherwise have its offender attributed to the wrong
        # block, which is worse than not naming one at all. The offender is a
        # VELOCITY: a joint_pos one sits at 6 + i whatever the count is, so only
        # a block after the first discriminates a wrong count from the right one.
        joints = [f"k{i}" for i in range(9)]
        obs: dict[str, Any] = {f"{n}{suf}": v for n in joints for suf, v in (("", 0.1), (".vel", 0.0))}
        obs["base_ang_vel"] = [0.0, 0.0, 0.0]
        obs["base_quat"] = [1.0, 0.0, 0.0, 0.0]
        obs[f"{joints[7]}.vel"] = float("nan")
        with pytest.raises(ValueError, match=rf"joint_vel \({joints[7]}\)"):
            build_observation(
                obs,
                joint_names=joints,
                default_pose=np.zeros(len(joints), np.float32),
                last_action=np.zeros(len(joints), np.float32),
                command=np.zeros(_ALPHA_COMMAND_WIDTH, np.float32),
            )

    def test_a_command_offender_is_named(self) -> None:
        cmd = np.zeros(_ALPHA_COMMAND_WIDTH, np.float32)
        cmd[7] = np.float32("nan")
        with pytest.raises(ValueError, match=r"command at \[7\]"):
            _build(command=cmd)

    def test_two_offenders_in_different_blocks_are_both_named(self) -> None:
        joint = _joints()[1]
        with pytest.raises(ValueError) as excinfo:
            _build({"base_ang_vel": [float("nan"), 0.0, 0.0], joint: float("inf")})
        text = str(excinfo.value)
        assert "base_ang_vel" in text
        assert f"joint_pos ({joint})" in text
        assert "2 non-finite" in text

    def test_the_message_says_what_the_value_would_have_done(self) -> None:
        # The reason a nan matters here is that get_actions refuses it as the
        # graph's output, so the message names that consequence.
        with pytest.raises(ValueError, match="the ONNX graph consumes"):
            _build({"base_ang_vel": [float("nan"), 0.0, 0.0]})


class TestTheGuardSitsOnTheAssembledVector:
    """One check, after the concatenate, reading the vector every path feeds.

    Guarding each input separately would be six checks that drift apart, and it
    would still miss whatever a later block adds. The structural cells hold the
    placement so a future edit cannot move the check ahead of a block.
    """

    def test_the_builder_has_exactly_one_finiteness_owner(self) -> None:
        src = textwrap.dedent(inspect.getsource(obs_mod))
        calls = [
            node
            for node in ast.walk(ast.parse(src))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_non_finite_observation_error"
        ]
        assert len(calls) == 1, "the finiteness check has one owner and one call site"

    def test_the_check_follows_the_concatenate_it_reads(self) -> None:
        src = textwrap.dedent(inspect.getsource(obs_mod.build_observation))
        assert "np.concatenate" in src, "the builder still assembles by concatenate"
        assert "_non_finite_observation_error(" in src, "the builder consults the check"
        assert src.index("np.concatenate") < src.index("_non_finite_observation_error("), (
            "the check must read the assembled vector, so it follows the concatenate"
        )

    def test_the_refusal_precedes_the_return(self) -> None:
        src = textwrap.dedent(inspect.getsource(obs_mod.build_observation))
        assert src.index("raise ValueError(reason)") < src.rindex("return observation"), (
            "a refused observation must not be returned"
        )


class TestTheSharedDomainWouldAgreeHere:
    """The fast path is equivalent to the shared vector domain at this point.

    A hand-rolled finiteness test is only safe when it cannot silently accept
    something the shared domain rejects. Here the value is a ``float32`` 1-D
    array the builder produced, so the nested sequences, ``bool``s, 0-d scalars
    and non-numerics that domain exists to judge are unreachable - and these
    cells measure the agreement rather than asserting it.
    """

    def test_the_assembled_value_is_already_normalised(self) -> None:
        vec = _build()
        assert vec.dtype == np.float32
        assert vec.ndim == 1

    @pytest.mark.parametrize("bad", [None, *_NON_FINITE], ids=["clean", "nan", "inf"])
    def test_both_readers_reach_the_same_verdict(self, bad: float | None) -> None:
        vec = _build()
        if bad is not None:
            vec = vec.copy()
            vec[7] = np.float32(bad)
        shared_ok = finite_vector_error("build_observation", "the observation", vec) is None
        fast_ok = bool(np.isfinite(vec).all())
        assert shared_ok == fast_ok


class TestThePremisesTheDefectRestedOn:
    """What made the non-finite value reachable and invisible."""

    def test_the_layout_the_message_reads_covers_the_whole_vector(self) -> None:
        # The block names are only useful if their widths sum to the vector, or a
        # trailing offender would be attributed to the wrong block (or to none).
        # Holds on both trees: it is a statement about the layout, not the guard.
        joints = _joints()
        vec = _build()
        widths = (3, 3, len(joints), len(joints), len(joints), _ALPHA_COMMAND_WIDTH)
        assert sum(widths) == vec.shape[0] == _FIXED_OBS_WIDTH + _ALPHA_COMMAND_WIDTH

    def test_a_non_finite_float_survives_the_float32_cast(self) -> None:
        # Both spellings are representable in float32, so neither is lost by the
        # builder's own astype - which is why they reached the graph.
        for bad in _NON_FINITE:
            assert not np.isfinite(np.float32(bad))

    def test_the_returned_width_does_not_depend_on_the_values(self) -> None:
        # This is why a non-finite component was invisible: the width is a
        # function of the block widths alone, so a poisoned observation is shaped
        # exactly like a healthy one and nothing downstream can screen it by
        # shape. Stated with finite values so it holds on both trees.
        joints = _joints()
        assert _build().shape == _build({joints[0]: 1e30, joints[1]: -1e30}).shape

    def test_the_graph_output_guard_is_the_one_that_used_to_fire(self) -> None:
        # get_actions holds the graph's returned action to the shared finiteness
        # domain (#2882). That guard is what produced the misattributed refusal:
        # with the builder silent, a caller's nan came back named as the graph's.
        from strands_robots.policies.microduck import policy as policy_mod

        src = textwrap.dedent(inspect.getsource(policy_mod.MicroduckPolicy.get_actions))
        assert "finite_vector_error(" in src, "the graph's action is held to the domain"
        assert "the ONNX action" in src, "and it is named as the graph's action"

    def test_the_sibling_policy_holds_its_caller_vectors_to_finiteness(self) -> None:
        from strands_robots.policies.motionbricks import observation as mb_obs

        doc = " ".join((mb_obs.__doc__ or "").split())
        assert "finite" in doc, "the sibling observation builder states this rule"


class TestWhatIsUnchanged:
    """The accepted contract, and the refusals that were already there."""

    def test_a_contract_observation_still_builds(self) -> None:
        vec = _build()
        assert vec.shape == (_FIXED_OBS_WIDTH + _ALPHA_COMMAND_WIDTH,)
        assert bool(np.isfinite(vec).all())

    def test_a_large_but_finite_value_is_accepted(self) -> None:
        # The guard refuses nan/inf, not magnitude: a big reading is a reading.
        vec = _build({_joints()[0]: 1e30})
        assert bool(np.isfinite(vec).all())

    def test_the_most_negative_float32_is_accepted(self) -> None:
        vec = _build({_joints()[2]: float(np.finfo(np.float32).min)})
        assert bool(np.isfinite(vec).all())

    def test_a_wrong_width_base_block_is_still_refused_by_width(self) -> None:
        with pytest.raises(ValueError, match="component"):
            _build({"base_quat": [1.0, 0.0, 0.0]})

    def test_the_width_refusal_still_precedes_the_finiteness_one(self) -> None:
        # A block that is both wrong-width and non-finite is reported by width:
        # the width is read first, and it is the more basic mistake.
        with pytest.raises(ValueError, match="wide"):
            _build({"base_quat": [1.0, float("nan"), 0.0]})

    def test_an_absent_key_is_still_a_key_error(self) -> None:
        joints = _joints()
        d = _obs_dict()
        del d[joints[0]]
        with pytest.raises(KeyError):
            build_observation(
                d,
                joint_names=joints,
                default_pose=np.zeros(_N_JOINTS, np.float32),
                last_action=np.zeros(_N_JOINTS, np.float32),
                command=np.zeros(_ALPHA_COMMAND_WIDTH, np.float32),
            )
