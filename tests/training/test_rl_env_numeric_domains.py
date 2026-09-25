"""Every numeric ``SimEnv`` stores has a domain, checked before the engine is touched.

``SimEnv.__init__`` already refused an empty ``actor_obs_keys``, an empty
``reward_terms`` and a non-positive ``n_substeps``. It silently coerced the other
three numerics it stores -- ``action_scale`` with a bare ``float()``,
``max_episode_steps`` and ``action_dim`` with a bare ``int()`` -- and
``n_substeps``' own hand-rolled ``int(n_substeps) < 1`` comparison had the ``bool``
hole every such comparison has.

``action_scale`` is the consequential one. It multiplies *every* action the env
sends, and the constructor's own ``n_substeps`` docstring already states the
pathology it guards against ("the PD controller needs several substeps to track
it, so a single substep barely moves the arm") -- which is what an unusable scale
does, more completely, by scaling the target itself. Measured on a real MuJoCo
two-joint arm, 60 steps of a constant ``[0.9, -0.7]`` command:

===============  ==============  ==========  ==============  =================
``action_scale``  ``send_action``  reward      shoulder (rad)  elbow (rad)
===============  ==============  ==========  ==============  =================
``1.0``           60 ok           60.0        ``+0.5235``     ``-0.6910``
``0``             60 ok           60.0        ``-0.0``        ``+0.0117``
``-1.0``          60 ok           60.0        ``-0.4511``     ``+0.6558``
``nan``           **60 errors**   **60.0**    ``0.0``         ``0.0``
``inf``           **60 errors**   **60.0**    ``0.0``         ``0.0``
===============  ==============  ==========  ==============  =================

A non-finite scale makes every command unsendable -- ``send_action`` refuses a
non-finite action -- and :meth:`SimEnv.step` discards that status, so the arm
never moved and the rollout still banked the full return. That is verbatim the
pathology the ``num_actions`` comment three lines below in the same constructor
describes ("every step wrote no target while the reward was still collected"),
reached through the scale instead of the width. ``0`` disconnects the policy from
the robot at full cost (the elbow's ``+0.0117`` is gravity sag, not tracking),
``-1.0`` inverts every commanded DOF, and ``True`` is a silent scale of ``1.0``.

The tests below pin the contract in three parts: every unusable value is refused
with a message naming the class and the parameter; a usable value (including the
``np.float32`` the continuous domain deliberately admits and the ``action_dim=None``
sentinel) is untouched; and the refusal happens before the engine is read, so a
rejected env cannot leave a stepped simulation behind. A structural test then
derives the set of numerics from the live signature, so a numeric added later
cannot ship without a domain.
"""

from __future__ import annotations

import ast
import inspect
import math
import re
from typing import Any, cast

import pytest

torch = pytest.importorskip("torch")

import numpy as np  # noqa: E402 - after torch importorskip

import strands_robots.training.rl.env as rl_env  # noqa: E402
from strands_robots.simulation.base import SimEngine  # noqa: E402
from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine  # noqa: E402
from strands_robots.training.rl import SimEnv  # noqa: E402
from strands_robots.utils import positive_whole_number_error  # noqa: E402

# Values no numeric of this constructor can honor. ``0`` and a negative are in
# both lists: for the continuous scale they make the multiplier degenerate or
# inverting, for a count they make the loop empty or nonsensical.
UNUSABLE_SCALES: list[Any] = [0, 0.0, -1.0, -0.25, math.nan, math.inf, -math.inf, True, False, "0.5", None, [1.0]]
# ``3.0`` and ``np.int64(3)`` are deliberately absent: they are *usable* counts
# for the two whole-number knobs (and for the ``send_action`` one forwards to),
# and refused only by ``action_dim``'s narrower domain - pinned separately.
UNUSABLE_COUNTS: list[Any] = [0, -5, True, False, 2.7, math.nan, math.inf, "10", None, [2]]


class _Recorder:
    """Fake engine that records every method the constructor calls on it.

    Two robots' worth of shape is unnecessary here; what matters is that each
    call is observable, so a test can assert a *refused* construction touched the
    engine not at all.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.actions: list[list[float]] = []
        self.statuses: list[str] = []

    def list_robots(self) -> list[str]:
        self.calls.append("list_robots")
        return ["fake"]

    def robot_joint_names(self, robot_name: str) -> list[str]:
        self.calls.append("robot_joint_names")
        return ["A", "B"]

    def robot_action_keys(self, robot_name: str) -> list[str]:
        self.calls.append("robot_action_keys")
        return ["A", "B"]

    def reset(self) -> dict[str, Any]:
        self.calls.append("reset")
        return {"status": "success"}

    def get_observation(self, robot_name: str | None = None, *, skip_images: bool = False) -> dict[str, Any]:
        self.calls.append("get_observation")
        return {"A": 0.0, "B": 0.0}

    def send_action(self, action: Any, robot_name: str | None = None, n_substeps: int = 1) -> dict[str, Any]:
        self.calls.append("send_action")
        vals = [float(v) for v in action]
        self.actions.append(vals)
        # Mirror the real ``SimEngine._coerce_action`` contract: a non-finite
        # command is refused rather than clamped.
        status = "success" if all(math.isfinite(v) for v in vals) else "error"
        self.statuses.append(status)
        return {"status": status}


def _env(engine: _Recorder, **kwargs: Any) -> SimEnv:
    """Build a ``SimEnv`` over the recorder.

    ``kwargs`` is splatted so a deliberately off-type value reaches the runtime
    guard as a caller would supply it, without the type checker objecting at each
    individual call site.
    """
    return SimEnv(
        cast(SimEngine, engine),
        actor_obs_keys=["A", "B"],
        reward_terms=[lambda _e: 1.0],
        **kwargs,
    )


class TestEveryUnusableNumericIsRefused:
    """A value no numeric can honor is refused, naming the class and the parameter."""

    @pytest.mark.parametrize("value", UNUSABLE_SCALES, ids=[repr(v) for v in UNUSABLE_SCALES])
    def test_action_scale(self, value: Any) -> None:
        with pytest.raises(ValueError, match=r"SimEnv: action_scale"):
            _env(_Recorder(), action_scale=value)

    @pytest.mark.parametrize("value", UNUSABLE_COUNTS, ids=[repr(v) for v in UNUSABLE_COUNTS])
    def test_max_episode_steps(self, value: Any) -> None:
        with pytest.raises(ValueError, match=r"SimEnv: max_episode_steps"):
            _env(_Recorder(), max_episode_steps=value)

    @pytest.mark.parametrize("value", UNUSABLE_COUNTS, ids=[repr(v) for v in UNUSABLE_COUNTS])
    def test_n_substeps(self, value: Any) -> None:
        with pytest.raises(ValueError, match=r"SimEnv: n_substeps"):
            _env(_Recorder(), n_substeps=value)

    # ``None`` is excluded deliberately: for ``action_dim`` alone it is the
    # documented "size the head from the robot's action keys" spelling, so it is a
    # sentinel rather than a value with a domain (pinned as accepted below).
    @pytest.mark.parametrize(
        "value",
        [v for v in UNUSABLE_COUNTS if v is not None],
        ids=[repr(v) for v in UNUSABLE_COUNTS if v is not None],
    )
    def test_action_dim(self, value: Any) -> None:
        with pytest.raises(ValueError, match=r"SimEnv: action_dim"):
            _env(_Recorder(), action_dim=value)

    def test_the_message_names_the_reason_not_only_the_parameter(self) -> None:
        with pytest.raises(ValueError, match=r"action_scale must be > 0, got 0\.0"):
            _env(_Recorder(), action_scale=0.0)
        with pytest.raises(ValueError, match=r"max_episode_steps must be a positive whole number, got 0"):
            _env(_Recorder(), max_episode_steps=0)
        with pytest.raises(ValueError, match=r"action_dim must be a positive integer, got 0"):
            _env(_Recorder(), action_dim=0)


class TestUsableValuesAreUntouched:
    """The change is additive: nothing that could be honored is now refused."""

    def test_the_defaults_build(self) -> None:
        env = _env(_Recorder())
        assert (env.action_scale, env.max_episode_steps, env.n_substeps) == (1.0, 200, 5)

    def test_a_fractional_scale_is_honored(self) -> None:
        assert _env(_Recorder(), action_scale=0.25).action_scale == 0.25

    def test_a_numpy_real_scale_is_honored_and_normalized(self) -> None:
        # ``positive_finite_number_error`` deliberately admits any real scalar so a
        # rate or scale read from a config array passes; the stored value is a
        # plain ``float``.
        env = _env(_Recorder(), action_scale=np.float32(0.25))
        assert isinstance(env.action_scale, float)
        assert env.action_scale == pytest.approx(0.25)

    def test_a_single_step_episode_is_honored(self) -> None:
        assert _env(_Recorder(), max_episode_steps=1, n_substeps=1).max_episode_steps == 1

    def test_an_omitted_action_dim_still_sizes_from_the_action_keys(self) -> None:
        # ``None`` is the documented "size the head from the robot's action keys"
        # spelling, not a value with a domain.
        engine = _Recorder()
        assert _env(engine, action_dim=None).num_actions == 2
        assert "robot_action_keys" in engine.calls

    def test_an_explicit_action_dim_is_honored(self) -> None:
        assert _env(_Recorder(), action_dim=3).num_actions == 3

    @pytest.mark.parametrize("value", [3.0, np.int64(3), np.float64(4.0)], ids=["3.0", "np.int64", "np.float64"])
    def test_an_integral_real_substep_count_is_honored(self, value: Any) -> None:
        # ``send_action`` honors these spellings, so this constructor must too;
        # the stored value is normalized to a plain ``int``.
        env = _env(_Recorder(), n_substeps=value)
        assert env.n_substeps == 3 or env.n_substeps == 4
        assert isinstance(env.n_substeps, int)

    @pytest.mark.parametrize("value", [50.0, np.int64(50)], ids=["50.0", "np.int64"])
    def test_an_integral_real_episode_ceiling_is_honored(self, value: Any) -> None:
        # Only ever compared against the step counter, so an integral real is
        # equally usable; normalized to a plain ``int``.
        env = _env(_Recorder(), max_episode_steps=value)
        assert env.max_episode_steps == 50
        assert isinstance(env.max_episode_steps, int)

    @pytest.mark.parametrize("value", [3.0, np.int64(3)], ids=["3.0", "np.int64"])
    def test_an_integral_real_action_dim_is_refused(self, value: Any) -> None:
        # The one knob that is deliberately narrower: it sizes the trainers'
        # action head, where an integral float raises rather than being coerced.
        with pytest.raises(ValueError, match=r"SimEnv: action_dim"):
            _env(_Recorder(), action_dim=value)

    def test_a_usable_env_still_steps(self) -> None:
        engine = _Recorder()
        env = _env(engine, action_scale=0.5, max_episode_steps=4, n_substeps=2)
        env.reset()
        _obs, reward, done, info = env.step(torch.tensor([1.0, -1.0]))
        assert engine.actions[-1] == pytest.approx([0.5, -0.5])
        assert engine.statuses == ["success"]
        assert float(reward.item()) == 1.0
        assert not bool(done.item()) and not info["time_out"]


class TestTheRefusalPrecedesTheEngine:
    """A refused numeric must not leave a stepped simulation behind."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"action_scale": 0.0},
            {"action_scale": math.nan},
            {"max_episode_steps": 0},
            {"n_substeps": 0},
            {"action_dim": 0},
        ],
        ids=["scale=0", "scale=nan", "max_episode_steps=0", "n_substeps=0", "action_dim=0"],
    )
    def test_no_engine_call_is_made(self, kwargs: dict[str, Any]) -> None:
        engine = _Recorder()
        with pytest.raises(ValueError):
            _env(engine, **kwargs)
        assert engine.calls == []

    def test_a_usable_construction_does_read_the_engine(self) -> None:
        # Non-vacuity: the assertion above would hold for any refusal reason if the
        # constructor never read the engine at all.
        engine = _Recorder()
        _env(engine)
        assert "get_observation" in engine.calls


class TestWhatTheUnusableScalesDid:
    """Ground the refused values in what the env and engine really do with them.

    These drive a *usable* env and reproduce the arithmetic the constructor used
    to permit, so the reason each value is refused is measured rather than
    asserted.
    """

    def test_a_non_finite_scale_makes_every_command_unsendable(self) -> None:
        engine = _Recorder()
        scaled = (np.array([0.9, -0.7], dtype=np.float64) * math.nan).tolist()
        assert engine.send_action(scaled)["status"] == "error"

    def test_step_discards_the_send_action_status(self) -> None:
        # Why a non-finite scale is silent: the refusal never reaches the caller.
        # ``step`` banks the reward for a step whose command the engine rejected.
        class _AlwaysRefuses(_Recorder):
            def send_action(self, action: Any, robot_name: str | None = None, n_substeps: int = 1) -> dict[str, Any]:
                super().send_action(action, robot_name, n_substeps)
                self.statuses[-1] = "error"
                return {"status": "error", "content": [{"text": "refused"}]}

        engine = _AlwaysRefuses()
        env = _env(engine)
        env.reset()
        _obs, reward, _done, _info = env.step(torch.tensor([0.9, -0.7]))
        assert engine.statuses == ["error"]
        assert float(reward.item()) == 1.0

    def test_a_zero_scale_commands_the_same_target_every_step(self) -> None:
        # The policy's output cannot reach the robot: whatever it asks for, the
        # command is the zero vector.
        engine = _Recorder()
        env = _env(engine, action_scale=1.0)
        env.reset()
        env.step(torch.tensor([0.9, -0.7]) * 0.0)
        assert engine.actions[-1] == pytest.approx([0.0, 0.0])

    def test_a_negative_scale_inverts_every_commanded_dof(self) -> None:
        engine = _Recorder()
        env = _env(engine, action_scale=1.0)
        env.reset()
        env.step(torch.tensor([0.9, -0.7]) * -1.0)
        assert engine.actions[-1] == pytest.approx([-0.9, 0.7])

    def test_a_zero_max_episode_steps_would_time_out_before_the_first_step(self) -> None:
        # ``time_out = step_count >= max_episode_steps`` is evaluated after the
        # increment, so a ceiling of 1 is the smallest that runs a step; 0 or below
        # reports a truncation the trainer would value-bootstrap.
        env = _env(_Recorder(), max_episode_steps=1)
        env.reset()
        _obs, _r, done, info = env.step(torch.tensor([0.0, 0.0]))
        assert bool(done.item()) and info["time_out"]


# ---------------------------------------------------------------------------
# Structural: no numeric of this constructor may ship without a domain.
# ---------------------------------------------------------------------------
#: Annotations that make a parameter a *magnitude* this contract governs. A
#: ``bool`` flag (``skip_images``) is excluded by construction: it selects a mode
#: rather than sizing or scaling anything.
_NUMERIC_ANNOTATIONS = frozenset({"int", "float", "int | None", "float | None"})


def _numeric_parameters() -> set[str]:
    """Numeric parameters of the live ``SimEnv.__init__`` signature."""
    sig = inspect.signature(SimEnv.__init__)
    return {
        name
        for name, p in sig.parameters.items()
        if isinstance(p.annotation, str) and p.annotation in _NUMERIC_ANNOTATIONS
    }


def _consulted_parameters() -> set[str]:
    """Parameter names the constructor body puts through the domain table.

    Both spellings count: a key of the literal mapping, and a key added to it by
    subscript (``supplied["action_dim"] = action_dim``, which is how the one
    parameter with a sentinel default is added only when it was supplied).
    """
    init = ast.parse(inspect.getsource(SimEnv.__init__).lstrip())
    names: set[str] = set()
    for node in ast.walk(init):
        if isinstance(node, ast.Dict):
            keys = node.keys
        elif isinstance(node, ast.Subscript):
            keys = [node.slice]
        else:
            continue
        for key in keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                names.add(key.value)
    return names


class TestEveryNumericHasADomain:
    """The table and the signature cannot drift apart."""

    def test_the_signature_is_the_expected_set(self) -> None:
        # Non-vacuity: a scan that resolved to nothing would satisfy the two
        # set relations below trivially.
        assert _numeric_parameters() == {"action_dim", "max_episode_steps", "action_scale", "n_substeps"}

    def test_every_numeric_parameter_has_a_domain(self) -> None:
        missing = _numeric_parameters() - set(rl_env._NUMERIC_DOMAINS)
        assert not missing, f"numeric parameters with no entry in _NUMERIC_DOMAINS: {sorted(missing)}"

    def test_the_table_describes_only_real_parameters(self) -> None:
        stale = set(rl_env._NUMERIC_DOMAINS) - set(inspect.signature(SimEnv.__init__).parameters)
        assert not stale, f"_NUMERIC_DOMAINS names parameters the constructor does not take: {sorted(stale)}"

    def test_the_constructor_consults_the_table_for_every_numeric(self) -> None:
        unchecked = _numeric_parameters() - _consulted_parameters()
        assert not unchecked, f"numerics the constructor never puts through a domain: {sorted(unchecked)}"

    def test_the_scanner_detects_a_numeric_left_out(self) -> None:
        # A planted numeric with no table entry must be reported, so an empty
        # ``missing`` set above means the signature really is covered.
        planted = _numeric_parameters() | {"action_jitter"}
        assert planted - set(rl_env._NUMERIC_DOMAINS) == {"action_jitter"}

    def test_each_domain_is_the_one_its_consumer_can_honor(self) -> None:
        # Not one blanket check: each knob is on the shared rule its own consumer
        # accepts, so none is refused here for a value the code downstream honors.
        from strands_robots.utils import (
            positive_count_error,
            positive_finite_number_error,
            positive_whole_number_error,
        )

        assert rl_env._NUMERIC_DOMAINS == {
            "action_scale": positive_finite_number_error,
            "max_episode_steps": positive_whole_number_error,
            "n_substeps": positive_whole_number_error,
            "action_dim": positive_count_error,
        }

    def test_the_substep_domain_is_the_one_send_action_uses(self) -> None:
        # ``n_substeps`` is forwarded verbatim to ``send_action``. Were this guard
        # the narrower one, an ``np.int64`` or a ``3.0`` from a config would be
        # refused here and honored there - a guard stricter than its own applier.
        engine_src = inspect.getsource(MuJoCoSimEngine.send_action)
        assert "positive_whole_number_error" in engine_src
        assert rl_env._NUMERIC_DOMAINS["n_substeps"] is positive_whole_number_error


#: Wordings that state a *per-step displacement* bound rather than a bound on the
#: command. Narrowed by attribution, not by keyword: each is read only inside the
#: ``action_scale`` block of the constructor's own ``Args`` section, so a true
#: statement about another parameter's step behaviour ("the PD controller needs
#: several substeps") is not flagged for containing the word "step".
_PER_STEP_CLAIMS = (
    r"how far one\s+action\s+step",
    r"how far one\s+step",
    r"per[-\s]step",
)


def _argument_docs() -> dict[str, str]:
    """The ``Args:`` section of ``SimEnv``, one whitespace-collapsed block per parameter.

    The constructor's arguments are documented on the class docstring (``__init__``
    carries none), which is what ``help(SimEnv)`` and the API reference render.
    """
    doc = inspect.getdoc(SimEnv) or ""
    body = doc.split("Args:", 1)[1] if "Args:" in doc else ""
    blocks: dict[str, str] = {}
    current: str | None = None
    for line in body.splitlines():
        head = re.match(r"^ {4}(\w+): (.*)$", line)
        if head is not None:
            current = head.group(1)
            blocks[current] = head.group(2)
        elif current is not None and line.strip():
            blocks[current] += " " + line.strip()
        elif current is not None:
            current = None
    return {name: re.sub(r"\s+", " ", text) for name, text in blocks.items()}


class TestTheScaleBoundsTheCommandNotTheStep:
    """``action_scale`` multiplies the actuator command, so it caps what the policy can reach.

    ``SimEnv.step`` sends ``action * action_scale`` to ``send_action``, which is an
    actuator *command* - a position target on a position-actuated robot, a torque
    on a torque-actuated one - and never a displacement from the current state.
    The difference is not cosmetic. Measured on a real so100 in MuJoCo, a constant
    ``Elbow`` command of ``1.0`` held for 200 steps:

    ===============  =====================  ======================  ==================
    ``action_scale``  largest step (rad)     ``Elbow`` after 200     reaches 0.9 rad?
    ===============  =====================  ======================  ==================
    ``0.1``           ``0.0085``             ``0.1084``              **never**
    ``1.0``           ``0.0412``             ``0.9424``              yes
    ===============  =====================  ======================  ==================

    So the scale is neither the per-step bound it was documented as (``0.0085`` is
    not ``0.1``) nor a rate limit a patient policy could integrate past: at
    ``0.1`` the joint converges on ``0.1`` and stays there for any number of
    steps. The consequence is the one a reader needs: the backend clamps the
    product to each actuator's ``ctrlrange``, so a bounded actor - the
    ``tanh``-squashed FastSAC / FastTD3 actors emit ``[-1, 1]`` - can only command
    the overlap of that range with ``[-action_scale, action_scale]``: 46.4% of the
    so100's six ranges at the default scale, 57.6% of the g1's twenty-nine, and
    3.5% of the go2's twelve torque limits.
    """

    def test_the_command_sent_is_the_action_times_the_scale(self) -> None:
        recorder = _Recorder()
        env = _env(recorder, action_scale=0.1, max_episode_steps=100)
        env.reset()
        for _ in range(4):
            env.step(torch.ones((1, 2)))
        # Every command is the same 0.1: the scale does not accumulate over steps
        # (a displacement would have reached 0.4) and does not rate-limit toward a
        # larger target (a bounded actor never commands past the scale).
        assert recorder.actions == [[0.1, 0.1]] * 4

    def test_the_parser_reads_every_documented_numeric(self) -> None:
        # Non-vacuity: the prose cells below assert over these blocks, and an
        # empty parse would satisfy every one of them.
        assert set(rl_env._NUMERIC_DOMAINS) <= set(_argument_docs())

    def test_the_scale_is_not_documented_as_a_per_step_bound(self) -> None:
        block = _argument_docs()["action_scale"]
        offenders = [p for p in _PER_STEP_CLAIMS if re.search(p, block, re.IGNORECASE)]
        assert not offenders, f"action_scale is documented as a per-step displacement bound {offenders}: {block}"

    def test_the_scale_says_what_it_bounds(self) -> None:
        block = _argument_docs()["action_scale"]
        assert "reach" in block.lower(), f"action_scale does not say it bounds what the policy can reach: {block}"

    def test_a_planted_per_step_claim_is_flagged(self) -> None:
        # The patterns really do discriminate, so a clean ``action_scale`` block
        # above is a statement about the prose and not about the regexes.
        planted = "it is the magnitude bound on how far one action step may move a joint"
        assert [p for p in _PER_STEP_CLAIMS if re.search(p, planted, re.IGNORECASE)]

    def test_a_clean_block_is_not_flagged(self) -> None:
        clean = "it scales the command itself, so it bounds what the policy can reach, not one step's motion"
        assert not [p for p in _PER_STEP_CLAIMS if re.search(p, clean, re.IGNORECASE)]

    def test_the_position_target_claim_is_qualified_by_the_actuator_kind(self) -> None:
        # go2's twelve actuators have no position bias - the command is a torque -
        # so an unqualified "the action is a position target" is false for the
        # library's own locomotion robot.
        block = _argument_docs()["n_substeps"]
        if "position target" in block:
            assert "position-actuated" in block, (
                f"n_substeps calls the action a position target unconditionally: {block}"
            )

    def test_the_action_contract_states_the_clamp(self) -> None:
        # The module docstring owns the action contract, so the ``ctrlrange``
        # ceiling a bounded actor runs into belongs beside it.
        doc = rl_env.__doc__ or ""
        assert "ctrlrange" in doc, "the action contract does not say the command is clamped to each actuator's range"
