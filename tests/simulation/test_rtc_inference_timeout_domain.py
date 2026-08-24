"""Accepted domain of ``rtc_inference_timeout_s``, the async-RTC prefetch deadline.

The async-RTC chunk pipeline swaps a prefetched chunk in with
``future.result(timeout=rtc_inference_timeout_s)`` and turns a
``concurrent.futures.TimeoutError`` into a structured rollout error reading
"policy inference is stuck. Raise the timeout or check the policy/server."

That sentence is only true of a deadline a healthy inference could have met.
``0``, a negative value and ``nan`` all make ``Future.result`` give up
immediately, so a policy that answered on time is reported as stuck; ``inf``
reaches ``time_t`` arithmetic and raises ``OverflowError`` naming nothing;
``True`` is a silent one-second budget; a string or a list leaks a bare
``TypeError`` from the comparison. Every sibling wall-clock knob of the same
rollout - ``duration``, ``control_frequency``, ``control_substeps`` - is already
bounded by :func:`~strands_robots.utils.positive_finite_number_error`.

These tests pin, on the values themselves rather than on wording:

* the premise - what ``Future.result`` really does with each of them,
* both public envelope surfaces (``run_policy`` / ``eval_policy``) refusing,
* both directly-drivable runner surfaces raising,
* the refusal landing before any inference is queried,
* the two usable spellings (``None`` and a positive finite number) still working,
* the check being unconditional rather than gated on ``async_rtc``,
* and a structural sweep so a future surface accepting the parameter cannot
  forget the domain.
"""

from __future__ import annotations

import ast
import concurrent.futures as futures
import inspect
import pathlib
import threading
from typing import Any

import numpy as np
import pytest

import strands_robots.simulation.base as base_mod
from strands_robots.simulation.base import SimEngine
from strands_robots.simulation.policy_runner import PolicyRunner

from .test_policy_runner_async_rtc import _ChunkPolicy, _CountingSim

# Values no prefetch deadline can be built from. ``0`` and the negatives make
# the wait give up before the policy can answer; ``nan`` because every
# comparison against it is false; ``inf`` because it overflows ``time_t``;
# ``True`` because ``bool`` is an ``int`` subclass; the rest because they are
# not numbers at all.
UNUSABLE: list[Any] = [0, 0.0, -1.0, -0.5, float("nan"), float("inf"), float("-inf"), True, False, "5", [5], {}]

# Values a deadline CAN be built from. ``None`` is the documented "wait without
# a deadline" spelling and the parameter's own default, so it must stay valid.
USABLE: list[Any] = [None, 0.02, 2.5, 30, np.float32(1.5), np.float64(0.25), np.int64(2)]


def _runner_arm() -> tuple[_CountingSim, _ChunkPolicy]:
    """A GL-free engine plus a chunk-emitting policy that records its calls."""
    sim = _CountingSim()
    policy = _ChunkPolicy(sim, infer_sleep=0.0)
    policy.set_robot_state_keys(sim.robot_joint_names("arm"))
    return sim, policy


def _text(result: dict[str, Any]) -> str:
    return next(block["text"] for block in result["content"] if "text" in block)


class TestTheConsumerReallyRefusesTheseDeadlines:
    """The premise the domain rests on, measured against ``Future.result``.

    Without this the accepted set would be an assertion about the standard
    library rather than a measurement of it, and a future Python that started
    honoring one of these values would leave the domain quietly over-strict.
    """

    @pytest.mark.parametrize("value", [0, 0.0, -1.0, float("nan"), float("-inf"), False])
    def test_a_deadline_below_the_domain_gives_up_before_a_healthy_answer(self, value: Any) -> None:
        gate = threading.Event()
        with futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(lambda: (gate.wait(30), "chunk")[1])
            try:
                with pytest.raises(futures.TimeoutError):
                    future.result(timeout=value)
            finally:
                gate.set()

    def test_an_infinite_deadline_overflows_rather_than_waiting_forever(self) -> None:
        gate = threading.Event()
        with futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(lambda: (gate.wait(30), "chunk")[1])
            try:
                with pytest.raises(OverflowError):
                    future.result(timeout=float("inf"))
            finally:
                gate.set()

    def test_none_is_the_wait_without_a_deadline_spelling(self) -> None:
        with futures.ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(lambda: "chunk").result(timeout=None) == "chunk"


class TestTheFacadesRefuseAnUnusableDeadline:
    """``run_policy`` / ``eval_policy`` report it as a caller error."""

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_run_policy_refuses(self, value: Any) -> None:
        sim, policy = _runner_arm()
        result = sim.run_policy(
            robot_name="arm", policy_object=policy, n_steps=4, fast_mode=True, rtc_inference_timeout_s=value
        )
        assert result["status"] == "error", result
        assert "rtc_inference_timeout_s" in _text(result)
        assert "run_policy" in _text(result)
        assert repr(value) in _text(result)

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_eval_policy_refuses(self, value: Any) -> None:
        sim, policy = _runner_arm()
        result = sim.eval_policy(
            robot_name="arm", policy_object=policy, n_episodes=1, max_steps=4, rtc_inference_timeout_s=value
        )
        assert result["status"] == "error", result
        assert "rtc_inference_timeout_s" in _text(result)
        assert "eval_policy" in _text(result)

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_the_refusal_lands_before_the_policy_is_queried(self, value: Any) -> None:
        """No inference, no frame, no dataset row for a deadline that cannot hold."""
        sim, policy = _runner_arm()
        result = sim.run_policy(
            robot_name="arm", policy_object=policy, n_steps=4, fast_mode=True, rtc_inference_timeout_s=value
        )
        assert result["status"] == "error"
        assert policy.infer_starts == []
        assert sim.send_count == 0

    def test_the_message_does_not_blame_the_policy(self) -> None:
        """The pre-fix report accused a healthy model; the refusal must not."""
        sim, policy = _runner_arm()
        result = sim.run_policy(
            robot_name="arm", policy_object=policy, n_steps=4, fast_mode=True, rtc_inference_timeout_s=0
        )
        text = _text(result)
        assert "inference is stuck" not in text
        assert "check the policy/server" not in text


class TestTheRunnerRaisesForTheSameValues:
    """``PolicyRunner`` is drivable directly, so it enforces the domain too.

    A direct caller has no structured envelope to read a refusal from, which is
    why this layer raises where the facades return - the same split the seed
    domain already uses on these two methods.
    """

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_run_raises(self, value: Any) -> None:
        sim, policy = _runner_arm()
        with pytest.raises(ValueError, match="rtc_inference_timeout_s"):
            PolicyRunner(sim).run("arm", policy, n_steps=4, fast_mode=True, rtc_inference_timeout_s=value)
        assert policy.infer_starts == []

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_evaluate_raises(self, value: Any) -> None:
        sim, policy = _runner_arm()
        with pytest.raises(ValueError, match="rtc_inference_timeout_s"):
            PolicyRunner(sim).evaluate("arm", policy, n_episodes=1, max_steps=4, rtc_inference_timeout_s=value)
        assert policy.infer_starts == []

    def test_the_raised_message_names_the_method(self) -> None:
        sim, policy = _runner_arm()
        with pytest.raises(ValueError, match=r"PolicyRunner\.run: rtc_inference_timeout_s"):
            PolicyRunner(sim).run("arm", policy, n_steps=4, fast_mode=True, rtc_inference_timeout_s=-1.0)


class TestAUsableDeadlineIsStillHonored:
    """The accepted domain, including the ``None`` default and NumPy scalars."""

    @pytest.mark.parametrize("value", USABLE)
    def test_run_policy_accepts(self, value: Any) -> None:
        sim, policy = _runner_arm()
        result = sim.run_policy(
            robot_name="arm", policy_object=policy, n_steps=4, fast_mode=True, rtc_inference_timeout_s=value
        )
        assert result["status"] == "success", result

    @pytest.mark.parametrize("value", USABLE)
    def test_the_runner_accepts(self, value: Any) -> None:
        sim, policy = _runner_arm()
        result = PolicyRunner(sim).run("arm", policy, n_steps=4, fast_mode=True, rtc_inference_timeout_s=value)
        assert result["status"] == "success", result

    def test_a_usable_deadline_still_bounds_a_genuinely_stuck_inference(self) -> None:
        """The feature the parameter exists for keeps working.

        A policy slower than its own deadline is still aborted, and the report
        still names the parameter - the guard bounds which values reach the
        seam, not what the seam does with an honored one.
        """
        sim = _CountingSim()
        policy = _ChunkPolicy(sim, infer_sleep=0.3)
        policy.set_robot_state_keys(sim.robot_joint_names("arm"))
        result = PolicyRunner(sim).run(
            "arm", policy, n_steps=16, fast_mode=True, async_rtc=True, rtc_inference_timeout_s=0.01
        )
        assert result["status"] == "error", result
        assert "rtc_inference_timeout_s" in _text(result)


class TestTheDomainIsNotGatedOnTheAsyncFlag:
    """One value, one answer - whichever path the rollout resolves to.

    ``async_rtc=None`` (the default) auto-resolves from the policy's own
    chunk-emitting shape after the policy is constructed, so the facade cannot
    know whether the deadline will be read. Gating the check on the flag would
    leave that dominant path unguarded, so the check is unconditional; this
    pins that so a later reader cannot narrow it back.
    """

    @pytest.mark.parametrize("async_rtc", [None, True, False])
    def test_every_async_setting_refuses_the_same_value(self, async_rtc: bool | None) -> None:
        sim, policy = _runner_arm()
        result = sim.run_policy(
            robot_name="arm",
            policy_object=policy,
            n_steps=4,
            fast_mode=True,
            async_rtc=async_rtc,
            rtc_inference_timeout_s=0,
        )
        assert result["status"] == "error", result
        assert "rtc_inference_timeout_s" in _text(result)


class TestTheFacadeAndTheRunnerAgree:
    """Neither layer may accept a deadline the other refuses."""

    @pytest.mark.parametrize("value", [*USABLE, *UNUSABLE])
    def test_the_two_layers_reach_the_same_verdict(self, value: Any) -> None:
        sim, policy = _runner_arm()
        facade_refused = (
            sim.run_policy(
                robot_name="arm",
                policy_object=policy,
                n_steps=4,
                fast_mode=True,
                rtc_inference_timeout_s=value,
            )["status"]
            == "error"
        )

        sim2, policy2 = _runner_arm()
        try:
            PolicyRunner(sim2).run("arm", policy2, n_steps=4, fast_mode=True, rtc_inference_timeout_s=value)
            runner_refused = False
        except ValueError:
            runner_refused = True

        assert facade_refused is runner_refused, f"verdicts differ for {value!r}"


def _surfaces_taking_the_deadline() -> dict[str, list[str]]:
    """Public methods across ``strands_robots.simulation`` that accept the deadline."""
    root = pathlib.Path(inspect.getfile(base_mod)).parent
    found: dict[str, list[str]] = {}
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef) or node.name.startswith("_"):
                continue
            for method in node.body:
                if not isinstance(method, ast.FunctionDef) or method.name.startswith("_"):
                    continue
                names = [a.arg for a in method.args.args + method.args.kwonlyargs]
                if "rtc_inference_timeout_s" in names:
                    found.setdefault(str(path.relative_to(root)), []).append(f"{node.name}.{method.name}")
    return found


def _guards_or_forwards(source: str, func: ast.FunctionDef) -> tuple[bool, bool]:
    """Whether *func* validates the deadline itself, or forwards it by keyword."""
    guards = forwards = False
    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        name = target.id if isinstance(target, ast.Name) else getattr(target, "attr", "")
        segment = ast.get_source_segment(source, node) or ""
        if "rtc_inference_timeout_s" not in segment:
            continue
        if name in {"_validate_rtc_inference_timeout", "positive_finite_number_error"}:
            guards = True
        for keyword in node.keywords:
            if keyword.arg == "rtc_inference_timeout_s" and isinstance(keyword.value, ast.Name):
                if keyword.value.id == "rtc_inference_timeout_s":
                    forwards = True
    return guards, forwards


class TestEveryPublicSurfaceOwnsTheDomain:
    """A structural sweep, so a new surface cannot forget the deadline's domain.

    Checked structurally rather than by a hand-kept list of method names: a
    surface either validates the value or hands it verbatim to one that does.
    """

    def test_the_expected_surfaces_are_discovered(self) -> None:
        """Non-vacuity: an empty or mis-rooted sweep must not read as clean."""
        found = _surfaces_taking_the_deadline()
        assert found == {
            "base.py": ["SimEngine.run_policy", "SimEngine.eval_policy"],
            "mujoco/simulation.py": ["MuJoCoSimEngine.run_policy"],
            "policy_runner.py": ["PolicyRunner.run", "PolicyRunner.evaluate"],
        }, found

    def test_each_surface_validates_or_forwards(self) -> None:
        root = pathlib.Path(inspect.getfile(base_mod)).parent
        adrift: list[str] = []
        for module, qualnames in _surfaces_taking_the_deadline().items():
            source = (root / module).read_text()
            tree = ast.parse(source)
            for qualname in qualnames:
                cls, method_name = qualname.split(".")
                func = next(
                    m
                    for n in ast.walk(tree)
                    if isinstance(n, ast.ClassDef) and n.name == cls
                    for m in n.body
                    if isinstance(m, ast.FunctionDef) and m.name == method_name
                )
                guards, forwards = _guards_or_forwards(source, func)
                if not (guards or forwards):
                    adrift.append(f"{module}::{qualname}")
        assert not adrift, f"accept rtc_inference_timeout_s without validating or forwarding it: {adrift}"

    def test_the_sweep_detects_a_planted_surface(self) -> None:
        """The scanner must fail on an unguarded surface, not merely find none."""
        planted = ast.parse(
            "class Engine:\n"
            "    def run_policy(self, rtc_inference_timeout_s=None):\n"
            "        return self._go(timeout=rtc_inference_timeout_s)\n"
        )
        planted_class = planted.body[0]
        assert isinstance(planted_class, ast.ClassDef)
        func = planted_class.body[0]
        assert isinstance(func, ast.FunctionDef)
        guards, forwards = _guards_or_forwards(ast.unparse(planted), func)
        assert not guards and not forwards


class TestTheGuardAddsOnlyTheNoneCarveOut:
    """Everything but the ``None`` passthrough is the shared domain verbatim."""

    @pytest.mark.parametrize("value", [*USABLE, *UNUSABLE])
    def test_it_agrees_with_the_shared_rule_except_for_none(self, value: Any) -> None:
        from strands_robots.utils import positive_finite_number_error

        local_refuses = SimEngine._validate_rtc_inference_timeout(value, "run_policy") is not None
        shared_refuses = positive_finite_number_error(value, "rtc_inference_timeout_s", "run_policy") is not None
        if value is None:
            assert shared_refuses and not local_refuses
        else:
            assert local_refuses is shared_refuses, f"diverged for {value!r}"

    def test_the_message_is_the_shared_one_verbatim(self) -> None:
        from strands_robots.utils import positive_finite_number_error

        error = SimEngine._validate_rtc_inference_timeout(-1.0, "run_policy")
        assert error is not None
        assert error["content"][0]["text"] == positive_finite_number_error(
            -1.0, "rtc_inference_timeout_s", "run_policy"
        )
