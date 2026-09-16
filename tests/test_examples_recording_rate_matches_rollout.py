"""Every shipped example records at the rate its own rollout captures.

The dataset recorder writes one frame per control step with **no decimation**,
so ``start_recording(fps=N)`` is only honoured by a ``run_policy`` that captures
at ``control_frequency=N``. A mismatch is refused outright rather than written
at a distorted timestamp rate, which means the episode lands with zero frames
and the example demonstrates nothing it claims to.

That refusal arrived with the recorder-rate guard and three shipped examples
were left declaring 30 fps while their rollouts ran at the 50 Hz default, so
each recorded an empty dataset: ``03_record_dataset.py`` exited on
``stop_recording`` reporting ``0 frames`` against a docstring promising
``100 frames``, ``07_post_tune_any_policy.py`` printed
``Recorded LeRobotDataset ->`` and then failed two steps later inside lerobot
with an unrelated-looking Hub 404 for the empty local set, and
``locomotion/vla_g1_workflow.py`` printed ``Episode N/N recorded.`` per episode
for a dataset with no frames in it.

The rates are compared as the example states them. An example that names no
rate at all is graded against what an unset rate resolves to, read from the code
rather than restated here: ``control_frequency=None`` adopts the fps of a
recording that is already open, so a rollout after ``start_recording`` cannot
mismatch, and falls back to ``SimEngine.DEFAULT_CONTROL_FREQUENCY`` when no
recording is open yet - which a rollout placed BEFORE the recording still can.
A rate the example does state is never adopted, so stating the wrong one is
still refused and still flagged here.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any

import pytest

from strands_robots.simulation.base import SimEngine

_EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

#: What ``run_policy`` declares for a caller who names no rate. ``None`` means
#: "resolve at call time" - adopt an open recording's fps, else fall back.
_UNSET_CONTROL_FREQUENCY = inspect.signature(SimEngine.run_policy).parameters["control_frequency"].default

#: The rate an unset rollout captures at with NO recording open. Read from the
#: engine so this file cannot disagree with the code it grades.
_NO_RECORDING_RATE = SimEngine.DEFAULT_CONTROL_FREQUENCY

#: A refactor that stops reaching the examples must fail loudly rather than
#: report a clean sweep over nothing.
_MINIMUM_GRADED_EXAMPLES = 3

_DYNAMIC = object()


def _literal(node: ast.AST) -> Any:
    """The literal value of ``node``, or ``_DYNAMIC`` when it is computed."""
    try:
        return ast.literal_eval(node)
    except Exception:
        return _DYNAMIC


def _keyword(call: ast.Call, name: str) -> Any:
    for kw in call.keywords:
        if kw.arg == name:
            return _literal(kw.value)
    return None


def _splatted_names(call: ast.Call) -> list[str]:
    """Names of ``**kwargs`` splats in ``call`` (``kw.arg is None``)."""
    return [kw.value.id for kw in call.keywords if kw.arg is None and isinstance(kw.value, ast.Name)]


def _dict_assignments(tree: ast.Module, name: str) -> list[dict[str, Any]]:
    """Every literal dict assigned to ``name`` anywhere in ``tree``."""
    out: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(t, ast.Name) and t.id == name for t in targets):
            continue
        if isinstance(node.value, ast.Dict):
            keys = [_literal(k) if k is not None else _DYNAMIC for k in node.value.keys]
            vals = [_literal(v) for v in node.value.values]
            out.append({k: v for k, v in zip(keys, vals, strict=True) if isinstance(k, str)})
    return out


def _capture_rates(tree: ast.Module, call: ast.Call, unset_rate: float) -> set[Any]:
    """Every rate ``call`` can capture at, given the module's assignments.

    A caller may pass the rate directly, omit it (resolving to ``unset_rate``),
    or splat a dict that carries it - a locomotion example builds one dict per
    data source, so every branch it can take is graded. An explicit ``None`` is
    the same request as omitting it and resolves the same way.
    """
    direct = _keyword(call, "control_frequency")
    if direct is not None:
        return {direct}
    splats = _splatted_names(call)
    if not splats:
        return {unset_rate}
    rates: set[Any] = set()
    for name in splats:
        assignments = _dict_assignments(tree, name)
        if not assignments:
            rates.add(_DYNAMIC)
            continue
        for mapping in assignments:
            rates.add(mapping.get("control_frequency") or unset_rate)
    return rates


def _graded_examples() -> list[tuple[Path, float, set[Any]]]:
    """``(path, declared fps, capture rates)`` per example that records."""
    graded: list[tuple[Path, float, set[Any]]] = []
    for path in sorted(_EXAMPLES.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        recordings, rollouts, opened_at = [], [], []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "start_recording":
                    recordings.append(_keyword(node, "fps"))
                    opened_at.append(node.lineno)
                elif node.func.attr == "run_policy":
                    rollouts.append(node)
        declared = {f for f in recordings if isinstance(f, (int, float))}
        # One rate per example is what a static reading can attribute; an
        # example declaring several is left ungraded rather than guessed at.
        if len(declared) != 1 or not rollouts:
            continue
        fps = float(declared.pop())
        first_open = min(opened_at)
        rates: set[Any] = set()
        for call in rollouts:
            # A rollout after the recording opens adopts its fps; one before it
            # has no recording to adopt from and takes the engine's fallback.
            unset_rate = fps if first_open < call.lineno else _NO_RECORDING_RATE
            rates |= _capture_rates(tree, call, unset_rate)
        graded.append((path, fps, rates))
    return graded


def test_every_recording_example_captures_at_the_rate_it_declares() -> None:
    """No example declares one rate and captures at another."""
    graded = _graded_examples()
    assert len(graded) >= _MINIMUM_GRADED_EXAMPLES, (
        f"only {len(graded)} example(s) were graded, expected at least "
        f"{_MINIMUM_GRADED_EXAMPLES}: the scan is no longer reaching "
        f"{_EXAMPLES}, so a clean result would prove nothing"
    )
    mismatched = []
    for path, fps, rates in graded:
        wrong = sorted(r for r in rates if r is not _DYNAMIC and isinstance(r, (int, float)) and float(r) != fps)
        if wrong:
            mismatched.append(
                f"{path.relative_to(_EXAMPLES.parent)}: records at {fps:g} fps but its "
                f"rollout captures at {', '.join(f'{w:g}' for w in wrong)} Hz - the "
                f"recorder refuses the mismatch and the episode lands with zero frames"
            )
    assert not mismatched, "example(s) record at a rate their rollout does not capture at:\n  " + "\n  ".join(
        mismatched
    )


def test_the_unset_rate_is_read_from_the_code_not_restated() -> None:
    """An unset rate resolves at call time, and its fallback lives on the engine."""
    assert _UNSET_CONTROL_FREQUENCY is None, (
        "run_policy no longer defers an unset control_frequency to call time, so it "
        "cannot adopt an open recording's fps and this file grades the wrong rule"
    )
    assert isinstance(_NO_RECORDING_RATE, (int, float))
    assert _NO_RECORDING_RATE > 0


@pytest.mark.parametrize(
    ("source", "should_flag"),
    [
        # Omitting the rate after a recording opens adopts its fps, at any value.
        ("sim.start_recording(fps=30)\nsim.run_policy(robot_name='r')", False),
        (f"sim.start_recording(fps={_NO_RECORDING_RATE:g})\nsim.run_policy(robot_name='r')", False),
        # Omitting it BEFORE any recording opens has no fps to adopt, so the
        # rollout takes the engine fallback and a differing declaration is flagged.
        (f"sim.run_policy(robot_name='r')\nsim.start_recording(fps={_NO_RECORDING_RATE + 20:g})", True),
        (f"sim.run_policy(robot_name='r')\nsim.start_recording(fps={_NO_RECORDING_RATE:g})", False),
        # An explicit None is the same request as omitting it.
        ("sim.start_recording(fps=30)\nsim.run_policy(control_frequency=None)", False),
        # Stating it wrongly is flagged; stating it correctly is not.
        ("sim.start_recording(fps=30)\nsim.run_policy(control_frequency=50.0)", True),
        ("sim.start_recording(fps=30)\nsim.run_policy(control_frequency=30.0)", False),
        # A splatted dict is resolved through its assignment.
        ("kw = {'control_frequency': 50.0}\nsim.start_recording(fps=30)\nsim.run_policy(**kw)", True),
        ("kw = {'control_frequency': 30.0}\nsim.start_recording(fps=30)\nsim.run_policy(**kw)", False),
    ],
)
def test_the_grader_flags_a_mismatch_and_leaves_a_match_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str, should_flag: bool
) -> None:
    """A clean sweep means the examples agree, not that the grader is blind."""
    examples = tmp_path / "examples"
    examples.mkdir()
    (examples / "planted.py").write_text(source, encoding="utf-8")
    monkeypatch.setattr(f"{__name__}._EXAMPLES", examples)
    (path, fps, rates) = _graded_examples()[0]
    flagged = any(r is not _DYNAMIC and isinstance(r, (int, float)) and float(r) != fps for r in rates)
    assert flagged is should_flag, f"grader returned rates={rates!r} against fps={fps} for:\n{source}"
