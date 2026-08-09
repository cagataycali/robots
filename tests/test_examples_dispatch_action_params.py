"""Examples must not pass a parameter the dispatch router refuses.

``sim._dispatch_action(action, params)`` reports an unknown parameter by
RETURNING ``{"status": "error", ...}`` rather than raising, so an example that
discards the result silently no-ops. The observed defect (#244): a sim VLA
example passed ``camera_name`` to ``add_camera``, whose parameter is ``name``.
``camera_name`` is the render-side spelling (``render`` / ``render_depth`` /
``get_frame`` / ``get_camera_params`` / ``get_world_point``), so the mistake
reads plausible -- and because the envelope was dropped the world kept only its
default camera and the policy failed much later complaining about missing image
keys, pointing at the policy instead of the refused call.

This statically scans every example for ``_dispatch_action`` calls written with
a literal action name and a literal parameter dict, and checks each parameter
name against the router's own acceptance rule. The accepted set is DERIVED from
the live engine class (its ``_ACTION_ALIASES`` / ``_FIELD_ALIASES`` /
``_ROUTER_PASSTHROUGH`` plus ``inspect.signature``) rather than hand-listed, so
the guard tracks the router instead of drifting from it.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine  # noqa: E402

_EXAMPLES_DIR = Path(inspect.getfile(MuJoCoSimEngine)).resolve().parents[3] / "examples"

# Actions that fold flat video keys into a structured ``video`` dict; the router
# accepts those flat keys at its boundary even though the method takes ``video``.
_VIDEO_FOLDING_ACTIONS = frozenset({"run_policy", "start_policy", "eval_policy", "evaluate_benchmark"})
_VIDEO_FLAT_KEYS = frozenset({"output_path", "fps", "camera_name"})


def _accepted_params(action: str) -> frozenset[str] | None:
    """Names the router accepts for ``action``, or ``None`` if it accepts anything.

    Mirrors ``MuJoCoSimEngine._validate_and_build_kwargs``: a method declaring
    ``**kwargs`` legitimately passes residual keys through, so the router skips
    the unknown-key check for it and so does this guard.
    """
    method_name = MuJoCoSimEngine._ACTION_ALIASES.get(action, action)
    method = getattr(MuJoCoSimEngine, method_name, None)
    if method is None or action.startswith("_"):
        return frozenset()  # unknown action: no parameter is acceptable
    sig = inspect.signature(method)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return None
    named = {
        n
        for n, p in sig.parameters.items()
        if n != "self" and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }
    accepted = named | set(MuJoCoSimEngine._FIELD_ALIASES) | set(MuJoCoSimEngine._ROUTER_PASSTHROUGH)
    if action in _VIDEO_FOLDING_ACTIONS:
        accepted |= _VIDEO_FLAT_KEYS
    # name/robot_name are aliased in both directions by the router.
    if "name" in named:
        accepted.add("robot_name")
    if "robot_name" in named:
        accepted.add("name")
    return frozenset(accepted)


def _is_router_action(action: str) -> bool:
    """True when ``action`` names a real dispatchable method on the engine."""
    if action.startswith("_"):
        return False
    method_name = MuJoCoSimEngine._ACTION_ALIASES.get(action, action)
    return callable(getattr(MuJoCoSimEngine, method_name, None))


def _literal_dispatch_calls(source: str) -> list[tuple[int, str, tuple[str, ...]]]:
    """``(lineno, action, param_names)`` for each literal action + params call.

    Matched by SHAPE rather than by callee name: any call passing an adjacent
    ``("<known action>", {literal dict})`` positional pair. Examples routinely
    wrap the router in a local ``_must(sim, action, params)`` helper that raises
    on an error envelope, and a scanner keyed on ``_dispatch_action`` would go
    blind on exactly the examples that handle errors correctly. Requiring the
    string to name a real engine method keeps the shape from matching unrelated
    two-argument calls.
    """
    found: list[tuple[int, str, tuple[str, ...]]] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        for first, second in zip(node.args, node.args[1:], strict=False):
            if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
                continue
            if not isinstance(second, ast.Dict) or not _is_router_action(first.value):
                continue
            keys = [k.value for k in second.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)]
            if len(keys) != len(second.keys):
                continue  # a non-literal key: nothing static to check
            found.append((node.lineno, first.value, tuple(keys)))
    return found


def _example_sources() -> list[Path]:
    return sorted(p for p in _EXAMPLES_DIR.rglob("*.py") if p.is_file())


def _all_calls() -> list[tuple[Path, int, str, tuple[str, ...]]]:
    calls: list[tuple[Path, int, str, tuple[str, ...]]] = []
    for path in _example_sources():
        for lineno, action, keys in _literal_dispatch_calls(path.read_text(encoding="utf-8")):
            calls.append((path, lineno, action, keys))
    return calls


class TestExamplesUseParametersTheRouterAccepts:
    """Every literal ``_dispatch_action`` in an example must name real parameters."""

    def test_no_example_passes_a_parameter_the_router_refuses(self) -> None:
        offenders: list[str] = []
        for path, lineno, action, keys in _all_calls():
            accepted = _accepted_params(action)
            if accepted is None:
                continue
            for key in keys:
                if key not in accepted:
                    rel = path.relative_to(_EXAMPLES_DIR.parent)
                    offenders.append(
                        f"{rel}:{lineno}: action {action!r} does not accept {key!r}; valid: {sorted(accepted)}"
                    )
        assert not offenders, "examples pass parameters the dispatch router refuses:\n" + "\n".join(offenders)

    def test_the_scan_finds_the_examples_it_is_meant_to_cover(self) -> None:
        """Non-vacuity: a mis-rooted scan must not report a clean sweep over nothing."""
        calls = _all_calls()
        assert calls, f"no literal _dispatch_action calls found under {_EXAMPLES_DIR}"
        actions = {action for _, _, action, _ in calls}
        assert "add_camera" in actions, f"add_camera call not discovered; found {sorted(actions)}"

    def test_a_planted_bad_parameter_is_detected(self) -> None:
        """Meta: an empty offender list must mean clean sources, not a blind scanner."""
        for planted in (
            'sim._dispatch_action("add_camera", {"camera_name": "front"})\n',
            '_must(sim, "add_camera", {"camera_name": "front"})\n',  # the wrapper form
        ):
            calls = _literal_dispatch_calls(planted)
            assert calls == [(1, "add_camera", ("camera_name",))], (planted, calls)
        accepted = _accepted_params("add_camera")
        assert accepted is not None
        assert "camera_name" not in accepted
        assert "name" in accepted


class TestTheAcceptanceMirrorAgreesWithTheRouter:
    """The derived set must match what the live router actually does."""

    @pytest.mark.parametrize(
        ("action", "param", "accepted"),
        [
            ("add_camera", "name", True),
            ("add_camera", "camera_name", False),  # the #244 defect
            ("add_camera", "position", True),
            ("get_observation", "robot_name", True),
            ("get_observation", "camera_name", False),  # same file, same mistake
            ("render", "camera_name", True),  # the render-side spelling is correct here
            ("get_world_point", "camera_name", True),
            ("run_policy", "camera_name", True),  # folded into video.camera
        ],
    )
    def test_mirror_verdict(self, action: str, param: str, accepted: bool) -> None:
        allowed = _accepted_params(action)
        assert allowed is not None, f"{action} accepts **kwargs; nothing to assert"
        assert (param in allowed) is accepted
