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
import threading
from collections.abc import Sequence
from pathlib import Path

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine  # noqa: E402

_EXAMPLES_DIR = Path(inspect.getfile(MuJoCoSimEngine)).resolve().parents[3] / "examples"

# Actions that fold flat video keys into a structured ``video`` dict, and the flat
# keys they fold. Restated here so the table below grades the engine rather than
# importing its answer; ``test_the_restated_fold_matches_the_engine`` pins the two
# together. Acceptance is NOT simply "these keys for these actions": the fold is
# payload-dependent, so ``_accepted_params`` runs the engine's own fold instead of
# reasoning from these names.
_VIDEO_FOLDING_ACTIONS = frozenset({"run_policy", "start_policy", "eval_policy", "evaluate_benchmark"})
_VIDEO_FLAT_KEYS = frozenset({"output_path", "fps", "camera_name"})


def _accepted_params(action: str, param_names: Sequence[str] = ()) -> frozenset[str] | None:
    """Names the router accepts for ``action`` in a payload carrying *param_names*.

    Mirrors ``MuJoCoSimEngine._validate_and_build_kwargs``: a method declaring
    ``**kwargs`` legitimately passes residual keys through, so the router skips
    the unknown-key check for it and so does this guard.

    Acceptance of a flat video key is payload-dependent, so the payload's key set
    is part of the question: a bare ``camera_name`` has no path to attach itself
    to, and a flat key sent alongside an explicit ``video=`` is not folded at all.
    Rather than restate that rule, this RUNS it -
    ``MuJoCoSimEngine._fold_flat_video_keys`` is the router's own code, and a key
    it removes from the probe is a key validation never sees. Restating it is the
    defect this replaces: the mirror modelled the three flat keys as accepted for
    all four folding actions, while the router accepted a bare ``camera_name``
    for ``run_policy`` alone.
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
    # Keys the router's own fold consumes before validation are accepted by
    # construction: validation never sees them.
    probe: dict[str, object] = dict.fromkeys(param_names, "probe")
    MuJoCoSimEngine._fold_flat_video_keys(action, probe)
    accepted |= {key for key in param_names if key not in probe}
    # ...plus the residue ``run_policy`` alone accepts at the boundary.
    if action == "run_policy":
        accepted |= set(MuJoCoSimEngine._RUN_POLICY_RESIDUAL_VIDEO_KEYS)
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
            accepted = _accepted_params(action, keys)
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


# Payload shapes that reach the flat-video rule. The fold consumes what it can
# honor and leaves the rest to the unknown-key check, so the same key is accepted
# in one shape and refused in another - which is why a mirror keyed on the key
# NAME alone cannot track the router.
_FLAT_VIDEO_PAYLOADS: tuple[tuple[str, dict[str, object]], ...] = (
    ("output_path alone", {"output_path": "r.mp4"}),
    ("fps alone", {"fps": 15}),
    ("camera_name alone", {"camera_name": "wrist"}),
    ("output_path + camera_name", {"output_path": "r.mp4", "camera_name": "wrist"}),
    ("output_path + fps", {"output_path": "r.mp4", "fps": 15}),
    ("video + output_path", {"video": {"path": "a.mp4"}, "output_path": "b.mp4"}),
    ("video + fps", {"video": {"path": "a.mp4"}, "fps": 15}),
)


def _router_refuses(action: str, payload: dict[str, object]) -> bool:
    """True when the LIVE router rejects *payload* for *action* as an unknown key.

    The unknown-key check runs before the method body, so a bare engine with no
    world answers the routing question without a scene, a GL context or a fixture:
    an accepted payload falls through to the method's own "no world" refusal.
    """
    engine = MuJoCoSimEngine.__new__(MuJoCoSimEngine)
    engine._lock = threading.RLock()
    engine._world = None
    result = engine._dispatch_action(action, dict(payload))
    text = next((b["text"] for b in result.get("content", []) if isinstance(b, dict) and "text" in b), "")
    return "Unknown parameter" in text


class TestTheAcceptanceMirrorAgreesWithTheRouter:
    """The derived set must match what the live router actually does."""

    @pytest.mark.parametrize(
        ("action", "params", "param", "accepted"),
        [
            ("add_camera", ("name",), "name", True),
            ("add_camera", ("camera_name",), "camera_name", False),  # the #244 defect
            ("add_camera", ("position",), "position", True),
            ("get_observation", ("robot_name",), "robot_name", True),
            ("get_observation", ("camera_name",), "camera_name", False),  # same file, same mistake
            ("render", ("camera_name",), "camera_name", True),  # the render-side spelling is correct here
            ("get_world_point", ("camera_name",), "camera_name", True),
            # A bare camera_name has no path to attach itself to, so the fold leaves
            # it behind. run_policy swallows the residue; its three siblings refuse
            # it. The mirror modelled all four as accepting it.
            ("run_policy", ("camera_name",), "camera_name", True),
            ("start_policy", ("camera_name",), "camera_name", False),
            ("eval_policy", ("camera_name",), "camera_name", False),
            ("evaluate_benchmark", ("camera_name",), "camera_name", False),
            # Paired with a path the fold consumes it, for every folding action.
            ("eval_policy", ("output_path", "camera_name"), "camera_name", True),
            ("evaluate_benchmark", ("output_path", "camera_name"), "camera_name", True),
            # output_path and fps are always folded, so always accepted.
            ("eval_policy", ("output_path",), "output_path", True),
            ("start_policy", ("fps",), "fps", True),
        ],
    )
    def test_mirror_verdict(self, action: str, params: tuple[str, ...], param: str, accepted: bool) -> None:
        allowed = _accepted_params(action, params)
        assert allowed is not None, f"{action} accepts **kwargs; nothing to assert"
        assert (param in allowed) is accepted

    @pytest.mark.parametrize("action", sorted(_VIDEO_FOLDING_ACTIONS))
    @pytest.mark.parametrize(("label", "extra"), _FLAT_VIDEO_PAYLOADS, ids=[label for label, _ in _FLAT_VIDEO_PAYLOADS])
    def test_the_mirror_matches_the_router_for_every_folding_payload(
        self, action: str, label: str, extra: dict[str, object]
    ) -> None:
        """Drive the live router and require the mirror to reach the same verdict.

        The class name is a promise about the router, so the router is the oracle.
        Keying acceptance on the key name alone made the mirror bless a bare
        ``camera_name`` on the three sibling folding actions, where the router
        refuses it - so an example naming the video camera for an eval rollout
        passed this guard and no-oped at runtime.
        """
        base: dict[str, object] = (
            {"benchmark_name": "bench"} if action == "evaluate_benchmark" else {"robot_name": "arm"}
        )
        payload = {**base, **extra}
        allowed = _accepted_params(action, tuple(payload))
        assert allowed is not None, f"{action} accepts **kwargs; nothing to assert"
        mirror_refuses = sorted(key for key in payload if key not in allowed)
        router_refuses = _router_refuses(action, payload)
        assert bool(mirror_refuses) is router_refuses, (
            f"{action} with {label}: the mirror says refused={bool(mirror_refuses)} {mirror_refuses}, "
            f"the router says refused={router_refuses}"
        )

    def test_a_planted_flat_video_key_on_a_sibling_action_is_reported(self) -> None:
        """The #244 shape, on a folding action: the mirror must not bless a refusal."""
        planted = 'sim._dispatch_action("eval_policy", {"robot_name": "arm", "camera_name": "wrist"})\n'
        assert _literal_dispatch_calls(planted) == [(1, "eval_policy", ("robot_name", "camera_name"))]
        router_refuses = _router_refuses("eval_policy", {"robot_name": "arm", "camera_name": "wrist"})
        assert router_refuses, "premise: the router must refuse a bare camera_name for eval_policy"
        allowed = _accepted_params("eval_policy", ("robot_name", "camera_name"))
        assert allowed is not None
        assert "camera_name" not in allowed, (
            "the mirror accepts a bare camera_name for eval_policy while the router refuses it, "
            "so an example written that way passes this guard and no-ops at runtime"
        )

    def test_the_restated_folding_actions_match_the_engine(self) -> None:
        """Non-vacuity: the table's restatement must track the engine's own owner."""
        assert _VIDEO_FOLDING_ACTIONS == MuJoCoSimEngine._VIDEO_FOLDING_ACTIONS
        # The keys ``run_policy`` accepts as residue are the same three the fold folds.
        assert _VIDEO_FLAT_KEYS == MuJoCoSimEngine._RUN_POLICY_RESIDUAL_VIDEO_KEYS
