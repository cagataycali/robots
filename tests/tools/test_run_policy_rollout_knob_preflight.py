# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests: the ``run_policy`` tool checks every rollout knob it forwards.

The tool owns the episode loop *and* the recording lifecycle, and it starts that
recording with ``overwrite=True``. So a knob the facade refuses is not merely
reported late - it is reported after an existing dataset at ``dataset_root`` has
been removed and replaced with an empty one, and after every episode has failed
with the same message. Measured before the fix, over a real MuJoCo rollout that
had recorded one episode of four frames:

* ``control_frequency`` in ``{0, -5, nan, inf, True, '30', None, [30]}`` and
  ``action_horizon`` in ``{0, -5, 2.7, nan, True, '8', None, [8]}`` - 16 of 16 -
  each destroyed that episode (``meta/info.json`` went from
  ``total_episodes=1, total_frames=4`` to ``0, 0``) and returned
  ``run_policy: 0/1 episodes ok | parquet-truth: total_episodes=0``.
* The reason was already correct, just buried: every per-episode record read
  ``run_policy: control_frequency must be > 0, got 0.0.``

The tool's own pre-flight block states the principle four times over - for
``seed`` ("reached NumPy inside the loop after step 2 had already created a
dataset"), ``video``, the provider keyword bags ("otherwise the tool leaves an
empty dataset behind and reports every episode as raised") and ``stop_when``.
These two knobs were the ones left out, so they are now checked on the same
shared domains the facade applies, which is what keeps the reported reason
byte-identical while moving it ahead of the destruction.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# Imported as a module rather than also pulling the tool out by name: the
# structural sweep below reads the shipped source via ``inspect.getfile``, and
# one import of a module is clearer than two forms of it.
import strands_robots.tools.run_policy as rp_mod
from strands_robots.simulation.base import SimEngine
from strands_robots.utils import positive_count_error, positive_finite_number_error
from tests.tools.test_run_policy import _FakeSim

#: Values no control loop can be driven at. ``inf`` is here because a rate is a
#: divisor: ``1 / inf`` is 0, so it is a rate no period can be built from rather
#: than a very fast one. ``True`` is an ``int`` subclass that would act as a
#: silent 1 Hz.
UNUSABLE_FREQUENCIES: list[Any] = [0.0, -5.0, float("nan"), float("inf"), True, "30", None, [30]]

#: Values that are not a positive count of actions. ``2.7`` is here because the
#: horizon indexes into a policy chunk, and ``True`` because it would act as a
#: silent horizon of 1.
UNUSABLE_HORIZONS: list[Any] = [0, -5, 2.7, float("nan"), True, "8", None, [8]]

#: Probe set for the parity premise below - the unusable values plus the
#: spellings a caller legitimately reads out of a config, so parity is asserted
#: for the accepted side too (a NumPy float rate is usable; a NumPy integer
#: horizon is not, because the horizon is a strict-``int`` count).
PARITY_FREQUENCIES: list[Any] = [*UNUSABLE_FREQUENCIES, 30.0, 50, np.float32(50.0), np.float64(30.0), np.int64(50)]
PARITY_HORIZONS: list[Any] = [*UNUSABLE_HORIZONS, 8, 1, np.int64(8), np.float64(8.0)]


def _run_tool(sim: Any, **kwargs: Any) -> dict[str, Any]:
    """Call the tool with kwargs mypy cannot narrow.

    The unusable values are deliberately outside the declared parameter types -
    that is the point of the test - so they are splatted through one funnel
    rather than suppressed at every call site.
    """
    return dict(rp_mod.run_policy(sim, **kwargs))


def _text(result: dict[str, Any]) -> str:
    return str((result.get("content") or [{}])[0].get("text", ""))


def _facade_text(err: dict[str, Any] | None) -> str | None:
    """Flatten the facade's structured error dict to its message, or ``None``."""
    if err is None:
        return None
    return str((err.get("content") or [{}])[0].get("text", ""))


class TestTheDomainIsTheFacadesOwn:
    """The tool's early check must be the facade's rule, not a second one.

    If these two ever diverged, the tool would refuse a value the rollout would
    have accepted (or accept one it cannot run), and the pre-flight would stop
    being a strictly-earlier report of the same verdict.
    """

    @pytest.mark.parametrize("value", PARITY_FREQUENCIES, ids=repr)
    def test_the_frequency_domain_matches_validate_positive_frequency(self, value: Any) -> None:
        assert positive_finite_number_error(value, "control_frequency", "run_policy") == _facade_text(
            SimEngine._validate_positive_frequency(value, "run_policy")
        )

    @pytest.mark.parametrize("value", PARITY_HORIZONS, ids=repr)
    def test_the_horizon_domain_matches_validate_action_horizon(self, value: Any) -> None:
        assert positive_count_error(value, "action_horizon", "run_policy") == _facade_text(
            SimEngine._validate_action_horizon(value, "run_policy")
        )


class TestUnusableKnobsAreRefused:
    """Each unusable value is reported, naming the parameter and the value."""

    @pytest.mark.parametrize("value", UNUSABLE_FREQUENCIES, ids=repr)
    def test_an_unusable_control_frequency_is_refused(self, value: Any) -> None:
        result = _run_tool(_FakeSim(), n_episodes=1, n_steps=4, control_frequency=value)
        assert result["status"] == "error"
        assert _text(result) == f"run_policy: control_frequency must be > 0, got {value!r}."

    @pytest.mark.parametrize("value", UNUSABLE_HORIZONS, ids=repr)
    def test_an_unusable_action_horizon_is_refused(self, value: Any) -> None:
        result = _run_tool(_FakeSim(), n_episodes=1, n_steps=4, action_horizon=value)
        assert result["status"] == "error"
        assert _text(result) == f"run_policy: action_horizon must be a positive integer, got {value!r}."


class TestTheRefusalPrecedesTheRecording:
    """The reason this is a defect and not a tidiness question.

    ``start_recording`` is forwarded ``overwrite=True``, so reaching it with a
    knob the loop will refuse costs the caller the dataset already at that root.
    A refused call must therefore touch neither the recorder nor the facade.
    """

    @pytest.mark.parametrize("value", UNUSABLE_FREQUENCIES, ids=repr)
    def test_a_refused_frequency_starts_no_recording(self, value: Any) -> None:
        sim = _FakeSim()
        result = _run_tool(sim, n_episodes=1, n_steps=4, control_frequency=value, dataset_root="/tmp/does-not-exist")
        assert result["status"] == "error"
        assert sim.start_recording_calls == [], "the refused call reached start_recording(overwrite=True)"
        assert sim.run_policy_calls == []
        assert sim.stop_recording_calls == []

    @pytest.mark.parametrize("value", UNUSABLE_HORIZONS, ids=repr)
    def test_a_refused_horizon_starts_no_recording(self, value: Any) -> None:
        sim = _FakeSim()
        result = _run_tool(sim, n_episodes=1, n_steps=4, action_horizon=value, dataset_root="/tmp/does-not-exist")
        assert result["status"] == "error"
        assert sim.start_recording_calls == [], "the refused call reached start_recording(overwrite=True)"
        assert sim.run_policy_calls == []

    def test_the_refusal_is_reported_at_the_top_level_not_per_episode(self) -> None:
        """Before the fix the reason was only inside ``episodes[0]["text"]``."""
        result = _run_tool(_FakeSim(), n_episodes=3, n_steps=4, control_frequency=0.0)
        assert _text(result) == "run_policy: control_frequency must be > 0, got 0.0."
        payload = next((b["json"] for b in result.get("content") or [] if "json" in b), None)
        assert payload is None, "a refused pre-flight reports no episode records"


class TestUsableKnobsStillRun:
    """Over-reach control: nothing a caller could legitimately pass is refused."""

    @pytest.mark.parametrize("value", [30.0, 50, 12.5, np.float32(50.0), np.float64(30.0), np.int64(50)], ids=repr)
    def test_a_usable_frequency_drives_the_loop(self, value: Any) -> None:
        sim = _FakeSim()
        result = _run_tool(sim, n_episodes=2, n_steps=4, control_frequency=value)
        assert result["status"] == "success"
        assert len(sim.run_policy_calls) == 2
        assert sim.run_policy_calls[0]["control_frequency"] == value

    @pytest.mark.parametrize("value", [1, 8, 64], ids=repr)
    def test_a_usable_horizon_drives_the_loop(self, value: Any) -> None:
        sim = _FakeSim()
        result = _run_tool(sim, n_episodes=2, n_steps=4, action_horizon=value)
        assert result["status"] == "success"
        assert sim.run_policy_calls[0]["action_horizon"] == value

    def test_the_documented_defaults_are_inside_the_domain(self) -> None:
        """A caller who supplies neither knob must not be refused by the new checks.

        The defaults are read off the source rather than the decorated tool
        object, which exposes no undecorated function to introspect.
        """
        defaults = _declared_defaults(Path(inspect.getfile(rp_mod)).read_text())
        assert defaults["control_frequency"] == 30.0
        assert defaults["action_horizon"] == 8
        assert positive_finite_number_error(defaults["control_frequency"], "control_frequency", "run_policy") is None
        assert positive_count_error(defaults["action_horizon"], "action_horizon", "run_policy") is None

    def test_a_usable_rollout_still_records(self) -> None:
        sim = _FakeSim()
        result = _run_tool(
            sim, n_episodes=1, n_steps=4, control_frequency=30.0, action_horizon=8, dataset_root="/tmp/probe-usable"
        )
        assert len(sim.start_recording_calls) == 1
        assert sim.start_recording_calls[0]["overwrite"] is True
        assert len(sim.run_policy_calls) == 1
        # ``status`` is "error" only because the fake writes no parquet truth.
        assert "1/1 episodes ok" in _text(result)


# --------------------------------------------------------------------------
# Structural sweep: no numeric knob may be forwarded without a pre-flight
# --------------------------------------------------------------------------

#: The numeric tool parameters forwarded verbatim into the per-episode
#: ``Simulation.run_policy`` call. Derived below rather than listed, so a knob
#: added to that call is picked up instead of quietly skipped.
KNOWN_FORWARDED_NUMERIC_KNOBS = {"n_steps", "control_frequency", "action_horizon"}

#: Annotations that mark a parameter as numeric for the sweep. ``bool`` is not
#: here: ``fast_mode`` is forwarded too, but a flag has no numeric domain.
_NUMERIC_ANNOTATIONS = {"int", "float", "int | None", "float | None"}


def _tool_function(source: str) -> ast.FunctionDef:
    """Return the ``run_policy`` tool's own FunctionDef from ``source``."""
    tree = ast.parse(source)
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "run_policy"
        and any(arg.arg == "simulation" for arg in node.args.args)
    )


def _declared_defaults(source: str) -> dict[str, Any]:
    """Return the tool's keyword-only defaults, read from its signature."""
    fn = _tool_function(source)
    return {
        arg.arg: ast.literal_eval(default)
        for arg, default in zip(fn.args.kwonlyargs, fn.args.kw_defaults, strict=True)
        if default is not None
    }


def _forwarded_numeric_knobs(source: str) -> set[str]:
    """Numeric tool parameters passed verbatim to ``simulation.run_policy(...)``."""
    fn = _tool_function(source)
    numeric = {
        arg.arg
        for arg in fn.args.args + fn.args.kwonlyargs
        if arg.annotation is not None and ast.unparse(arg.annotation) in _NUMERIC_ANNOTATIONS
    }
    forwarded: set[str] = set()
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "run_policy":
            continue
        for keyword in node.keywords:
            if isinstance(keyword.value, ast.Name) and keyword.value.id in numeric:
                forwarded.add(keyword.value.id)
    return forwarded


def _preflighted_knobs(source: str) -> set[str]:
    """Parameters checked before the statement that starts the recording.

    A mention only counts when it sits inside a domain call (``*_error(...)``),
    an ``isinstance`` probe or a comparison - i.e. somewhere a verdict is
    reached - so merely forwarding or re-assigning the value does not qualify.
    """
    fn = _tool_function(source)
    # Enumerate from 1: ``fn.body[0]`` is the docstring, and it documents the
    # ``start_recording(root=dataset_root, ...)`` call by name.
    start_index = next(
        index
        for index, node in enumerate(fn.body)
        if index > 0 and "start_recording(" in (ast.get_source_segment(source, node) or "")
    )
    checked: set[str] = set()
    for statement in fn.body[1:start_index]:
        for node in ast.walk(statement):
            judging = (
                isinstance(node, ast.Compare)
                or (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "isinstance")
                or (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id.endswith("_error"))
            )
            if not judging:
                continue
            for child in ast.walk(node):
                if isinstance(child, ast.Name):
                    checked.add(child.id)
    return checked


class TestEveryForwardedNumericKnobIsPreflighted:
    """A numeric knob added to the forwarded call must be checked before step 2.

    The tool destroys and recreates the dataset at ``dataset_root`` before the
    episode loop runs, so "the facade will refuse it" is not enough: the refusal
    has to happen while nothing has been set up yet.
    """

    def test_the_sweep_finds_the_known_knobs(self) -> None:
        """Non-vacuity: a mis-rooted scan must not report a clean sweep."""
        source = Path(inspect.getfile(rp_mod)).read_text()
        assert _forwarded_numeric_knobs(source) == KNOWN_FORWARDED_NUMERIC_KNOBS

    def test_every_forwarded_numeric_knob_is_checked_first(self) -> None:
        source = Path(inspect.getfile(rp_mod)).read_text()
        adrift = sorted(_forwarded_numeric_knobs(source) - _preflighted_knobs(source))
        assert adrift == [], f"forwarded without a pre-flight check: {adrift}"

    def test_the_sweep_catches_a_removed_guard(self) -> None:
        """Planted defect: an empty result must mean a clean source, not a blind scan."""
        source = Path(inspect.getfile(rp_mod)).read_text()
        guard = '    if horizon_error := positive_count_error(action_horizon, "action_horizon", "run_policy"):\n        return _err(horizon_error)\n'
        assert source.count(guard) == 1
        planted = source.replace(guard, "")
        adrift = sorted(_forwarded_numeric_knobs(planted) - _preflighted_knobs(planted))
        assert adrift == ["action_horizon"]
