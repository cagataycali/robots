"""A render test's GL gate must be the shared probe's marker, not a local rebuild.

:mod:`tests.simulation.mujoco._gl_probe` owns the answer to "can this host render
offscreen" for the test suite. Its marker honours ``ROBOT_TEST_MUJOCO=0``, the
force-skip an operator sets "to keep a known-bad runner from attempting GL at
all" - the host class where a second renderer construction aborts the
interpreter uncatchably rather than raising.

A module that rebuilds the gate locally from
``strands_robots.simulation.mujoco.backend._can_render`` answers the same
question by a different route, and that route has no force-skip: the variable is
read by :func:`gl_available` and by nothing else, so a module holding its own
marker attempts GL on the runner that opted out. It also reaches into a private
production symbol for a test-harness concern, and it is invisible to
:mod:`tests.test_mujoco_render_assertions_are_gl_gated`, whose scan recognises
gating through the shared probe and would report such a module's render
assertion as ungated.

So the rule is one owner: exactly one module in the tree may build a GL-gating
skip marker, and it is the probe. The force-skip half is pinned where it lives -
``tests/simulation/mujoco/test_gl_probe.py::test_robot_test_mujoco_zero_forces_no_gl``
pins that the shared marker honours the variable - and composes with this rule to
cover every gated render test in the tree. One case here measures that
composition end to end through a child pytest, because the marker's condition is
evaluated at import time and no in-process assertion can observe it.

Reading ``_can_render`` is not itself the defect and is not reported:
``tests/simulation/mujoco/test_backend.py`` and
``tests/simulation/mujoco/test_software_render_warning.py`` are its contract
tests. The discriminator is building a skip marker out of it.
"""

from __future__ import annotations

import ast
import os
import pathlib
import subprocess
import sys

import pytest

#: Names that answer "can this host render offscreen". A ``skipif`` condition
#: mentioning one of these is a GL gate whichever spelling it uses.
GL_CAPABILITY_NAMES = frozenset({"_can_render", "gl_available"})

#: The single module allowed to build a GL-gating marker.
GATE_OWNER = "tests/simulation/mujoco/_gl_probe.py"

#: A module whose gate is the shared marker, used to measure the force-skip end
#: to end. Any consolidated module would do; this one is the cheapest.
A_GATED_MODULE = "tests/simulation/mujoco/test_renderer_hygiene.py"


def _tests_root() -> pathlib.Path:
    """This module's own directory - the tree the rule covers."""
    return pathlib.Path(__file__).parent


def gl_gating_markers(tree: ast.AST) -> list[int]:
    """Line numbers of every ``pytest.mark.skipif`` built from a GL probe."""
    lines = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "skipif":
            continue
        condition = node.args[0] if node.args else None
        if condition is None:
            continue
        for inner in ast.walk(condition):
            if isinstance(inner, ast.Name) and inner.id in GL_CAPABILITY_NAMES:
                lines.append(node.lineno)
                break
    return lines


def survey(root: pathlib.Path) -> dict[str, list[int]]:
    """Every module under *root* that builds a GL-gating marker, by path."""
    found: dict[str, list[int]] = {}
    for path in sorted(root.rglob("*.py")):
        lines = gl_gating_markers(ast.parse(path.read_text(encoding="utf-8")))
        if lines:
            found[path.relative_to(root.parent).as_posix()] = lines
    return found


class TestOnlyTheProbeBuildsAGlGate:
    def test_no_module_rebuilds_the_gate_locally(self) -> None:
        rebuilt = {path: lines for path, lines in survey(_tests_root()).items() if path != GATE_OWNER}
        assert not rebuilt, (
            f"these modules build their own GL-gating skip marker instead of taking the shared "
            f"one: {rebuilt}. A locally built gate does not honour ROBOT_TEST_MUJOCO=0, so it "
            f"attempts GL on the runner that opted out, and it is invisible to the render-gating "
            f"scan. Import requires_gl from tests.simulation.mujoco._gl_probe instead."
        )

    def test_the_probe_still_builds_the_one_gate(self) -> None:
        """Non-vacuity: a scan that matched nothing would read as a clean tree."""
        assert list(survey(_tests_root())) == [GATE_OWNER]


_PLANTED_LOCAL_GATE = """
import pytest

from strands_robots.simulation.mujoco.backend import _can_render

requires_gl = pytest.mark.skipif(not _can_render(), reason="no GL")
"""

_PLANTED_SHARED_GATE = """
from tests.simulation.mujoco._gl_probe import requires_gl


@requires_gl
def test_planted(): ...
"""

_PLANTED_PROBE_CONTRACT_TEST = """
from strands_robots.simulation.mujoco import backend


def test_the_probe_reports_a_bool():
    assert backend._can_render() in (True, False)
"""


class TestTheSurveyDetectsWhatItClaimsTo:
    @pytest.mark.parametrize(
        ("source", "reported"),
        [
            pytest.param(_PLANTED_LOCAL_GATE, True, id="local-gate"),
            pytest.param(_PLANTED_SHARED_GATE, False, id="shared-gate"),
            pytest.param(_PLANTED_PROBE_CONTRACT_TEST, False, id="probe-contract-test"),
        ],
    )
    def test_the_discriminator_is_building_the_marker(
        self, source: str, reported: bool, tmp_path: pathlib.Path
    ) -> None:
        """Reading the probe is fine; building a second gate out of it is not."""
        root = tmp_path / "tests"
        root.mkdir()
        (root / "test_planted.py").write_text(source, encoding="utf-8")
        assert bool(survey(root)) is reported


class TestTheForceSkipReachesAGatedModule:
    """The composition, measured: one owner plus a force-skip means every gate skips.

    ``test_gl_probe.py`` pins that the shared marker honours the variable, and
    the rule above pins that every gate is that marker. Nothing in-process can
    check the two together, because a ``skipif`` condition is evaluated when the
    module is imported - so this runs a child pytest over one gated module with
    the variable set, which is the only place the composition is observable.
    """

    def test_a_gated_module_skips_under_the_force_skip(self) -> None:
        root = _tests_root().parent
        module = root / A_GATED_MODULE
        assert module.is_file(), module
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", A_GATED_MODULE, "-q", "--no-cov", "-p", "no:cacheprovider"],
            capture_output=True,
            text=True,
            cwd=root,
            env={**os.environ, "ROBOT_TEST_MUJOCO": "0", "PYTHONPATH": str(root)},
            timeout=600,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert " passed" not in proc.stdout, (
            f"{A_GATED_MODULE} ran GL-dependent cases under ROBOT_TEST_MUJOCO=0, so its gate is "
            f"not the shared marker: {proc.stdout[-2000:]}"
        )
        assert " skipped" in proc.stdout, proc.stdout[-2000:]
