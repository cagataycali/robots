"""One quaternion domain for every entry point that takes an orientation.

Four finite components make a readable vector, not a rotation. A wxyz value
whose norm rounds to zero has no direction to recover, and nothing downstream
says so: MuJoCo refuses ``quat="0 0 0 0"`` through its XML door outright ("zero
quaternion is not allowed") but accepts it through the spec-attribute and
``qpos`` doors this package writes through, substituting identity and reporting
success. ``move_object`` then echoed the requested quaternion back in its
success text while ``get_body_state`` reported identity.

``move_to`` had always refused such a value, with a hand-rolled check sitting
directly after the shared pose guard - so the library held one wxyz quaternion
to two different domains depending on which entry point received it, the drift
:func:`~strands_robots.utils.coerce_pose_vector` exists to prevent. These tests
pin the shared domain and the rule that every orientation parameter reaches it.
"""

import ast
import math
import pathlib

import numpy as np
import pytest

from strands_robots.utils import (
    MIN_QUATERNION_NORM,
    coerce_orientation_quaternion,
    orientation_quaternion_error,
)

#: The two shared helpers that hold a value to the orientation domain.
QUATERNION_DOMAIN = frozenset({"coerce_orientation_quaternion", "orientation_quaternion_error"})

#: The general pose-vector guards. They read a width and finite components, which
#: is the whole contract for a position and only part of it for an orientation.
POSE_DOMAIN = frozenset({"coerce_pose_vector", "pose_vector_error"})

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent.parent / "strands_robots"


def _domain_calls():
    """Every call to a pose or orientation guard that names an ``"orientation"``.

    Returns:
        ``(sites, files_scanned)`` where each site is
        ``(module_path, lineno, callee_name)``.
    """
    sites, scanned = [], 0
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - the package parses
            continue
        scanned += 1
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            name = node.func.id
            if name not in POSE_DOMAIN | QUATERNION_DOMAIN:
                continue
            named = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
            if "orientation" in named:
                sites.append((str(path.relative_to(PACKAGE_ROOT.parent)), node.lineno, name))
    return sites, scanned


class TestTheDomainReadsARotationRatherThanFourNumbers:
    """What the shared orientation guard accepts and what it refuses."""

    def test_an_omitted_orientation_is_not_a_value(self):
        assert coerce_orientation_quaternion("add_object", "orientation", None) == (None, None)

    @pytest.mark.parametrize(
        "quat",
        [
            pytest.param([1.0, 0.0, 0.0, 0.0], id="identity"),
            pytest.param([0.0, 2.0, 0.0, 0.0], id="non-unit"),
            pytest.param([0.7071, 0.0, 0.7071, 0.0], id="quarter-turn"),
            pytest.param(np.array([0.0, 1.0, 0.0, 0.0]), id="numpy-array"),
        ],
    )
    def test_a_quaternion_with_a_direction_is_accepted_whatever_its_magnitude(self, quat):
        floats, err = coerce_orientation_quaternion("add_object", "orientation", quat)
        assert err is None
        assert floats == [float(component) for component in quat]

    @pytest.mark.parametrize(
        "quat",
        [
            pytest.param([0.0, 0.0, 0.0, 0.0], id="all-zero"),
            pytest.param([0.0, 0.0, 0.0, -0.0], id="negative-zero"),
            pytest.param([1e-12, 0.0, 0.0, 0.0], id="below-the-bound"),
            pytest.param(np.zeros(4), id="numpy-zeros"),
        ],
    )
    def test_a_quaternion_with_no_direction_is_refused(self, quat):
        floats, err = coerce_orientation_quaternion("move_object", "orientation", quat)
        assert floats is None
        assert err == "move_object: 'orientation' quaternion has ~zero norm; pass a valid [w, x, y, z]."

    def test_the_bound_is_a_norm_not_a_component(self):
        # Four components each BELOW the bound still make a usable direction:
        # their norm is above it. Reading the largest component instead of the
        # norm would refuse this value, which is why the check is a norm.
        component = 0.6 * MIN_QUATERNION_NORM
        norm = math.sqrt(4 * component**2)
        assert component < MIN_QUATERNION_NORM < norm, "premise: every component below the bound, the norm above it"
        floats, err = coerce_orientation_quaternion("add_robot", "orientation", [component] * 4)
        assert err is None and floats is not None

    @pytest.mark.parametrize(
        "quat",
        [
            pytest.param([0.0, 1.0, 0.0], id="too-short"),
            pytest.param([float("nan"), 1.0, 0.0, 0.0], id="nan-component"),
            pytest.param([True, 1.0, 0.0, 0.0], id="bool-component"),
            pytest.param(0.5, id="scalar"),
        ],
    )
    def test_the_pose_vector_rules_still_apply(self, quat):
        floats, err = coerce_orientation_quaternion("add_object", "orientation", quat)
        assert floats is None and err is not None
        assert "add_object" in err and "orientation" in err

    def test_the_error_only_wrapper_agrees_on_every_supplied_value(self):
        for quat in ([0.0] * 4, [0.0, 1.0, 0.0, 0.0], [1.0, 2.0], 0.5, np.zeros(4)):
            assert (
                orientation_quaternion_error("op", "quat", quat)
                == (coerce_orientation_quaternion("op", "quat", quat)[1])
            )

    def test_the_two_wrappers_read_a_none_differently_and_that_is_the_point(self):
        # The same split as pose_vector_error / coerce_pose_vector. A keyword
        # argument left at None was not supplied; a key PRESENT in an op dict
        # carries a value, and reading it as an omission would apply identity
        # under a success result.
        assert coerce_orientation_quaternion("add_object", "orientation", None) == (None, None)
        refusal = orientation_quaternion_error("set_body_quat", "quat", None)
        assert refusal is not None and "None" in refusal


class TestEveryOrientationParameterReachesThatDomain:
    """The rule that keeps the two contracts from drifting apart again."""

    def test_no_orientation_is_held_to_the_position_contract(self):
        sites, scanned = _domain_calls()
        assert scanned >= 50, f"the scan reached only {scanned} modules; it is not measuring the package"
        assert len(sites) >= 11, f"only {len(sites)} orientation guard calls found; the scan is not finding them"
        general = [site for site in sites if site[2] in POSE_DOMAIN]
        assert general == [], (
            "these orientation parameters are held to the position contract (four finite components) "
            f"instead of the quaternion domain: {general}"
        )

    def test_the_general_guards_are_no_longer_asked_for_four_components(self):
        # A width of 4 asked of a pose guard is an orientation by construction:
        # the only 4-component pose vector this library has is a wxyz quaternion.
        # The shared reader both domains sit on (``_read_pose_vector``) is not in
        # POSE_DOMAIN, so the quaternion domain's own read is not an offender.
        offenders, scanned = [], 0
        for path in sorted(PACKAGE_ROOT.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            scanned += 1
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                    continue
                if node.func.id not in POSE_DOMAIN or not node.args:
                    continue
                last = node.args[-1]
                if isinstance(last, ast.Constant) and last.value == 4:
                    offenders.append((str(path.relative_to(PACKAGE_ROOT.parent)), node.lineno))
        assert scanned >= 50, f"the scan reached only {scanned} modules; it is not measuring the package"
        assert offenders == [], f"a four-component pose guard outside the quaternion domain: {offenders}"
