"""``DeclarativeBenchmark`` holds its four string fields to one shared domain.

``DeclarativeBenchmark`` has two construction paths that applied different
checks to the same values. ``from_dict`` refused a non-string ``name`` /
``default_robot`` / ``scene`` / ``instruction``; ``__init__`` stored each one
raw. Only ``max_steps`` and ``supported_robots`` were mirrored across the two,
and the comment on the first states the reason for all of them - a directly
constructed benchmark must not carry a value the evaluation loop, or the policy
it drives, has to deal with later.

Measured on ``3ce3da7`` (mujoco 3.11.0, ``MUJOCO_GL=egl``), constructing
directly and then running ``evaluate_benchmark`` over a two-link arm:

* ``instruction=42`` was accepted, and ``PolicyRunner`` handed it to the policy
  verbatim: ``get_actions`` received ``42`` where a task command belongs. The
  fallback that does it (``spec.instruction or ""``) exists so a
  language-conditioned policy receives the command rather than an empty string,
  and cannot tell an ``int`` from an instruction. The evaluation reported
  ``status="success"``.
* ``scene=[]`` was accepted and then *skipped*: ``on_episode_start`` loads the
  scene under ``if self._scene:``, so a falsy non-string meant the declared
  scene was never loaded, again under ``status="success"``.
* ``scene=42`` was the one loud case - truthy, so ``load_scene(42)`` failed and
  the evaluation returned an error. That is the behaviour the previous boundary
  pin described for all four; it holds for this one value only.
* ``name=7`` was accepted, stored, and advertised by ``list_benchmarks()``.

The same values through ``from_dict`` were all refused. These tests pin the two
paths to one rule.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any

import pytest

from strands_robots.simulation import benchmark_spec as spec_mod
from strands_robots.simulation.benchmark_spec import DeclarativeBenchmark

# The four string fields and the two axes they differ along: whether the empty
# string is a value they can carry, and whether ``None`` is how the field is
# omitted. Both are read off the shipped contract rather than asserted here -
# ``instruction`` defaults to ``""`` and ``scene`` defaults to ``None``.
FIELDS: dict[str, dict[str, bool]] = {
    "name": {"allow_empty": False, "allow_none": False},
    "default_robot": {"allow_empty": False, "allow_none": False},
    "scene": {"allow_empty": True, "allow_none": True},
    "instruction": {"allow_empty": True, "allow_none": False},
}

# Values no string field can be honored as, whatever its axes.
NOT_A_STRING: list[Any] = [
    pytest.param(7, id="int"),
    pytest.param(3.5, id="float"),
    pytest.param(True, id="bool"),
    pytest.param(["probe"], id="list"),
    pytest.param({"name": "probe"}, id="mapping"),
    pytest.param(object(), id="object"),
]


def _make(**overrides: Any) -> DeclarativeBenchmark:
    """Construct a benchmark directly, with ``overrides`` splatted in.

    Splatted rather than passed positionally so a deliberately off-type value
    reaches the runtime guard as an agent or a caller would supply it, instead
    of being reported by the type checker at each call site.
    """
    kwargs: dict[str, Any] = {
        "name": "probe",
        "supported_robots": [],
        "default_robot": "panda",
        "max_steps": 10,
        "success_fn": lambda _sim: False,
        "failure_fn": lambda _sim: False,
        "reward_terms": [],
    }
    kwargs.update(overrides)
    return DeclarativeBenchmark(**kwargs)


def _from_dict(**overrides: Any) -> DeclarativeBenchmark:
    """Compile the same benchmark from a spec dict, with ``overrides`` merged in."""
    spec: dict[str, Any] = {"name": "probe", "default_robot": "panda", "max_steps": 10}
    spec.update(overrides)
    return DeclarativeBenchmark.from_dict(spec)


def _refuses(build: Any, **overrides: Any) -> str | None:
    """Return the refusal message ``build`` raises, or ``None`` if it accepted."""
    try:
        build(**overrides)
    except ValueError as exc:
        return str(exc)
    return None


class TestDirectConstructionRefusesANonString:
    """Every field, every value that is not a ``str`` at all."""

    @pytest.mark.parametrize("field", sorted(FIELDS))
    @pytest.mark.parametrize("value", NOT_A_STRING)
    def test_it_is_refused(self, field: str, value: Any) -> None:
        text = _refuses(_make, **{field: value})
        assert text is not None, f"{field}={value!r} was accepted"

    @pytest.mark.parametrize("field", sorted(FIELDS))
    @pytest.mark.parametrize("value", NOT_A_STRING)
    def test_the_message_names_the_field_and_the_type(self, field: str, value: Any) -> None:
        text = _refuses(_make, **{field: value})
        assert text is not None
        assert field in text, text
        assert type(value).__name__ in text, text
        assert "DeclarativeBenchmark" in text, text


class TestDirectConstructionRefusesAnEmptyIdentifier:
    """``name`` and ``default_robot`` name something, so ``""`` names nothing."""

    @pytest.mark.parametrize("field", ["name", "default_robot"])
    def test_the_empty_string_is_refused(self, field: str) -> None:
        text = _refuses(_make, **{field: ""})
        assert text is not None, f"{field}='' was accepted"
        assert "non-empty" in text, text

    @pytest.mark.parametrize("field", ["scene", "instruction"])
    def test_the_empty_string_stays_accepted_for_content(self, field: str) -> None:
        """``instruction=""`` is its default and ``scene=""`` declares no scene."""
        assert _refuses(_make, **{field: ""}) is None


class TestNoneIsOnlyAValueForTheOptionalField:
    """``scene`` is the one field a spec omits by writing ``None``."""

    def test_a_none_scene_is_accepted(self) -> None:
        assert _refuses(_make, scene=None) is None

    @pytest.mark.parametrize("field", ["name", "default_robot", "instruction"])
    def test_a_none_elsewhere_is_refused(self, field: str) -> None:
        text = _refuses(_make, **{field: None})
        assert text is not None, f"{field}=None was accepted"
        assert "NoneType" in text, text


class TestUsableValuesAreUntouched:
    """The accepted side of the domain, so the guard cannot be read as a ban."""

    def test_the_default_construction_is_accepted(self) -> None:
        bench = _make()
        assert bench.name == "probe"
        assert bench.default_robot == "panda"
        assert bench.instruction == ""

    def test_a_declared_scene_and_instruction_are_kept(self) -> None:
        bench = _make(scene="scenes/table.xml", instruction="pick the cube")
        assert bench.instruction == "pick the cube"
        assert bench.on_episode_start is not None


class TestBothConstructionPathsAgree:
    """The property the mirror exists for: one rule, two paths.

    Parametrized over every field and every unusable value rather than asserting
    the two messages match - the contexts differ by design (``"spec"`` for a
    spec file, the class name for a direct construction), so what has to agree
    is the verdict.
    """

    @pytest.mark.parametrize("field", sorted(FIELDS))
    @pytest.mark.parametrize("value", NOT_A_STRING)
    def test_a_non_string_is_refused_by_both(self, field: str, value: Any) -> None:
        direct = _refuses(_make, **{field: value})
        spec = _refuses(_from_dict, **{field: value})
        assert (direct is None) == (spec is None), f"{field}={value!r}: __init__={direct!r} from_dict={spec!r}"
        assert direct is not None

    @pytest.mark.parametrize("field", sorted(FIELDS))
    def test_the_empty_string_verdict_agrees(self, field: str) -> None:
        direct = _refuses(_make, **{field: ""})
        spec = _refuses(_from_dict, **{field: ""})
        assert (direct is None) == (spec is None), f"{field}='': __init__={direct!r} from_dict={spec!r}"

    @pytest.mark.parametrize("field", sorted(FIELDS))
    def test_the_none_verdict_agrees(self, field: str) -> None:
        direct = _refuses(_make, **{field: None})
        spec = _refuses(_from_dict, **{field: None})
        assert (direct is None) == (spec is None), f"{field}=None: __init__={direct!r} from_dict={spec!r}"


class TestTheGuardPrecedesTheMembershipCheck:
    """A non-string ``default_robot`` names the wrong type, not a missing robot.

    Ordering rather than domain: the membership test below the guard compares
    ``default_robot`` against ``supported_robots``, so on ``default_robot=7``
    it would report ``7 not in ['panda']`` - the symptom rather than the
    mistake. Same reason the ``supported_robots`` shape check runs first.
    """

    def test_the_type_is_reported_rather_than_the_membership(self) -> None:
        text = _refuses(_make, supported_robots=["panda"], default_robot=7)
        assert text is not None
        assert "must be a string" in text, text
        assert "not in" not in text, text

    def test_a_genuine_membership_failure_still_reports_membership(self) -> None:
        """The check the guard runs ahead of is unchanged."""
        text = _refuses(_make, supported_robots=["panda"], default_robot="aloha")
        assert text is not None
        assert "not in" in text, text


class TestTheDomainIsTheWholeContribution:
    """``_spec_string_error`` decides only the two axes it documents.

    Parity between the helper and the two construction paths, so the rule cannot
    be widened at one call site without the helper agreeing.
    """

    @pytest.mark.parametrize("field", sorted(FIELDS))
    @pytest.mark.parametrize("value", NOT_A_STRING)
    def test_the_helper_and_the_constructor_agree(self, field: str, value: Any) -> None:
        helper = spec_mod._spec_string_error(value, field, "DeclarativeBenchmark", **FIELDS[field])
        direct = _refuses(_make, **{field: value})
        assert (helper is None) == (direct is None), (helper, direct)

    @pytest.mark.parametrize("field", sorted(FIELDS))
    @pytest.mark.parametrize("value", ["", None])
    def test_the_helper_and_the_constructor_agree_on_the_axes(self, field: str, value: Any) -> None:
        helper = spec_mod._spec_string_error(value, field, "DeclarativeBenchmark", **FIELDS[field])
        direct = _refuses(_make, **{field: value})
        assert (helper is None) == (direct is None), (field, value, helper, direct)

    def test_a_usable_string_is_accepted_by_the_helper(self) -> None:
        assert spec_mod._spec_string_error("probe", "name", "ctx") is None


# --- structural guard: no string field of either path may skip the helper ---

_STRING_PARAMS = frozenset(FIELDS)


def _routed_params(func_name: str) -> set[str]:
    """Names passed to ``_spec_string_error`` inside ``func_name``.

    Reads the shipped source rather than a copy, so a field added to either
    construction path without the domain is reported by name.
    """
    source = inspect.getsource(DeclarativeBenchmark)
    tree = ast.parse("class _S:\n" + "\n".join("    " + line for line in source.splitlines()))
    cls = tree.body[0]
    assert isinstance(cls, ast.ClassDef)
    inner = next(n for n in ast.walk(cls) if isinstance(n, ast.ClassDef) and n.name == "DeclarativeBenchmark")
    func = next(n for n in inner.body if isinstance(n, ast.FunctionDef) and n.name == func_name)
    routed: set[str] = set()
    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Name) and node.func.id == "_spec_string_error"):
            continue
        if node.args and isinstance(node.args[0], ast.Name):
            routed.add(node.args[0].id)
    return routed


class TestNoStringFieldSkipsTheDomain:
    """Both construction paths must route every string field through the helper."""

    @pytest.mark.parametrize("func_name", ["__init__", "from_dict"])
    def test_every_string_field_is_routed(self, func_name: str) -> None:
        routed = _routed_params(func_name)
        missing = _STRING_PARAMS - routed
        assert not missing, f"{func_name} does not route {sorted(missing)} through _spec_string_error"

    def test_the_scan_finds_the_known_fields(self) -> None:
        """Non-vacuity: a scan resolving nothing would satisfy the check above."""
        assert _routed_params("__init__") == _STRING_PARAMS
        assert _routed_params("from_dict") == _STRING_PARAMS

    def test_the_shipped_source_is_the_scanned_source(self) -> None:
        """The scan reads the module under test, not a copy of it."""
        module = inspect.getmodule(DeclarativeBenchmark)
        assert module is not None and module.__file__ is not None
        assert Path(module.__file__).name == "benchmark_spec.py"
