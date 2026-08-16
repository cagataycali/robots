"""``DeclarativeBenchmark`` holds ``supported_robots`` to the shared name-list domain.

``DeclarativeBenchmark`` has two construction paths and they applied different
checks to the same value. ``from_dict`` refused a ``supported_robots`` that was
not a list of strings; ``__init__`` stored ``list(supported_robots)`` raw. Only
``max_steps`` was mirrored across the two, and its comment states the reason for
all of them - a directly constructed benchmark must not carry a value the
evaluation loop has to refuse later.

Measured on ``24766c3`` (mujoco 3.11.0, ``MUJOCO_GL=egl``), constructing directly
with ``supported_robots="panda"`` - one name, spelled without the list:

* ``supported_robots`` read back as ``['p', 'a', 'n', 'd', 'a']``, five names the
  caller never wrote, because ``str`` is iterable per character.
* ``sim.list_benchmarks()`` - the discovery surface - advertised
  ``robots=['p', 'a', 'n', 'd', 'a']``.
* ``sim.evaluate_benchmark(benchmark_name=..., robot_name="panda")`` returned
  ``status="error"`` with ``robot 'panda' has data_config='panda', but benchmark
  DeclarativeBenchmark supports ['p', 'a', 'n', 'd', 'a']`` - the evaluation
  refusing the benchmark's *own* ``default_robot`` and naming the five
  one-letter robots as the allowed set.
* ``supported_robots=""`` was worse in the other direction: it stored ``[]``,
  which is this parameter's documented "any robot" spelling, so a mistyped name
  silently widened the benchmark to every robot instead of restricting it.

The same values through ``from_dict`` were all refused. These tests pin the two
paths to one rule.
"""

from __future__ import annotations

from typing import Any

import pytest

from strands_robots.simulation.benchmark_spec import DeclarativeBenchmark
from strands_robots.utils import name_list_error

# Values ``supported_robots`` cannot be honored as. Each is refused by the shared
# domain for a stated reason; the ids name the mistake rather than the value.
UNUSABLE: list[Any] = [
    pytest.param("panda", id="bare-string"),
    pytest.param("", id="empty-string"),
    pytest.param({"panda": 1}, id="mapping"),
    pytest.param(["panda", "panda"], id="repeated-name"),
    pytest.param(["panda", 7], id="non-string-entry"),
    pytest.param(["panda", ""], id="blank-entry"),
    pytest.param([None], id="none-entry"),
    pytest.param(7, id="int"),
    pytest.param(3.5, id="float"),
    pytest.param(None, id="none"),
]

# Spellings that name a robot set the evaluation loop can honor. ``[]`` is the
# documented "any robot" case and must stay accepted - it is why the shape check
# is not gated on a truthy value the way the ``image_keys`` callers gate theirs.
USABLE: list[Any] = [
    pytest.param(["panda"], id="one-name"),
    pytest.param(["panda", "aloha"], id="two-names"),
    pytest.param(("panda",), id="tuple"),
    pytest.param([], id="empty-any-robot"),
]


def _make(**overrides: Any) -> DeclarativeBenchmark:
    """Construct a benchmark directly, with ``overrides`` splatted in.

    Splatted rather than passed positionally so a deliberately off-type value
    reaches the runtime guard as an agent or a caller would supply it, instead
    of being reported by the type checker at each call site.
    """
    kwargs: dict[str, Any] = {
        "name": "probe",
        "supported_robots": ["panda"],
        "default_robot": "panda",
        "max_steps": 10,
        "success_fn": lambda _sim: False,
        "failure_fn": lambda _sim: False,
        "reward_terms": [],
    }
    kwargs.update(overrides)
    return DeclarativeBenchmark(**kwargs)


def _from_spec(**overrides: Any) -> DeclarativeBenchmark:
    """Compile the equivalent spec dict, so both paths see the same value."""
    spec: dict[str, Any] = {
        "name": "probe",
        "default_robot": "panda",
        "max_steps": 10,
        "supported_robots": ["panda"],
    }
    spec.update(overrides)
    return DeclarativeBenchmark.from_dict(spec)


def _refuses(build: Any, **overrides: Any) -> str | None:
    """Return the refusal text, or ``None`` when the value was accepted."""
    try:
        build(**overrides)
    except ValueError as exc:
        return str(exc)
    return None


class TestAnUnusableRobotSetIsRefusedAtConstruction:
    """Direct construction refuses what it cannot honor, naming the parameter."""

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_the_value_is_refused(self, value: Any) -> None:
        text = _refuses(_make, supported_robots=value)
        assert text is not None, f"{value!r} was accepted as supported_robots"
        assert "supported_robots" in text, text
        assert "DeclarativeBenchmark" in text, text

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_the_refusal_is_the_shared_domains_own_answer(self, value: Any) -> None:
        """Byte-identical to the shared rule, so no second wording can drift in."""
        assert _refuses(_make, supported_robots=value) == name_list_error(
            value, "supported_robots", "DeclarativeBenchmark"
        )


class TestAUsableRobotSetIsUnchanged:
    """Every spelling the evaluation loop can honor still constructs."""

    @pytest.mark.parametrize("value", USABLE)
    def test_the_value_is_accepted(self, value: Any) -> None:
        bench = _make(supported_robots=value, default_robot=(list(value) or ["panda"])[0])
        assert bench.supported_robots == list(value)

    def test_an_empty_list_still_means_any_robot(self) -> None:
        """The documented any-robot case, with a default outside the (empty) set."""
        bench = _make(supported_robots=[], default_robot="anything")
        assert bench.supported_robots == []
        assert bench.default_robot == "anything"

    def test_a_tuple_is_normalized_to_a_list(self) -> None:
        """``list()`` is load-bearing: the domain accepts a tuple, the property
        is a ``list[str]``, and callers mutate the copy they are handed."""
        bench = _make(supported_robots=("panda", "aloha"))
        assert bench.supported_robots == ["panda", "aloha"]
        assert isinstance(bench.supported_robots, list)


class TestTheTwoConstructionPathsAgree:
    """One rule, so a benchmark is not refused from a spec file and accepted
    when constructed directly - the property ``max_steps`` already had."""

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_neither_path_accepts_what_the_other_refuses(self, value: Any) -> None:
        direct = _refuses(_make, supported_robots=value)
        spec = _refuses(_from_spec, supported_robots=value)
        assert (direct is None) == (spec is None), (
            f"{value!r}: direct={direct!r} spec={spec!r} - the two paths disagree"
        )
        assert direct is not None and spec is not None

    @pytest.mark.parametrize("value", USABLE)
    def test_both_paths_accept_a_usable_set(self, value: Any) -> None:
        default = (list(value) or ["panda"])[0]
        assert _refuses(_make, supported_robots=value, default_robot=default) is None
        # A spec file cannot express a tuple, so compare on the list form.
        assert _refuses(_from_spec, supported_robots=list(value), default_robot=default) is None

    def test_the_spec_path_names_the_spec_key(self) -> None:
        """Both paths share the rule but keep their own context, so the message
        says where the value came from - as the ``max_steps`` pair does."""
        spec = _refuses(_from_spec, supported_robots="panda")
        assert spec == name_list_error("panda", "supported_robots", "spec")
        assert spec is not None and spec.startswith("spec: ")


class TestTheShapeIsCheckedBeforeMembership:
    """Order matters: on a bare string the membership check would describe the
    symptom (``'panda' not in ['p', 'a', 'n', 'd', 'a']``) rather than the
    mistake, so the shape refusal has to come first."""

    def test_a_bare_string_reports_the_bare_string(self) -> None:
        text = _refuses(_make, supported_robots="panda")
        assert text is not None
        assert "not a single string" in text, text
        assert "not in supported_robots" not in text, text

    def test_the_character_reading_is_quoted_so_the_mistake_is_visible(self) -> None:
        text = _refuses(_make, supported_robots="panda")
        assert text is not None
        assert "['p', 'a', 'n', 'd', 'a']" in text, text


class TestABenchmarkCannotExcludeItsOwnDefaultRobot:
    """The invariant the bare string broke: a declared robot set that the
    benchmark's own ``default_robot`` is outside of is refused, as
    ``from_dict`` already refused it."""

    def test_a_default_outside_a_non_empty_set_is_refused(self) -> None:
        text = _refuses(_make, supported_robots=["panda"], default_robot="aloha")
        assert text is not None
        assert "not in supported_robots" in text, text
        assert "'aloha'" in text, text

    def test_both_paths_refuse_it(self) -> None:
        assert _refuses(_make, supported_robots=["panda"], default_robot="aloha") is not None
        assert _refuses(_from_spec, supported_robots=["panda"], default_robot="aloha") is not None

    def test_a_default_inside_the_set_is_accepted(self) -> None:
        bench = _make(supported_robots=["panda", "aloha"], default_robot="aloha")
        assert bench.default_robot == "aloha"


class TestNeighbouringFieldsAreMirroredToo:
    """``name`` / ``default_robot`` / ``scene`` / ``instruction`` are mirrored now.

    This class previously pinned the opposite - that a non-string in any of the
    four was still accepted by ``__init__`` - and scoped the boundary of the
    ``supported_robots`` change. Its stated reason was that each "fails loudly
    at its consumer rather than silently widening the robot set", and that turned
    out to hold for one value of one field: a truthy non-string ``scene`` fails
    in ``load_scene``. ``instruction=42`` was handed to the policy verbatim as
    its task command and ``scene=[]`` was skipped by a truthiness test, both
    under ``status="success"``.

    Kept as the pointer to where those four now live, so the boundary this file
    draws stays measured: ``supported_robots`` is the name-list domain, and the
    string fields are the module's own string domain, pinned in
    ``test_benchmark_string_field_domains``.
    """

    @pytest.mark.parametrize("field", ["name", "default_robot", "scene", "instruction"])
    def test_a_non_string_is_refused_directly(self, field: str) -> None:
        assert _refuses(_make, **{field: 42}) is not None

    def test_max_steps_keeps_its_own_mirrored_domain(self) -> None:
        """The one mirror that predated both changes, unchanged by either."""
        text = _refuses(_make, max_steps=2.7)
        assert text is not None
        assert "max_steps" in text, text

    def test_supported_robots_keeps_the_name_list_domain(self) -> None:
        """The mirror this file is about, unchanged by the string domain."""
        text = _refuses(_make, supported_robots="panda")
        assert text is not None
        assert "supported_robots" in text, text
