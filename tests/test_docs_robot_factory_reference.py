"""The factory reference page must not state a contract the factory does not implement.

``docs/reference/api/robot.md`` is the reference a caller reads before writing
``Robot(...)``. It renders :func:`strands_robots.robot.Robot` with mkdocstrings,
so the signature and the parameter list the reader sees are the docstring's
``Args:`` section at build time. That moves the drift from a hand-written table
to the docstring: a parameter with no entry renders as one that does not exist,
an entry for a renamed parameter renders as one that does, and a ``(default)``
spelled in prose can name a value the caller never gets.

A wrong default is worse than a missing one. It names a state the caller is
never in, and it hides the knob that would reach the state the page describes.
A documented refusal that does not happen fails the same way in reverse: the
caller believes a typo is caught at construction and ships the typo.

These tests grade the page against :func:`strands_robots.robot.Robot` itself,
its signature, its docstring and its observed behaviour, rather than against a
hand-copied expectation, so changing a default cannot silently invalidate them.
``docs/learn/mesh/index.md`` states the mesh opt-in and
``docs/learn/hardware/drivers.md`` states the ``driver="strands"`` refusal; both
claims are graded here by reaching the state they describe.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import re
from pathlib import Path

import pytest

from strands_robots.robot import Robot, _mesh_env_opt_in

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DOC = _REPO_ROOT / "docs" / "reference" / "api" / "robot.md"
#: The page whose fence turns mesh on and whose table spells ``STRANDS_MESH``.
_MESH_DOC = _REPO_ROOT / "docs" / "learn" / "mesh" / "index.md"
#: The ``driver="strands"`` contract, including the refusal-by-name claim the
#: last class here grades. The factory docstring states the choice; this page
#: states the contract and names the robots that take the native path.
_NATIVE_DRIVERS_DOC = _REPO_ROOT / "docs" / "learn" / "hardware" / "drivers.md"
_DRIVERS_HOOK = _REPO_ROOT / "docs" / "hooks" / "drivers.py"

_VARIADIC = (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
_DIRECTIVE = "::: strands_robots.robot.Robot"

# The guard is only meaningful while it still reaches the Args section. If a
# docstring rewrite drops the entries, fail loudly instead of reporting clean.
_MINIMUM_GRADED_ENTRIES = 8

_PROBE_MJCF = """<mujoco model="probe">
  <worldbody>
    <light pos="0 0 3"/>
    <geom type="plane" size="1 1 0.1"/>
    <body name="link0" pos="0 0 0.1">
      <joint name="joint0" type="hinge" axis="0 0 1"/>
      <geom type="capsule" size="0.02" fromto="0 0 0  0 0 0.2"/>
    </body>
  </worldbody>
  <actuator><motor joint="joint0" ctrlrange="-1 1"/></actuator>
</mujoco>"""


def _args_entries() -> dict[str, str]:
    """Return the docstring's ``Args:`` entries, the text mkdocstrings renders as rows.

    Returns:
        ``{parameter_name: entry_text}`` for every ``name: ...`` entry in the
        ``Args:`` section (``**kwargs`` keeps its name, asterisks stripped).
    """
    doc = inspect.getdoc(Robot) or ""
    match = re.search(r"^Args:\n(.*?)(?=^\S)", doc, re.M | re.S)
    if match is None:
        return {}
    entries: dict[str, str] = {}
    current: str | None = None
    for line in match.group(1).splitlines():
        head = re.match(r"^ {0,4}(\*{0,2}[A-Za-z_][A-Za-z0-9_]*):(?=\s|$)", line)
        if head is not None:
            current = head.group(1).lstrip("*")
            entries[current] = line[head.end() :].strip()
        elif current is not None:
            entries[current] += "\n" + line.strip()
    return entries


def _stated_defaults() -> list[tuple[str, str]]:
    """Return ``(name, literal)`` for every entry that marks a value as the default.

    Returns:
        The literal immediately before a ``(default`` marker, for entries whose
        prose spells one (``"sim" (default - safe)``, ````None`` (default)``).
        Entries that only describe a derived default in words are not returned.
    """
    literal = r"(?:``)?(\"[^\"`]*\"|'[^'`]*'|None|True|False|-?\d+(?:\.\d+)?)(?:``)?"
    out: list[tuple[str, str]] = []
    for name, text in _args_entries().items():
        for match in re.finditer(literal + r"\s*\([^()]*?\bdefault\b", text):
            out.append((name, match.group(1)))
    return out


def _graded() -> list[tuple[str, inspect.Parameter]]:
    """Return the entries that name a real, non-variadic parameter."""
    out = []
    for name in _args_entries():
        param = inspect.signature(Robot).parameters.get(name)
        if param is not None and param.kind not in _VARIADIC:
            out.append((name, param))
    return out


def _mesh_env_row() -> str:
    """Return the table row of the mesh page whose first cell is ``STRANDS_MESH``."""
    for line in _MESH_DOC.read_text(encoding="utf-8").splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and cells[0] == "`STRANDS_MESH`":
            return line
    return ""


class TestThePageRendersTheSignatureFromTheSource:
    """Premises. A clean result must mean the page was read, not skipped."""

    def test_the_page_ships(self) -> None:
        assert _DOC.is_file(), f"premise: {_DOC.relative_to(_REPO_ROOT)} is the page under test"

    def test_the_page_renders_the_factory_with_mkdocstrings(self) -> None:
        text = _DOC.read_text(encoding="utf-8")
        assert _DIRECTIVE in text, (
            f"premise: the page renders `{_DIRECTIVE}`; the signature the reader sees is the "
            "source's, and the guards below grade that source"
        )

    def test_the_page_says_it_is_the_factory(self) -> None:
        text = _DOC.read_text(encoding="utf-8")
        assert "factory" in text and 'mode="real"' in text, (
            "the page must say Robot(...) is a factory and name the mode that reaches hardware"
        )

    def test_enough_entries_are_graded(self) -> None:
        graded = _graded()
        assert len(graded) >= _MINIMUM_GRADED_ENTRIES, (
            f"premise: only {len(graded)} Args entr(ies) resolved to a Robot() parameter, below "
            f"the {_MINIMUM_GRADED_ENTRIES} this guard expects. A clean run would prove nothing."
        )


class TestEveryDocumentedDefaultIsTheRealDefault:
    """A stated default is the value the caller gets by omitting the parameter."""

    def test_some_entry_states_a_default(self) -> None:
        assert _stated_defaults(), "premise: the Args section marks at least one value as the default"

    def test_no_entry_states_a_default_the_signature_contradicts(self) -> None:
        wrong: list[str] = []
        params = inspect.signature(Robot).parameters
        for name, shown in _stated_defaults():
            param = params.get(name)
            if param is None or param.kind in _VARIADIC:
                continue
            if param.default is inspect.Parameter.empty:
                wrong.append(f"`{name}` documents {shown} as the default but has no default")
                continue
            try:
                parsed = ast.literal_eval(shown)
            except (SyntaxError, ValueError):
                wrong.append(f"`{name}` documents {shown}, which is not a literal value")
                continue
            if parsed != param.default or type(parsed) is not type(param.default):
                wrong.append(f"`{name}` documents {shown} but omitting it yields {param.default!r}")
        assert not wrong, (
            "The rendered parameter list names a default the caller never gets, so the entry "
            "describes a state a bare Robot() is not in:\n  " + "\n  ".join(wrong)
        )

    def test_every_parameter_has_an_entry(self) -> None:
        documented = set(_args_entries())
        missing = [
            name
            for name, param in inspect.signature(Robot).parameters.items()
            if param.kind not in _VARIADIC and name not in documented
        ]
        assert not missing, f"Robot() accepts {missing}, which the rendered parameter list never shows"

    def test_every_entry_names_a_parameter(self) -> None:
        real = set(inspect.signature(Robot).parameters)
        stale = [name for name in _args_entries() if name not in real]
        assert not stale, f"the Args section documents {stale}, which Robot() does not accept"

    def test_the_variadic_parameter_has_an_entry(self) -> None:
        entries = _args_entries()
        variadic = [name for name, p in inspect.signature(Robot).parameters.items() if p.kind in _VARIADIC]
        assert variadic, "premise: Robot() forwards **kwargs"
        for name in variadic:
            assert name in entries, f"Robot() forwards `**{name}` but the Args section never says where"


class TestADocumentedRefusalReallyRefuses:
    """A claim that promises an exception is graded by raising it, not by wording."""

    def test_an_unknown_native_driver_kwarg_is_refused(self, tmp_path: Path) -> None:
        text = _NATIVE_DRIVERS_DOC.read_text(encoding="utf-8")
        assert "does not declare is refused" in text, (
            "premise: the drivers page promises that a keyword the driver does not declare is refused"
        )
        pytest.importorskip("mujoco")
        # The refusal is the factory's, raised before a driver is constructed, so
        # it reaches no hardware. It must name the offending keyword.
        with pytest.raises(ValueError, match="definitely_not_a_forwardable_kwarg"):
            Robot("so101", mode="real", driver="strands", definitely_not_a_forwardable_kwarg=1)
        # The docstring scopes ``**kwargs`` to the backend constructor, so a sim
        # keyword the backend does not recognize is forwarded, not refused here.
        mjcf = tmp_path / "probe.xml"
        mjcf.write_text(_PROBE_MJCF, encoding="utf-8")
        sim = None
        try:
            sim = Robot("so100", mode="sim", urdf_path=str(mjcf), definitely_not_a_forwardable_kwarg=1)
            assert sim is not None
        finally:
            if sim is not None:
                sim.destroy()

    def test_the_raises_section_names_the_native_driver_refusal(self) -> None:
        doc = inspect.getdoc(Robot) or ""
        raises = re.search(r"^Raises:\n(.*?)(?=^\S)", doc, re.M | re.S)
        assert raises is not None, "premise: the docstring has a Raises section the page renders"
        assert "ValueError" in raises.group(1) and 'driver="strands"' in raises.group(1), (
            "the rendered Raises section must promise the ValueError for a driver='strands' robot with no native driver"
        )


class TestTheMeshPageNamesTheSpellingThatEnablesMesh:
    """The mesh page must reach the state its fence prints."""

    def test_the_fence_opts_in_by_argument(self) -> None:
        text = _MESH_DOC.read_text(encoding="utf-8")
        assert re.search(r"Robot\([^)]*mesh\s*=\s*True", text) is not None, (
            "The mesh page reads .mesh attributes but never passes mesh=True in a Robot(...) call. "
            "Copied as written it raises AttributeError on None."
        )

    def test_the_env_row_spellings_do_what_the_row_says(self, monkeypatch: pytest.MonkeyPatch) -> None:
        row = _mesh_env_row()
        assert row, f"premise: {_MESH_DOC.relative_to(_REPO_ROOT)} has a table row for `STRANDS_MESH`"
        spellings = [s for s in re.findall(r"`([A-Za-z01]+)`", row) if s != "STRANDS_MESH"]
        assert spellings, f"the STRANDS_MESH row names no value: {row}"
        enabling: list[str] = []
        for raw in spellings:
            monkeypatch.setenv("STRANDS_MESH", raw)
            if _mesh_env_opt_in():
                enabling.append(raw)
        assert enabling, f"none of the STRANDS_MESH spellings the row shows ({spellings}) opts in"
        # The row calls ``false`` a hard kill switch, so it must not opt in.
        if "false" in spellings:
            monkeypatch.setenv("STRANDS_MESH", "false")
            assert not _mesh_env_opt_in(), "the row calls STRANDS_MESH=false a kill switch, but it opts in"

    def test_a_bare_robot_leaves_mesh_off(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """The reality the page has to describe: the docstring says None keeps mesh off."""
        pytest.importorskip("mujoco")
        monkeypatch.delenv("STRANDS_MESH", raising=False)
        mjcf = tmp_path / "probe.xml"
        mjcf.write_text(_PROBE_MJCF, encoding="utf-8")
        sim = Robot("so100", mode="sim", urdf_path=str(mjcf))
        try:
            assert sim.mesh is None
        finally:
            sim.destroy()


class TestTheNativeDriverRefusalClaimIsStillTrue:
    """The ``driver="strands"`` page names robots by path, and the names are graded.

    ``docs/learn/hardware/drivers.md`` says a ``driver="strands"`` request for a
    robot with no native driver is refused by name, never served the lerobot
    path quietly, and it lists the robots whose registry entry makes the native
    driver the default. Those names can rot without a word changing: a robot
    named as native loses its driver, or a robot named as lerobot-only gains
    one. So the names are graded, not the wording, and the refusal is graded by
    raising it.

    Correctness is graded, deliberately not completeness. The shipped table is
    generated from the source by ``docs/hooks/drivers.py``, and the page names
    ``list_native_drivers()`` as the live answer, so the prose list only has to
    be right about every robot it names.
    """

    @staticmethod
    def _registry_default_native_names() -> list[str]:
        """Return the robots the page says declare ``hardware.driver = "strands"``."""
        text = _NATIVE_DRIVERS_DOC.read_text(encoding="utf-8")
        match = re.search(r"Robots lerobot has no type for \((.*?)\) declare", text, re.S)
        assert match is not None, "premise: the page lists the robots whose registry entry picks the native driver"
        return re.findall(r"`([A-Za-z0-9_]+)`", match.group(1))

    def test_every_robot_the_page_calls_natively_driven_really_is(self) -> None:
        """A name in the list must have a driver, or it sends a reader to a dead end."""
        from strands_robots.drivers import get_native_driver_class

        names = self._registry_default_native_names()
        assert names, "no robot name was read out of the page; the guard would prove nothing"
        wrong = [name for name in names if get_native_driver_class(name) is None]
        assert not wrong, f"The page lists {wrong} among natively driven robots, but none is registered for them."

    def test_every_robot_the_page_says_defaults_to_native_really_does(self) -> None:
        """The page promises a bare ``Robot(name, mode="real")`` builds the native driver."""
        from strands_robots.drivers import resolve_driver

        names = self._registry_default_native_names()
        not_default = [name for name in names if resolve_driver(name, None) != "strands"]
        assert not not_default, (
            f"The page says {not_default} declare hardware.driver = 'strands', so a bare "
            "Robot(name, mode='real') builds the native driver; the registry declares nothing "
            f"for them and resolve_driver() routes them to lerobot."
        )

    def test_the_generated_table_names_only_robots_with_a_driver(self) -> None:
        """The hook reads source text; the classes it names must be the registered ones."""
        from strands_robots.drivers import get_native_driver_class

        spec = importlib.util.spec_from_file_location("drivers_hook", _DRIVERS_HOOK)
        assert spec is not None and spec.loader is not None
        hook = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hook)
        rows = hook.rows()
        assert rows, "the drivers table is empty; the guard would prove nothing"
        wrong = []
        for robot, cls, *_ in rows:
            registered = get_native_driver_class(robot)
            if registered is None or registered.__name__ != cls:
                wrong.append(f"{robot}: table says {cls}, registry has {getattr(registered, '__name__', None)}")
        assert not wrong, "the shipped table disagrees with the registry:\n  " + "\n  ".join(wrong)

    def test_the_refusal_names_the_robot_and_does_not_fall_back(self) -> None:
        """Graded by raising it, so a reworded refusal cannot leave the page stale."""
        from strands_robots.drivers import get_native_driver_class
        from strands_robots.registry import list_robots

        text = _NATIVE_DRIVERS_DOC.read_text(encoding="utf-8")
        assert "refused by name" in text, "premise: the page promises a refusal by name"
        candidates = [row["name"] for row in list_robots() if get_native_driver_class(row["name"]) is None]
        assert candidates, "every registered robot has a native driver; the claim has nothing to refuse"
        name = candidates[0]
        with pytest.raises(ValueError) as excinfo:
            Robot(name, mode="real", driver="strands")
        raised = str(excinfo.value)
        assert repr(name) in raised, f"the refusal does not name the robot: {raised}"
        assert "driver='strands'" in raised, f"the refusal does not name the choice that failed: {raised}"
        assert "register_native_driver" in raised, f"the refusal does not say how to add a driver: {raised}"

    def test_the_page_names_the_live_listing_helper(self) -> None:
        """A prose list is only honest if the page says where the live one is."""
        assert "list_native_drivers()" in _NATIVE_DRIVERS_DOC.read_text(encoding="utf-8"), (
            "The page's list of natively driven robots is a capture, so the page must name "
            "list_native_drivers() as the way to get the current one."
        )
