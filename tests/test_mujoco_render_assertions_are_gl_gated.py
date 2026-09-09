"""A MuJoCo read that needs an offscreen GL context must be gated on the shared probe.

``Simulation.render`` returns ``{"status": "error"}`` on a host with no usable
offscreen GL context - headless without EGL/OSMesa - for a reason that has
nothing to do with whatever contract the calling test is pinning. Asserting
``render(...)["status"] == "success"`` inline therefore conflates the property
under test with a host graphics capability: it passes only where a GL context
happens to exist, and elsewhere it reports a bare ``'error' != 'success'`` that
names neither GL nor the contract.

Indexing a camera out of an observation is the same dependency in a second
spelling, and it degrades worse. ``get_observation`` renders each camera and
``_render_cameras`` logs a per-camera failure at debug level and *omits the key
it could not produce*, so ``get_observation(...)["camera1"]`` raises
``KeyError: 'camera1'`` - which names neither GL nor the contract nor even the
fact that a render was attempted. Both spellings are in scope here.

:mod:`tests.simulation.mujoco._gl_probe` exists for exactly this and is the
convention in its sibling modules. This guard keeps it the convention: every
GL-dependent read in a module that requires ``mujoco`` must sit behind that
probe, so a test added later cannot re-open the confusion.

Scope is the mujoco requirement, not a directory. A module that reads a rendered
value *without* requiring ``mujoco`` renders through some other backend,
whose availability the MuJoCo probe does not describe - ``tests/simulation/newton``
gates its own render tests on a Newton-availability marker instead. Keying on the
requirement rather than on a path excludes those by construction, so this rule
needs no exemption list.

Three gating forms are accepted, all of which stop the read from running without
a context: the probe's marker on the test function, the same marker on its
class, or an in-function ``gl_available()`` skip. All three are in use.

For the observation spelling the discriminator is the *key*, not the call. Sixteen
modules index an observation and the key names a camera in two of them; the other
fourteen read a joint angle or a base pose, or compute the key at runtime, and no
graphics context is involved in producing any of those. Keying on the call would
report seven such reads across five modules as needing a gate they do not need,
which is the shape of a rule contributors learn to route around. So a subscript
counts only when its key is a camera name the module itself registers - or the
free view every world registers - and only when the call did not ask to skip
images. A key computed at runtime is out of scope by construction, and both
boundaries are pinned below rather than left to be rediscovered.

The rule's own non-vacuity pin is keyed on modules, never on a count of
assertions: splitting one render call into its own gated case is what the remedy
below asks for, and must not move a number a contributor then has to chase.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

#: Envelope-returning render entry points. ``get_frame`` is excluded: it returns
#: raw arrays rather than a status envelope, so it cannot carry this assertion.
RENDER_ATTRS = frozenset({"render", "render_depth", "render_all"})

#: The observation entry point that renders every camera as a side effect.
OBSERVATION_ATTR = "get_observation"

#: Keyword that tells ``get_observation`` to render nothing, so a read from such
#: a call carries no graphics dependency whatever key it names.
SKIP_IMAGES_KEYWORD = "skip_images"

#: The free view every MuJoCo world registers itself - ``create_world`` writes
#: ``cameras["default"]`` - so a module reading it declares no camera of its own.
IMPLICIT_CAMERA_NAMES = frozenset({"default"})

#: Modules that require ``mujoco`` and assert a render succeeded. Pinned so a
#: scan rooted somewhere unexpected fails loudly instead of reporting a clean
#: sweep over nothing.
EXPECTED_IN_SCOPE = frozenset(
    {
        "tests/simulation/mujoco/test_entity_name_lookup_type_safety.py",
        "tests/simulation/mujoco/test_remove_camera_refused_recompile.py",
        "tests/simulation/mujoco/test_reset_forwards_derived_state.py",
        "tests/simulation/test_camera_pixel_count_domain.py",
        "tests/simulation/test_unhashable_entity_name_is_reported.py",
    }
)

#: Asserts a render succeeded through another backend, so the MuJoCo probe does
#: not describe its host requirement. Pinned as the discriminator this rule
#: rests on rather than as an exemption.
OTHER_BACKEND = "tests/simulation/newton/test_domain_randomization.py"


def _tests_root() -> pathlib.Path:
    """This module's own directory - the tree the rule covers."""
    return pathlib.Path(__file__).parent


def requires_mujoco(tree: ast.AST) -> bool:
    """True when *tree* calls ``pytest.importorskip("mujoco")`` anywhere."""
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "importorskip":
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and arg.value == "mujoco":
                return True
    return False


def render_success_assertions(tree: ast.AST) -> list[int]:
    """Line numbers of every ``<x>.render*(...)["status"] == "success"`` compare."""
    lines = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Compare) and len(node.comparators) == 1):
            continue
        right = node.comparators[0]
        if not (isinstance(right, ast.Constant) and right.value == "success"):
            continue
        subscript = node.left
        if not isinstance(subscript, ast.Subscript):
            continue
        key = subscript.slice
        if not (isinstance(key, ast.Constant) and key.value == "status"):
            continue
        call = subscript.value
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr in RENDER_ATTRS:
            lines.append(node.lineno)
    return lines


def camera_names(tree: ast.AST) -> set[str]:
    """Camera names *tree* can read an image for: those it registers, plus the free view."""
    found = set(IMPLICIT_CAMERA_NAMES)
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "add_camera":
            continue
        positional = node.args[0] if node.args else None
        if isinstance(positional, ast.Constant) and isinstance(positional.value, str):
            found.add(positional.value)
        for keyword in node.keywords:
            value = keyword.value
            if keyword.arg == "name" and isinstance(value, ast.Constant) and isinstance(value.value, str):
                found.add(value.value)
    return found


def _skips_images(call: ast.Call) -> bool:
    """True when *call* passes ``skip_images=True``, so it rendered nothing."""
    for keyword in call.keywords:
        if keyword.arg != SKIP_IMAGES_KEYWORD:
            continue
        return isinstance(keyword.value, ast.Constant) and keyword.value.value is True
    return False


def camera_image_reads(tree: ast.AST) -> list[int]:
    """Line numbers of every ``<x>.get_observation(...)[<camera name>]`` subscript.

    The key carries the dependency, not the call: a subscript naming a joint is
    proprioception and needs no context, which is why this is not simply every
    ``get_observation`` index. A key the module cannot be shown to be a camera -
    computed, or a name this module never registers - is left out rather than
    guessed at.
    """
    cameras = camera_names(tree)
    lines = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        call = node.value
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)):
            continue
        if call.func.attr != OBSERVATION_ATTR or _skips_images(call):
            continue
        key = node.slice
        if isinstance(key, ast.Constant) and key.value in cameras:
            lines.append(node.lineno)
    return lines


def gl_dependent_reads(tree: ast.AST) -> list[int]:
    """Every line in *tree* whose value cannot be produced without a GL context."""
    return sorted(set(render_success_assertions(tree)) | set(camera_image_reads(tree)))


def probe_names(tree: ast.AST) -> set[str]:
    """Local names bound by importing from the shared GL probe module."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("_gl_probe"):
            names.update(alias.asname or alias.name for alias in node.names)
    return names


Decorated = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef


def _decorator_names(node: Decorated) -> set[str]:
    """Every decorator on *node*, as the bare name applied."""
    found: set[str] = set()
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(target, ast.Name):
            found.add(target.id)
        elif isinstance(target, ast.Attribute):
            found.add(target.attr)
    return found


def _enclosing(tree: ast.AST, lineno: int) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef | None, ast.ClassDef | None]:
    """The innermost function containing *lineno*, and its enclosing class."""
    function: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    enclosing_class: ast.ClassDef | None = None
    for node in ast.walk(tree):
        if not isinstance(node, Decorated):
            continue
        if not node.lineno <= lineno <= (node.end_lineno or node.lineno):
            continue
        if isinstance(node, ast.ClassDef):
            enclosing_class = node
        elif function is None or node.lineno > function.lineno:
            function = node
    return function, enclosing_class


def is_gated(tree: ast.AST, lineno: int, names: set[str]) -> bool:
    """True when the assertion at *lineno* cannot run without a GL context."""
    function, enclosing_class = _enclosing(tree, lineno)
    if function is None:
        return False
    if _decorator_names(function) & names:
        return True
    if enclosing_class is not None and _decorator_names(enclosing_class) & names:
        return True
    return any(isinstance(node, ast.Name) and node.id in names for node in ast.walk(function))


def survey(root: pathlib.Path) -> tuple[dict[str, list[int]], dict[str, list[int]], list[str]]:
    """Return (ungated, gated, out-of-scope) GL-dependent reads under *root*."""
    ungated: dict[str, list[int]] = {}
    gated: dict[str, list[int]] = {}
    other_backend: list[str] = []
    for path in sorted(root.rglob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        lines = gl_dependent_reads(tree)
        if not lines:
            continue
        label = path.relative_to(root.parent).as_posix()
        if not requires_mujoco(tree):
            other_backend.append(label)
            continue
        names = probe_names(tree)
        for lineno in lines:
            bucket = gated if is_gated(tree, lineno, names) else ungated
            bucket.setdefault(label, []).append(lineno)
    return ungated, gated, other_backend


def unaccounted_modules(gated: dict[str, list[int]], expected: frozenset[str]) -> tuple[set[str], set[str]]:
    """Return (in-scope modules with no gated assertion, gated modules not expected).

    Both sides are keyed on modules. A count of gated assertions is a different
    quantity from a count of modules: splitting one render assertion into its own
    case -- exactly what this guard's remedy text asks a contributor to do -- moves
    the first and not the second, so comparing them would fail a diff that did
    precisely the right thing, with no pin left to update.
    """
    return set(expected) - set(gated), set(gated) - set(expected)


class TestEveryMuJoCoGlDependentReadIsGated:
    def test_no_module_reads_a_rendered_value_without_the_probe(self) -> None:
        ungated, _, _ = survey(_tests_root())
        assert not ungated, (
            f"these modules read a value that needs an offscreen GL context without gating it "
            f"on the shared GL probe: {ungated}. On a headless host without EGL/OSMesa a render "
            f"reports an error and an observation omits the camera key, for a reason unrelated "
            f"to the contract under test, so the read fails there as 'error' != 'success' or as "
            f"a bare KeyError and names neither cause. Import requires_gl from "
            f"tests.simulation.mujoco._gl_probe and split the GL-dependent read into its own case."
        )

    def test_every_module_the_survey_covers_contributes_a_gated_assertion(self) -> None:
        """Non-vacuity: a scan that found nothing must not read as a clean sweep.

        Keyed on modules in both directions, which is also the module-set check
        this replaces: a module whose assertion stopped being gated leaves
        ``gated`` and is reported here by name, and a scan rooted somewhere
        unexpected reports every entry as missing.
        """
        _, gated, _ = survey(_tests_root())
        missing, unexpected = unaccounted_modules(gated, EXPECTED_IN_SCOPE)
        assert not missing and not unexpected, (
            f"the survey no longer accounts for EXPECTED_IN_SCOPE: {sorted(missing)} contribute no "
            f"gated GL-dependent read, and {sorted(unexpected)} are not listed. Add or drop the "
            f"module path. Adding a second gated assertion to a module already listed is not a "
            f"change to this pin."
        )


class TestTheScopeIsTheMujocoRequirement:
    def test_another_backends_render_assertion_is_out_of_scope(self) -> None:
        """The discriminator, pinned: no mujoco requirement, so a different probe."""
        _, _, other_backend = survey(_tests_root())
        assert OTHER_BACKEND in other_backend
        tree = ast.parse((_tests_root().parent / OTHER_BACKEND).read_text(encoding="utf-8"))
        assert render_success_assertions(tree)
        assert not requires_mujoco(tree)


_PLANTED_UNGATED = """
import pytest

mujoco = pytest.importorskip("mujoco")


def test_planted(sim):
    assert sim.render(camera_name="default")["status"] == "success"
"""

_PLANTED_GATED = """
import pytest

mujoco = pytest.importorskip("mujoco")

from tests.simulation.mujoco._gl_probe import requires_gl  # noqa: E402


@requires_gl
def test_planted(sim):
    assert sim.render(camera_name="default")["status"] == "success"
"""


_PLANTED_TWO_GATED = """
import pytest

mujoco = pytest.importorskip("mujoco")

from tests.simulation.mujoco._gl_probe import requires_gl  # noqa: E402


@requires_gl
def test_planted_one(sim):
    assert sim.render(camera_name="default")["status"] == "success"


@requires_gl
def test_planted_two(sim):
    assert sim.render_depth(camera_name="default")["status"] == "success"
"""


class TestTheSurveyDetectsWhatItClaimsTo:
    """A scanner that silently matched nothing would look like a clean tree."""

    def test_a_planted_ungated_assertion_is_reported(self, tmp_path: pathlib.Path) -> None:
        root = tmp_path / "tests"
        root.mkdir()
        (root / "test_planted.py").write_text(_PLANTED_UNGATED, encoding="utf-8")
        ungated, gated, _ = survey(root)
        assert list(ungated) == ["tests/test_planted.py"]
        assert not gated

    def test_the_same_assertion_behind_the_probe_is_accepted(self, tmp_path: pathlib.Path) -> None:
        root = tmp_path / "tests"
        root.mkdir()
        (root / "test_planted.py").write_text(_PLANTED_GATED, encoding="utf-8")
        ungated, gated, _ = survey(root)
        assert not ungated
        assert list(gated) == ["tests/test_planted.py"]

    @pytest.mark.parametrize("attr", sorted(RENDER_ATTRS))
    def test_each_render_entry_point_is_recognised(self, attr: str) -> None:
        tree = ast.parse(f'assert sim.{attr}(camera_name="c")["status"] == "success"\n')
        assert render_success_assertions(tree) == [1]

    def test_a_non_render_status_assertion_is_not_matched(self) -> None:
        """The rule is about rendering, not about every envelope in the suite."""
        tree = ast.parse('assert sim.step(n_steps=5)["status"] == "success"\n')
        assert render_success_assertions(tree) == []


class TestThePinIsKeyedOnModulesNotOnAnAssertionCount:
    """Splitting a render assertion into its own case must need no pin updated.

    That split is what this guard's remedy text instructs, so a pin that moved
    with the number of assertions would turn the required check red on a diff
    that complied with it -- and report only two bare integers.
    """

    def test_a_second_gated_assertion_in_a_listed_module_is_accounted_for(self, tmp_path: pathlib.Path) -> None:
        root = tmp_path / "tests"
        root.mkdir()
        (root / "test_planted.py").write_text(_PLANTED_TWO_GATED, encoding="utf-8")
        ungated, gated, _ = survey(root)
        assert not ungated
        assert unaccounted_modules(gated, frozenset({"tests/test_planted.py"})) == (set(), set())

    def test_the_module_and_assertion_counts_are_different_quantities(self, tmp_path: pathlib.Path) -> None:
        """One module, two gated assertions: why a count cannot stand in for a set."""
        root = tmp_path / "tests"
        root.mkdir()
        (root / "test_planted.py").write_text(_PLANTED_TWO_GATED, encoding="utf-8")
        _, gated, _ = survey(root)
        assert set(gated) == {"tests/test_planted.py"}
        assert sum(len(lines) for lines in gated.values()) == 2

    def test_a_listed_module_contributing_nothing_gated_is_reported(self, tmp_path: pathlib.Path) -> None:
        root = tmp_path / "tests"
        root.mkdir()
        (root / "test_planted.py").write_text(_PLANTED_UNGATED, encoding="utf-8")
        _, gated, _ = survey(root)
        missing, unexpected = unaccounted_modules(gated, frozenset({"tests/test_planted.py"}))
        assert missing == {"tests/test_planted.py"}
        assert not unexpected

    def test_a_module_that_is_not_listed_is_reported(self, tmp_path: pathlib.Path) -> None:
        root = tmp_path / "tests"
        root.mkdir()
        (root / "test_planted.py").write_text(_PLANTED_GATED, encoding="utf-8")
        _, gated, _ = survey(root)
        missing, unexpected = unaccounted_modules(gated, frozenset())
        assert not missing
        assert unexpected == {"tests/test_planted.py"}


_PLANTED_UNGATED_CAMERA_READ = """
import pytest

mujoco = pytest.importorskip("mujoco")


def test_planted(sim):
    sim.add_camera(name="cam1", position=[1, 0, 1])
    assert sim.get_observation(robot_name="arm")["cam1"].shape == (128, 128, 3)
"""

_PLANTED_GATED_CAMERA_READ = """
import pytest

mujoco = pytest.importorskip("mujoco")

from tests.simulation.mujoco._gl_probe import requires_gl  # noqa: E402


@requires_gl
def test_planted(sim):
    sim.add_camera(name="cam1", position=[1, 0, 1])
    assert sim.get_observation(robot_name="arm")["cam1"].shape == (128, 128, 3)
"""

_PLANTED_PROPRIOCEPTION_READ = """
import pytest

mujoco = pytest.importorskip("mujoco")


def test_planted(sim):
    sim.add_camera(name="cam1", position=[1, 0, 1])
    assert sim.get_observation(robot_name="arm")["yaw"] == 0.0
"""

_PLANTED_FREE_VIEW_READ = """
import pytest

mujoco = pytest.importorskip("mujoco")


def test_planted(sim):
    assert sim.get_observation(robot_name="arm")["default"].shape == (480, 640, 3)
"""

_PLANTED_SKIPPED_IMAGE_READ = """
import pytest

mujoco = pytest.importorskip("mujoco")


def test_planted(sim):
    sim.add_camera(name="cam1", position=[1, 0, 1])
    assert sim.get_observation(robot_name="arm", skip_images=True)["cam1"] is None
"""

_PLANTED_COMPUTED_KEY_READ = """
import pytest

mujoco = pytest.importorskip("mujoco")


def test_planted(sim, camera_name):
    sim.add_camera(name="cam1", position=[1, 0, 1])
    assert sim.get_observation(robot_name="arm")[camera_name].shape == (128, 128, 3)
"""


class TestACameraImageReadIsTheSameDependency:
    """``get_observation(...)[<camera>]`` needs a context exactly as ``render`` does.

    It fails worse, though: ``_render_cameras`` omits a key it could not produce,
    so the read raises ``KeyError`` rather than reporting a missing context.
    """

    def _survey_one(self, tmp_path: pathlib.Path, source: str) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
        root = tmp_path / "tests"
        root.mkdir()
        (root / "test_planted.py").write_text(source, encoding="utf-8")
        ungated, gated, _ = survey(root)
        return ungated, gated

    def test_a_planted_ungated_camera_image_read_is_reported(self, tmp_path: pathlib.Path) -> None:
        ungated, gated = self._survey_one(tmp_path, _PLANTED_UNGATED_CAMERA_READ)
        assert list(ungated) == ["tests/test_planted.py"]
        assert not gated

    def test_the_same_read_behind_the_probe_is_accepted(self, tmp_path: pathlib.Path) -> None:
        ungated, gated = self._survey_one(tmp_path, _PLANTED_GATED_CAMERA_READ)
        assert not ungated
        assert list(gated) == ["tests/test_planted.py"]

    def test_the_free_view_needs_no_declaration(self, tmp_path: pathlib.Path) -> None:
        """``create_world`` registers ``cameras["default"]``, so no ``add_camera`` precedes it."""
        ungated, _ = self._survey_one(tmp_path, _PLANTED_FREE_VIEW_READ)
        assert list(ungated) == ["tests/test_planted.py"]


class TestTheKeyIsTheDiscriminatorNotTheCall:
    """Why this is not simply every ``get_observation`` subscript.

    Sixteen modules index an observation and the key names a camera in two of
    them. Reporting every literal key instead - the widening this rule does not
    take - reports seven reads across five modules whose keys are joint angles and
    base poses, so the rule would refuse reads no graphics context is involved in
    producing. These cells pin the three clauses that keep it from doing so.
    """

    def _ungated(self, tmp_path: pathlib.Path, source: str) -> dict[str, list[int]]:
        root = tmp_path / "tests"
        root.mkdir()
        (root / "test_planted.py").write_text(source, encoding="utf-8")
        ungated, _, _ = survey(root)
        return ungated

    def test_a_proprioception_key_is_not_a_camera_read(self, tmp_path: pathlib.Path) -> None:
        """The module registers a camera and reads a joint: not a graphics dependency."""
        assert self._ungated(tmp_path, _PLANTED_PROPRIOCEPTION_READ) == {}

    def test_a_skipped_image_read_is_not_a_graphics_dependency(self, tmp_path: pathlib.Path) -> None:
        """``skip_images=True`` renders nothing, so no key it returns needed a context."""
        assert self._ungated(tmp_path, _PLANTED_SKIPPED_IMAGE_READ) == {}

    def test_a_computed_key_is_out_of_scope(self, tmp_path: pathlib.Path) -> None:
        """The boundary, pinned rather than left to be discovered.

        A key is a runtime string in general. This rule reads the ones it can
        prove are cameras and says so, instead of guessing at the rest.
        """
        assert self._ungated(tmp_path, _PLANTED_COMPUTED_KEY_READ) == {}

    def test_only_a_camera_this_module_registers_counts(self) -> None:
        tree = ast.parse(_PLANTED_UNGATED_CAMERA_READ)
        assert camera_names(tree) == {"cam1", "default"}
        assert camera_image_reads(tree) == [9]
        assert render_success_assertions(tree) == []
        assert gl_dependent_reads(tree) == [9]


class TestTheRuleCoversTheTreesCameraImageReads:
    """The real tree, not a planted one: both spellings reach the same rule."""

    def test_the_reset_render_readiness_read_is_in_scope_and_gated(self) -> None:
        """Deleting its ``@requires_gl`` left this guard green before the rule read the key."""
        path = _tests_root() / "simulation/mujoco/test_reset_forwards_derived_state.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assert camera_image_reads(tree), "the camera-image read this module is listed for is gone"
        assert not render_success_assertions(tree), "no render envelope here - the read is the whole dependency"
        _, gated, _ = survey(_tests_root())
        assert "tests/simulation/mujoco/test_reset_forwards_derived_state.py" in gated

    def test_a_proprioception_read_in_the_tree_is_not_in_scope(self) -> None:
        """The measured discriminator, on a real module that reads ``base_pos`` ungated."""
        path = _tests_root() / "simulation/test_base_pose_is_the_robots_own_free_joint.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assert camera_image_reads(tree) == []
        assert gl_dependent_reads(tree) == []
