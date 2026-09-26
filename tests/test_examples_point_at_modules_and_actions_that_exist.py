"""A shipped example points only at modules and actions that are there.

An example's prose is run, not read: a reader copies the ``python -m ...`` line
out of its README, and a model is handed the system prompt the example builds
for it. Both name things by hand, and nothing graded either name against the
tree, so a deletion elsewhere leaves the prose selling what is gone. Two shapes,
both measured on ``edd8ae092``, and a third on ``982013eeb``:

1. ``examples/mujoco_gs/README.md`` ran ``python -m
   examples.mujoco_gs.app_groot_libero`` and ``python -m
   examples.mujoco_gs.libero_groot`` on three lines, under a section promising
   "Pick a task, press **Run GR00T policy**" and a measured ``success_rate=1.00``
   recipe. Both modules left with the vendored LIBERO adapter (#3056), so every
   one of those commands is ``No module named``. The sibling guard in
   ``test_docs_module_commands_are_dispatched`` grades the *subcommand* of
   ``python -m strands_robots <command>`` and never reads the module path, so a
   page naming a module that does not exist passed it.

2. The scene block ``examples/mujoco_gs/scene.py`` writes into that demo's agent
   system prompt closed with "then call ``hybrid_render``" - a name the
   ``Simulation`` tool does not publish, while the same prompt's own driving
   section says ``render(camera_name="front")``. A model that followed the scene
   block spent its turn on ``Unknown action: hybrid_render``, and the demo's
   README says out loud that the compositing is a display layer and "not an
   agent tool".

3. ``examples/registry/lerobot_hardware_catalog.py --g1`` closed by printing
   ``Robot('g1', mode='real', robot_ip='192.168.123.164',
   controller='GrootLocomotionController')``. Neither keyword exists: the
   factory grades a native driver's kwargs against its own signature and
   refuses with ``ValueError: Unknown kwarg(s) for 'unitree_g1' on
   driver='strands': ['controller', 'robot_ip']``, and
   ``GrootLocomotionController`` was a name nothing else in the tree carried.
   The working spelling is ``port=``, which is what the G1 bring-up page has
   always documented.

Every rule is one-directional: prose need not name every module, every action or
every keyword, only nothing the tree lacks. Each carries a floor over what it
read, so a reflow that hides the commands (or a rename of the prompt builders)
reports a shrunken sweep instead of a clean one.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

from strands_robots.drivers import constructor_keywords, get_native_driver_class, resolve_driver
from strands_robots.simulation.mujoco.simulation import _PUBLISHED_ACTIONS

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EXAMPLES_DIR = _REPO_ROOT / "examples"

#: Directories in this repository whose modules the tree itself must supply.
_LOCAL_ROOTS = frozenset({"examples", "strands_robots", "scripts", "tests", "tools"})

_FENCE = re.compile(r"^\s*(?:```|~~~)")
_INLINE = re.compile(r"`([^`\n]+)`")
#: Shell noise a documented command line may carry before the interpreter.
_PREFIX = re.compile(r"^(?:\$\s*|sudo\s+|uv\s+run\s+|[A-Z_][A-Z0-9_]*=(?:\"[^\"]*\"|'[^']*'|\S*)\s+)+")
_MODULE_RUN = re.compile(r"^python3?\s+-m\s+([A-Za-z_][A-Za-z0-9_.]*)")

#: A documented call inside prompt text: ``` `render(` ```.
_PROMPT_CALL = re.compile(r"`([a-z_][a-z0-9_]*)\(")
#: The other way a prompt names one: ``call `hybrid_render` ``.
_PROMPT_CALLED = re.compile(r"\bcall\s+`([a-z_][a-z0-9_]*)`")
#: The functions and constants that build the text a model is handed.
_PROMPT_BUILDER = re.compile(r"prompt|scene_description", re.IGNORECASE)

#: A ``Robot(...)`` call an example spells inside text it prints or documents,
#: up to the first closing paren - a recipe a reader copies rather than runs.
_RECIPE = re.compile(r"Robot\(\s*['\"](?P<robot>[a-z0-9_]+)['\"](?P<tail>[^)]*)\)", re.S)
#: A keyword named in that call. ``(?<![\w.])`` keeps ``==`` comparisons and
#: attribute defaults out.
_RECIPE_KEYWORD = re.compile(r"(?<![\w.])(?P<keyword>[a-z_][a-z0-9_]*)\s*=")
#: Keywords :class:`~strands_robots.robot.Robot` itself reads, so no driver has
#: to declare them.
_FACTORY_KEYWORDS = frozenset({"mode", "driver"})

# Floors: the population each sweep must still reach, well under what the tree
# carries, so a reflow or a rename is reported rather than read as a clean pass.
_MINIMUM_RUNS = 20
_MINIMUM_LOCAL_RUNS = 6
_MINIMUM_PAGES = 3
_MINIMUM_PROMPT_ACTIONS = 5
_MINIMUM_PROMPT_FILES = 2
_MINIMUM_RECIPES = 4
_MINIMUM_GRADED_RECIPES = 2


@dataclass(frozen=True)
class _ModuleRun:
    """A ``python -m <module>`` command a page tells the reader to run."""

    page: str
    module: str
    line: str

    @property
    def is_local(self) -> bool:
        """Whether this repository is the thing that has to supply the module."""
        return self.module.split(".")[0] in _LOCAL_ROOTS

    def resolves(self, root: Path) -> bool:
        """Whether the module is a file or an importable package in ``root``."""
        target = root.joinpath(*self.module.split("."))
        return target.with_suffix(".py").is_file() or (target / "__init__.py").is_file()

    def __str__(self) -> str:
        return f"{self.page}: {self.line.strip()} -> {self.module}"


@dataclass(frozen=True)
class _PromptAction:
    """An action a prompt an example hands to a model tells it to call."""

    source: str
    action: str

    def __str__(self) -> str:
        return f"{self.source}: {self.action!r}"


@dataclass(frozen=True)
class _RealModeRecipe:
    """A ``Robot(name, mode="real", ...)`` call an example's text spells out."""

    source: str
    robot: str
    keywords: frozenset[str]
    names_native_driver: bool

    @property
    def driver_class(self) -> type | None:
        """The native driver this recipe builds, or ``None`` when lerobot does.

        A recipe is graded only when a native driver is what the factory reaches
        for it: the lerobot driver's keyword roster is its config dataclass,
        which needs lerobot installed, and this repository's test environment is
        not where that dependency is decided.
        """
        if not (self.names_native_driver or resolve_driver(self.robot) == "strands"):
            return None
        return get_native_driver_class(self.robot)

    def unaccepted(self) -> list[str]:
        """The keywords this recipe names that its driver would refuse."""
        driver = self.driver_class
        if driver is None:
            return []
        return sorted(self.keywords - set(constructor_keywords(driver)) - _FACTORY_KEYWORDS)

    def __str__(self) -> str:
        return f"{self.source}: Robot({self.robot!r}, ...) names {self.unaccepted()}"


def real_mode_recipes_in(source: str, name: str) -> list[_RealModeRecipe]:
    """The ``mode="real"`` recipes one module's printed and documented text gives.

    Read from string literals rather than from calls: the recipe is text a reader
    copies out of the output or the docstring, and it is routinely split across
    several ``print`` lines, so the literals are joined and searched as one
    document.
    """
    text = "\n".join(
        node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    )
    found: list[_RealModeRecipe] = []
    for match in _RECIPE.finditer(text):
        tail = match.group("tail")
        keywords = frozenset(_RECIPE_KEYWORD.findall(tail))
        if "mode" not in keywords or "real" not in tail:
            continue
        found.append(
            _RealModeRecipe(
                source=name,
                robot=match.group("robot"),
                keywords=keywords,
                names_native_driver="'strands'" in tail or '"strands"' in tail,
            )
        )
    return found


def _example_real_mode_recipes() -> list[_RealModeRecipe]:
    """Every ``mode="real"`` recipe the shipped examples hand a reader."""
    found: list[_RealModeRecipe] = []
    for module in sorted(_EXAMPLES_DIR.glob("**/*.py")):
        found += real_mode_recipes_in(module.read_text(encoding="utf-8"), module.relative_to(_REPO_ROOT).as_posix())
    return found


def module_runs_in(text: str, page: str) -> list[_ModuleRun]:
    """The ``python -m`` commands ``text`` runs, read from code contexts only."""
    found: list[_ModuleRun] = []
    in_fence = False
    for line in text.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        spans = [line] if in_fence else [m.group(1) for m in _INLINE.finditer(line)]
        for span in spans:
            match = _MODULE_RUN.match(_PREFIX.sub("", span.strip()))
            if match is not None:
                found.append(_ModuleRun(page, match.group(1), line))
    return found


def _documented_module_runs() -> list[_ModuleRun]:
    """Every ``python -m`` command the shipped markdown runs."""
    found: list[_ModuleRun] = []
    for page in sorted(_REPO_ROOT.glob("**/*.md")):
        if any(part in {".git", ".venv", "node_modules", "site"} for part in page.parts):
            continue
        rel = page.relative_to(_REPO_ROOT).as_posix()
        found += module_runs_in(page.read_text(encoding="utf-8"), rel)
    return found


def prompt_actions_in(source: str, name: str) -> list[_PromptAction]:
    """The actions the prompt-building text of one module names."""
    strings: list[str] = []
    tree = ast.parse(source)
    for node in ast.walk(tree):
        holder: ast.AST | None = None
        if isinstance(node, ast.FunctionDef) and _PROMPT_BUILDER.search(node.name):
            holder = node
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and _PROMPT_BUILDER.search(target.id) for target in node.targets
        ):
            holder = node.value
        if holder is None:
            continue
        strings += [
            child.value
            for child in ast.walk(holder)
            if isinstance(child, ast.Constant) and isinstance(child.value, str)
        ]
    named: set[str] = set()
    for text in strings:
        named |= set(_PROMPT_CALL.findall(text)) | set(_PROMPT_CALLED.findall(text))
    return [_PromptAction(name, action) for action in sorted(named)]


def _example_prompt_actions() -> list[_PromptAction]:
    """Every action the shipped examples' agent prompts name."""
    found: list[_PromptAction] = []
    for module in sorted(_EXAMPLES_DIR.glob("**/*.py")):
        found += prompt_actions_in(module.read_text(encoding="utf-8"), module.relative_to(_REPO_ROOT).as_posix())
    return found


class TestEveryDocumentedModuleRunResolves:
    """A ``python -m`` line this repository owns must name a module it has."""

    def test_no_page_runs_a_module_the_tree_does_not_have(self) -> None:
        """The shape the LIBERO section used must fail on arrival."""
        offenders = [run for run in _documented_module_runs() if run.is_local and not run.resolves(_REPO_ROOT)]
        assert not offenders, "documentation runs modules this tree does not have:\n" + "\n".join(
            f"  {offender}" for offender in offenders
        )

    def test_the_sweep_reaches_the_commands(self) -> None:
        """The proof the sweep read anything, so an empty read is not a pass."""
        runs = _documented_module_runs()
        local = [run for run in runs if run.is_local]
        assert len(runs) >= _MINIMUM_RUNS, f"read {len(runs)} python -m commands"
        assert len(local) >= _MINIMUM_LOCAL_RUNS, f"read {len(local)} in-tree commands"
        assert len({run.page for run in local}) >= _MINIMUM_PAGES, "the sweep must reach more than two pages"


class TestEveryPromptActionIsPublished:
    """An action an example's agent prompt names must be one the tool publishes."""

    def test_no_prompt_names_an_unpublished_action(self) -> None:
        """A model handed the prompt must not be sent to a refusal."""
        offenders = [named for named in _example_prompt_actions() if named.action not in _PUBLISHED_ACTIONS]
        assert not offenders, "example agent prompts name actions the Simulation tool does not publish:\n" + "\n".join(
            f"  {offender}" for offender in offenders
        )

    def test_the_sweep_reaches_the_prompts(self) -> None:
        """A rename of the prompt builders must shrink the sweep, not empty it."""
        found = _example_prompt_actions()
        assert len(found) >= _MINIMUM_PROMPT_ACTIONS, f"read {len(found)} prompt actions"
        assert len({named.source for named in found}) >= _MINIMUM_PROMPT_FILES, "the sweep must reach two prompt files"


class TestEveryPrintedRealModeRecipeIsAccepted:
    """A keyword a printed ``mode="real"`` recipe names must reach the driver."""

    def test_no_example_prints_a_recipe_the_factory_refuses(self) -> None:
        """The shape the G1 recipe had must fail on arrival."""
        offenders = [recipe for recipe in _example_real_mode_recipes() if recipe.unaccepted()]
        assert not offenders, "examples print mode='real' recipes the factory refuses:\n" + "\n".join(
            f"  {offender}" for offender in offenders
        )

    def test_the_sweep_reaches_the_recipes(self) -> None:
        """A reflow that hides the recipes reports a shrunken sweep, not a pass."""
        recipes = _example_real_mode_recipes()
        graded = [recipe for recipe in recipes if recipe.driver_class is not None]
        assert len(recipes) >= _MINIMUM_RECIPES, f"read {len(recipes)} mode='real' recipes"
        assert len(graded) >= _MINIMUM_GRADED_RECIPES, f"graded {len(graded)} native-driver recipes"


class TestTheGradersAreLoadBearing:
    """Each sweep reports a planted offender, and only the offender."""

    def test_a_planted_missing_module_is_reported(self) -> None:
        """The deleted-script shape is reported; its live sibling is not."""
        planted = module_runs_in(
            "```bash\npython -m examples.mujoco_gs.libero_groot --provider mock\n```\n"
            "and `python -m examples.mujoco_gs.app` still runs.\n",
            "planted.md",
        )
        assert [run.module for run in planted] == [
            "examples.mujoco_gs.libero_groot",
            "examples.mujoco_gs.app",
        ]
        assert [run for run in planted if not run.resolves(_REPO_ROOT)] == [planted[0]]

    def test_an_external_module_is_not_this_trees_to_supply(self) -> None:
        """A page may run a dependency's module; only in-tree names are graded."""
        planted = module_runs_in("`python -m lerobot.scripts.lerobot_train`", "planted.md")
        assert [run.is_local for run in planted] == [False]

    def test_a_planted_unpublished_prompt_action_is_reported(self) -> None:
        """The scene block's shape is reported; the published call beside it is not."""
        planted = prompt_actions_in(
            "def make_scene_description() -> str:\n"
            '    return "Drive it with `step`, then call `hybrid_render`, or `render(camera_name=...)`."\n',
            "planted.py",
        )
        assert [named.action for named in planted] == ["hybrid_render", "render"]
        assert [named.action for named in planted if named.action not in _PUBLISHED_ACTIONS] == ["hybrid_render"]

    def test_prose_outside_a_prompt_builder_is_not_read(self) -> None:
        """A docstring elsewhere in an example names whatever it likes."""
        planted = prompt_actions_in('def encode() -> None:\n    """Writes through `mimsave(...)`."""\n', "planted.py")
        assert planted == []

    def test_a_planted_refused_keyword_is_reported(self) -> None:
        """The G1 recipe's shape is reported; the ``port=`` spelling beside it is not."""
        planted = real_mode_recipes_in(
            "def show() -> None:\n"
            "    print(\"    g1 = Robot('g1', mode='real',\")\n"
            "    print(\"               robot_ip='192.168.123.164')\")\n"
            "    print(\"    ok = Robot('g1', mode='real', port='192.168.123.164')\")\n",
            "planted.py",
        )
        assert [recipe.unaccepted() for recipe in planted] == [["robot_ip"], []]

    def test_a_lerobot_recipe_is_not_graded_here(self) -> None:
        """lerobot owns its own keyword roster, and it need not be installed."""
        planted = real_mode_recipes_in(
            "print(\"arm = Robot('so100', mode='real', port='/dev/ttyACM0')\")\n", "planted.py"
        )
        assert [recipe.driver_class for recipe in planted] == [None]

    def test_a_sim_recipe_is_not_a_hardware_recipe(self) -> None:
        """Only a ``mode="real"`` call names a driver keyword at all."""
        assert real_mode_recipes_in("print(\"sim = Robot('g1', position=[0, 0, 0])\")\n", "planted.py") == []
