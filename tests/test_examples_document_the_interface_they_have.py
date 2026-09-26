"""An example's own docstring and flags describe the script that is there.

An example is run by copying its header, so the header is an interface: the
install line is what a reader's environment ends up containing, and the
``--flags`` in ``--help`` are what a reader believes they can set. Three ways
that interface drifts from the file, the first two measured on ``74136572a``:

1. ``examples/vla/cosmos3_diffusers_mujoco_rollout.py`` documented
   ``uv pip install "strands-robots[cosmos3-diffusers,cosmos3-sim]"`` and then
   imported ``robot_descriptions`` (the Panda MJCF) and ``imageio`` (encoding
   ``--render``). Neither cosmos3 extra declares either distribution, and both
   imports sit after the model forward pass, so a reader who followed the line
   exactly lost the pipeline load plus sampling to
   ``ModuleNotFoundError: No module named 'robot_descriptions'``. Both are
   declared by ``sim-mujoco``, which the line now names.
2. The same file advertised ``--steps`` ("diffusion sampling steps") and never
   read ``args.steps``: the sampler count is a ``Cosmos3DiffusersBackend``
   parameter and ``Cosmos3Policy`` forwards only ``embodiment``/``model``/
   ``mode``, so every run sampled the backend default of 35 whatever the flag
   said.

3. ``examples/07_post_tune_any_policy.py`` and
   ``examples/17_judge_recorded_episodes.py`` documented
   ``pip install "strands-robots[sim-mujoco,lerobot]"`` and then trained through
   ``create_trainer("lerobot_local")``. LeRobot's ``train()`` opens with
   ``require_package("accelerate", extra="training")`` - on CPU as well as GPU -
   and no strands extra supplies it, so both examples ran every earlier stage
   and then exited 1 on a ``TrainSpec rejected`` the line could not satisfy.
   ``accelerate`` is invisible to the import scan above because the example
   never imports it: the trainer names it in
   ``_LEROBOT_CALL_TIME_PACKAGES``, which is what the rule below reads.

4. The three ``examples/locomotion/`` scripts steered the G1 through a
   ``locomotion_style`` goal key. Its only consumer, the MotionBricks policy, was
   removed with the rest of the policy tree-shake; no policy in the tree reads the
   key today, and ``get_actions(obs, instruction, **policy_kwargs)`` drops an
   unknown one in silence. Measured on ``55f38d17b`` with the published
   GR00T-WBC Balance/Walk ONNX on a Unitree G1 in MuJoCo, the example's own
   four-segment schedule run twice - once as shipped, once with a style added to
   every segment - traced 408 control steps whose ``(x, y, z, yaw)`` differed by
   ``0.000000``. ``keyboard_g1.py`` advertised eight keys for it, and
   ``agent_g1.py`` named it plus a seven-value vocabulary in the system prompt an
   LLM is handed, so the agent reported style switches it never made.

5. ``examples/wbc/wbc_g1_gait.py`` documented no install line at all and drew its
   PNG with ``matplotlib``. ``--plot-clock`` is the mode the file advertises as
   needing no checkpoint and the command ``docs/policies/wbc_gait.md`` credits its
   figure to, and in a venv built from the two sibling scripts' own line
   (``pip install "strands-robots[wbc,sim-mujoco]"``) it exited 1 on
   ``ModuleNotFoundError: No module named 'matplotlib'``. No extra of this project
   declares matplotlib, so the import scan above skipped it by design.

6. ``examples/robots/neon.py`` documented ``pip install "strands-robots[mesh]"
   cyclonedds unitree_sdk2py``, and ``unitree_sdk2py`` is not on PyPI under that
   name - ``uv pip install`` ends the whole command with ``unitree-sdk2py was not
   found in the package registry``, so nothing is installed and the reader never
   reaches the G1. The driver's own missing-SDK refusal already names the working
   recipe (``[ros2]`` for the CycloneDDS binding plus the vendor checkout), which
   is what the rule below reads. The line was invisible to the scan above because
   it wrapped: ``pip install`` ended one source line and its arguments began the
   next, and a reader ending at the newline resolved the only unsatisfiable line
   in ``examples/`` to no arguments at all - the one docstring of 52 whose
   install command read as empty.

Why the install rule is keyed on distributions this project declares: an example
may legitimately import something no extra covers (an optional third-party tool
the header installs separately, or a module only the reader's own environment
has). What it may not do is import a distribution ``pyproject.toml`` knows how to
install and leave that out of its own line - that gap is always the line's bug,
and the fix is always naming the extra.

The undeclared half is the header's own obligation, which is the rule
``test_a_script_names_every_dependency_no_extra_can_install`` adds: a module this
project cannot install is exactly the one a reader has to be told about, because
no extra spelling of the install line reaches it. It is scoped to scripts - an
app's own submodule (``examples/isaac_gs/scene.py`` importing ``isaacsim``) is
not run directly and its README carries the install - and the import-name roster
below resolves the cases where a module and its distribution are spelled
differently, so a declared dependency is not read as an undeclared one.
"""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from pathlib import Path

import pytest

from strands_robots.drivers.unitree._common import UNITREE_SDK_INSTALL, sdk_missing
from strands_robots.training.lerobot import _LEROBOT_CALL_TIME_PACKAGES

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EXAMPLES_DIR = _REPO_ROOT / "examples"
_PYPROJECT = _REPO_ROOT / "pyproject.toml"

_INSTALL_START = re.compile(r"(?:uv )?pip install")
_CODE_FENCES = ("``", "`")
_SELF_EXTRAS = re.compile(r"strands[-_]robots\[(?P<extras>[^\]]+)\]")

# The import names whose distribution is spelled differently, so that a declared
# dependency is not read as one this project cannot install. Each entry is
# checked against the manifest by
# ``test_each_import_name_alias_resolves_to_a_declared_distribution``.
_DISTRIBUTION_BY_IMPORT_NAME = {
    "PIL": "pillow",
    "cv2": "opencv-python-headless",
    "yaml": "pyyaml",
}

# The vendor SDKs this project refuses to pip-install, each paired with the
# refusal that says so and the recipe that refusal names. Both halves are
# checked against the library by
# ``test_each_unpublished_sdk_is_one_the_library_refuses_to_install``, so the
# roster cannot become a list of names nothing in the tree stands behind.
_UNPUBLISHED_VENDOR_SDKS = {
    "unitree_sdk2py": (sdk_missing, UNITREE_SDK_INSTALL),
}


def _canonical(requirement: str) -> str:
    """The distribution name a requirement string installs, module-spelled."""
    return re.split(r"[<>=!;@\[ ]", requirement.strip().strip("\"'"))[0].replace("-", "_").lower()


def _declared() -> tuple[dict[str, frozenset[str]], frozenset[str]]:
    """``{extra: distributions it pulls in}`` plus the always-installed base."""
    project = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))["project"]
    extras = project["optional-dependencies"]

    def resolve(extra: str, seen: frozenset[str]) -> frozenset[str]:
        names: set[str] = set()
        for requirement in extras.get(extra, []):
            nested = _SELF_EXTRAS.match(requirement.strip())
            if nested:
                for child in nested.group("extras").split(","):
                    child = child.strip()
                    if child not in seen:
                        names |= resolve(child, seen | {extra})
            else:
                names.add(_canonical(requirement))
        return frozenset(names)

    return (
        {extra: resolve(extra, frozenset()) for extra in extras},
        frozenset(_canonical(r) for r in project["dependencies"]),
    )


def _install_arguments(text: str) -> str | None:
    """Everything the first ``pip install`` in ``text`` is handed, verbatim.

    A docstring wraps, so a command ends at its own delimiter and not at the
    line break: opened inside a code span the arguments run to the matching
    closer across as many source lines as the span takes, bounded by the blank
    line no code span crosses. Only an undelimited command ends at the newline.
    Ending every command at the newline instead drops every argument of a
    wrapped one, and a command that installs nothing is graded as a docstring
    with nothing to install rather than as the line it is.

    ``None`` when ``text`` documents no install command.
    """
    match = _INSTALL_START.search(text)
    if match is None:
        return None
    fence = next((f for f in _CODE_FENCES if text[: match.start()].endswith(f)), "")
    rest = text[match.end() :]
    limits = [len(rest)]
    if fence:
        limits += [index for index in (rest.find(fence), rest.find("\n\n")) if index != -1]
    else:
        limits += [index for index in (rest.find("\n"),) if index != -1]
    return rest[: min(limits)].replace("\\\n", " ")


def _pip_targets(arguments: str) -> list[str]:
    """The requirement arguments of an install command, flags and quotes gone."""
    targets = []
    for token in re.findall(r"\"[^\"]+\"|'[^']+'|\S+", arguments):
        token = token.strip("\"'")
        if token and not token.startswith("-"):
            targets.append(token)
    return targets


def _install_line_provides(docstring: str) -> frozenset[str] | None:
    """Distributions the docstring's install line ends up installing.

    ``None`` when the docstring documents no install line, which is most
    examples: the quickstart install is assumed and there is nothing to grade.
    """
    arguments = _install_arguments(docstring)
    if arguments is None:
        return None
    by_extra, base = _declared()
    provided = set(base)
    for token in _pip_targets(arguments):
        extras = _SELF_EXTRAS.search(token)
        if extras:
            for extra in extras.group("extras").split(","):
                provided |= by_extra.get(extra.strip(), frozenset())
        else:
            provided.add(_canonical(token))
    return frozenset(provided)


def _imported_top_level(tree: ast.AST) -> frozenset[str]:
    """Top-level module names the file imports, wherever the import sits."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return frozenset(names)


def _advertised_flags(tree: ast.AST) -> list[tuple[str, int]]:
    """``(attribute name, lineno)`` for every ``--flag`` an argparse call adds."""
    flags = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "add_argument":
            continue
        dest = next(
            (kw.value.value for kw in node.keywords if kw.arg == "dest" and isinstance(kw.value, ast.Constant)),
            None,
        )
        if dest is None:
            dest = next(
                (
                    arg.value[2:].replace("-", "_")
                    for arg in node.args
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("--")
                ),
                None,
            )
        if isinstance(dest, str):
            flags.append((dest, node.lineno))
    return flags


def _read_names(tree: ast.AST) -> frozenset[str]:
    """Every attribute and bare name read anywhere in the file."""
    return frozenset(
        node.attr if isinstance(node, ast.Attribute) else node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute | ast.Name)
    )


def _examples() -> list[tuple[Path, ast.Module]]:
    """Every example paired with its parsed module."""
    sources: list[tuple[Path, ast.Module]] = []
    for path in sorted(_EXAMPLES_DIR.rglob("*.py")):
        sources.append((path, ast.parse(path.read_text(encoding="utf-8"), filename=str(path))))
    return sources


def test_an_install_line_declares_every_distribution_the_example_imports() -> None:
    """A documented install line leaves nothing the file imports uninstalled."""
    by_extra, _ = _declared()
    installable = {dist for dists in by_extra.values() for dist in dists}
    offenders = []
    for path, tree in _examples():
        provided = _install_line_provides(ast.get_docstring(tree) or "")
        if provided is None:
            continue
        for module in sorted(_imported_top_level(tree)):
            dist = module.replace("-", "_").lower()
            if dist == "strands_robots" or dist in provided or dist not in installable:
                continue
            extras = sorted(extra for extra, dists in by_extra.items() if dist in dists)
            offenders.append(f"{path.relative_to(_REPO_ROOT).as_posix()} imports {module} (declared by {extras})")
    assert not offenders, "an example's install line must install what the example imports: " + "; ".join(offenders)


def _install_line_text(docstring: str) -> str | None:
    """The docstring's install line verbatim, or ``None`` when it has none."""
    arguments = _install_arguments(docstring)
    return None if arguments is None else "pip install" + arguments


def _lerobot_extras_named(install_line: str) -> frozenset[str]:
    """Extras of the ``lerobot`` distribution the line asks for by name."""
    return frozenset(
        extra.strip()
        for group in re.findall(r"lerobot\[(?P<extras>[^\]]+)\]", install_line)
        for extra in group.split(",")
    )


def _trains_through_lerobot(tree: ast.Module) -> bool:
    """Whether the example reaches lerobot's ``train()`` via the trainer factory.

    The provider reaches ``create_trainer`` either literally or through a
    module-level constant (``PROVIDER = "lerobot_local"``, the spelling
    example 07 uses so a reader retargets the flow by editing one line), so
    both are resolved.
    """
    constants = {
        target.id: node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
        for target in node.targets
        if isinstance(target, ast.Name)
    }

    def names_lerobot(arg: ast.expr) -> bool:
        if isinstance(arg, ast.Constant):
            return arg.value == "lerobot_local"
        return isinstance(arg, ast.Name) and constants.get(arg.id) == "lerobot_local"

    return any(
        isinstance(node, ast.Call)
        and (node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None))
        == "create_trainer"
        and any(names_lerobot(arg) for arg in node.args)
        for node in ast.walk(tree)
    )


def test_an_install_line_declares_what_the_trainer_needs_at_call_time() -> None:
    """An example that trains names the extra lerobot's ``train()`` requires.

    The import scan cannot see these: the example never imports ``accelerate``,
    lerobot's ``train()`` does, as its first statement and whatever the device.
    The trainer publishes the pair in ``_LEROBOT_CALL_TIME_PACKAGES`` and
    refuses a spec without it, so an install line missing the extra buys a
    ``TrainSpec rejected`` at the end of an otherwise working run.
    """
    graded, offenders = [], []
    for path, tree in _examples():
        install_line = _install_line_text(ast.get_docstring(tree) or "")
        if install_line is None or not _trains_through_lerobot(tree):
            continue
        graded.append(path)
        named = _lerobot_extras_named(install_line)
        for package, extra in _LEROBOT_CALL_TIME_PACKAGES:
            if extra not in named and package not in install_line:
                offenders.append(
                    f"{path.relative_to(_REPO_ROOT).as_posix()} trains but does not install "
                    f"{package} (lerobot[{extra}])"
                )
    assert graded, "no example trains through create_trainer('lerobot_local'); the rule grades nothing"
    assert not offenders, "a training example's install line must reach train(): " + "; ".join(offenders)


def test_the_call_time_rule_separates_a_naming_line_from_a_silent_one() -> None:
    """Planted pair: the extra is what the rule reads, in either spelling."""
    assert ("accelerate", "training") in _LEROBOT_CALL_TIME_PACKAGES, "the trainer's call-time roster moved"
    assert _lerobot_extras_named('pip install "strands-robots[lerobot]"') == frozenset()
    assert "training" in _lerobot_extras_named('pip install "strands-robots[lerobot]" "lerobot[training]"')
    assert "training" in _lerobot_extras_named('pip install "lerobot[pi,training]"')
    assert _trains_through_lerobot(ast.parse('x = create_trainer("lerobot_local")'))
    assert _trains_through_lerobot(ast.parse('P = "lerobot_local"\nx = create_trainer(P, device="cpu")'))
    assert not _trains_through_lerobot(ast.parse('x = create_trainer("ppo")'))
    assert not _trains_through_lerobot(ast.parse('P = "ppo"\nx = create_trainer(P)'))


def test_every_flag_an_example_advertises_is_read() -> None:
    """No example offers a ``--flag`` in ``--help`` that changes nothing."""
    offenders = []
    for path, tree in _examples():
        read = _read_names(tree)
        for dest, lineno in _advertised_flags(tree):
            if dest not in read:
                offenders.append(f"{path.relative_to(_REPO_ROOT).as_posix()}:{lineno} --{dest.replace('_', '-')}")
    assert not offenders, "a flag nobody reads is a knob that silently does nothing: " + "; ".join(offenders)


def test_the_scan_reaches_install_lines_and_flags() -> None:
    """Non-vacuity: both scans resolve real examples, not an empty tree."""
    examples = _examples()
    with_install = [p for p, tree in examples if _install_line_provides(ast.get_docstring(tree) or "") is not None]
    with_flags = [p for p, tree in examples if _advertised_flags(tree)]
    assert with_install, f"no install line found under {_EXAMPLES_DIR}"
    assert with_flags, f"no argparse flag found under {_EXAMPLES_DIR}"


def test_the_install_rule_separates_a_covered_import_from_a_missing_one() -> None:
    """Planted positive: naming the extra is what makes the line sufficient."""
    by_extra, base = _declared()
    assert "robot_descriptions" in by_extra["sim-mujoco"], "sim-mujoco no longer ships the MJCF assets"
    bare = _install_line_provides('uv pip install "strands-robots[cosmos3-sim]"')
    fixed = _install_line_provides('uv pip install "strands-robots[cosmos3-sim,sim-mujoco]"')
    named = _install_line_provides("pip install robot_descriptions")
    assert bare is not None and fixed is not None and named is not None
    assert "robot_descriptions" not in bare
    assert "robot_descriptions" in fixed
    assert "robot_descriptions" in named, "a distribution named directly on the line counts as installed"
    assert _install_line_provides("no install line here") is None
    assert base, "the base dependency list is empty; every example would look under-installed"


def test_no_install_line_names_a_vendor_sdk_pip_cannot_reach() -> None:
    """No example tells a reader to ``pip install`` an unpublished vendor SDK.

    A vendor SDK that is not on PyPI under the name it is imported by has one
    install, and the library already holds it: the missing-SDK refusal names a
    recipe, and an install line that names the module instead resolves to
    nothing at all. That is worse than a missing extra, because the reader's
    whole command fails before anything is installed.
    """
    offenders = []
    for path, tree in _examples():
        arguments = _install_arguments(ast.get_docstring(tree) or "")
        if arguments is None:
            continue
        for target in _pip_targets(arguments):
            module = _canonical(target).replace("-", "_")
            if module in _UNPUBLISHED_VENDOR_SDKS:
                offenders.append(f"{path.relative_to(_REPO_ROOT).as_posix()} installs {target}")
    assert not offenders, (
        "a vendor SDK pip cannot reach must be installed by the recipe its refusal names, "
        "not named as a target: " + "; ".join(offenders)
    )


def test_each_unpublished_sdk_is_one_the_library_refuses_to_install() -> None:
    """Premise: every roster entry is a module the library says pip cannot get."""
    assert _UNPUBLISHED_VENDOR_SDKS, "the roster is empty; the rule above grades nothing"
    for module, (refusal, recipe) in _UNPUBLISHED_VENDOR_SDKS.items():
        text = refusal("No module named 'x'")
        assert module in text, f"{module} is not what {refusal.__name__} refuses"
        assert "not a strands-robots extra" in text, f"{refusal.__name__} no longer says pip cannot reach {module}"
        assert module not in _pip_targets(recipe), (
            f"{module} is a pip target of its own recipe; drop it from the roster"
        )


@pytest.mark.parametrize(
    ("docstring", "targets"),
    [
        pytest.param(
            'Dependencies: ``pip install\n"strands-robots[mesh]" cyclonedds unitree_sdk2py``. The SDK is\nlazy-imported.',
            ["strands-robots[mesh]", "cyclonedds", "unitree_sdk2py"],
            id="a-wrapped-code-span-keeps-every-target",
        ),
        pytest.param(
            'Dependencies: ``pip install "strands-robots[mesh,ros2]"``\n\nThen clone the SDK.',
            ["strands-robots[mesh,ros2]"],
            id="a-closed-code-span-stops-at-its-closer",
        ),
        pytest.param(
            "Dependencies::\n\n    pip install 'strands-robots[mesh,ros2]'\n    git clone https://example/sdk\n",
            ["strands-robots[mesh,ros2]"],
            id="an-undelimited-command-stops-at-the-newline",
        ),
        pytest.param(
            "Dependencies: pip install --no-deps -e ./unitree_sdk2_python\nRuntime: forever.",
            ["./unitree_sdk2_python"],
            id="flags-are-not-targets",
        ),
        pytest.param(
            "Dependencies: ``pip install robot_descriptions`` + a working GPU.",
            ["robot_descriptions"],
            id="prose-after-the-closer-is-not-a-target",
        ),
        pytest.param("Runtime: ~3 seconds.", None, id="no-command-reads-as-none"),
    ],
)
def test_the_install_reader_ends_a_command_at_its_own_delimiter(docstring: str, targets: list[str] | None) -> None:
    """Planted pair: the wrap is what the line-scoped reader used to lose."""
    arguments = _install_arguments(docstring)
    assert (None if arguments is None else _pip_targets(arguments)) == targets


def test_the_unpublished_sdk_rule_separates_the_recipe_from_the_module() -> None:
    """Planted pair: the checkout installs, the module name does not."""
    bad = _install_arguments('Dependencies: ``pip install\n"strands-robots[mesh]" cyclonedds unitree_sdk2py``.')
    good = _install_arguments("Dependencies: ``pip install 'strands-robots[mesh,ros2]'``")
    assert bad is not None and good is not None
    assert "unitree_sdk2py" in _pip_targets(bad)
    assert not set(_pip_targets(good)) & set(_UNPUBLISHED_VENDOR_SDKS)
    assert not set(_pip_targets(UNITREE_SDK_INSTALL)) & set(_UNPUBLISHED_VENDOR_SDKS)


def _goal_keys_policies_read() -> frozenset[str]:
    """Every ``policy_kwargs`` key some policy in the tree actually reads.

    Harvested from the ``kwargs.get("x")`` / ``kwargs.pop("x")`` calls in
    ``strands_robots/policies/``, which is where a provider pulls its goal out of
    the payload ``run_policy`` forwards verbatim.
    """
    keys: set[str] = set()
    for path in sorted((_REPO_ROOT / "strands_robots" / "policies").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            receiver = node.func.value
            if node.func.attr not in {"get", "pop"} or not isinstance(receiver, ast.Name):
                continue
            if receiver.id not in {"kwargs", "policy_kwargs"} or not node.args:
                continue
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                keys.add(first.value)
    return frozenset(keys)


_DICT_KEY = re.compile(r'"([a-z_][a-z_0-9]*)"\s*:')
_INLINE_CODE = re.compile(r"``([a-z_][a-z_0-9]*)``")
_GOAL_DICT_IN_PROSE = re.compile(r"policy_kwargs\s*=\s*\{([^}]*)\}")
_INLINE_CODE_RUN = re.compile(r"``[a-z_0-9]+``(?:\s*/\s*``[a-z_0-9]+``)+")


def _goal_citations(source: str, tree: ast.Module, anchors: frozenset[str]) -> list[tuple[int, str, list[str]]]:
    """``(line, kind, keys)`` for every place one file spells the goal channel.

    A citation is recognised by carrying an ``anchor`` - a goal key whose name
    starts with ``target_``. Those are unambiguous: nothing else in the tree
    keys a dict on one, while ``height`` / ``command`` / ``video`` / ``seed`` are
    read by a policy AND spelled by unrelated dicts, so anchoring on them would
    grade camera configs and JSON-RPC envelopes. Three spellings are read: a dict
    literal (the goal the script passes), a ``policy_kwargs={...}`` fragment
    inside a string (the instructions an agent example hands a model), and a
    slash-separated run of inline-code names (the channel a docstring describes).
    """
    found: list[tuple[int, str, list[str]]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            keys = [k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)]
            if anchors.intersection(keys):
                found.append((node.lineno, "goal dict", keys))
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for fragment in _GOAL_DICT_IN_PROSE.findall(node.value):
                keys = _DICT_KEY.findall(fragment)
                if anchors.intersection(keys):
                    found.append((node.lineno, "goal dict in a string", keys))
    for lineno, line in enumerate(source.splitlines(), start=1):
        for run in _INLINE_CODE_RUN.findall(line):
            keys = _INLINE_CODE.findall(run)
            if anchors.intersection(keys):
                found.append((lineno, "prose channel list", keys))
    return found


def _anchors(read: frozenset[str]) -> frozenset[str]:
    """The goal keys a citation is recognised by - see :func:`_goal_citations`."""
    return frozenset(key for key in read if key.startswith("target_"))


def test_every_goal_key_an_example_names_is_read_by_a_policy() -> None:
    """No example steers through a ``policy_kwargs`` key nothing consumes."""
    read = _goal_keys_policies_read()
    anchors = _anchors(read)
    offenders = []
    for path, tree in _examples():
        source = path.read_text(encoding="utf-8")
        for lineno, kind, keys in _goal_citations(source, tree, anchors):
            for key in keys:
                if key not in read:
                    where = f"{path.relative_to(_REPO_ROOT).as_posix()}:{lineno}"
                    offenders.append(f"{where} [{kind}] {key}")
    assert not offenders, (
        "a goal key no policy reads is dropped in silence by "
        "get_actions(obs, instruction, **policy_kwargs): " + "; ".join(sorted(set(offenders)))
    )


def test_the_goal_scan_reaches_real_citations() -> None:
    """Non-vacuity: the rule above grades a populated set, on both sides."""
    read = _goal_keys_policies_read()
    anchors = _anchors(read)
    assert {"target_velocity", "target_pose", "target_joints"} <= anchors, f"the anchor set moved: {sorted(anchors)}"
    assert {"height", "world_update", "gait_frequency"} <= read, f"the read set moved: {sorted(read)}"
    cited = {path for path, tree in _examples() if _goal_citations(path.read_text(encoding="utf-8"), tree, anchors)}
    assert len(cited) >= 8, f"only {len(cited)} example(s) spell the goal channel; the scan stopped reaching them"


def test_the_goal_rule_separates_a_read_key_from_a_dead_one() -> None:
    """Planted cases: each spelling is recognised, and only a dead key fails."""
    read = _goal_keys_policies_read()
    anchors = _anchors(read)

    def dead(text: str) -> list[str]:
        tree = ast.parse(text)
        return [key for _, _, keys in _goal_citations(text, tree, anchors) for key in keys if key not in read]

    assert dead('G = {"target_velocity": [0.4, 0.0, 0.0], "height": 0.7}') == []
    assert dead('G = {"target_velocity": [0.4, 0.0, 0.0], "locomotion_style": "run"}') == ["locomotion_style"]
    assert dead('S = \'policy_kwargs={"target_velocity": [vx, vy, wz], "locomotion_style": <s>}\'') == [
        "locomotion_style"
    ]
    assert dead('"""The ``target_velocity`` / ``locomotion_style`` channel."""') == ["locomotion_style"]
    assert dead('"""The ``target_velocity`` / ``height`` channel."""') == []
    assert dead('CFG = {"width": 640, "height": 480, "locomotion_style": "run"}') == [], (
        "a camera config carries no target_* anchor and must not be read as a goal dict"
    )


def _is_script(tree: ast.Module) -> bool:
    """Whether the file is run directly, rather than imported by an app."""
    return any(
        isinstance(node, ast.If) and ast.unparse(node.test).replace('"', "'") == "__name__ == '__main__'"
        for node in tree.body
    )


def _repo_local(module: str) -> bool:
    """Whether the import resolves inside this repository, not to a dependency."""
    return (_REPO_ROOT / module).is_dir() or any(_EXAMPLES_DIR.rglob(f"{module}.py"))


def _installable() -> frozenset[str]:
    """Every distribution the manifest knows how to install, module-spelled."""
    by_extra, base = _declared()
    return frozenset(base | {dist for dists in by_extra.values() for dist in dists})


def _uninstallable_imports(tree: ast.Module) -> list[str]:
    """Third-party modules the file imports that no extra of this project declares."""
    installable = _installable()
    unreachable = []
    for module in sorted(_imported_top_level(tree)):
        if module in sys.stdlib_module_names or module in {"strands_robots", "strands"} or _repo_local(module):
            continue
        if _canonical(_DISTRIBUTION_BY_IMPORT_NAME.get(module, module)) in installable:
            continue
        unreachable.append(module)
    return unreachable


def _unnamed_dependencies(tree: ast.Module) -> list[str]:
    """Uninstallable imports the file's own docstring never mentions."""
    docstring = (ast.get_docstring(tree) or "").lower()
    return [module for module in _uninstallable_imports(tree) if module.lower() not in docstring]


def test_a_script_names_every_dependency_no_extra_can_install() -> None:
    """A runnable example names what ``pyproject.toml`` cannot install for it.

    No spelling of ``pip install "strands-robots[...]"`` reaches these, so the
    header is the only place a reader can learn about them - and the failure
    lands wherever the import sits, which for a plotting or encoding import is
    after the work the reader was waiting for.
    """
    graded, named, offenders = [], [], []
    for path, tree in _examples():
        if not _is_script(tree):
            continue
        graded.append(path)
        where = path.relative_to(_REPO_ROOT).as_posix()
        named += [
            f"{where}:{module}" for module in _uninstallable_imports(tree) if module not in _unnamed_dependencies(tree)
        ]
        offenders += [f"{where} imports {module}" for module in _unnamed_dependencies(tree)]
    assert graded, f"no runnable example found under {_EXAMPLES_DIR}"
    assert named, "no script names an undeclared dependency; the scan is reading nothing"
    assert not offenders, (
        "a script must name a dependency no extra of this project declares, because no install "
        "line can reach it: " + "; ".join(offenders)
    )


def test_each_import_name_alias_resolves_to_a_declared_distribution() -> None:
    """The roster resolves spellings; it never excuses an undeclared module."""
    installable = _installable()
    stale = {
        module: dist for module, dist in _DISTRIBUTION_BY_IMPORT_NAME.items() if _canonical(dist) not in installable
    }
    assert not stale, (
        f"{stale} map onto distributions the manifest does not declare, so the roster hides an "
        "undeclared dependency instead of resolving a spelling"
    )


@pytest.mark.parametrize(
    ("source", "unnamed"),
    [
        ('"""Plots it."""\nimport seaborn\n', ["seaborn"]),
        ('"""Plots it with seaborn: pip install seaborn."""\nimport seaborn\n', []),
        ('"""Renders it."""\nimport mujoco\n', []),
        ('"""Reads an image."""\nfrom PIL import Image\n', []),
        ('"""Parses a config."""\nimport yaml\n', []),
        ('"""Reads the clock."""\nimport json\nimport math\n', []),
        ('"""Drives the arm."""\nfrom strands_robots import Robot\n', []),
    ],
    ids=["undeclared", "named", "declared", "alias", "alias-yaml", "stdlib", "package"],
)
def test_the_undeclared_rule_separates_a_named_dependency_from_a_silent_one(source: str, unnamed: list[str]) -> None:
    """Planted cases: only a dependency nothing can install and nothing names fails."""
    assert _unnamed_dependencies(ast.parse(source)) == unnamed


@pytest.mark.parametrize(
    ("source", "is_script"),
    [
        ('if __name__ == "__main__":\n    main()\n', True),
        ("if __name__ == '__main__':\n    main()\n", True),
        ("def main() -> None:\n    pass\n", False),
    ],
    ids=["double-quoted", "single-quoted", "module"],
)
def test_the_script_gate_reads_the_entry_point_in_either_spelling(source: str, is_script: bool) -> None:
    """An app's submodule is graded by its README, not by a header it has not got."""
    assert _is_script(ast.parse(source)) is is_script
