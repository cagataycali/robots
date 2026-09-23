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

Why the install rule is keyed on distributions this project declares: an example
may legitimately import something no extra covers (an optional third-party tool
the header installs separately, or a module only the reader's own environment
has). What it may not do is import a distribution ``pyproject.toml`` knows how to
install and leave that out of its own line - that gap is always the line's bug,
and the fix is always naming the extra.
"""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

from strands_robots.training.lerobot import _LEROBOT_CALL_TIME_PACKAGES

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EXAMPLES_DIR = _REPO_ROOT / "examples"
_PYPROJECT = _REPO_ROOT / "pyproject.toml"

_INSTALL_LINE = re.compile(r"(?:uv )?pip install(?P<rest>[^\n]*(?:\\\n[^\n]*)*)")
_SELF_EXTRAS = re.compile(r"strands[-_]robots\[(?P<extras>[^\]]+)\]")


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


def _install_line_provides(docstring: str) -> frozenset[str] | None:
    """Distributions the docstring's install line ends up installing.

    ``None`` when the docstring documents no install line, which is most
    examples: the quickstart install is assumed and there is nothing to grade.
    """
    match = _INSTALL_LINE.search(docstring)
    if not match:
        return None
    by_extra, base = _declared()
    provided = set(base)
    rest = match.group("rest").replace("\\\n", " ")
    for token in re.findall(r"\"[^\"]+\"|'[^']+'|\S+", rest):
        token = token.strip("\"'")
        if token.startswith("-"):
            continue
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
    match = _INSTALL_LINE.search(docstring)
    return None if match is None else match.group(0).replace("\\\n", " ")


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
