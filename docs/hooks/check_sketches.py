#!/usr/bin/env python3
"""Statically verify every ``python`` and ``bash`` fence under docs/ against the package.

``check_fences.py`` runs the 105 bare ``python`` fences. The 102 fences with a title
(``python title="sketch"``) are never run, because they need an arm, a GPU or a cloud
account, so nothing noticed when one named a method that does not exist or swapped a
checkpoint. This hook reads every fence, runnable or not, and checks what can be checked
without executing it and without the network:

- the fence parses (``ast``);
- every ``from X import Y`` resolves against the installed package (a module of another
  distribution that is not installed is skipped and listed, not failed);
- every ``Robot("name", ...)`` names a registry robot, its ``mode=`` / ``driver=`` /
  ``backend=`` values are ones the factory accepts, and its keywords are parameters of the
  factory or of the class it builds (the MuJoCo engine in sim, the lerobot ``Robot`` or the
  native driver on hardware);
- every attribute read on a name bound to ``Robot(...)``, ``create_simulation(...)``,
  ``create_policy(...)``, ``create_trainer(...)``, ``Agent(...)`` or ``Mesh(...)`` exists on
  that surface, and a call's keywords are parameters of the method when it is a plain
  function without ``**kwargs``;
- every ``action="..."`` passed to a robot tool is an action that tool publishes;
- every Hub id (``org/name``) is recorded; ``--online`` HEADs it on huggingface.co;
- in ``bash`` fences: every ``strands-robots[extra]`` is an extra in pyproject.toml, every
  ``strands-robots <command> --flag`` parses with the command's own argparse parser, and
  every ``pip install`` name is recorded (installed names are checked, ``--online`` asks
  PyPI for the rest).

One row per finding: page:line, fence index, kind, what, remedy. Exit 1 when there is any.

    python3 docs/hooks/check_sketches.py [--only start/first-robot.md] [--online] [--skipped]
"""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.metadata
import importlib.util
import inspect
import io
import re
import shlex
import sys
import tomllib
import urllib.request
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs"

FENCE = re.compile(r"^```([^\n]*)\n(.*?)^```[ \t]*$", re.M | re.S)
HUB_ID = re.compile(r"\A[\w.-]+/[\w.-]+\Z")
HUB_KEYWORDS = frozenset(
    {"pretrained_name_or_path", "base_model", "repo_id", "dataset_repo_id", "checkpoint", "model_id", "hub_repo"}
)
EXTRA_SPEC = re.compile(r"strands-robots\[([^\]]+)\]")
CLI_LINE = re.compile(r"(?:^|&&|;|\|)\s*(?:python(?:3)?\s+-m\s+strands_robots|strands-robots)\s+(\S+)(.*)")
# Attributes the factory attaches to the instance it returns (robot.py _attach_mesh and
# _attach_device_connect); no class declares them.
FACTORY_ATTACHED = frozenset({"mesh", "peer_id", "run"})
# A factory call whose bound name is followed to a surface class.
CONSTRUCTORS = ("Robot", "create_simulation", "create_policy", "create_trainer", "Agent", "Mesh")


@dataclass(frozen=True)
class Finding:
    """One defect in one fence, with the remedy a docs author applies."""

    page: str
    line: int
    fence: int
    kind: str
    what: str
    remedy: str

    def row(self) -> str:
        """The finding as one report line."""
        return f"{self.page}:{self.line} #{self.fence} [{self.kind}] {self.what} -> {self.remedy}"


@dataclass(frozen=True)
class Fence:
    """A fenced block with its page, opener line and 1-based index among its language."""

    page: Path
    line: int
    index: int
    info: str
    body: str

    @property
    def rel(self) -> str:
        """The page path relative to docs/."""
        return str(self.page.relative_to(DOCS))

    @property
    def lang(self) -> str:
        """The first word of the info string."""
        return self.info.split()[0] if self.info.strip() else ""


def fences(page: Path) -> list[Fence]:
    """Every fence on a page, with per-language indexes matching ``check_fences.py``."""
    text = page.read_text(encoding="utf-8")
    counts: dict[str, int] = {}
    out: list[Fence] = []
    for match in FENCE.finditer(text):
        info = match.group(1).strip()
        lang = info.split()[0] if info else ""
        counts[lang] = counts.get(lang, 0) + 1
        out.append(Fence(page, text.count("\n", 0, match.start()) + 1, counts[lang], info, match.group(2)))
    return out


class Surfaces:
    """The package objects fences are checked against, imported once."""

    def __init__(self) -> None:
        from strands_robots import robot as robot_module
        from strands_robots.drivers import get_native_driver_class
        from strands_robots.drivers.base import HardwareDriver
        from strands_robots.drivers.registry import resolve_driver
        from strands_robots.hardware_robot import _FORWARDABLE_KWARGS, _PUBLISHED_ACTIONS
        from strands_robots.hardware_robot import Robot as HardwareRobot
        from strands_robots.mesh.core import Mesh
        from strands_robots.policies import list_aliases, list_providers
        from strands_robots.policies.base import Policy
        from strands_robots.policies.factory import _is_smart_string
        from strands_robots.registry import DRIVER_CHOICES, get_hardware_type, list_robots, resolve_name
        from strands_robots.simulation import SimEngine, list_backends
        from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine
        from strands_robots.training import Trainer, import_trainer_class, list_trainers

        self.robot_factory = robot_module.Robot
        self.factory_params = set(inspect.signature(robot_module.Robot).parameters) - {"name", "kwargs"}
        self.sim_engine = MuJoCoSimEngine
        self.sim_engine_params = set(inspect.signature(MuJoCoSimEngine.__init__).parameters) - {"self", "kwargs"}
        self.sim_base = SimEngine
        self.hardware_robot = HardwareRobot
        self.hardware_params = set(inspect.signature(HardwareRobot.__init__).parameters) - {"self", "kwargs"}
        self.hardware_driver = HardwareDriver
        self.native_driver_class = get_native_driver_class
        self.resolve_driver = resolve_driver
        self.get_hardware_type = get_hardware_type
        self.forwardable_kwargs = set(_FORWARDABLE_KWARGS)
        self.policy = Policy
        self.providers = set(list_providers()) | set(list_aliases())
        self.is_smart_string = _is_smart_string  # a Hub id or ws:// / zmq:// URL create_policy also accepts
        self.trainers = set(list_trainers())
        self.trainer = Trainer
        self.import_trainer_class = import_trainer_class
        self.mesh = Mesh
        self.robots = {entry["name"] for entry in list_robots()}
        self.resolve_name = resolve_name
        self.driver_choices = set(DRIVER_CHOICES)
        self.backends = set(list_backends())
        self.real_actions = set(_PUBLISHED_ACTIONS)
        spec = (REPO / "strands_robots" / "simulation" / "mujoco" / "tool_spec.json").read_text(encoding="utf-8")
        self.sim_actions = _sim_actions(spec)
        try:
            from strands import Agent

            self.agent: type | None = Agent
        except ImportError:
            self.agent = None
        pyproject = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
        self.extras = set(pyproject["project"]["optional-dependencies"])
        self.cli_commands = _cli_commands()

    def backend_class(self, backend: str) -> type:
        """The engine class ``create_simulation(backend)`` builds; the ABC when it cannot be imported here."""
        try:
            from strands_robots.simulation.factory import _import_backend_class

            return _import_backend_class(backend)
        except Exception:  # noqa: BLE001 - a plugin backend that is not installed falls back to the ABC
            return self.sim_base

    def _driver(self, name: str | None, driver: str) -> str:
        """The driver the factory resolves for this robot: registry default unless pinned."""
        if name is not None and name in self.robots:
            try:
                return str(self.resolve_driver(name, driver))
            except ValueError:
                return driver
        return driver

    def robot_surface(self, name: str | None, mode: str, driver: str) -> tuple[type, ...]:
        """The classes a ``Robot(...)`` call can return, for attribute checks."""
        sim: tuple[type, ...] = (self.sim_engine,)
        if mode == "sim":
            return sim
        real: tuple[type, ...]
        if self._driver(name, driver) == "strands":
            native = self.native_driver_class(name) if name is not None and name in self.robots else None
            real = (native or self.hardware_driver,)
        else:
            real = (self.hardware_robot,)
        return real if mode == "real" else sim + real

    def lerobot_config_fields(self, name: str | None) -> set[str]:
        """The fields of the lerobot config class this robot builds, or the forwardable set when unknown."""
        if name is None:
            return set(self.forwardable_kwargs)
        robot_type = self.get_hardware_type(name) or name
        try:
            from lerobot.robots.config import RobotConfig

            from strands_robots.utils import ensure_lerobot_family_registered

            ensure_lerobot_family_registered("robots")
            config_cls = RobotConfig.get_choice_class(robot_type)
        except Exception:  # noqa: BLE001 - lerobot missing or no such type: fall back to the allowlist
            return set(self.forwardable_kwargs)
        import dataclasses

        if not dataclasses.is_dataclass(config_cls):
            return set(self.forwardable_kwargs)
        return {f.name for f in dataclasses.fields(config_cls)} | self.forwardable_kwargs

    def robot_keywords(self, name: str | None, mode: str, driver: str) -> set[str]:
        """Every keyword a ``Robot(...)`` call may carry in this mode."""
        allowed = set(self.factory_params)
        if mode in ("sim", "auto"):
            allowed |= self.sim_engine_params
        if mode in ("real", "auto"):
            allowed |= self.hardware_params
            if self._driver(name, driver) == "strands":
                native = self.native_driver_class(name) if name is not None and name in self.robots else None
                if native is not None:
                    allowed |= set(inspect.signature(native).parameters)
                else:
                    allowed |= self.forwardable_kwargs
            else:
                allowed |= self.lerobot_config_fields(name)
        return allowed


def _sim_actions(spec_text: str) -> set[str]:
    """The action enum of the MuJoCo simulation tool."""
    import json

    return set(json.loads(spec_text)["properties"]["action"]["enum"])


def _cli_commands() -> dict[str, argparse.ArgumentParser]:
    """The argparse parser behind each ``strands-robots <command>``.

    ``verify-dataset`` builds its parser inside ``main``; ``parse_args`` is
    intercepted once to capture it without running the command.
    """
    from strands_robots.__main__ import _COMMANDS
    from strands_robots.dashboard.cli import build_parser as dashboard_parser
    from strands_robots.doctor import _parser as doctor_parser
    from strands_robots.mesh.iot.cli import _parser as iot_parser

    parsers: dict[str, argparse.ArgumentParser] = {
        "doctor": doctor_parser(),
        "dashboard": dashboard_parser(),
        "iot": iot_parser(),
    }

    class _Captured(Exception):
        def __init__(self, parser: argparse.ArgumentParser) -> None:
            self.parser = parser

    def _capture(self: argparse.ArgumentParser, *_: Any, **__: Any) -> Any:
        raise _Captured(self)

    from strands_robots import verify_dataset

    original = argparse.ArgumentParser.parse_args
    argparse.ArgumentParser.parse_args = _capture  # type: ignore[method-assign]
    try:
        verify_dataset.main([])
    except _Captured as captured:
        parsers["verify-dataset"] = captured.parser
    finally:
        argparse.ArgumentParser.parse_args = original  # type: ignore[method-assign]
    missing = set(_COMMANDS) - set(parsers)
    if missing:
        raise RuntimeError(f"no parser captured for {sorted(missing)}; teach _cli_commands the new command")
    return parsers


class Checker:
    """Runs every check over a set of pages and collects findings."""

    def __init__(self, surfaces: Surfaces, online: bool = False) -> None:
        self.s = surfaces
        self.online = online
        self.findings: list[Finding] = []
        self.skipped: list[str] = []
        self.hub_ids: dict[str, set[str]] = {}
        self.pip_names: dict[str, set[str]] = {}
        self.fences_checked = 0
        self._import_cache: dict[str, Any] = {}

    # ----- driver ---------------------------------------------------------

    def check_page(self, page: Path) -> None:
        """Check every python and bash fence on one page."""
        for fence in fences(page):
            if fence.lang == "python":
                self.fences_checked += 1
                self._check_python(fence)
            elif fence.lang in ("bash", "sh", "shell", "console"):
                self.fences_checked += 1
                self._check_bash(fence)

    def _find(self, fence: Fence, kind: str, what: str, remedy: str, offset: int = 0) -> None:
        self.findings.append(Finding(fence.rel, fence.line + offset, fence.index, kind, what, remedy))

    # ----- python ---------------------------------------------------------

    def _check_python(self, fence: Fence) -> None:
        try:
            tree = ast.parse(fence.body)
        except SyntaxError as exc:
            self._find(fence, "syntax", str(exc.msg), "make the fence valid Python", exc.lineno or 0)
            return
        names: dict[str, Any] = {}  # imported name -> object (None when unresolvable)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    names[alias.asname or alias.name] = self._check_import_from(fence, node, alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    names[alias.asname or alias.name.split(".")[0]] = self._import(fence, node, alias.name)
        bound = self._bind(fence, tree, names)
        self._check_attributes(fence, tree, bound)
        self._check_actions(fence, tree, names, bound)
        self._record_hub_ids(fence, tree)

    def _import(self, fence: Fence, node: ast.stmt, module: str) -> Any:
        """Import ``module``; a finding when it is ours and does not exist."""
        if module in self._import_cache:
            return self._import_cache[module]
        result: Any = None
        top = module.split(".")[0]
        try:
            result = importlib.import_module(module)
        except ImportError as exc:
            if top in ("strands_robots",):
                spec = None
                try:
                    spec = importlib.util.find_spec(module)
                except (ImportError, ValueError):
                    spec = None
                if spec is None:
                    self._find(
                        fence, "import", f"module {module} does not exist", "import the module that exists", node.lineno
                    )
                else:
                    self.skipped.append(f"{fence.rel} #{fence.index}: {module} needs an optional dependency ({exc})")
            else:
                self.skipped.append(f"{fence.rel} #{fence.index}: {module} is not installed here")
        except Exception as exc:  # noqa: BLE001 - a module that fails at import is a finding, whatever it raised
            self._find(
                fence, "import", f"importing {module} raised {type(exc).__name__}", "fix the import path", node.lineno
            )
        self._import_cache[module] = result
        return result

    def _check_import_from(self, fence: Fence, node: ast.ImportFrom, name: str) -> Any:
        module = self._import(fence, node, node.module or "")
        if module is None:
            return None
        if hasattr(module, name):
            return getattr(module, name)
        try:
            sub = importlib.import_module(f"{node.module}.{name}")
        except ImportError:
            sub = None
        if sub is None:
            self._find(
                fence,
                "import",
                f"{node.module} has no name {name}",
                f"import a name {node.module} defines (dir() lists them)",
                node.lineno,
            )
        return sub

    def _bind(self, fence: Fence, tree: ast.AST, names: dict[str, Any]) -> dict[str, tuple[type, ...]]:
        """Variable -> surface classes, for every ``x = Constructor(...)`` in the fence."""
        bound: dict[str, tuple[type, ...]] = {}
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)):
                continue
            func = node.value.func
            ctor = func.id if isinstance(func, ast.Name) else None
            if ctor not in CONSTRUCTORS or not isinstance(node.targets[0], ast.Name):
                continue
            var = node.targets[0].id
            surface = self._surface_for(fence, ctor, node.value)
            if surface:
                bound[var] = surface
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            callee = node.func.id if isinstance(node.func, ast.Name) else None
            if callee == "Robot":
                self._check_robot_call(fence, node)
            elif callee == "create_policy":
                self._check_provider(fence, node, _literal_arg(node, 0), "create_policy")
            elif callee == "create_trainer":
                provider = _literal_arg(node, 0)
                if isinstance(provider, str) and provider not in self.s.trainers:
                    self._find(
                        fence,
                        "provider",
                        f"create_trainer({provider!r}) is not a trainer",
                        f"use one of {sorted(self.s.trainers)}",
                        node.lineno,
                    )
            elif isinstance(node.func, ast.Attribute) and node.func.attr == "add_robot":
                name = _literal_arg(node, 0)
                keywords = {k.arg for k in node.keywords}
                if (
                    isinstance(name, str)
                    and self.s.resolve_name(name) not in self.s.robots
                    and "urdf_path" not in keywords
                ):
                    self._find(
                        fence,
                        "robot",
                        f"add_robot({name!r}) is not a registry robot",
                        "use a name from list_robots()",
                        node.lineno,
                    )
            for keyword in node.keywords:
                if keyword.arg == "policy_provider" and isinstance(keyword.value, ast.Constant):
                    self._check_provider(fence, node, keyword.value.value, "policy_provider=")
                if keyword.arg == "provider" and callee == "train_policy" and isinstance(keyword.value, ast.Constant):
                    value = keyword.value.value
                    if isinstance(value, str) and value not in self.s.trainers:
                        self._find(
                            fence,
                            "provider",
                            f"train_policy(provider={value!r}) is not a trainer",
                            f"use one of {sorted(self.s.trainers)}",
                            node.lineno,
                        )
        return bound

    def _check_provider(self, fence: Fence, node: ast.Call, value: Any, label: str) -> None:
        if isinstance(value, str) and value not in self.s.providers and not self.s.is_smart_string(value):
            self._find(
                fence,
                "provider",
                f"{label} {value!r} is not a policy provider or alias",
                f"use one of {sorted(self.s.providers)}",
                node.lineno,
            )

    def _surface_for(self, fence: Fence, ctor: str, call: ast.Call) -> tuple[type, ...]:
        kw = _literal_keywords(call)
        if ctor == "Robot":
            name = _literal_arg(call, 0)
            mode = str(kw.get("mode", "sim"))
            driver = str(kw.get("driver", "auto"))
            if mode not in ("sim", "real", "auto"):
                return ()
            canonical = self.s.resolve_name(name) if isinstance(name, str) else None
            return self.s.robot_surface(canonical, mode, driver)
        if ctor == "create_simulation":
            backend = _literal_arg(call, 0) or kw.get("backend", "mujoco")
            return (self.s.backend_class(str(backend)),)
        if ctor == "create_policy":
            return (self.s.policy,)
        if ctor == "create_trainer":
            provider = _literal_arg(call, 0) or kw.get("provider") or kw.get("name")
            if isinstance(provider, str):
                try:
                    return (self.s.import_trainer_class(provider),)
                except Exception:  # noqa: BLE001 - an unknown or uninstallable provider falls back to the base
                    return (self.s.trainer,)
            return (self.s.trainer,)
        if ctor == "Agent":
            return (self.s.agent,) if self.s.agent is not None else ()
        if ctor == "Mesh":
            return (self.s.mesh,)
        return ()

    def _check_robot_call(self, fence: Fence, call: ast.Call) -> None:
        kw = _literal_keywords(call)
        name = _literal_arg(call, 0)
        line = call.lineno
        if isinstance(name, str):
            canonical = self.s.resolve_name(name)
            if (
                canonical not in self.s.robots
                and "urdf_path" not in kw
                and not any(k.arg == "urdf_path" for k in call.keywords)
            ):
                self._find(
                    fence, "robot", f"Robot({name!r}) is not a registry robot", "use a name from list_robots()", line
                )
        mode = kw.get("mode", "sim")
        if isinstance(mode, str) and mode not in ("sim", "real", "auto"):
            self._find(fence, "robot", f"mode={mode!r}", "use sim, real or auto", line)
            return
        driver = kw.get("driver", "auto")
        if isinstance(driver, str) and driver not in self.s.driver_choices:
            self._find(fence, "robot", f"driver={driver!r}", f"use one of {sorted(self.s.driver_choices)}", line)
        backend = kw.get("backend", "mujoco")
        if isinstance(backend, str) and backend not in self.s.backends:
            self._find(fence, "robot", f"backend={backend!r}", f"use one of {sorted(self.s.backends)}", line)
        if mode == "sim" and "cameras" in kw:
            self._find(fence, "robot", "cameras= in mode='sim'", "add cameras with add_camera after creation", line)
        allowed = self.s.robot_keywords(
            self.s.resolve_name(name) if isinstance(name, str) else None,
            str(mode) if isinstance(mode, str) else "auto",
            str(driver) if isinstance(driver, str) else "auto",
        )
        for keyword in call.keywords:
            if keyword.arg and keyword.arg not in allowed:
                self._find(
                    fence,
                    "robot",
                    f"Robot(..., {keyword.arg}=) is not a parameter in mode={mode!r}",
                    "use a keyword of Robot() or of the class it builds",
                    line,
                )

    def _check_attributes(self, fence: Fence, tree: ast.AST, bound: dict[str, tuple[type, ...]]) -> None:
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in bound):
                continue
            surfaces = bound[node.value.id]
            owners = [cls for cls in surfaces if _has_attribute(cls, node.attr)]
            if not owners:
                self._find(
                    fence,
                    "attribute",
                    f"{node.value.id}.{node.attr} does not exist on {' or '.join(c.__name__ for c in surfaces)}",
                    "call a method the object has (dir() lists them)",
                    node.lineno,
                )
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            target = node.func.value
            if not (isinstance(target, ast.Name) and target.id in bound):
                continue
            owners = [cls for cls in bound[target.id] if hasattr(cls, node.func.attr)]
            if not owners:
                continue
            self._check_call_keywords(fence, node, owners, f"{target.id}.{node.func.attr}")

    def _check_call_keywords(self, fence: Fence, call: ast.Call, owners: list[type], label: str) -> None:
        """Keywords of a call against every owner's signature; a finding only when no owner takes them."""
        given = [k.arg for k in call.keywords if k.arg]
        if not given:
            return
        verdicts: list[set[str]] = []
        for owner in owners:
            attr = inspect.getattr_static(owner, label.split(".")[-1], None)
            func = attr.__func__ if isinstance(attr, (staticmethod, classmethod)) else attr
            if not inspect.isfunction(func):
                return  # a property, a descriptor or a tool object: nothing to check statically
            try:
                params = inspect.signature(func).parameters
            except (TypeError, ValueError):
                return
            if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
                return
            verdicts.append({name for name in given if name not in params})
        unknown = set.intersection(*verdicts) if verdicts else set()
        for name in sorted(unknown):
            self._find(
                fence,
                "keyword",
                f"{label}({name}=) is not a parameter",
                "use a keyword the method declares",
                call.lineno,
            )

    def _check_actions(
        self, fence: Fence, tree: ast.AST, names: dict[str, Any], bound: dict[str, tuple[type, ...]]
    ) -> None:
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            action = next(
                (k.value.value for k in node.keywords if k.arg == "action" and isinstance(k.value, ast.Constant)), None
            )
            if not isinstance(action, str):
                continue
            func = node.func
            allowed: set[str] | None = None
            label = ast.unparse(func)
            if isinstance(func, ast.Name) and func.id in bound:
                allowed = set()
                for cls in bound[func.id]:
                    allowed |= self.s.sim_actions if cls is self.s.sim_engine else self.s.real_actions
            elif isinstance(func, ast.Attribute) and label.startswith("agent.tool.") and label.endswith("_sim"):
                allowed = self.s.sim_actions
            elif isinstance(func, ast.Name):
                tool = names.get(func.id)
                if tool is None:
                    try:
                        tool = getattr(importlib.import_module("strands_robots"), func.id)
                    except AttributeError:
                        tool = None
                spec = getattr(tool, "tool_spec", None)
                if isinstance(spec, dict):
                    prop = spec.get("inputSchema", {}).get("json", {}).get("properties", {}).get("action", {})
                    enum = prop.get("enum")
                    text = f"{spec.get('description', '')}\n{prop.get('description', '')}"
                    if enum:
                        allowed = set(enum)
                    elif re.search(rf"(?<![\w-]){re.escape(action)}(?![\w-])", text):
                        continue
                    else:
                        self._find(
                            fence,
                            "action",
                            f"{label}(action={action!r}) is not an action the tool's description names",
                            "use an action from the tool's Actions list",
                            node.lineno,
                        )
                        continue
            if allowed is not None and action not in allowed:
                self._find(
                    fence,
                    "action",
                    f"{label}(action={action!r}) is not published",
                    f"use one of {sorted(allowed)[:8]}...",
                    node.lineno,
                )

    def _record_hub_ids(self, fence: Fence, tree: ast.AST) -> None:
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for keyword in node.keywords:
                if keyword.arg in HUB_KEYWORDS and isinstance(keyword.value, ast.Constant):
                    value = keyword.value.value
                    if isinstance(value, str) and HUB_ID.match(value) and not value.startswith((".", "/", "~")):
                        self.hub_ids.setdefault(value, set()).add(f"{fence.rel} #{fence.index}")
                        if self.online and not _hub_exists(value):
                            self._find(
                                fence,
                                "hub",
                                f"{value} is not on huggingface.co",
                                "name a checkpoint that exists",
                                node.lineno,
                            )

    # ----- bash -----------------------------------------------------------

    def _check_bash(self, fence: Fence) -> None:
        lines = _join_continuations(fence.body)
        for offset, raw in lines:
            line = raw.split(" #", 1)[0].strip()
            if not line or line.startswith("#"):
                continue
            for match in EXTRA_SPEC.finditer(line):
                for extra in match.group(1).split(","):
                    if extra.strip() not in self.s.extras:
                        self._find(
                            fence,
                            "extra",
                            f"strands-robots[{extra.strip()}] is not an extra",
                            f"use one of {sorted(self.s.extras)}",
                            offset,
                        )
            if re.search(r"\bpip install\b", line):
                self._check_pip(fence, line, offset)
            cli = CLI_LINE.search(line)
            if cli:
                self._check_cli(fence, cli.group(1), cli.group(2), offset)

    def _check_pip(self, fence: Fence, line: str, offset: int) -> None:
        try:
            tokens = shlex.split(line.split("pip install", 1)[1])
        except ValueError:
            return
        skip_next = False
        for token in tokens:
            if skip_next:
                skip_next = False
                continue
            if token in ("--extra-index-url", "--index-url", "--python", "-r", "--requirement", "-c", "--constraint"):
                skip_next = True
                continue
            if (
                token.startswith("-")
                or token.startswith((".", "/", "~", "$"))
                or "*" in token
                or token.endswith(".whl")
            ):
                continue
            name = re.split(r"[\[<>=!~;@ ]", token, maxsplit=1)[0]
            if not name:
                continue
            normalised = re.sub(r"[-_.]+", "-", name).lower()
            self.pip_names.setdefault(normalised, set()).add(f"{fence.rel} #{fence.index}")
            if _distribution_installed(normalised):
                continue
            if self.online and not _pypi_exists(normalised):
                self._find(fence, "pip", f"{name} is not on PyPI", "spell the distribution as PyPI has it", offset)
            elif not self.online:
                self.skipped.append(f"{fence.rel} #{fence.index}: pip name {name} is not installed here (use --online)")

    def _check_cli(self, fence: Fence, command: str, rest: str, offset: int) -> None:
        if command in ("--help", "-h", "--version", "-V"):
            return
        parser = self.s.cli_commands.get(command)
        if parser is None:
            self._find(
                fence,
                "cli",
                f"strands-robots {command} is not a command",
                f"use one of {sorted(self.s.cli_commands)}",
                offset,
            )
            return
        rest = rest.split("#", 1)[0]
        if "..." in rest or "<" in rest:
            return
        try:
            tokens = shlex.split(rest)
        except ValueError:
            return
        sink = io.StringIO()
        try:
            with redirect_stderr(sink), redirect_stdout(sink):
                parser.parse_args(tokens)
        except SystemExit as exc:
            if exc.code not in (0, None):
                reason = sink.getvalue().strip().splitlines()[-1:] or ["unknown"]
                self._find(
                    fence,
                    "cli",
                    f"strands-robots {command} {rest.strip()}: {reason[0]}",
                    "match the parser's flags (--help)",
                    offset,
                )
        except Exception as exc:  # noqa: BLE001 - argparse types raise on a bad literal; that is the finding
            self._find(
                fence,
                "cli",
                f"strands-robots {command} {rest.strip()}: {exc}",
                "match the parser's flags (--help)",
                offset,
            )


# ----- helpers --------------------------------------------------------------


_INSTANCE_ATTRS: dict[type, set[str]] = {}


def _instance_attributes(cls: type) -> set[str]:
    """Every ``self.<name> = ...`` in the source of ``cls`` and its bases."""
    if cls in _INSTANCE_ATTRS:
        return _INSTANCE_ATTRS[cls]
    names: set[str] = set()
    for klass in getattr(cls, "__mro__", (cls,)):
        try:
            tree = ast.parse(inspect.getsource(klass))
        except (OSError, TypeError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.ctx, ast.Store)
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"
            ):
                names.add(node.attr)
    _INSTANCE_ATTRS[cls] = names
    return names


def _has_attribute(cls: type, name: str) -> bool:
    """True when the class, an instance of it, or the factory supplies ``name``."""
    return hasattr(cls, name) or name in FACTORY_ATTACHED or name in _instance_attributes(cls)


def _literal_keywords(call: ast.Call) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            continue
        try:
            out[keyword.arg] = ast.literal_eval(keyword.value)
        except (ValueError, SyntaxError):
            out[keyword.arg] = None
    return out


def _literal_arg(call: ast.Call, index: int) -> Any:
    if len(call.args) <= index:
        return None
    try:
        return ast.literal_eval(call.args[index])
    except (ValueError, SyntaxError):
        return None


def _join_continuations(body: str) -> list[tuple[int, str]]:
    """(line offset, logical line) with backslash continuations joined."""
    out: list[tuple[int, str]] = []
    pending = ""
    start = 0
    for number, line in enumerate(body.splitlines(), start=1):
        if not pending:
            start = number
        if line.rstrip().endswith("\\"):
            pending += line.rstrip()[:-1] + " "
            continue
        out.append((start, pending + line))
        pending = ""
    if pending:
        out.append((start, pending))
    return out


def _distribution_installed(normalised: str) -> bool:
    try:
        importlib.metadata.distribution(normalised)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def _head(url: str) -> bool:
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310 - https only, fixed hosts
            return bool(200 <= response.status < 400)
    except Exception:  # noqa: BLE001 - any failure is "not reachable"
        return False


def _hub_exists(repo_id: str) -> bool:
    return _head(f"https://huggingface.co/api/models/{repo_id}") or _head(
        f"https://huggingface.co/api/datasets/{repo_id}"
    )


def _pypi_exists(name: str) -> bool:
    return _head(f"https://pypi.org/pypi/{name}/json")


def check(pages: list[Path] | None = None, online: bool = False) -> Checker:
    """Check ``pages`` (default: every page under docs/) and return the checker with its findings."""
    surfaces = Surfaces()
    checker = Checker(surfaces, online=online)
    for page in pages or sorted(p for p in DOCS.rglob("*.md") if "hooks" not in p.parts):
        checker.check_page(page)
    return checker


def main(argv: list[str] | None = None) -> int:
    """Run the checks and print one row per finding; return the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--only", action="append", default=[], help="page path under docs/ (repeatable)")
    parser.add_argument("--online", action="store_true", help="HEAD every Hub id and unknown pip name")
    parser.add_argument("--skipped", action="store_true", help="also list what could not be checked")
    parser.add_argument("--hub-ids", action="store_true", help="list every Hub id the fences name")
    args = parser.parse_args(argv)

    pages = sorted(p for p in DOCS.rglob("*.md") if "hooks" not in p.parts)
    if args.only:
        wanted = {str((DOCS / o).resolve()) for o in args.only}
        pages = [p for p in pages if str(p.resolve()) in wanted]
    sys.path.insert(0, str(REPO))
    checker = check(pages, online=args.online)
    for finding in checker.findings:
        print(finding.row())
    if args.skipped:
        for line in checker.skipped:
            print(f"skipped  {line}")
    if args.hub_ids:
        for hub_id, where in sorted(checker.hub_ids.items()):
            print(f"hub      {hub_id}  ({len(where)} fences)")
    print()
    print(
        f"{checker.fences_checked} fences on {len(pages)} pages: {len(checker.findings)} findings, "
        f"{len(checker.skipped)} skipped, {len(checker.hub_ids)} Hub ids, {len(checker.pip_names)} pip names"
    )
    return 1 if checker.findings else 0


if __name__ == "__main__":
    sys.exit(main())
