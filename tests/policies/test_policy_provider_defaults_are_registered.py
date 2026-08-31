"""A defaulted ``policy_provider`` must name a provider the registry still has.

Removing a provider is a two-sided edit: the entry leaves
``strands_robots/registry/policies.json``, and every *default* that named it has
to move with it. Nothing links the two, and a stale default is invisible until a
caller omits the argument - at which point ``create_policy`` raises about a
provider name the caller never typed, from inside a hardware entry point that has
already connected the arm.

The hardware task entry points are where this bites hardest. They take no
pre-built policy (``start_task``'s ``policy_port`` is documented as required
"this entry point takes no pre-built policy, so the port is the only thing a
policy can be built from"), so the defaulted provider is the whole of what gets
built, and it is reached by exactly the calls that pass the fewest arguments.

Both surfaces are graded, because a default reaches a caller two ways:
the Python signature, and the ``default`` an agent reads out of a tool schema.
Neither is derived from the other - ``Robot.start_task``'s signature default and
its ``tool_spec`` entry are written separately - so a removal can update one and
leave the other.

The population is derived from the tree rather than listed, so a provider-dialing
method added later is held to the same rule on arrival.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any

import pytest

import strands_robots
from strands_robots.registry.policies import list_policy_providers

_PKG_ROOT = Path(strands_robots.__file__).resolve().parent

# A default of None is a sentinel meaning "not supplied", not a provider name.
_NOT_A_PROVIDER_NAME = (None,)


def _registered() -> set[str]:
    return set(list_policy_providers())


def _signature_defaults() -> dict[str, str]:
    """Map ``module::qualname`` to each literal ``policy_provider`` default in the package."""
    found: dict[str, str] = {}
    for path in sorted(_PKG_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        rel = path.relative_to(_PKG_ROOT.parent)
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            args = node.args
            # Defaults align to the tail of positional args; kwonly are paired.
            pairs: list[tuple[ast.arg, ast.expr | None]] = list(
                zip(args.args[len(args.args) - len(args.defaults) :], args.defaults, strict=True)
            )
            pairs += list(zip(args.kwonlyargs, args.kw_defaults, strict=True))
            for arg, default in pairs:
                if arg.arg != "policy_provider" or not isinstance(default, ast.Constant):
                    continue
                if default.value in _NOT_A_PROVIDER_NAME:
                    continue
                found[f"{rel}::{node.name}"] = str(default.value)
    return found


def _schema_defaults() -> dict[str, str]:
    """Map ``module::line`` to each ``policy_provider`` default declared in a tool schema.

    Read from the source rather than from a built ``tool_spec``: the specs are
    properties on classes whose construction touches hardware, and an agent's
    view of the default is the literal in the schema either way.
    """
    found: dict[str, str] = {}
    for path in sorted(_PKG_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        rel = path.relative_to(_PKG_ROOT.parent)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            for key, value in zip(node.keys, node.values, strict=True):
                if not (isinstance(key, ast.Constant) and key.value == "policy_provider"):
                    continue
                if not isinstance(value, ast.Dict):
                    continue
                for inner_key, inner_value in zip(value.keys, value.values, strict=True):
                    if (
                        isinstance(inner_key, ast.Constant)
                        and inner_key.value == "default"
                        and isinstance(inner_value, ast.Constant)
                        and isinstance(inner_value.value, str)
                    ):
                        found[f"{rel}:{inner_value.lineno}"] = inner_value.value
    return found


class TestEveryDefaultedProviderIsRegistered:
    """The rule, over both surfaces a default reaches a caller through."""

    def test_every_signature_default_resolves(self) -> None:
        registered = _registered()
        stale = {site: name for site, name in _signature_defaults().items() if name not in registered}
        assert not stale, (
            f"these policy_provider defaults name a provider the registry no longer has: {stale}. "
            f"Registered: {sorted(registered)}. A caller who omits the argument gets a failure "
            "naming a provider they never typed - point the default at a provider that still "
            "ships, or drop the default so the caller must choose."
        )

    def test_every_tool_schema_default_resolves(self) -> None:
        registered = _registered()
        stale = {site: name for site, name in _schema_defaults().items() if name not in registered}
        assert not stale, (
            f"these tool-schema policy_provider defaults name a provider the registry no longer "
            f"has: {stale}. An agent reads the schema default, so it is a second, independently "
            "written copy of the signature default and goes stale on its own."
        )

    def test_the_hardware_entry_points_default_to_a_provider_that_dials_a_port(self) -> None:
        """The task entry points build the policy from a port, so the default must want one.

        A default naming an in-process provider would ignore the ``policy_port``
        these entry points document as the only thing a policy can be built from.
        """
        from strands_robots.hardware_robot import Robot

        default = inspect.signature(Robot.start_task).parameters["policy_provider"].default
        from strands_robots.registry.policies import get_policy_provider

        entry = get_policy_provider(default)
        assert entry is not None, default
        keys = set(entry.get("config_keys", ())) | set(entry.get("defaults", {}))
        assert "port" in keys, (
            f"start_task defaults to {default!r}, which declares no 'port' - but the method's own "
            "docstring says the port is the only thing a policy can be built from here"
        )


class TestTheGradingIsNotVacuous:
    """A clean result has to mean the defaults were actually read."""

    def test_the_signature_scan_found_defaults(self) -> None:
        assert len(_signature_defaults()) >= 5, _signature_defaults()

    def test_the_schema_scan_found_a_default(self) -> None:
        assert _schema_defaults(), "no tool schema declared a policy_provider default"

    @pytest.mark.parametrize("bogus", ["groot", "definitely_not_a_provider"])
    def test_a_name_the_registry_lacks_is_not_registered(self, bogus: str) -> None:
        """Both a removed provider and a typo must read as absent."""
        assert bogus not in _registered()

    def test_none_is_not_read_as_a_provider_name(self) -> None:
        """``policy_provider: str | None = None`` is a sentinel, not a stale default."""
        tree = ast.parse("def f(policy_provider=None): pass")
        function = tree.body[0]
        assert isinstance(function, ast.FunctionDef)
        default = function.args.defaults[0]
        assert isinstance(default, ast.Constant)
        assert default.value in _NOT_A_PROVIDER_NAME
