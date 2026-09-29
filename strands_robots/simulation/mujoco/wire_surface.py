"""The simulation tool's surface as a mesh peer serves it.

A ``Simulation`` on the mesh is a peer. In process an agent calls any of the
actions ``tool_spec.json`` publishes; over the wire the peer serves the same
tool minus what it is not willing to serve to a caller on another host:
actions that replace or destroy the world its mesh robots live in, actions
that open a window or read a path on the peer host, rollouts that already
ride the ``execute`` / ``start`` / ``stop`` rail with their own allowlists,
and params that name a peer-host path, raw MJCF or an egress switch. Both
tables live in ``wire_surface.json`` next to the spec, one reason per entry,
and are read here and by :mod:`strands_robots.mesh.security` as a file, so the
mesh layer never imports MuJoCo to know them.

The served spec is what the peer advertises: presence carries its
``tool_spec_hash`` and the ``describe_tool`` mesh action returns the spec
itself, so a caller (the dashboard's fleet agent) projects the peer's real
tool rather than a copy kept somewhere else. The authority stays on the peer:
the ``call`` action is validated on the wire for shape and size only and the
peer refuses any function or param outside what :func:`build_wire_tool_spec`
lists for it.

This module imports no engine code; it reads two JSON files and inspects a
class.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_TOOL_SPEC_PATH = _HERE / "tool_spec.json"
_WIRE_SURFACE_PATH = _HERE / "wire_surface.json"

#: Bound on one param description in the served spec (a caller builds a
#: model-facing schema from it; the full sentence stays in ``tool_spec.json``).
MAX_WIRE_PARAM_DESCRIPTION_CHARS: int = 200


@functools.lru_cache(maxsize=1)
def _tool_spec() -> dict[str, Any]:
    with open(_TOOL_SPEC_PATH, encoding="utf-8") as handle:
        spec: dict[str, Any] = json.load(handle)
    return spec


@functools.lru_cache(maxsize=1)
def _wire_surface() -> dict[str, Any]:
    with open(_WIRE_SURFACE_PATH, encoding="utf-8") as handle:
        surface: dict[str, Any] = json.load(handle)
    return surface


def published_actions() -> frozenset[str]:
    """Every action ``tool_spec.json`` publishes."""
    return frozenset(_tool_spec()["properties"]["action"]["enum"])


def published_params() -> frozenset[str]:
    """Every param name ``tool_spec.json`` publishes, ``action`` excluded."""
    return frozenset(_tool_spec()["properties"]) - {"action"}


def denied_actions() -> dict[str, str]:
    """Published actions the peer does not serve over the wire, with the reason for each."""
    return dict(_wire_surface()["denied_actions"])


def rail_for() -> dict[str, str]:
    """The mesh action a denied rollout rides instead, named in the refusal."""
    return dict(_wire_surface()["rail_for"])


def denied_params() -> dict[str, str]:
    """Published params the peer refuses on every served function, with the reason for each."""
    return dict(_wire_surface()["denied_params"])


def served_actions() -> frozenset[str]:
    """The published actions a peer serves over the wire."""
    return published_actions() - frozenset(denied_actions())


def served_params() -> frozenset[str]:
    """The published params a peer accepts over the wire (any function)."""
    return published_params() - frozenset(denied_params())


def refusal_for_action(name: str) -> str | None:
    """The sentence a peer answers for a denied *name*, or ``None`` when it is served."""
    reason = denied_actions().get(name)
    if reason is None:
        return None
    rail = rail_for().get(name)
    tail = f" Use the `{rail}` action on this peer instead." if rail else ""
    return f"{name!r} is not served over the mesh: {reason}.{tail}"


def refusal_for_param(key: str) -> str | None:
    """The sentence a peer answers for a denied param *key*, or ``None`` when it is accepted."""
    reason = denied_params().get(key)
    if reason is None:
        return None
    return f"param {key!r} is not accepted over the mesh: {reason}. Leave it out and the peer's own default applies."


def _function_params(sim_cls: type, action: str) -> tuple[str, ...] | None:
    """The published, served params the method behind *action* declares, in signature order.

    Resolved the way the simulation's own router resolves a name (its alias
    table first) and read off the class, so no engine code runs. ``None`` when
    the class has no such method: a published name the peer cannot answer is
    not served, so the spec never advertises it.
    """
    aliases = getattr(sim_cls, "_ACTION_ALIASES", {}) or {}
    method = getattr(sim_cls, aliases.get(action, action), None)
    if not callable(method):
        return None
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return ()
    accepted = served_params()
    return tuple(name for name in signature.parameters if name in accepted)


def _param_entry(name: str) -> dict[str, Any]:
    prop = _tool_spec()["properties"].get(name, {})
    entry: dict[str, Any] = {}
    if "type" in prop:
        entry["type"] = prop["type"]
    description = prop.get("description")
    if isinstance(description, str) and description:
        cut = description.strip()
        if len(cut) > MAX_WIRE_PARAM_DESCRIPTION_CHARS:
            cut = cut[: MAX_WIRE_PARAM_DESCRIPTION_CHARS - 3].rstrip() + "..."
        entry["description"] = cut
    if "enum" in prop:
        entry["enum"] = list(prop["enum"])
    return entry


def build_wire_tool_spec(tool_name: str, sim_cls: type) -> dict[str, Any]:
    """The tool surface a peer of class *sim_cls* serves over the mesh.

    Shape (canonical, JSON-encodable, hashed by :func:`tool_spec_hash`)::

        {"tool_name": str,
         "functions": {name: {"params": {key: {"type", "description", "enum"?}}}},
         "denied": {name: reason}}

    ``functions`` holds every published action outside the peer-side deny
    table that *sim_cls* has a method for; each lists the served params its
    method declares. ``denied`` is the table itself, so a caller can say why a
    name is missing instead of guessing.
    """
    functions: dict[str, Any] = {}
    for action in sorted(served_actions()):
        declared = _function_params(sim_cls, action)
        if declared is None:
            continue
        functions[action] = {"params": {name: _param_entry(name) for name in declared}}
    return {
        "tool_name": str(tool_name),
        "functions": functions,
        "denied": dict(sorted(denied_actions().items())),
    }


def canonical_json(spec: dict[str, Any]) -> str:
    """The one encoding every peer hashes: sorted keys, no whitespace, UTF-8 kept."""
    return json.dumps(spec, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def tool_spec_hash(spec: dict[str, Any]) -> str:
    """sha256 of :func:`canonical_json`; what presence carries."""
    return hashlib.sha256(canonical_json(spec).encode("utf-8")).hexdigest()


__all__ = [
    "MAX_WIRE_PARAM_DESCRIPTION_CHARS",
    "build_wire_tool_spec",
    "canonical_json",
    "denied_actions",
    "denied_params",
    "published_actions",
    "published_params",
    "rail_for",
    "refusal_for_action",
    "refusal_for_param",
    "served_actions",
    "served_params",
    "tool_spec_hash",
]
