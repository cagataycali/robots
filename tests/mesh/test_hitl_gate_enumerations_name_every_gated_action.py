"""Every enumeration of the HITL-gated action set names every action in it.

``robot_mesh`` gates a set of actions behind an out-of-band operator
approval. Two constants own that set: the gateable vocabulary the env var
accepts, and the default subset gated when the operator configures nothing.
Several human-facing surfaces enumerate one of them - the generated
configuration row (``docs/reference/configuration.md``, rendered by
``docs/hooks/env_vars.py``), the fleet page's approval sentence and the agents
page's gate-table row (the successors of the old security guide's gate
bullet), the one-time warning logged when the gate is disabled, the
dispatcher's own comment and docstring, an example's operator note, and the
contract docstring of the suite that pins the resolver. The old audit-trail
bullet has no successor: the audit sections now say "every operator verdict"
rather than listing actions, so nothing there enumerates the set.

An enumeration that omits a member of the set it claims to list is a
security defect rather than a wording slip, for two measured reasons. An
operator narrowing the gate can only write tokens they have been shown, so
an omitted token is dropped from the gate silently - a subset naming the
documented actions parses cleanly, and no refusal fires to say a further
action was ungated too. And the warning logged when the gate is disabled is
the operator's only notice of what is now unguarded; a name missing there is
a re-opened surface that is never reported.

The required set is derived from the constants rather than restated here, so
an action added to the gate later fails these tests until every enumeration
names it.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

import strands_robots.tools.robot_mesh as rmt
from tests._package_ast import parse_file

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MESH_TOOL_SOURCE = Path(inspect.getsourcefile(rmt) or "")
_CONFIGURATION = _REPO_ROOT / "docs" / "reference" / "configuration.md"
_ENV_VARS_HOOK = _REPO_ROOT / "docs" / "hooks" / "env_vars.py"
_CONFIGURATION_ANCHOR = "| <code>STRANDS_MESH_HITL_ACTIONS</code> |"

# Cap on how far an enumeration may wrap past its anchor line. Wide enough
# for the warning's three string fragments, narrow enough that a neighbouring
# bullet cannot satisfy an anchor whose own list is incomplete.
_MAX_WRAPPED_LINES = 4


@dataclass(frozen=True)
class _Enumeration:
    """One human-facing surface that lists the gated actions.

    Attributes:
        label: How the surface is named in a failure message.
        path: File holding it.
        anchor: Unique substring on the line the enumeration starts on.
        required: Action names the surface has to name.
    """

    label: str
    path: Path
    anchor: str
    required: frozenset[str]


def _enumerations() -> tuple[_Enumeration, ...]:
    gateable = frozenset(rmt._GATEABLE_ACTIONS)
    default = frozenset(rmt._DEFAULT_INTERRUPT_ACTIONS)
    return (
        # The env var's whole accepted vocabulary: this row is what an
        # operator writing a subset copies from.
        _Enumeration(
            "the configuration.md row",
            _CONFIGURATION,
            _CONFIGURATION_ANCHOR,
            gateable,
        ),
        _Enumeration(
            "the fleet page's approval sentence",
            _REPO_ROOT / "docs" / "learn" / "mesh" / "fleet.md",
            "pause for operator approval by default",
            default,
        ),
        _Enumeration(
            "the agents page's gate-table row",
            _REPO_ROOT / "docs" / "learn" / "agents.md",
            "| `robot_mesh` | `emergency_stop`",
            default,
        ),
        _Enumeration(
            "the warning logged when the gate is disabled",
            _MESH_TOOL_SOURCE,
            "approval is DISABLED for all mesh actions",
            default,
        ),
        _Enumeration(
            "the dispatcher's interrupt-set comment",
            _MESH_TOOL_SOURCE,
            "default gates every physical-actuation action",
            default,
        ),
        _Enumeration(
            "the tool docstring's audit list",
            _MESH_TOOL_SOURCE,
            "* Every ``tell`` / ``send``",
            default,
        ),
        _Enumeration(
            "the hub_to_hardware example's operator note",
            _REPO_ROOT / "examples" / "lerobot" / "hub_to_hardware.py",
            "routes every physically-actuating",
            default,
        ),
        _Enumeration(
            "the HITL config suite's contract docstring",
            _REPO_ROOT / "tests" / "mesh" / "test_robot_mesh_hitl_config.py",
            "The default interrupt set gates every physical-actuation action",
            default,
        ),
    )


def _rendered(path: Path) -> str:
    """The file's text; the configuration page with ``{{env_vars}}`` expanded by its hook."""
    text = path.read_text(encoding="utf-8")
    if path != _CONFIGURATION:
        return text
    spec = importlib.util.spec_from_file_location("docs_env_vars_hook_hitl", _ENV_VARS_HOOK)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert "{{env_vars}}" in text, "docs/reference/configuration.md must carry the {{env_vars}} token"
    rendered = module.on_page_markdown(text, page=None, config=None, files=None)
    assert "{{env_vars}}" not in rendered, "the env_vars hook left the token unexpanded"
    return rendered


def _anchored_paragraph(path: Path, anchor: str) -> str | None:
    """Return the enumeration anchored at *anchor*, following its wrapping.

    A markdown bullet or table row holds its whole list on one line; a
    wrapped comment, docstring item or string literal continues onto the
    next. Extending until a blank line, a new bullet or a new table row
    covers both without letting the next list answer for this one.
    """
    lines = _rendered(path).splitlines()
    start = next((i for i, line in enumerate(lines) if anchor in line), None)
    if start is None:
        return None
    block = [lines[start]]
    for line in lines[start + 1 : start + _MAX_WRAPPED_LINES]:
        stripped = line.strip()
        if not stripped or stripped.startswith(("- ", "* ", "|")):
            break
        block.append(line)
    return " ".join(" ".join(block).split())


@pytest.mark.parametrize("enumeration", _enumerations(), ids=lambda e: e.label)
def test_every_enumeration_of_the_gate_names_every_action_it_gates(enumeration: _Enumeration) -> None:
    text = _anchored_paragraph(enumeration.path, enumeration.anchor)
    assert text is not None, f"premise: {enumeration.label} no longer holds {enumeration.anchor!r}"
    missing = sorted(action for action in enumeration.required if action not in text)
    assert not missing, (
        f"{enumeration.label} ({enumeration.path.relative_to(_REPO_ROOT)}) lists the gated actions "
        f"but omits {missing}: {text!r}. An operator can only narrow the gate to actions they have "
        f"been shown, and an omitted action is ungated without a refusal."
    )


def test_the_documented_vocabulary_can_express_the_shipped_default() -> None:
    """The README's subset tokens must be able to spell the default gate.

    An operator keeping the default minus one action writes the rest of the
    default from that row. A default action absent from the row is dropped
    from every such subset, silently.
    """
    row = _anchored_paragraph(_CONFIGURATION, _CONFIGURATION_ANCHOR)
    assert row is not None, "premise: docs/reference/configuration.md no longer documents STRANDS_MESH_HITL_ACTIONS"
    unspellable = sorted(action for action in rmt._DEFAULT_INTERRUPT_ACTIONS if action not in row)
    assert not unspellable, (
        f"the documented vocabulary cannot express the shipped default gate: {unspellable} "
        f"are gated by default but absent from the documented token list: {row!r}"
    )


def test_every_graded_surface_is_still_present() -> None:
    """A reflow that hides an enumeration must fail, not report clean."""
    missing = [
        f"{e.label} ({e.path.name}): {e.anchor!r}"
        for e in _enumerations()
        if _anchored_paragraph(e.path, e.anchor) is None
    ]
    assert not missing, f"anchors no longer found, so these surfaces are ungraded: {missing}"


def test_the_read_only_telemetry_actions_are_not_in_the_default_gate() -> None:
    """``subscribe`` / ``watch`` are gateable but opt-in.

    They have no actuation effect, so widening the default to make an
    enumeration true would gate every read-only observation instead.
    """
    assert {"subscribe", "watch"} <= rmt._GATEABLE_ACTIONS
    assert not {"subscribe", "watch"} & rmt._DEFAULT_INTERRUPT_ACTIONS


def test_rpc_is_a_gated_actuation_action_the_dispatcher_audits() -> None:
    """Premise: ``rpc`` really is gated by default and really is audited."""
    assert "rpc" in rmt._GATEABLE_ACTIONS
    assert "rpc" in rmt._DEFAULT_INTERRUPT_ACTIONS
    tree = parse_file(_MESH_TOOL_SOURCE)
    branches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and "action == 'rpc'" in ast.unparse(node.test).replace('"', "'")
    ]
    assert branches, 'premise: no `action == "rpc"` dispatch branch to audit'
    audited = any(
        isinstance(call.func, ast.Name) and call.func.id == "_audit_tool_action"
        for branch in branches
        for call in ast.walk(branch)
        if isinstance(call, ast.Call)
    )
    assert audited, "premise: the rpc branch does not reach _audit_tool_action"


def test_the_device_connect_guides_already_name_every_gated_action() -> None:
    """The Device Connect docs are the surfaces that got this right.

    ``rpc`` arrived with Device Connect, whose own guide names all six
    actuation actions twice; the shared mesh surfaces did not follow. Keeping
    them graded stops that asymmetry reappearing from the other side. The old
    ``docs/reference/device-connect.md`` page redirects to the generated API
    reference, so the package's ``GUIDE.md`` carries both enumerations now.
    """
    for path, anchor in (
        (_REPO_ROOT / "strands_robots" / "device_connect" / "GUIDE.md", "the actuation actions"),
        (_REPO_ROOT / "strands_robots" / "device_connect" / "GUIDE.md", "is gated"),
    ):
        text = _anchored_paragraph(path, anchor)
        assert text is not None, f"premise: {path.name} no longer holds {anchor!r}"
        missing = sorted(action for action in rmt._DEFAULT_INTERRUPT_ACTIONS if action not in text)
        assert not missing, f"{path.name} omits {missing}: {text!r}"
