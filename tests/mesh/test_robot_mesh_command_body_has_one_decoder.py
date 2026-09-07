"""A malformed ``robot_mesh`` command body is refused through the tool's envelope.

``robot_mesh`` takes ``command`` as free-text JSON an agent writes, and four
sites used to decode it inline behind ``except json.JSONDecodeError``. That
subclass covers malformed syntax and nothing else, while :func:`json.loads`
fails on a caller string three ways:

* malformed syntax -> :class:`json.JSONDecodeError`,
* a number wider than :func:`sys.get_int_max_str_digits` -> a plain
  :class:`ValueError` (well-formed JSON this build cannot construct an ``int``
  for; RFC 8259 bounds no range),
* a document nested past the interpreter stack -> :class:`RecursionError`, a
  :class:`RuntimeError` outside :class:`ValueError` entirely.

So two of the three escaped. On the two pre-pass sites inside :func:`robot_mesh`
the escape is an exception raised past the tool's own ``{"status": "error"}``
envelope, and no audit row for a ``broadcast`` or ``send`` that never reached
the wire -- while every refusal beside them writes one, including the
same-function ones for a missing body and a body that is not an object. On the
two Device Connect sites the dispatcher's outermost ``except Exception`` absorbs
it and reports a caller's malformed body as ``"Device Connect error"``, naming a
transport that was never dialled.

Why the suite did not catch it: every existing cell drives ``"not json"`` or a
non-object body (``tests/test_robot_mesh_dc_dispatch_errors.py``,
``tests/mesh/test_robot_mesh_command_validation.py``), which are exactly the two
spellings the narrow handler gets right. Those cells hold unchanged here and are
the controls below.

The wording rule is the one the ACL loader already applies: a document that is
nested too deeply, or that carries a number too wide, is *valid JSON* this build
cannot read, so reusing "is not valid JSON" would send the operator hunting a
syntax error that is not there.
"""

from __future__ import annotations

import ast
import inspect
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

import strands_robots.tools.robot_mesh as rmt

#: Nesting depth that exhausts the scanner's stack. 60k levels of ``[`` is
#: 120 KB of text -- depth, not size, is what it runs out of room for.
_DEEP = 60_000

#: A number literal one digit past what ``int()`` will build from a string.
_WIDE = "9" * (sys.get_int_max_str_digits() + 100)

#: The two spellings that used to escape, and the two that never did. Each entry
#: is ``(id, command text, must-not-be-called-valid-JSON)``.
_ESCAPED = [
    ("a number too wide to build", '{"action": ' + _WIDE + "}", True),
    ("a bare number too wide", _WIDE, True),
    ("nested past the stack", "[" * _DEEP + "]" * _DEEP, True),
]
_ALREADY_REFUSED = [
    ("malformed syntax", "{oops", False),
    ("a well-formed non-object", "[1, 2]", False),
]
_ALL_BODIES = _ESCAPED + _ALREADY_REFUSED


def _tool() -> Any:
    """The undecorated ``robot_mesh`` function."""
    return getattr(rmt.robot_mesh, "original", rmt.robot_mesh)


def _text(result: dict[str, Any]) -> str:
    return "\n".join(item.get("text", "") for item in result.get("content", []) if "text" in item)


def _call_tool(monkeypatch: pytest.MonkeyPatch, action: str, command: str) -> tuple[dict[str, Any], list[Any]]:
    """Drive the tool with the HITL gate open, recording audit rows.

    Returns the tool result and the audit rows written during the call. The gate
    is opened because the pre-pass under test runs *before* it; an escape there
    is reached whether or not an operator would be asked.
    """
    monkeypatch.setenv("STRANDS_MESH_HITL_ACTIONS", "none")
    rows: list[Any] = []
    monkeypatch.setattr(rmt, "_audit_tool_action", lambda *a: rows.append(a))
    rmt._reset_rate_limits()
    return _tool()(action=action, target="peer-b", command=command), rows


class _StubConnection:
    """Agent-side Device Connect connection that records nothing and dials nothing."""

    def list_devices(self, device_type: str | None = None) -> list[dict[str, Any]]:
        return [{"device_id": "dev-1"}]

    def invoke(
        self, device_id: str, function: str, params: dict[str, Any] | None = None, timeout: float = 30.0
    ) -> dict[str, Any]:
        raise AssertionError("a malformed command body must never reach invoke")


def _call_dispatch(action: str, command: str, **extra: Any) -> dict[str, Any]:
    """Drive ``_device_connect_dispatch`` against a stub connection."""
    with patch("device_connect_agent_tools.connection.get_connection", return_value=_StubConnection()):
        result = rmt._device_connect_dispatch(
            action, "dev-1", "", command, "mock", 0, 30.0, 5.0, extra.get("function", ""), None
        )
    assert result is not None, f"{action} unexpectedly deferred to the mesh path"
    return result


class TestJsonLoadsFailsThreeWaysOnACallerString:
    """Premise: the three failure modes are real, and two are not ``JSONDecodeError``.

    Holds on either side of the fix -- it measures :mod:`json`, not this tool.
    If a future interpreter collapses these into one class, this is what says so
    before the rest of the file starts passing for the wrong reason.
    """

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("{oops", json.JSONDecodeError),
            (_WIDE, ValueError),
            ("[" * _DEEP + "]" * _DEEP, RecursionError),
        ],
        ids=["syntax", "too-wide", "too-deep"],
    )
    def test_the_class_json_raises(self, text: str, expected: type[BaseException]) -> None:
        with pytest.raises(expected) as excinfo:
            json.loads(text)
        if expected is ValueError:
            assert not isinstance(excinfo.value, json.JSONDecodeError), (
                "a too-wide number must raise a plain ValueError, or the narrow handler already covered it"
            )

    def test_a_recursion_error_is_not_a_value_error(self) -> None:
        # This is why widening to ``except ValueError`` alone would not do:
        # the deepest input is outside that class entirely.
        assert not issubclass(RecursionError, ValueError)
        assert issubclass(RecursionError, RuntimeError)


class TestEveryMalformedBodyReachesTheToolEnvelope:
    """Every unreadable ``command`` is reported through the envelope, with an audit row.

    Pre-fix the three escaping spellings raise out of the tool and write no row,
    on both gated fleet actions.
    """

    @pytest.mark.parametrize("action", ["broadcast", "send"])
    @pytest.mark.parametrize(("label", "command", "_novalid"), _ALL_BODIES, ids=[c[0] for c in _ALL_BODIES])
    def test_the_tool_returns_an_error_envelope(
        self, monkeypatch: pytest.MonkeyPatch, action: str, label: str, command: str, _novalid: bool
    ) -> None:
        result, _rows = _call_tool(monkeypatch, action, command)
        assert isinstance(result, dict), f"{action} with {label} did not return a tool result"
        assert result["status"] == "error", f"{action} with {label} reported {result['status']}"
        assert "command" in _text(result), f"{action} with {label} refused without naming the parameter"

    @pytest.mark.parametrize("action", ["broadcast", "send"])
    @pytest.mark.parametrize(("label", "command", "_novalid"), _ALL_BODIES, ids=[c[0] for c in _ALL_BODIES])
    def test_the_refusal_is_audited(
        self, monkeypatch: pytest.MonkeyPatch, action: str, label: str, command: str, _novalid: bool
    ) -> None:
        # The forensic half: a refused gated action leaves a record. Every other
        # refusal in this pre-pass writes one, so a silent path is the gap.
        _result, rows = _call_tool(monkeypatch, action, command)
        assert len(rows) == 1, f"{action} with {label} wrote {len(rows)} audit rows, expected 1"
        assert rows[0][2] is False, "a refusal must be audited as a failure"


class TestTheDispatcherNamesTheBodyNotTheTransport:
    """A malformed body is the caller's, not a Device Connect failure.

    Pre-fix the dispatcher's outermost handler absorbs the escape and answers
    ``"[send] Device Connect error: Exceeds the limit (4300 digits) ..."`` -- a
    transport it never dialled, for a string it could have judged locally.
    """

    @pytest.mark.parametrize(("label", "command", "_novalid"), _ESCAPED, ids=[c[0] for c in _ESCAPED])
    def test_send_reports_the_command(self, label: str, command: str, _novalid: bool) -> None:
        result = _call_dispatch("send", command)
        assert result["status"] == "error"
        assert "Device Connect error" not in _text(result), f"send with {label} blamed the transport"
        assert "command" in _text(result)

    @pytest.mark.parametrize(("label", "command", "_novalid"), _ESCAPED, ids=[c[0] for c in _ESCAPED])
    def test_rpc_reports_its_own_label(self, label: str, command: str, _novalid: bool) -> None:
        result = _call_dispatch("rpc", command, function="nod")
        assert result["status"] == "error"
        assert "Device Connect error" not in _text(result), f"rpc with {label} blamed the transport"
        assert "rpc params (command)" in _text(result), "the rpc body keeps its own label"


class TestACauseIsNotReportedAsASyntaxError:
    """A body that is valid JSON is never called invalid.

    The rule the ACL loader already applies to the same class of input: a
    document nested too deeply, or carrying a number too wide, parses as JSON
    and is refused for what it holds, so reusing the syntax wording would send
    the operator looking for a syntax error that is not there.
    """

    @pytest.mark.parametrize(("label", "command", "_novalid"), _ESCAPED, ids=[c[0] for c in _ESCAPED])
    def test_the_refusal_does_not_say_the_body_is_invalid_json(
        self, monkeypatch: pytest.MonkeyPatch, label: str, command: str, _novalid: bool
    ) -> None:
        result, _rows = _call_tool(monkeypatch, "broadcast", command)
        assert "is not valid JSON" not in _text(result), f"{label} was reported as a syntax error"

    def test_each_cause_reads_differently(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Three causes, three messages: an operator who cannot tell them apart
        # cannot pick a remedy.
        seen = {
            label: _text(_call_tool(monkeypatch, "broadcast", command)[0])
            for label, command, _ in _ESCAPED + [_ALREADY_REFUSED[0]]
        }
        assert len(set(seen.values())) == len(seen) - 1, f"causes share a message: {seen}"
        assert "nested too deeply" in seen["nested past the stack"]
        assert "too wide" in seen["a number too wide to build"]


class TestTheAlreadyRefusedWordingIsUnchanged:
    """Controls: the two spellings the narrow handler always caught read as before.

    These hold on both sides of the fix. They are what stops the change being a
    reword of the messages the rest of the suite pins.
    """

    def test_a_syntax_error_still_says_not_valid_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        result, _rows = _call_tool(monkeypatch, "broadcast", "{oops")
        assert "command is not valid JSON: " in _text(result)

    def test_a_non_object_still_says_it_must_be_an_object(self, monkeypatch: pytest.MonkeyPatch) -> None:
        result, _rows = _call_tool(monkeypatch, "send", "[1, 2]")
        assert "command must decode to a JSON object (dict)" in _text(result)

    def test_the_dispatcher_keeps_both_wordings(self) -> None:
        assert "not valid JSON" in _text(_call_dispatch("send", "not json"))
        assert "must decode to a JSON object" in _text(_call_dispatch("send", "[1, 2, 3]"))
        assert "rpc params (command) is not valid JSON" in _text(_call_dispatch("rpc", "nope", function="nod"))

    def test_a_readable_body_still_gets_through_the_decoder(self) -> None:
        assert rmt._decoded_command_body('{"action": "status"}', "command") == {"action": "status"}


class TestOneOwnerDecodesEveryCommandBody:
    """No site decodes the ``command`` parameter itself.

    Graded on the call graph rather than the source text: this module's own
    docstring writes ``json.loads(command)`` as prose while describing the
    dispatch table, so a text scan would report the documentation as the defect.
    """

    def test_no_other_function_calls_json_loads_on_the_command_parameter(self) -> None:
        module_path = Path(inspect.getfile(rmt))
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        owner = rmt._decoded_command_body.__name__
        offenders: list[str] = []
        found_in_owner = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            for call in ast.walk(node):
                if not isinstance(call, ast.Call):
                    continue
                if ast.unparse(call.func) not in ("json.loads", "json.load"):
                    continue
                if not any(ast.unparse(a) == "command" for a in call.args):
                    continue
                if node.name == owner:
                    found_in_owner = True
                else:
                    offenders.append(f"{node.name}:{call.lineno}")
        assert found_in_owner, f"this scan found no decode inside {owner}, so it is looking in the wrong place"
        assert offenders == [], f"the command body is decoded outside {owner}: {offenders}"
