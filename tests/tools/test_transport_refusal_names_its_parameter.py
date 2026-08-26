"""A refusal that takes a parameter name spends it on the message.

Eleven helpers in this package document a ``param`` argument with one
sentence - "The parameter name it came from, used in the message" - and
interpolate it, so the caller is told which of their own parameters was
refused. ``use_rosbridge._transport_port_error`` made the same promise and
hard-coded the word ``port`` instead, which is only correct while every
caller happens to name its parameter ``port``.

That is not a hypothetical constraint on a private helper. Its own docstring
says it is *shared* with
:class:`~strands_robots.mesh.rosbridge_robot.RosbridgeRobot` precisely so the
tool and the bridge "cannot disagree about which ports it can carry" - so a
third caller is anticipated by design, and a caller whose parameter is spelled
anything else would be handed a refusal naming a parameter it does not have,
one line after the shared 16-bit domain named the parameter correctly.

The population is derived from the promise rather than listed, so a twelfth
helper that copies the sentence is held to it the moment it lands.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import re
from typing import Any

import pytest

import strands_robots.tools.use_rosbridge as ur
from strands_robots.utils import tcp_port_error

# The sentence that constitutes the promise. Any helper whose docstring carries
# it has told its callers the message names their parameter.
PROMISE = "The parameter name it came from, used in the message"

PACKAGE_ROOT = pathlib.Path(ur.__file__).resolve().parent.parent

# One below the shared 16-bit ceiling: the highest port this transport carries.
ADDRESSABLE_PORT = 65534
# A legal TCP port the transport cannot address - what drives the refusal.
UNADDRESSABLE_PORT = 65535


def _is_declaration_only(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """True when ``fn`` has no body beyond its docstring.

    A ``TYPE_CHECKING`` protocol declaration states a signature for a checker
    and never builds a message, so it cannot honor - or break - the promise.
    """
    body = [s for s in fn.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
    return not body


def _reads_param(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """True when ``fn``'s body reads its ``param`` argument at least once."""
    for statement in fn.body:
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
            continue
        for node in ast.walk(statement):
            if isinstance(node, ast.Name) and node.id == "param":
                return True
    return False


def _promising_helpers() -> list[tuple[str, str, bool]]:
    """Every implemented helper whose docstring carries :data:`PROMISE`.

    Returns:
        ``(module_path, function_name, reads_param)`` triples.
    """
    found: list[tuple[str, str, bool]] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - a syntax error is another test's problem
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            doc = ast.get_docstring(node) or ""
            if PROMISE not in doc or _is_declaration_only(node):
                continue
            names = [a.arg for a in node.args.posonlyargs + node.args.args + node.args.kwonlyargs]
            if "param" not in names:
                continue
            found.append((str(path.relative_to(PACKAGE_ROOT)), node.name, _reads_param(node)))
    return found


PROMISING = _promising_helpers()


class TestTheRefusalNamesTheParameterItWasGiven:
    """The regression: a caller's own parameter name reaches the message."""

    @pytest.mark.parametrize("param", ["bridge_port", "ws_port", "rosbridge_port"])
    def test_a_differently_spelled_parameter_is_named(self, param: str) -> None:
        message = ur._transport_port_error(UNADDRESSABLE_PORT, param, "publish")

        assert message is not None
        assert param in message

    def test_the_message_does_not_name_a_parameter_the_caller_never_passed(self) -> None:
        """Hard-coding one spelling misnames every other one.

        The boundary matters: ``bridge_port 65535`` *contains* the substring
        ``port 65535``, so only a standalone ``port`` token is the wrong name.
        """
        message = ur._transport_port_error(UNADDRESSABLE_PORT, "bridge_port", "publish")

        assert message is not None
        assert re.search(rf"\bport {UNADDRESSABLE_PORT}", message) is None

    def test_both_guards_on_one_parameter_name_the_same_parameter(self) -> None:
        """The shared 16-bit domain runs one line earlier on the same value.

        A caller who trips the wide domain and a caller who trips the
        transport's narrower ceiling are the same caller with the same
        parameter, so the two refusals cannot name different things.
        """
        param = "bridge_port"

        wide = tcp_port_error(70000, param, "publish")
        narrow = ur._transport_port_error(UNADDRESSABLE_PORT, param, "publish")

        assert wide is not None and narrow is not None
        assert param in wide
        assert param in narrow


class TestEveryHelperThatPromisesThisKeepsIt:
    """Derived contract: the promise is a property of the sentence, not a list."""

    def test_the_scan_finds_the_family_rather_than_one_member(self) -> None:
        """Non-vacuity: an empty or single-member scan would prove nothing."""
        assert len(PROMISING) >= 10, PROMISING

    @pytest.mark.parametrize("module_path,name", [(m, n) for m, n, _ in PROMISING])
    def test_the_helper_interpolates_the_parameter_it_documents(self, module_path: str, name: str) -> None:
        reads = next(r for m, n, r in PROMISING if (m, n) == (module_path, name))

        assert reads, f"{module_path}:{name} documents param as used in the message and never reads it"

    def test_the_predicate_can_report_both_outcomes(self) -> None:
        """Grade constructed exemplars, so the rule is not true by absence.

        After the fix the package carries no violator, so the scan alone can
        never exercise its own failing branch.
        """
        keeps = ast.parse(
            "def keeps(value, param, context):\n"
            f'    """Doc.\n\n    Args:\n        param: {PROMISE}.\n    """\n'
            '    return f"{context}: invalid {param}"\n'
        ).body[0]
        breaks = ast.parse(
            "def breaks(value, param, context):\n"
            f'    """Doc.\n\n    Args:\n        param: {PROMISE}.\n    """\n'
            '    return f"{context}: invalid port"\n'
        ).body[0]
        assert isinstance(keeps, ast.FunctionDef) and isinstance(breaks, ast.FunctionDef)

        outcomes = {_reads_param(keeps), _reads_param(breaks)}

        assert outcomes == {True, False}

    def test_a_declaration_only_stub_is_not_held_to_the_promise(self) -> None:
        """A protocol signature builds no message, so it cannot break one."""
        stub = ast.parse('def f(value, param, context):\n    """Doc."""\n').body[0]
        live = ast.parse("def f(value, param, context):\n    return param\n").body[0]
        assert isinstance(stub, ast.FunctionDef) and isinstance(live, ast.FunctionDef)

        assert _is_declaration_only(stub)
        assert not _is_declaration_only(live)


class TestTheShippingCallersSeeNoChange:
    """Both call sites pass ``"port"``, so their refusal text is untouched."""

    @pytest.mark.parametrize("context", ["publish", "service_call", "RosbridgeRobot"])
    def test_the_port_spelling_still_reads_exactly_as_before(self, context: str) -> None:
        message = ur._transport_port_error(UNADDRESSABLE_PORT, "port", context)

        assert message == (
            f"{context}: port {UNADDRESSABLE_PORT!r} is a legal TCP port that the rosbridge "
            f"WebSocket transport cannot address (it addresses 1-{ur._TRANSPORT_MAX_PORT}; "
            "autobahn's URL builder excludes the top of the range)"
        )

    def test_every_shipping_call_site_passes_the_literal_port(self) -> None:
        """Why the change is text-identical in production, read off the source."""
        spellings = set()
        for module in (ur, __import__("strands_robots.mesh.rosbridge_robot", fromlist=["x"])):
            tree = ast.parse(inspect.getsource(module))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "_transport_port_error"
                    and len(node.args) >= 2
                    and isinstance(node.args[1], ast.Constant)
                ):
                    spellings.add(node.args[1].value)

        assert spellings == {"port"}


class TestTheDomainIsUnchanged:
    """Over-reach guard: naming the parameter must not move the boundary."""

    @pytest.mark.parametrize("port", [1, 8080, ADDRESSABLE_PORT])
    def test_an_addressable_port_is_still_accepted(self, port: int) -> None:
        assert ur._transport_port_error(port, "port", "ctx") is None

    def test_the_unaddressable_port_is_still_refused(self) -> None:
        assert ur._transport_port_error(UNADDRESSABLE_PORT, "port", "ctx") is not None

    def test_the_shared_owner_still_carries_the_whole_port_space(self) -> None:
        assert tcp_port_error(UNADDRESSABLE_PORT, "port", "ctx") is None


def test_the_helper_still_returns_none_for_a_usable_port_under_any_spelling() -> None:
    """The parameter name is diagnostic only - it cannot change the verdict."""
    verdicts: set[Any] = {ur._transport_port_error(ADDRESSABLE_PORT, p, "ctx") for p in ("port", "bridge_port", "x")}

    assert verdicts == {None}
