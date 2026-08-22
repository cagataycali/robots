"""Pin: both validate-before-HITL contract guards in ``robot_mesh``.

``robot_mesh`` parses and validates the ``send`` / ``broadcast`` JSON command
body BEFORE raising the human-in-the-loop interrupt, so the operator approves
the validated form rather than the raw model-authored string. Each handler then
re-reads that pre-validated command from a sentinel local
(``validated_send_cmd`` / ``validated_broadcast_cmd``) and refuses outright when
the sentinel is still ``None``: the pre-pass did not run, so the
validate-before-HITL contract is broken and there is no validated command to
dispatch.

Both guards are deliberately an explicit ``raise`` rather than an ``assert``.
``assert`` is stripped under ``python -O`` / ``PYTHONOPTIMIZE=1``, which would
turn the guard into a no-op on exactly the interpreters a production deployment
is most likely to run. :class:`TestTheAssertStrippingPremise` measures that
stripping, so the choice is a pinned property rather than a comment.

Neither guard had a test. Both are second-line: on today's code the pre-pass
either sets the sentinel or returns an error, so nothing reaches a handler with
it unset and a coverage report shows both raise bodies dead. They exist for the
refactor that changes that -- and the measured cost of losing one is not a tidy
failure. With the ``broadcast`` guard deleted and ``validate_command`` returning
``None`` (a validator refactored to validate in place), the handler calls
``mesh.broadcast(None, timeout=30.0)``: a fleet-wide dispatch of a non-command
leaves the tool, and the ``AttributeError`` that follows lands on the audit line
*after* the dispatch, so the dispatch is never recorded. The ``send`` half is the
same against a single targeted peer.

The guards raise rather than returning the tool's structured error envelope on
purpose: an unset sentinel is a broken internal contract, not a malformed agent
tool call, so there is no caller mistake to report and failing loudly is the
fail-closed answer. The structured-error contract for malformed *input* is
pinned separately.

Reaching a guard needs the refactor its comment names, modelled here by pointing
``validate_command`` at a validator that returns ``None`` instead of the
sanitised copy. That leaves the pre-pass succeeding with the sentinel unset --
exactly the state the guard exists to catch.
"""

from __future__ import annotations

import ast
import inspect
import subprocess
import sys
import textwrap
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import strands_robots.tools.robot_mesh as rmt

# The two gated actions whose handler re-reads a pre-validated command body,
# with the transport method each one dispatches through and the call-arg index
# holding the command. ``send`` targets one peer (``send(target, cmd, ...)``);
# ``broadcast`` is fleet-wide (``broadcast(cmd, ...)``).
_PRE_VALIDATED_ACTIONS: list[tuple[str, str, dict[str, Any], int]] = [
    ("send", "send", {"target": "peer-b", "command": '{"action": "status"}'}, 1),
    ("broadcast", "broadcast", {"command": '{"action": "status"}'}, 0),
    ("sim_call", "send", {"target": "sim-peer", "function": "list_objects"}, 1),
]

_ACTION_IDS = [a for a, _, _, _ in _PRE_VALIDATED_ACTIONS]


@pytest.fixture(autouse=True)
def _reset_caches() -> Any:
    """Each gated call consumes a per-action rate-limit slot; reset so the cases
    stay independent of collection order."""
    rmt._reset_rate_limits()
    rmt._reset_interrupt_actions_cache()
    yield
    rmt._reset_rate_limits()
    rmt._reset_interrupt_actions_cache()


def _approving_ctx() -> MagicMock:
    """A ToolContext stand-in whose interrupt returns an operator approval, so
    the gated action reaches its handler rather than stopping at the HITL gate."""
    ctx = MagicMock(name="ToolContext")
    ctx.interrupt.return_value = "y"
    return ctx


def _call(**kwargs: Any) -> Any:
    """Invoke the underlying tool fn (Strands ``@tool`` wraps it as ``.original``)."""
    fn = getattr(rmt.robot_mesh, "original", rmt.robot_mesh)
    return fn(tool_context=_approving_ctx(), **kwargs)


@pytest.fixture
def transport() -> Any:
    """A local mesh stand-in that records what reaches ``send`` / ``broadcast``."""
    mesh = MagicMock(name="LocalMesh")
    mesh.peer_id = "local-a"
    mesh.peer_type = "sim"
    mesh.inbox = {}
    mesh.send.return_value = {"ok": True}
    mesh.broadcast.return_value = [{"ok": True}]
    with (
        patch("strands_robots.mesh.get_local_robots", return_value={"local-a": mesh}),
        patch("strands_robots.mesh.session.get_peers", return_value=[]),
    ):
        yield mesh


def _validator_returning_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Model the refactor the guards' comment names: a ``validate_command`` that
    validates in place and returns ``None`` rather than a sanitised copy.

    The pre-pass then completes without error while leaving its sentinel unset --
    the only way a handler is reached with no validated command to dispatch.
    """
    monkeypatch.setattr(rmt._security, "validate_command", lambda cmd: None)


def _robot_mesh_ast() -> ast.FunctionDef:
    """The ``robot_mesh`` FunctionDef, parsed from the shipped module source."""
    tree = ast.parse(inspect.getsource(rmt))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "robot_mesh":
            return node
    raise AssertionError("robot_mesh function not found in the module source")


def _is_none_test(node: ast.expr) -> str | None:
    """Return the sentinel name for a ``<name> is None`` test, else None."""
    if not isinstance(node, ast.Compare) or len(node.ops) != 1:
        return None
    if not isinstance(node.ops[0], ast.Is) or not isinstance(node.left, ast.Name):
        return None
    comparator = node.comparators[0]
    if isinstance(comparator, ast.Constant) and comparator.value is None:
        return node.left.id
    return None


def _sentinels_assigned() -> set[str]:
    """Every ``validated_*_cmd`` sentinel the pre-pass assigns."""
    found: set[str] = set()
    for node in ast.walk(_robot_mesh_ast()):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        for target in targets:
            if isinstance(target, ast.Name) and target.id.startswith("validated_") and target.id.endswith("_cmd"):
                found.add(target.id)
    return found


def _sentinels_guarded_by_a_raise() -> set[str]:
    """Every sentinel whose ``is None`` branch raises rather than falling through."""
    found: set[str] = set()
    for node in ast.walk(_robot_mesh_ast()):
        if not isinstance(node, ast.If):
            continue
        name = _is_none_test(node.test)
        if name is None or not (name.startswith("validated_") and name.endswith("_cmd")):
            continue
        if any(isinstance(inner, ast.Raise) for inner in ast.walk(ast.Module(body=node.body, type_ignores=[]))):
            found.add(name)
    return found


class TestTheAssertStrippingPremise:
    """The reason both guards are an explicit ``raise``: ``-O`` strips ``assert``.

    Measured rather than asserted in a comment, so the choice cannot quietly
    stop being true for the interpreter the package is shipped against.
    """

    _SNIPPET = textwrap.dedent(
        """
        def handler(sentinel):
            assert sentinel is not None, "contract broken"
            return "dispatched"

        print(handler(None))
        """
    )

    def test_an_assert_fires_on_a_default_interpreter(self) -> None:
        proc = subprocess.run([sys.executable, "-c", self._SNIPPET], capture_output=True, text=True, check=False)
        assert proc.returncode != 0
        assert "AssertionError" in proc.stderr
        assert "dispatched" not in proc.stdout

    def test_the_same_assert_is_a_no_op_under_dash_o(self) -> None:
        proc = subprocess.run([sys.executable, "-O", "-c", self._SNIPPET], capture_output=True, text=True, check=False)
        assert proc.returncode == 0
        assert proc.stdout.strip() == "dispatched"


class TestTheGuardsAreExplicitRaisesNotAsserts:
    """Every pre-validated sentinel is guarded, and guarded by a ``raise``.

    Derived from the shipped source, so a third gated action that grows a
    validated command body fails here until its handler carries the guard too.
    """

    def test_the_sentinel_set_is_the_two_known_command_bodies(self) -> None:
        assert _sentinels_assigned() == {
            "validated_send_cmd",
            "validated_broadcast_cmd",
            "validated_sim_call_cmd",
        }

    def test_every_assigned_sentinel_is_guarded_by_a_raise(self) -> None:
        assigned = _sentinels_assigned()
        assert _sentinels_guarded_by_a_raise() == assigned

    def test_no_sentinel_is_guarded_by_an_assert(self) -> None:
        """An ``assert`` here would be stripped under ``-O`` (see the premise class)."""
        asserted = {
            name
            for node in ast.walk(_robot_mesh_ast())
            if isinstance(node, ast.Assert)
            for name in ast.walk(node.test)
            if isinstance(name, ast.Name) and name.id.startswith("validated_") and name.id.endswith("_cmd")
        }
        assert asserted == set()


class TestBothHandlersRefuseWithoutPreValidation:
    """A handler reached with its sentinel unset refuses, naming the contract."""

    @pytest.mark.parametrize(
        ("action", "_transport_attr", "kwargs", "_cmd_index"), _PRE_VALIDATED_ACTIONS, ids=_ACTION_IDS
    )
    def test_the_guard_names_the_action_and_the_broken_contract(
        self,
        transport: Any,
        monkeypatch: pytest.MonkeyPatch,
        action: str,
        _transport_attr: str,
        kwargs: dict[str, Any],
        _cmd_index: int,
    ) -> None:
        _validator_returning_none(monkeypatch)
        with pytest.raises(RuntimeError) as excinfo:
            _call(action=action, **kwargs)
        message = str(excinfo.value)
        assert action in message
        assert "validate-before-HITL contract broken" in message
        assert "without pre-validation" in message


class TestNothingIsDispatchedWhenTheContractIsBroken:
    """The guard runs before the transport, so a broken contract dispatches nothing.

    This is the property whose absence is expensive: without the guard the
    handler hands ``None`` to the transport, the dispatch really happens, and the
    ``AttributeError`` that follows lands after it -- so a fleet-wide (or
    targeted) command is issued and never audited.
    """

    @pytest.mark.parametrize(
        ("action", "transport_attr", "kwargs", "_cmd_index"), _PRE_VALIDATED_ACTIONS, ids=_ACTION_IDS
    )
    def test_the_transport_is_never_called(
        self,
        transport: Any,
        monkeypatch: pytest.MonkeyPatch,
        action: str,
        transport_attr: str,
        kwargs: dict[str, Any],
        _cmd_index: int,
    ) -> None:
        _validator_returning_none(monkeypatch)
        with pytest.raises(RuntimeError):
            _call(action=action, **kwargs)
        assert getattr(transport, transport_attr).call_args_list == []

    @pytest.mark.parametrize(
        ("action", "transport_attr", "kwargs", "_cmd_index"), _PRE_VALIDATED_ACTIONS, ids=_ACTION_IDS
    )
    def test_no_dispatch_success_is_audited(
        self,
        transport: Any,
        monkeypatch: pytest.MonkeyPatch,
        action: str,
        transport_attr: str,
        kwargs: dict[str, Any],
        _cmd_index: int,
    ) -> None:
        """No ``success=True`` record for a dispatch that never happened."""
        audited: list[tuple[str, str, bool, str]] = []
        monkeypatch.setattr(
            "strands_robots.tools.robot_mesh._audit_tool_action",
            lambda a, t, ok, detail: audited.append((a, t, ok, detail)),
        )
        _validator_returning_none(monkeypatch)
        with pytest.raises(RuntimeError):
            _call(action=action, **kwargs)
        assert [
            record for record in audited if record[0] == action and record[2] is True and "approved" not in record[3]
        ] == []


class TestTheHonoredPathStillDispatchesTheValidatedCommand:
    """Non-vacuity: with the real validator each action dispatches, and what
    reaches the transport is the validator's sanitised copy -- not the raw
    model-authored string and not ``None``."""

    @pytest.mark.parametrize(
        ("action", "transport_attr", "kwargs", "cmd_index"), _PRE_VALIDATED_ACTIONS, ids=_ACTION_IDS
    )
    def test_the_transport_receives_the_validated_copy(
        self,
        transport: Any,
        monkeypatch: pytest.MonkeyPatch,
        action: str,
        transport_attr: str,
        kwargs: dict[str, Any],
        cmd_index: int,
    ) -> None:
        real = rmt._security.validate_command
        returned: list[Any] = []

        def _spy(cmd: dict[str, Any]) -> Any:
            validated = real(cmd)
            returned.append(validated)
            return validated

        monkeypatch.setattr(rmt._security, "validate_command", _spy)

        out = _call(action=action, **kwargs)

        assert out["status"] == "success"
        assert len(returned) == 1
        dispatched = getattr(transport, transport_attr).call_args.args[cmd_index]
        assert dispatched is returned[0]
        assert isinstance(dispatched, dict)
        # send/broadcast dispatch the command BODY the caller wrote (a status
        # probe here); sim_call synthesises its own command whose action IS
        # the verb, with the caller's function under sim_action.
        if "command" in kwargs:
            assert dispatched["action"] == "status"
        else:
            assert dispatched["action"] == action
