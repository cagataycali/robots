"""One stand-in for the transport callable a mesh bridge forwards every call to.

``RosBridgedRobot``, ``RosbridgeRobot``, ``AckermannRosRobot`` and ``RtpsRobot``
own no transport state: each resolves :func:`strands_robots.ros.ros_action`,
:func:`strands_robots.rosbridge.rosbridge_action` or
:func:`strands_robots.rtps.participant.rtps_action` through its own module, so a
test replaces that symbol and reads what the bridge asked for.

Ten test modules grew a recorder for exactly that, in eight spellings, and every
one of them accepted ``**kwargs``. A recorder shaped like nothing accepts calls
the transport would not: dropping the ``gate`` all three require from one
forward leaves the whole mesh suite green, because no stand-in has the real
callable's parameters. :class:`Transport` binds each call against the signature
of the symbol it replaces, so it refuses what the transport refuses and the
arguments it records are the ones that would have reached the wire.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

#: Arguments a bridge hands its transport that never reach the robot. A gate is a
#: fresh closure per call, so two identical commands are never equal dicts while
#: it is in them.
OFF_THE_WIRE = frozenset({"gate", "tool_context"})


def wire_only(call: dict[str, Any]) -> dict[str, Any]:
    """One recorded call without the arguments that stay off the wire."""
    return {name: value for name, value in call.items() if name not in OFF_THE_WIRE}


class Transport:
    """Records every forwarded call, answering with a success envelope.

    Args:
        target: The real transport callable being stood in for, or a stand-in
            already installed over it - a test that patches one symbol several
            times would otherwise take the previous stand-in's own
            ``(*args, **kwargs)`` shape and grade nothing. Its signature is what
            every recorded call is bound against.
        text: Text of the default success envelope.

    Attributes:
        target: The real transport callable whose shape is being honored.
        calls: The arguments of each forwarded call, by parameter name.
        responses: Envelopes to answer with, in order, before the default one.
    """

    def __init__(self, target: Any, *, text: str = "ok") -> None:
        self.target: Any = target.target if isinstance(target, Transport) else target
        self._signature = inspect.signature(self.target)
        self._text = text
        self.calls: list[dict[str, Any]] = []
        self.responses: list[dict[str, Any]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        # The real callable raises TypeError on an argument it has no parameter
        # for, and on a missing ``gate``; binding here is what makes a recorded
        # call a call that could have happened.
        bound = self._signature.bind(*args, **kwargs)
        self.calls.append(dict(bound.arguments))
        if self.responses:
            return self.responses.pop(0)
        return {"status": "success", "content": [{"text": self._text}]}

    @property
    def on_the_wire(self) -> list[dict[str, Any]]:
        """The recorded calls without the arguments that stay off the wire."""
        return [wire_only(call) for call in self.calls]


def stands_in_for(monkeypatch: pytest.MonkeyPatch, module: Any, symbol: str, *, text: str = "ok") -> Transport:
    """Replace ``module.symbol`` with a stand-in shaped like the callable there.

    The stand-in takes its shape from the symbol it replaces, so it cannot be
    wired to one transport while grading the arguments of another.

    Args:
        monkeypatch: The patcher undoing the replacement after the test.
        module: Bridge module resolving the transport symbol.
        symbol: Name of the forwarded transport callable in ``module``.
        text: Text of the default success envelope.

    Returns:
        The installed :class:`Transport`.
    """
    stand_in = Transport(getattr(module, symbol), text=text)
    monkeypatch.setattr(module, symbol, stand_in)
    return stand_in
