"""The dashboard's own lockout is painted only from what the server said about it.

The button reads E-STOP or RESUME and a click posts `/api/safety/estop` or
`/api/safety/resume`. The two were decided in different places: the label was
written only by the click handler's own answer, while every other path that
painted the lockout line - page load, a telemetry frame, an error handler -
left the label as it was. So a page loaded under an e-stop engaged elsewhere
read "e-stop engaged" beside a button that read E-STOP, and pressing that
button thawed every frozen session: the operator's one reflex under an e-stop,
inverted. The error handlers compounded it by painting the line `locked` for
any failure - a 429 session cap, a network error - so the button could flip
into resume mode while the server's lockout was clear.

The surface that fires this rail is the e-stop sheet (`frontend/src/components/
EstopSheet.tsx`; the in-process Sim tab that once held its own button left with
the agent's sim tools). The rule there reads: the sheet holds ONE `simLockout`
state, every write of it is the server's answer to the `/api/safety/<action>`
it just posted, and a failed request becomes a message, never a lockout. These
cells read `EstopSheet.tsx` for that shape, so the rule holds without a browser
in the suite.
"""

from __future__ import annotations

import pathlib
import re

FRONTEND_SRC = pathlib.Path(__file__).parent.parent / "strands_robots" / "dashboard" / "frontend" / "src"
ESTOP_SHEET = FRONTEND_SRC / "components" / "EstopSheet.tsx"


def _function_body(source: str, header: str) -> str:
    """The text from ``header`` to the end of its braced block, braces balanced."""
    start = source.index(header)
    depth = 0
    for i in range(source.index("{", start), len(source)):
        depth += {"{": 1, "}": -1}.get(source[i], 0)
        if depth == 0:
            return source[start : i + 1]
    raise AssertionError(f"unbalanced braces after {header!r}")


class TestTheEstopButtonReadsOneState:
    def test_one_lockout_state_and_every_write_is_a_server_answer(self) -> None:
        """The sheet keeps one ``simLockout`` state, and nothing writes it but what the server said."""
        source = ESTOP_SHEET.read_text(encoding="utf-8")
        states = re.findall(r"useState<SimLockout \| null>", source)
        assert len(states) == 1, f"the lockout is held in {len(states)} states, so two can disagree"
        writes = [line.strip() for line in source.splitlines() if "setSimLockout(" in line]
        assert writes, "nothing writes the lockout, so the sheet paints a state that never changes"
        for write in writes:
            assert re.search(r"setSimLockout\(sim\.lockout\)", write), (
                f"a lockout write that is not a server answer: {write}"
            )

    def test_the_lockout_the_sheet_paints_is_the_answer_to_the_route_it_posted(self) -> None:
        """Each write follows a ``post<{ lockout: SimLockout }>('/api/safety/<action>')`` on the same rail."""
        source = ESTOP_SHEET.read_text(encoding="utf-8")
        for action in ("estop", "resume"):
            handler = _function_body(
                source, "const fire = async () =>" if action == "estop" else "const resume = async () =>"
            )
            assert f"post<{{ lockout: SimLockout }}>('/api/safety/{action}')" in handler, (
                f"the {action} handler does not read the lockout from the server's answer"
            )
            assert "className" not in handler, "the handler reads a painted class, which the label does not follow"

    def test_no_handler_fabricates_a_lockout(self) -> None:
        """A failed request is reported as a message; only a server answer becomes the lockout."""
        source = ESTOP_SHEET.read_text(encoding="utf-8")
        fabricated = [
            f"{n}: {line.strip()}"
            for n, line in enumerate(source.splitlines(), 1)
            if re.search(r"setSimLockout\(\s*\{\s*state\s*:", line)
        ]
        assert fabricated == [], f"a lockout the server never reported is written: {fabricated}"
        for header in ("const fire = async () =>", "const resume = async () =>"):
            handler = _function_body(source, header)
            assert "catch (e" in handler, f"{header} has no failure branch to report a failed request"
            assert "setSimLockout" not in handler.split("catch")[1], "a failed request is painted as a lockout"
