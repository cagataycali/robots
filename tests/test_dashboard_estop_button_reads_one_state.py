"""The red button's label and the action a click takes come from one lockout state.

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

Now `lockoutLine()` is the one writer of the client's lockout state, it writes
the label, and the click handler reads the action from that same state. A
failed request is not an e-stop: it is shown as a message and the line is
re-read from `/api/safety`. These cells read the shipped `app.js` for that shape,
line by line as the file is written, so the rule holds without a browser in the
suite.
"""

from __future__ import annotations

import pathlib
import re

APP_JS = pathlib.Path(__file__).parent.parent / "strands_robots" / "dashboard" / "static" / "app.js"
INDEX_HTML = APP_JS.with_name("index.html")


def _function_body(source: str, name: str) -> str:
    """The text of ``function <name>(...) { ... }``, braces balanced."""
    start = source.index(f"function {name}(")
    depth, i = 0, source.index("{", start)
    for i in range(i, len(source)):
        depth += {"{": 1, "}": -1}.get(source[i], 0)
        if depth == 0:
            return source[start : i + 1]
    raise AssertionError(f"unbalanced braces after function {name}")


class TestTheEstopButtonReadsOneState:
    def test_the_label_is_written_where_the_lockout_is_painted(self) -> None:
        """Every write of the button's label sits inside lockoutLine(), and it writes one."""
        source = APP_JS.read_text(encoding="utf-8")
        body = _function_body(source, "lockoutLine")
        writes = re.compile(r'\$\("#estop"\)\.textContent\s*=')
        assert writes.search(body), "lockoutLine() paints the line but not the button beside it"
        assert len(writes.findall(source)) == len(writes.findall(body)), (
            "the button's label is written outside lockoutLine(), so a paint can leave it stale"
        )

    def test_the_click_reads_the_state_the_paint_wrote(self) -> None:
        """The action is chosen from the recorded lockout, not from the line's class list."""
        source = APP_JS.read_text(encoding="utf-8")
        handler = source[source.index('$("#estop").addEventListener("click"') :]
        handler = handler[: handler.index("\n});") + 4]
        assert "className" not in handler, "the click reads the painted class, which the label does not follow"
        assert re.search(r'lockout\.state\s*===\s*"locked"', handler), "the click does not read the recorded lockout"
        assert "textContent" not in handler, "the click writes the label itself instead of leaving it to the paint"

    def test_no_handler_fabricates_a_lockout(self) -> None:
        """A failed request is reported as a message; only a server answer is painted as the lockout."""
        source = APP_JS.read_text(encoding="utf-8")
        fabricated = [
            f"{n}: {line.strip()}"
            for n, line in enumerate(source.splitlines(), 1)
            if re.search(r'lockoutLine\(\s*\{\s*state\s*:\s*"', line)
        ]
        assert fabricated == [], f"a lockout the server never reported is painted: {fabricated}"
        assert 'id="sim-msg"' in INDEX_HTML.read_text(encoding="utf-8"), "the page has nowhere to show a refusal"
        assert "simMessage(e.message)" in source, "a failed request's reason is not shown"
