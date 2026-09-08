"""A flight command the radio would not carry is refused, not raised.

Every caller-facing verb on this driver is documented to return "a success
envelope, or an error envelope naming the refusal", and each one ends in a CRTP
write. A write is where a link that opened and then stopped answering shows up:
the handle is still live, ``is_connected`` still reads True, and the write
raises.

The driver already decided both halves of this. ``_repeat_loop`` ends its own
loop on :data:`~strands_robots.drivers.crazyflie.LINK_WRITE_ERRORS` and says
why - "the next ``send_action`` reports the refusal through an envelope a caller
can read, which a background thread cannot" - so the raise set is settled and so
is where the report belongs. Three more places are already written for that
envelope: ``stop_task`` turns a refused descent into its own refusal, the agent
tool's ``land`` action reaches for ``stop_task``'s verdict *because* "the descent
itself can be refused - a disconnected link", and ``cleanup`` documents that
"every step tolerates a half-built driver".

The refusal a caller reads is not the same fact as the exception the SDK raised,
which is why the messages are pinned rather than only the status. Two of them
carry a fact no traceback does: a refused priority handover means the high-level
command was *not* attempted (the firmware would have ignored it underneath the
low-level stream), and a refused motor cut means the motors are still turning
while the setpoint stream has already been stopped.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest

from strands_robots.drivers import crazyflie as module

#: What a link that opened and then stopped answering raises on a write. One
#: instance reused across the cases so the reason text is checkable.
LINK_DOWN = OSError("link down: no response from the aircraft")

#: ``(verb, which commander fails, how to call it, what the refusal must name)``.
#: Both commanders appear because the two-step verbs write to each in turn, and a
#: guard on one is not a guard on the other.
CASES: tuple[tuple[str, str, str, str], ...] = (
    ("send_action", "commander", "send_action", "send_action"),
    ("set_twist", "commander", "set_twist", "send_action"),
    ("takeoff, priority handover", "commander", "takeoff", "takeoff"),
    ("takeoff, the climb", "high_level", "takeoff", "takeoff"),
    ("land, priority handover", "commander", "land", "land"),
    ("land, the descent", "high_level", "land", "land"),
    ("emergency_stop, the motor cut", "commander", "emergency_stop", "emergency_stop"),
)


def _call(driver: object, verb: str) -> dict[str, object]:
    """Invoke *verb* with arguments the driver accepts, so only the write can fail."""
    if verb == "send_action":
        return driver.send_action({"vx": 0.2, "z": 0.5})  # type: ignore[attr-defined,no-any-return]
    if verb == "set_twist":
        return driver.set_twist(vx=0.2, z=0.5)  # type: ignore[attr-defined,no-any-return]
    if verb == "takeoff":
        return driver.takeoff(height=0.5, duration=2.0)  # type: ignore[attr-defined,no-any-return]
    if verb == "land":
        return driver.land(duration=2.0)  # type: ignore[attr-defined,no-any-return]
    if verb == "emergency_stop":
        return driver.emergency_stop()  # type: ignore[attr-defined,no-any-return]
    raise AssertionError(f"unhandled verb {verb}")


def _armed(connected, which: str, **kwargs):  # type: ignore[no-untyped-def]
    """A connected, armed driver whose *which* commander refuses every write."""
    raises = {f"{which}_raises" if which == "commander" else "high_level_raises": LINK_DOWN}
    driver, fake, reason = connected(setpoint_hz=100, **raises, **kwargs)
    assert reason is None, f"the link must come up before a write can be refused: {reason}"
    return driver, fake


class TestARefusedWriteIsReturned:
    """The verb answers with its envelope instead of raising the SDK's error."""

    @pytest.mark.parametrize(("label", "which", "verb", "named"), CASES, ids=[c[0] for c in CASES])
    def test_the_verb_refuses_rather_than_raising(  # type: ignore[no-untyped-def]
        self, connected, label: str, which: str, verb: str, named: str
    ) -> None:
        driver, _ = _armed(connected, which)

        envelope = _call(driver, verb)

        assert envelope["status"] == "error", (
            f"{label}: the write was refused, so the verb owes the caller its error envelope. "
            "Raising the SDK's exception past the return leaves every consumer written for the "
            "envelope - stop_task, the agent tool's land action, cleanup - with nothing to read."
        )
        text = envelope["content"][0]["text"]  # type: ignore[index]
        assert named in text, f"{label}: the refusal must name the verb, got {text!r}"
        assert "link down: no response from the aircraft" in text, (
            f"{label}: the refusal must carry what the link said, got {text!r}"
        )

    def test_a_healthy_link_is_untouched(self, connected, recorder) -> None:  # type: ignore[no-untyped-def]
        """The control: with a link that answers, every verb still reaches the wire.

        Holds on both sides of the fix by construction. It is what separates
        "a refused write is reported" from "the write is no longer attempted".
        """
        driver, _, reason = connected(setpoint_hz=100)
        assert reason is None

        assert driver.send_action({"vx": 0.2, "z": 0.5})["status"] == "success"
        assert driver.takeoff(height=0.5, duration=2.0)["status"] == "success"
        assert driver.land(duration=2.0)["status"] == "success"
        assert driver.emergency_stop()["status"] == "success"
        assert recorder.count("high_level.takeoff") == 1
        assert recorder.count("high_level.land") == 1
        assert recorder.count("commander.send_stop_setpoint") == 1


class TestTheRefusalCarriesWhatTheTracebackDoesNot:
    """Two refusals state a consequence no SDK exception mentions."""

    @pytest.mark.parametrize("verb", ["takeoff", "land"])
    def test_a_refused_handover_does_not_command_the_high_level_move(  # type: ignore[no-untyped-def]
        self, connected, recorder, verb: str
    ) -> None:
        """The handover is a precondition, so a refused one ends the verb.

        ``send_notify_setpoint_stop`` is what hands setpoint priority back. Both
        verbs' docstrings record that a high-level command issued while the
        low-level stream still owns priority is *ignored* and the aircraft keeps
        flying the last twist. So a refused handover must not be followed by the
        move: commanding it anyway would return whatever the second write did
        about a command the firmware discards.
        """
        driver, _ = _armed(connected, "commander")

        envelope = _call(driver, verb)

        assert envelope["status"] == "error"
        assert recorder.count(f"high_level.{verb}") == 0, (
            f"a refused priority handover must not be followed by high_level.{verb}; "
            "the firmware ignores a high-level command underneath the low-level stream"
        )
        text = envelope["content"][0]["text"]  # type: ignore[index]
        assert "handover" in text and "ignored" in text, (
            f"the refusal must say the move was not commanded and why, got {text!r}"
        )

    def test_a_refused_motor_cut_says_the_motors_are_still_turning(self, connected) -> None:  # type: ignore[no-untyped-def]
        """``emergency_stop`` halts the stream first, so a refused cut is the worst state.

        The repeater is stopped before the cut is written, which is correct - it
        would otherwise re-latch a twist immediately after the stop. It also
        means a refused cut leaves an aircraft with neither a setpoint stream nor
        a motor cut, and only the firmware supervisor decides what happens next.
        A caller reading ``status`` alone cannot act on that; the refusal says it.
        """
        driver, _ = _armed(connected, "commander")

        envelope = driver.emergency_stop()

        assert envelope["status"] == "error"
        text = envelope["content"][0]["text"]  # type: ignore[index]
        assert "NOT cut" in text, f"the refusal must say the motors were not cut, got {text!r}"
        assert "hardware cutoff" in text, (
            f"a refused motor cut leaves no software path to stop the aircraft, got {text!r}"
        )


class TestTheConsumersAlreadyWrittenForIt:
    """Three callers in the tree read the envelope this fix lets the verbs make."""

    def test_stop_task_composes_the_refused_descent(self, connected) -> None:  # type: ignore[no-untyped-def]
        """``stop_task`` already branches on ``land()``'s status; it needed one to read."""
        driver, _ = _armed(connected, "high_level")

        envelope = driver.stop_task()

        assert envelope["status"] == "error"
        text = envelope["content"][0]["text"]  # type: ignore[index]
        assert "stop_task: the descent was refused" in text, f"stop_task's own composition must run, got {text!r}"
        assert "land: the link refused the descent" in text, (
            f"the composed refusal must carry land()'s reason, got {text!r}"
        )

    def test_cleanup_releases_the_link_when_the_descent_is_refused(  # type: ignore[no-untyped-def]
        self, connected, recorder
    ) -> None:
        """``cleanup``'s first statement is the descent; a raise there leaked the radio.

        The method documents that "every step tolerates a half-built driver,
        because cleanup is what runs after a failed connect", and each later step
        is individually guarded. The descent was not, so a refused one skipped the
        repeater halt, the telemetry block, ``close_link`` and the state clear -
        leaving the process holding a radio link and reporting itself connected.
        """
        driver, _ = _armed(connected, "high_level")

        driver.cleanup()

        assert recorder.count("close_link") == 1, "the radio link must be released"
        assert recorder.count("log.stop") == 1, "the telemetry block must be stopped"
        assert driver.is_connected is False, "a driver that has been cleaned up must not report itself connected"


class TestARefusedSetpointLeavesTheStreamAlone:
    """What the repeater is holding is what keeps the aircraft up."""

    def test_the_previous_setpoint_survives_a_refused_one(self, connected, recorder) -> None:  # type: ignore[no-untyped-def]
        """A refused *later* command must not drop a hover that is working.

        The firmware supervisor cuts thrust when the setpoint stream goes quiet,
        so the latched setpoint is load-bearing. Clearing it because a subsequent
        write was refused would turn one refused command into a thrust cut, which
        is why the refusal path restores what was latched before.
        """
        driver, fake, reason = connected(setpoint_hz=100)
        assert reason is None
        assert driver.send_action({"vx": 0.2, "z": 0.5})["status"] == "success"
        holding = driver._setpoint  # noqa: SLF001 - the state the repeater feeds
        assert holding is not None

        fake.commander._raises = LINK_DOWN  # noqa: SLF001 - the link stops answering
        assert driver.send_action({"vx": 0.9, "z": 0.5})["status"] == "error"

        assert driver._setpoint == holding, (  # noqa: SLF001
            "a refused setpoint must leave the stream holding what it was holding; "
            "dropping it would stop feeding the supervisor and cut thrust"
        )


class TestOneOwnerForTheRaiseSet:
    """Every guard around a commander write reads the same set."""

    #: How a write reaches the aircraft: the two commander accessors, and the
    #: helper the repeater shares with the setpoint path.
    WRITE_PATHS = ("self._commander()", "self._high_level()", "self._send_setpoint")

    def test_every_guarded_commander_write_reads_the_shared_set(self) -> None:
        """A write's raise set has one owner, so the two halves cannot drift.

        Keyed on the *operation*, not on the exception names: this driver has
        four other guards whose sets legitimately differ because they cover
        different operations (opening the link, closing it, the arming request,
        the log block). What must not differ is the set around a write through
        the commander - the repeater and the caller-facing verbs reach the same
        SDK objects, and a second literal tuple is how one of them came to
        handle ``AttributeError`` while the other handled nothing at all.
        """
        tree = ast.parse(textwrap.dedent(inspect.getsource(module)))

        guarded: list[tuple[int, str]] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            body = "\n".join(ast.unparse(stmt) for stmt in node.body)
            if not any(path in body for path in self.WRITE_PATHS):
                continue
            for handler in node.handlers:
                guarded.append((handler.lineno, ast.unparse(handler.type) if handler.type else "bare"))

        assert guarded, (
            "this scan is looking in the wrong place: no guarded commander write found, so it "
            "would report a driver with none as compliant"
        )
        drifted = sorted({spelling for _, spelling in guarded if spelling != "LINK_WRITE_ERRORS"})
        assert drifted == [], (
            "a guard around a commander write enumerates its own raise set instead of reading "
            f"LINK_WRITE_ERRORS: {drifted}"
        )

    def test_every_write_path_is_guarded(self) -> None:
        """No commander write is left outside a handler.

        The companion to the rule above: naming the shared set everywhere is
        only half of it while a write can still sit outside a ``try`` at all,
        which is exactly the state the caller-facing verbs were in.
        """
        tree = ast.parse(textwrap.dedent(inspect.getsource(module)))
        cls = next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == "CrazyflieDriver")

        guarded_lines: set[int] = set()
        for node in ast.walk(cls):
            if isinstance(node, ast.Try):
                for stmt in node.body:
                    guarded_lines.update(range(stmt.lineno, (stmt.end_lineno or stmt.lineno) + 1))

        unguarded: list[tuple[int, str]] = []
        for fn in ast.walk(cls):
            # ``_send_setpoint`` is the raw write itself; its two callers own the
            # handler, which is what lets the repeater end its loop where a verb
            # returns a refusal.
            if isinstance(fn, ast.FunctionDef) and fn.name == "_send_setpoint":
                continue
            if not isinstance(fn, ast.Call):
                continue
            spelling = ast.unparse(fn.func)
            if not any(spelling.startswith(path) for path in self.WRITE_PATHS):
                continue
            if fn.lineno not in guarded_lines:
                unguarded.append((fn.lineno, spelling))

        assert unguarded == [], (
            "a commander write is outside every handler, so the link's refusal leaves the verb "
            f"by raising instead of being returned: {unguarded}"
        )
