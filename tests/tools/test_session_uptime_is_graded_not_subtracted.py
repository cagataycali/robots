# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A session record's start time is graded, not subtracted from the clock.

Both session tools keep their sessions in a JSON file, and ``list`` and
``status`` report each one's ``Uptime`` by subtracting the record's
``start_time`` from :func:`time.time`. That stamp is a file's declaration rather
than a caller's argument, and subtracting one that is not a stamp is how a
report comes to state a duration the session never had.

Measured against the code these tests pin, every spelling a session store can
hold other than a real stamp was reported wrongly, in one of three ways:

* A record with **no** ``start_time`` at all defaulted to ``0`` in the
  arithmetic, so ``time.time() - 0`` reported ``Uptime: 29813993.5 min``, a
  little under fifty-seven years, under ``status: success`` and beside a running
  flag that was right. ``true`` and ``0`` landed on the same answer.
* ``NaN`` reported ``Uptime: nan min``, ``Infinity`` reported ``-inf min``, and a
  stamp ahead of this clock - what a backwards NTP correction between the write
  and the read leaves behind - reported ``Uptime: -10.0 min``.
* ``null``, a ``str`` and a ``list`` raised out of the verb: ``list`` answered
  ``Tool execution failed: unsupported operand type(s) for -: 'float' and
  'NoneType'``, naming neither the session nor the field nor a remedy.

The two verbs of one tool also disagreed about one record, because they spelled
the same read differently - ``info.get("start_time", 0)`` in ``list`` against
``float(session_info.get("start_time") or 0)`` in ``status``. A store holding
``"1788000000"`` made ``list`` abort while ``status`` reported ``13993.5 min``,
nine and a half days, which is an entirely ordinary length for a training run.

The abort is the reach of it. A store holds every tracked session, and its own
load step is documented to degrade to empty rather than raise so that a record
damaged outside its pid still stops. One damaged ``start_time`` among three
records made ``list`` answer ``status: error`` and report **none** of them, so
two sessions that were genuinely running became invisible to the verb that
enumerates them.

:func:`~strands_robots.tools._process_stop.session_uptime` is the one reader of
that field for both tools, in the shape
:func:`~strands_robots.tools._process_stop.recorded_pid` uses for the ``pid``
beside it: the span, or ``None`` for no usable one, plus the sentence saying
which way the record did not state it. These tests pin the domain, that all four
surfaces reach the same verdict, that no spelling raises anywhere, that a
damaged record no longer hides its healthy neighbours, that the ``json`` block
carries no span the reader could not measure, and that a real stamp still
renders exactly as it did.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

from strands_robots.tools import _process_stop
from strands_robots.tools._process_stop import PID_STARTED_SINCE_BOOT, process_started_since_boot


def session_uptime(info: Any) -> tuple[float | None, str]:
    """Reach the owner through the module.

    Bound here rather than imported by name so that a tree without the owner
    fails each cell below with what is missing, instead of failing to collect the
    file and grading nothing.
    """
    return _process_stop.session_uptime(info)  # type: ignore[no-any-return]


train_mod = importlib.import_module("strands_robots.tools.lerobot_train")
teleop_mod = importlib.import_module("strands_robots.tools.lerobot_teleoperate")

#: The two tool modules whose verbs report a session's uptime, and the callable
#: to reach each one through. ``@tool`` wraps the function, so the undecorated
#: body is reached through ``original`` where the decorator provides it.
_TOOLS: tuple[tuple[str, Any, dict[str, Any]], ...] = (
    ("lerobot_train", train_mod, {"dataset_root": "/tmp/nonexistent-dataset"}),
    ("lerobot_teleoperate", teleop_mod, {}),
)

#: Spellings a session store can hold under ``start_time``, with the fragment the
#: refusal must carry. ``<<ABSENT>>`` means the key is not written at all.
UNUSABLE: tuple[tuple[str, Any, str], ...] = (
    ("absent", "<<ABSENT>>", "states no start time"),
    ("null", None, "states no start time"),
    ("true", True, "states a bool"),
    ("false", False, "states a bool"),
    ("a stamp as a string", "1788000000", "states a str"),
    ("an unparsable string", "soon", "states a str"),
    ("a list", [], "states a list"),
    ("a mapping", {}, "states a dict"),
    ("the epoch itself", 0, "not a wall-clock stamp"),
    ("before the epoch", -1.0, "not a wall-clock stamp"),
    ("NaN", float("nan"), "not a wall-clock stamp"),
    ("Infinity", float("inf"), "not a wall-clock stamp"),
    ("negative Infinity", float("-inf"), "not a wall-clock stamp"),
)


def _record(value: Any = "<<ABSENT>>", **extra: Any) -> dict[str, Any]:
    """A session record for this process, carrying ``value`` as its start time.

    The pid is this test's own, with the identity the running verdict needs
    captured beside it, so the record reads as a live session without any probe
    being stood in for - the uptime read is what is under test, not the pid.
    """
    pid = os.getpid()
    info: dict[str, Any] = {
        "action": "train",
        "pid": pid,
        "command": "lerobot-train",
        "log_file": "/dev/null",
        PID_STARTED_SINCE_BOOT: process_started_since_boot(pid),
        "policy_type": "act",
        "output_dir": "/tmp/out",
        "robot_type": "so101_follower",
        "teleop_type": "so101_leader",
        **extra,
    }
    if value != "<<ABSENT>>":
        info["start_time"] = value
    return info


def _texts(result: dict[str, Any]) -> str:
    return "\n".join(item.get("text", "") for item in result.get("content", []) if "text" in item)


def _uptime_lines(result: dict[str, Any]) -> list[str]:
    return [line.strip() for line in _texts(result).splitlines() if "Uptime" in line]


def _json_block(result: dict[str, Any]) -> dict[str, Any]:
    for item in result.get("content", []):
        if isinstance(item, dict) and "json" in item:
            return dict(item["json"])
    return {}


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point both tools' session stores at one file under ``tmp_path``.

    Both modules read a module-level ``SESSION_DIR`` when a manager is built, so
    redirecting the attribute on each is what keeps the verbs off the store of
    whatever machine runs this.
    """
    session_dir = tmp_path / ".sessions"
    session_dir.mkdir()
    monkeypatch.setattr(train_mod, "SESSION_DIR", session_dir)
    monkeypatch.setattr(teleop_mod, "SESSION_DIR", session_dir)
    path = session_dir / "active_sessions.json"

    def write(records: dict[str, Any]) -> None:
        path.write_text(json.dumps(records), encoding="utf-8")

    return write


def _call(module: Any, extra: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    """Invoke a tool module's verb, reaching the undecorated body."""
    tool = getattr(module, module.__name__.rsplit(".", 1)[-1])
    fn = getattr(tool, "original", tool)
    return fn(**extra, **kwargs)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# The domain
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(("label", "value", "fragment"), UNUSABLE, ids=[case[0] for case in UNUSABLE])
def test_a_record_that_states_no_usable_stamp_measures_no_span(label: str, value: Any, fragment: str) -> None:
    """Nothing outside the domain becomes a duration, and the text says which way."""
    span, text = session_uptime(_record(value))
    assert span is None, f"{label} was subtracted from the clock and reported as {span} s"
    assert text.startswith("unknown ("), f"{label} produced {text!r}, which does not read as an absence"
    assert fragment in text, f"{label} produced {text!r}, which does not name how the record went wrong"


def test_a_real_stamp_is_the_span_it_states() -> None:
    """The one spelling a writer produces is measured, and rendered as it always was."""
    span, text = session_uptime(_record(time.time() - 600.0))
    assert span is not None
    assert 590.0 < span < 660.0, f"a stamp ten minutes old measured {span} s"
    assert text == f"{span / 60:.1f} min"


def test_a_stamp_ahead_of_this_clock_is_not_a_negative_duration() -> None:
    """A backwards clock step is reported as one, not rendered as negative uptime."""
    span, text = session_uptime(_record(time.time() + 600.0))
    assert span is None, "a start time in the future was reported as a duration"
    assert "ahead of this clock" in text
    assert "-" not in text, f"a negative duration reached the report: {text!r}"


def test_the_bool_refusal_is_about_the_type_not_the_number_it_carries() -> None:
    """``true`` is refused as a bool; the ``1`` it subclasses is a stamp and is measured.

    The accepted boundary, pinned so a later change cannot quietly turn this
    owner into a plausibility check. ``1`` is one second past the epoch and no
    writer produces it, but the only non-arbitrary floor above the epoch is this
    machine's boot time, and comparing against that mis-refuses a genuine stamp
    after a backwards clock correction - see :func:`session_uptime`.
    """
    span, text = session_uptime(_record(True))
    assert span is None
    assert "bool" in text, f"true was refused, but not as a bool: {text!r}"
    span, text = session_uptime(_record(1))
    assert span is not None, "a positive stamp is a span, however implausible the date"
    assert text.endswith(" min")


# ---------------------------------------------------------------------------
# Both tools, both verbs
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(("label", "value", "fragment"), UNUSABLE, ids=[case[0] for case in UNUSABLE])
@pytest.mark.parametrize("verb", ["list", "status"])
@pytest.mark.parametrize("tool", _TOOLS, ids=[case[0] for case in _TOOLS])
def test_every_verb_of_every_tool_reaches_the_same_verdict(
    tool: tuple[str, Any, dict[str, Any]],
    verb: str,
    label: str,
    value: Any,
    fragment: str,
    store: Any,
) -> None:
    """No surface raises, and none of them reports a span for a record without one."""
    name, module, extra = tool
    store({"s1": _record(value)})
    kwargs = {"action": verb} | ({"session_name": "s1"} if verb == "status" else {})
    result = _call(module, extra, **kwargs)
    assert result["status"] == "success", f"{name} {verb} refused the whole verb over {label}: {_texts(result)}"
    lines = _uptime_lines(result)
    assert lines, f"{name} {verb} dropped the uptime field entirely for {label}"
    for line in lines:
        assert fragment in line, f"{name} {verb} reported {line!r} for {label}"


@pytest.mark.parametrize("verb", ["list", "status"])
@pytest.mark.parametrize("tool", _TOOLS, ids=[case[0] for case in _TOOLS])
def test_every_verb_of_every_tool_still_reports_a_real_span(
    tool: tuple[str, Any, dict[str, Any]], verb: str, store: Any
) -> None:
    """The control: a record a writer produced reads the same on all four surfaces."""
    name, module, extra = tool
    store({"s1": _record(time.time() - 1800.0)})
    kwargs = {"action": verb} | ({"session_name": "s1"} if verb == "status" else {})
    result = _call(module, extra, **kwargs)
    assert result["status"] == "success"
    assert any("30.0 min" in line for line in _uptime_lines(result)), (
        f"{name} {verb} did not report a stamp half an hour old: {_uptime_lines(result)}"
    )


# ---------------------------------------------------------------------------
# What the abort cost
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("tool", _TOOLS, ids=[case[0] for case in _TOOLS])
def test_one_damaged_record_does_not_hide_the_running_ones(tool: tuple[str, Any, dict[str, Any]], store: Any) -> None:
    """``list`` enumerates every tracked session, damaged neighbour or not."""
    name, module, extra = tool
    store(
        {
            "healthy": _record(time.time() - 1800.0),
            "damaged": _record(None),
            "also_healthy": _record(time.time() - 60.0),
        }
    )
    result = _call(module, extra, action="list")
    assert result["status"] == "success", f"{name} list aborted over one damaged record"
    text = _texts(result)
    for expected in ("healthy", "damaged", "also_healthy"):
        assert expected in text, f"{name} list omitted '{expected}': {text}"
    assert "30.0 min" in text and "1.0 min" in text, f"{name} list lost a healthy session's span: {text}"


@pytest.mark.parametrize("tool", _TOOLS, ids=[case[0] for case in _TOOLS])
def test_the_status_json_carries_no_span_it_could_not_measure(
    tool: tuple[str, Any, dict[str, Any]], store: Any
) -> None:
    """A machine reading ``uptime`` gets ``null``, not the epoch as a duration."""
    name, module, extra = tool
    store({"s1": _record(None)})
    block = _json_block(_call(module, extra, action="status", session_name="s1"))
    assert "uptime" in block, f"{name} status stopped reporting an uptime key"
    assert block["uptime"] is None, f"{name} status published {block['uptime']} s for a record with no start time"


# ---------------------------------------------------------------------------
# One owner
# ---------------------------------------------------------------------------
def _start_time_reads(module: Any) -> list[str]:
    """Every expression in ``module`` that reads ``start_time`` out of a record."""
    tree = ast.parse(inspect.getsource(module))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get":
            if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "start_time":
                found.append(ast.unparse(node))
    return found


@pytest.mark.parametrize("tool", _TOOLS, ids=[case[0] for case in _TOOLS])
def test_no_verb_reads_the_stamp_itself(tool: tuple[str, Any, dict[str, Any]]) -> None:
    """The grading is not bypassable by a second reader of the same field.

    The field went wrong in the first place because two verbs of one tool each
    read it their own way, so a tool reading it again is the drift this closes.
    Writing it stays the tool's own business; only the read is owned.
    """
    name, module, _ = tool
    reads = _start_time_reads(module)
    assert not reads, f"{name} reads start_time directly instead of through session_uptime: {reads}"


def test_the_owner_is_the_one_that_reads_it() -> None:
    """Non-vacuity: the rule above is satisfied by delegation, not by nobody reading."""
    assert _start_time_reads(_process_stop), "session_uptime no longer reads start_time, so the rule above is empty"
    for name, module, _ in _TOOLS:
        source = inspect.getsource(module)
        assert "session_uptime(" in source, f"{name} does not ask the owner for its uptime"
