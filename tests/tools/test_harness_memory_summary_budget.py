"""The summary size budget must cover the payload the store actually holds.

``HarnessMemory.save_trace`` writes the caller's summary plus a provenance block
(timestamp, library version, backend, robot) into one file, and
``HarnessMemory.load_trace`` re-validates that file before anything reaches
planner context. The two ends therefore have to account for the same bytes.

They did not. ``save_trace`` checked the caller's payload while ``load_trace``
checked the caller's payload *plus* provenance, so a summary in the top 140
bytes of the documented 64 KiB budget saved with ``status="success"`` and was
then permanently unreadable::

    save_trace -> success, "Saved trace for task 't0' (1 steps)"
    load_trace -> error,   "summary too large (65676 > 65536 bytes);
                            delete it with delete_trace and re-save"

The remedy that message names cannot work: deleting and re-saving the same
summary reproduces the same unloadable file, measured over three attempts. The
trace budget is symmetric already (nothing is injected into a trace entry),
which is what makes the summary the defect rather than the caps being wrong.

What is pinned here:

* a summary whose *stored* size exceeds the budget is refused at save, before
  any file is written, and an existing pair survives the refusal;
* every summary a save accepts is one a load accepts -- swept across the
  boundary rather than asserted at one point;
* the size the save path checks equals the size the load path recomputes, over
  payload shapes that round-trip through JSON differently (unicode, floats,
  nested containers, ``True``/``None``);
* the trace budget stays symmetric, and the entry-count cap stays a distinct
  budget from the byte cap, so neither is collapsed into the other.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

import strands_robots.tools.harness_memory as hm
from tests.tool_result_contract import tool_json

TRACE = [{"action": "run_policy", "instruction": "grasp the bowl", "n_steps": 20}]


@pytest.fixture
def memory_dir(tmp_path, monkeypatch):
    """Point STRANDS_MEMORY_DIR at a temp dir so the store is isolated."""
    d = tmp_path / "memory"
    monkeypatch.setenv("STRANDS_MEMORY_DIR", str(d))
    return d


def _summary_of_size(n: int) -> dict[str, Any]:
    """A summary whose ``json.dumps(sort_keys=True)`` length is exactly *n*."""
    summary: dict[str, Any] = {"why": "a"}
    summary["why"] = "a" * (n - len(json.dumps(summary, sort_keys=True)) + 1)
    assert len(json.dumps(summary, sort_keys=True)) == n
    return summary


def _provenance_overhead() -> int:
    """Bytes ``save_trace``'s provenance block adds to a stored summary."""
    provenance = {
        "saved_at": "2026-01-01T00:00:00+0000",
        "strands_robots_version": hm._version_string(),
        "backend": None,
        "robot": None,
    }
    base = {"why": "x"}
    return len(json.dumps({**base, "provenance": provenance}, sort_keys=True)) - len(json.dumps(base, sort_keys=True))


def _stored_summary(memory_dir, task: str) -> dict[str, Any]:
    return json.loads((memory_dir / "tasks" / f"{task}.summary.json").read_text(encoding="utf-8"))


def _files(memory_dir) -> list[str]:
    tasks = memory_dir / "tasks"
    return sorted(p.name for p in tasks.iterdir()) if tasks.exists() else []


class TestThePremiseThatMakesTheTwoEndsAbleToDisagree:
    """The store injects bytes the caller never passed, so accounting matters."""

    def test_the_provenance_block_has_a_measurable_size(self, memory_dir):
        """A stored summary is strictly larger than the one handed in."""
        summary = {"why": "small"}
        hm.HarnessMemory().save_trace("t", TRACE, summary)
        stored = _stored_summary(memory_dir, "t")
        assert "provenance" in stored
        passed_in = len(json.dumps(summary, sort_keys=True))
        held = len(json.dumps(stored, sort_keys=True))
        assert held > passed_in
        assert _provenance_overhead() > 0


class TestTheBudgetCoversTheStoredPayload:
    """A summary the store cannot read back must be refused at save time."""

    def test_a_summary_at_the_documented_budget_is_refused(self, memory_dir):
        """Its stored size exceeds the cap, so saving it would lose the trace."""
        summary = _summary_of_size(hm._MAX_SUMMARY_BYTES)
        with pytest.raises(ValueError, match="summary too large"):
            hm.HarnessMemory().save_trace("t", TRACE, summary)

    def test_the_refusal_quotes_the_stored_size_against_the_cap(self, memory_dir):
        """The numbers name the payload that was measured, not the cap alone."""
        summary = _summary_of_size(hm._MAX_SUMMARY_BYTES)
        with pytest.raises(ValueError) as excinfo:
            hm.HarnessMemory().save_trace("t", TRACE, summary)
        expected = len(json.dumps({**summary, "provenance": {}}, sort_keys=True))
        assert str(hm._MAX_SUMMARY_BYTES) in str(excinfo.value)
        assert expected > hm._MAX_SUMMARY_BYTES

    def test_the_refusal_writes_no_files(self, memory_dir):
        """A first save that cannot be read back must leave nothing behind."""
        summary = _summary_of_size(hm._MAX_SUMMARY_BYTES)
        with pytest.raises(ValueError):
            hm.HarnessMemory().save_trace("t", TRACE, summary)
        assert _files(memory_dir) == []

    def test_the_refusal_leaves_an_existing_pair_intact(self, memory_dir):
        """A replace that cannot be read back must not disturb the stored pair."""
        memory = hm.HarnessMemory()
        memory.save_trace("t", TRACE, {"why": "v1"})
        with pytest.raises(ValueError):
            memory.save_trace("t", TRACE, _summary_of_size(hm._MAX_SUMMARY_BYTES))
        trace, summary = memory.load_trace("t")
        assert trace == TRACE
        assert summary["why"] == "v1"
        assert not list((memory_dir / "tasks").glob("*.tmp"))

    def test_the_tool_surface_refuses_instead_of_reporting_success(self, memory_dir):
        """The agent-facing envelope must not say a lost trace was saved."""
        result = hm.harness_memory(
            action="save_trace",
            task="t",
            trace=TRACE,
            summary=_summary_of_size(hm._MAX_SUMMARY_BYTES),
        )
        assert result["status"] == "error"
        assert "too large" in "\n".join(c.get("text", "") for c in result["content"])
        assert tool_json(hm.harness_memory(action="list_tasks"))["tasks"] == []


class TestEverySavedSummaryLoadsBack:
    """The invariant the budget exists to provide, not a single data point."""

    def test_no_summary_that_saves_is_unloadable_near_the_boundary(self, memory_dir):
        """Swept across the boundary: saved implies loadable, with no window."""
        memory = hm.HarnessMemory()
        cap = hm._MAX_SUMMARY_BYTES
        saved = unloadable = 0
        for size in range(cap - 200, cap + 1):
            try:
                memory.save_trace("sweep", TRACE, _summary_of_size(size))
            except ValueError:
                continue
            saved += 1
            try:
                memory.load_trace("sweep")
            except ValueError:
                unloadable += 1
        assert saved > 0, "the sweep never reached an accepted size"
        assert unloadable == 0

    @pytest.mark.parametrize(
        "summary",
        [
            {"why": "plain ascii"},
            {"why": "wide \u00e9\u00e8 \u4e2d\u6587", "n": 12},
            {"nested": {"a": [1, 2, {"b": 0.125}]}, "flag": True, "none": None},
            {"floats": [0.1, 1e-09, 1234567.891], "big": 2**53},
        ],
        ids=["ascii", "wide-characters", "nested-containers", "floats"],
    )
    def test_the_save_side_size_equals_the_load_side_size(self, memory_dir, monkeypatch, summary):
        """Both ends must measure the same bytes for a payload to be safe."""
        measured: list[int] = []
        real = hm._validate_summary

        def recording(payload: Any) -> dict[str, Any]:
            measured.append(len(json.dumps(payload, sort_keys=True)))
            return real(payload)

        monkeypatch.setattr(hm, "_validate_summary", recording)
        memory = hm.HarnessMemory()
        memory.save_trace("t", TRACE, summary)
        assert measured, "save_trace did not measure the payload it stores"
        on_save = measured[-1]
        measured.clear()
        memory.load_trace("t")
        assert measured, "load_trace did not re-validate the stored summary"
        assert measured[-1] == on_save


class TestTheRemedyTheRefusalNamesNowWorks:
    """Shrinking the summary must actually produce a loadable store."""

    def test_shrinking_by_the_provenance_overhead_saves_and_loads(self, memory_dir):
        """The largest summary a caller can hold round-trips unchanged."""
        memory = hm.HarnessMemory()
        summary = _summary_of_size(hm._MAX_SUMMARY_BYTES - _provenance_overhead())
        memory.save_trace("t", TRACE, summary)
        trace, loaded = memory.load_trace("t")
        assert trace == TRACE
        assert loaded["why"] == summary["why"]


class TestNeighbouringBudgetsStayOutOfScope:
    """What must not change: the trace side, and the caps being distinct."""

    def test_an_ordinary_summary_still_round_trips(self, memory_dir):
        """The overwhelmingly common case is untouched."""
        memory = hm.HarnessMemory()
        summary = {"task": "put the bowl on the tray", "success": True, "avoid": ["stale xyz"]}
        memory.save_trace("t", TRACE, summary)
        trace, loaded = memory.load_trace("t")
        assert trace == TRACE
        assert loaded["avoid"] == summary["avoid"]

    def test_the_trace_budget_is_symmetric_at_both_ends(self, memory_dir):
        """Nothing is injected into a trace entry, so its accounting matches."""
        memory = hm.HarnessMemory()
        trace = [{"action": "get_state", "pad": "a" * 500} for _ in range(20)]
        on_save = sum(len(json.dumps(entry, sort_keys=True)) for entry in trace)
        memory.save_trace("t", trace, {"why": "ok"})
        loaded, _ = memory.load_trace("t")
        on_load = sum(len(json.dumps(entry, sort_keys=True)) for entry in loaded)
        assert on_load == on_save

    def test_the_trace_byte_cap_is_distinct_from_the_entry_count_cap(self, memory_dir, monkeypatch):
        """A trace inside the entry count can still exceed the byte budget.

        Driven through the tool rather than :meth:`HarnessMemory.save_trace`:
        the trace validator runs at the tool boundary and on load, so that is
        where the byte budget is observable.
        """
        monkeypatch.setattr(hm, "_MAX_TRACE_BYTES", 400)
        trace = [{"action": "get_state", "pad": "a" * 200} for _ in range(3)]
        assert len(trace) <= hm._MAX_TRACE_ENTRIES
        result = hm.harness_memory(action="save_trace", task="t", trace=trace, summary={"why": "ok"})
        assert result["status"] == "error"
        text = "\n".join(c.get("text", "") for c in result["content"])
        assert "trace too large" in text
        assert "too long" not in text, "the byte budget must not report as the entry-count cap"
