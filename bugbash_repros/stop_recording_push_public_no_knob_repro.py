"""Repro: SimEngine.stop_recording(push_to_hub=True) silently publishes PUBLIC.

The agent-facing sim.stop_recording signature advertises:
    (push_to_hub=False, bucket=None, run_id=None) -> dict

No `private=` knob. Internally at
strands_robots/simulation/recording.py:1976 the call is literally:

    recorder.push_to_hub(tags=["strands-robots", "sim"])

`DatasetRecorder.push_to_hub(private: bool = False, ...)` defaults
`private=False`, so every sim-recorded dataset published via the
documented agent path lands in the user's namespace PUBLIC. The user
has no way to opt into private via the documented surface.

Adjacent paths disagree on the default:

- `DatasetRecorder.push_to_hub` default: private=False (public)   <--- used here, hard-coded
- `DatasetRecorder.sync_to_bucket` default: private=True (private)
- `sync_dataset_to_bucket` default:        private=True (private)

The docs page docs/learn/data/stream-and-sync.md:26 lists
`sim.stop_recording(bucket="you/collection", run_id="...")` and
`DatasetRecorder.sync_to_bucket(...)` as equivalent paths, and the
paragraph below says "private=True by default" and "delete=True
forwards --delete" -- both of which hold for sync_to_bucket but not
for push_to_hub, which the same `stop_recording` call also routes.

No subprocess, no Hub call: this repro verifies the asymmetry
in-process by inspecting signatures and the hard-coded call site.
"""

import inspect
import re
from pathlib import Path

from strands_robots.dataset_recorder import DatasetRecorder


def main() -> int:
    # 1. DatasetRecorder.push_to_hub defaults private=False.
    sig = inspect.signature(DatasetRecorder.push_to_hub)
    private_default = sig.parameters["private"].default
    assert private_default is False, (
        f"DatasetRecorder.push_to_hub(private=...) default drifted from False "
        f"(got {private_default!r}); the repro premise may be stale."
    )

    # 2. DatasetRecorder.sync_to_bucket defaults private=True (asymmetric).
    sig_b = inspect.signature(DatasetRecorder.sync_to_bucket)
    bucket_private_default = sig_b.parameters["private"].default
    assert bucket_private_default is True, (
        f"DatasetRecorder.sync_to_bucket(private=...) default drifted from True "
        f"(got {bucket_private_default!r}); the repro premise may be stale."
    )

    # 3. SimEngine.stop_recording signature does NOT expose private=/tags=.
    #    Loaded via regex rather than import to avoid the mujoco weight.
    recording_py = Path(__file__).resolve().parents[1] / "strands_robots" / "simulation" / "recording.py"
    text = recording_py.read_text()
    stop_rec_sig = re.search(
        r"def stop_recording\(\s*self,\s*(.*?)\)\s*->", text, flags=re.DOTALL
    )
    assert stop_rec_sig is not None, "could not find stop_recording signature"
    advertised = stop_rec_sig.group(1)
    assert "private" not in advertised, (
        f"stop_recording signature now exposes 'private' -- repro may be stale:\n{advertised}"
    )
    assert "tags" not in advertised, (
        f"stop_recording signature now exposes 'tags' -- repro may be stale:\n{advertised}"
    )

    # 4. The hard-coded call site at ~1976 omits private= entirely.
    hardcoded_call = re.search(
        r'recorder\.push_to_hub\(\s*tags=\[[^\]]*\]\s*\)', text
    )
    assert hardcoded_call is not None, (
        "could not find the hard-coded recorder.push_to_hub(tags=[...]) call "
        "at strands_robots/simulation/recording.py:1976 -- repro may be stale."
    )
    call_src = hardcoded_call.group(0)
    assert "private" not in call_src, (
        f"stop_recording internals now pass private=; repro may be stale:\n{call_src}"
    )

    # 5. The docs page lists the two paths as alternatives on one row and
    #    advertises `private=` as if it were a shared kwarg -- it is not
    #    reachable via `sim.stop_recording`.
    docs = Path(__file__).resolve().parents[1] / "docs" / "learn" / "data" / "stream-and-sync.md"
    docs_text = docs.read_text()
    assert "sim.stop_recording(push_to_hub=True)" in docs_text, (
        "stream-and-sync.md no longer references sim.stop_recording(push_to_hub=True); "
        "repro may be stale."
    )
    assert "DatasetRecorder.push_to_hub(tags=, private=)" in docs_text, (
        "stream-and-sync.md no longer lists DatasetRecorder.push_to_hub(tags=, private=) "
        "beside the sim path; repro may be stale."
    )
    assert "`private=True` by default" in docs_text, (
        "stream-and-sync.md no longer promises '`private=True` by default'; "
        "repro may be stale."
    )

    print(
        "DEFECT confirmed: sim.stop_recording(push_to_hub=True) publishes PUBLIC "
        "with no agent-facing knob.\n\n"
        "- DatasetRecorder.push_to_hub default:        private=False (public)\n"
        "- DatasetRecorder.sync_to_bucket default:     private=True  (private)\n"
        "- Hard-coded sim call site:                   push_to_hub(tags=[...])  <- no private=\n"
        "- stop_recording agent signature:             (push_to_hub, bucket, run_id)  <- no private=\n"
        "- docs/learn/data/stream-and-sync.md:25       lists two paths on one row and\n"
        "                                              names `private=` on the DatasetRecorder\n"
        "                                              side -- not reachable via sim.stop_recording\n"
        "- docs/learn/data/stream-and-sync.md:30       says `private=True by default` for the\n"
        "                                              bucket (true); same sentence is read as\n"
        "                                              applying to push_to_hub too (false)\n\n"
        "Expected: either (a) default private=True for symmetry with sync_to_bucket, or\n"
        "(b) expose `private=` on stop_recording and thread it through, or\n"
        "(c) rewrite docs to warn the two defaults differ.\n\n"
        "Observed:  silent-public publish of a possibly-sensitive recorded dataset."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
