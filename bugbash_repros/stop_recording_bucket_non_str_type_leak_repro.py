"""
Reproducer: stop_recording(bucket=<non-str>) raises raw TypeError out of the agent-tool envelope.

Compare sibling kwargs in the same signature:
  stop_recording(push_to_hub=<non-str>)  -> cleanly refused (dataset_recording_posture_error)
  stop_recording(private=<non-str>)      -> cleanly refused (dataset_recording_posture_error)
  stop_recording(bucket=<non-str>)       -> raw TypeError from re.match (THIS BUG)
  stop_recording(run_id=<non-str>)       -> see below

The validator block at strands_robots/simulation/recording.py:1827-1829 enforces
posture-type correctness for push_to_hub and private; bucket and run_id are
threaded unchecked into sync_dataset_to_bucket() which calls
_BUCKET_RE.match(bucket) at dataset_transfer.py:125 - re.match raises
TypeError on non-str, and the exception propagates out of the agent-tool envelope.

Reachable via two code paths:
  1) Idle path (_stop_recording_idle:2161 -> sync_dataset_to_bucket) when
     this sim has a last_dataset_root (prior finalized session).
  2) Active path (stop_recording:2008-2013 -> recorder.sync_to_bucket ->
     sync_dataset_to_bucket) when a recording is open.

The agent surface promises {"status": "error", "content": [...]} on error (every
sibling validator in the same method follows this). A raw TypeError means an
LLM dispatcher must catch bare exceptions on this one method or crash the loop.

Expected:
  All non-str bucket / run_id values returned as:
    {"status": "error", "content": [{"text": "stop_recording: 'bucket' must be a string, got <TYPE>."}]}

Actual:
  Raw TypeError("expected string or bytes-like object, got 'int'")
"""

import os
import tempfile
import pathlib
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

from strands_robots import Robot


def main() -> int:
    sim = Robot("so101", mesh=False)

    # Simulate a prior finalized session so bucket= reaches sync_dataset_to_bucket.
    fake_root = tempfile.mkdtemp(prefix="bugbash_bucket_type_")
    (pathlib.Path(fake_root) / "meta").mkdir()
    (pathlib.Path(fake_root) / "meta" / "info.json").write_text("{}")
    sim._recording_state()["last_dataset_root"] = fake_root

    cases = [
        {"bucket": 123},
        {"bucket": ["a", "b"]},
        {"bucket": True},
        {"bucket": {"owner": "me"}},
        {"bucket": 1.5},
    ]

    failures = []
    for kw in cases:
        try:
            res = sim.stop_recording(**kw)
            status = res.get("status")
            text = res.get("content", [{}])[0].get("text", "")
            if status != "error" or "must be a string" not in text:
                failures.append(
                    f"stop_recording({kw}) returned status={status!r} without "
                    f"naming the type violation. text={text[:120]!r}"
                )
        except TypeError as e:
            failures.append(
                f"stop_recording({kw}) raised raw TypeError out of agent envelope: {e}"
            )

    # Sibling path confirmation: push_to_hub=<non-str> is cleanly refused.
    for bad in (123, "yes", ["x"]):
        res = sim.stop_recording(push_to_hub=bad)
        if res.get("status") != "error":
            failures.append(f"sibling: stop_recording(push_to_hub={bad!r}) NOT refused")

    if failures:
        print("DEFECT REPRODUCED:")
        for f in failures:
            print(f"  * {f}")
        return 1
    print("No defect (something changed).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
