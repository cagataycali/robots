#!/bin/bash
set -e
MINE=$(cat /tmp/p2087/minepath)
cd "$MINE"
F=tests/test_bucket_publication_posture_flags.py
cp "$F" /tmp/p2087/round2_fixed.py

# Pre-fix = the state before the sibling's factory landed (the subclass at
# 80bc91b0), plus only the new regression class, its calls re-pointed.
git show 80bc91b07112a2e41e4f56c5b693bb63f350d87d:"$F" > "$F"
grep -q "class _Recorder(recorder_mod.DatasetRecorder)" "$F" || { echo "REVERT DID NOT TAKE"; exit 1; }

python3 - <<'PY'
import pathlib
p = pathlib.Path("tests/test_bucket_publication_posture_flags.py")
s = p.read_text()
block = '''

class TestTheDoubleIsARealRecorder:
    """Pre-fix probe: the same two assertions, against the subclass."""

    def test_the_double_carries_every_attribute_production_sets(self) -> None:
        reference = recorder_mod.DatasetRecorder(_FakeHubDataset())
        assert set(vars(_Recorder(_FakeHubDataset()))) == set(vars(reference))

    def test_the_double_reports_the_counts_it_was_given(self) -> None:
        recorder = _Recorder(_FakeHubDataset(), frames=7, episodes=3)
        assert (recorder.frame_count, recorder.episode_count) == (7, 3)
'''
assert "TestTheDoubleIsARealRecorder" not in s
p.write_text(s + block)
PY

echo "=== rule mirror on the PRE-FIX tree (should match CodeQL alert 881) ==="
python3 /tmp/p2087/rulemirror.py "$F"
echo ""
echo "=== PRE-FIX run of the regression class ==="
set +e
MUJOCO_GL=egl python -m pytest "$F" -q --no-cov -p no:randomly -k TestTheDoubleIsARealRecorder 2>&1 | tail -8
echo ""
echo "=== PRE-FIX: whole file (the 23 pre-existing tests still pass = silent, not broken) ==="
MUJOCO_GL=egl python -m pytest "$F" -q --no-cov -p no:randomly 2>&1 | tail -3
set -e

cp /tmp/p2087/round2_fixed.py "$F"
grep -q "TestTheDoubleIsARealRecorder" "$F" || { echo "RESTORE LOST THE TESTS"; exit 1; }
grep -q "_Recorder" "$F" && { echo "RESTORE LEFT THE SUBCLASS"; exit 1; }
echo "restored OK"
git status --porcelain
