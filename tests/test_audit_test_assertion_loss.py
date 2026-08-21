"""The discriminator in scripts/audit_test_assertion_loss.py (Q106).

Both real cases from the first run are pinned here, because they are the two shapes the tool exists to
tell apart: my clobbered passkey test (+88/-134, six assertions gone — SUSPECT) and a de-flaking refactor
that pulled two asserts into one helper (+23/-2, one assertion fewer — a note, strictly stronger).
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "audit_test_assertion_loss",
    Path(__file__).resolve().parents[1] / "scripts" / "audit_test_assertion_loss.py",
)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)
classify = mod.classify


def test_the_real_clobber_is_a_suspect():
    # 38e45595: passkey.test.mjs, 28 -> 22 assertions, +88/-134 lines.
    assert classify(28, 22, 88, 134) == "suspect"


def test_the_real_refactor_is_only_a_note():
    # 7f547a95: two `assert len(...) == N` became one shared _assert_one_rollout helper.
    assert classify(24, 23, 23, 2) == "note"


def test_growing_a_test_is_never_flagged():
    assert classify(20, 40, 30, 1) == "ok"
    assert classify(20, 20, 5, 5) == "ok", "a rewrite that keeps every assertion is fine"


def test_deleting_every_assertion_is_a_suspect_even_in_a_tiny_diff():
    assert classify(3, 0, 1, 4) == "suspect"


def test_the_boundary_is_shrinkage_not_equality():
    # Equal insert/delete counts with a lost assertion: the file did not shrink, so it reads as a
    # refactor. Deliberate — this tool warns, it does not accuse.
    assert classify(10, 9, 5, 5) == "note"
    assert classify(10, 9, 4, 5) == "suspect"


def test_a_repaired_clobber_is_healed_not_fatal():
    # My passkey clobber: 28 -> 22, restored to 46 in 548f2fff. History cannot be rewritten, and a
    # permanently red check is one people stop reading.
    assert classify(28, 22, 88, 134, current=46) == "healed"
    # Still short of where it was = still a suspect.
    assert classify(28, 22, 88, 134, current=27) == "suspect"
    # Exactly restored counts as healed.
    assert classify(28, 22, 88, 134, current=28) == "healed"


def test_every_verdict_is_printed_with_a_word_the_runner_forwards():
    """A finding audit_all cannot see is a finding nobody reads.

    audit_all.py forwards a line only when its FIRST TOKEN is one of its news words. The first real run
    of this audit inside the runner proved the point: the 'healed' finding was printed and then dropped,
    because 'healed' is not in that vocabulary. Importing NEWS_WORDS here means a change to either side
    breaks a test rather than quietly muting an audit.
    """
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "audit_all", Path(__file__).resolve().parents[1] / "scripts" / "audit_all.py")
    audit_all = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit_all)

    for verdict in ("suspect", "note", "healed"):
        first = mod.line_tag(verdict).split()[0]
        assert first in audit_all.NEWS_WORDS, (
            f"verdict {verdict!r} prints lines starting {first!r}, which audit_all does not forward — "
            f"the finding would be invisible in the runner. Its vocabulary: {sorted(audit_all.NEWS_WORDS)}")
    assert audit_all.interesting("  " + mod.line_tag("healed") + " x.test.mjs: 28 -> 22"), \
        "and the runner must actually pass the assembled line"
