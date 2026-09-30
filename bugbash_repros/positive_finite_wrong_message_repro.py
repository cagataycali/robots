"""Minimal repro: strands_robots.utils.positive_finite_number_error emits a
misleading "must be > 0" message for values that ARE > 0 (inf) or are not
numbers at all (None, "50", [50]) or NaN.

The function's name and docstring both promise "positive AND finite", but
the emitted message only cites the sign check. Compare with the sibling
finite_number_error at :653 which reports "must be a finite number, got nan"
in the same two families.

The blast radius is 87 call sites in strands_robots that surface this text
to end users through refusal envelopes (teleoperate hz/duration/fps,
run_policy control_frequency, add_camera fps, recorder fps, etc.).

Run: python bugbash_repros/positive_finite_wrong_message_repro.py
"""
from strands_robots.utils import positive_finite_number_error, finite_number_error

CASES = [
    # (label,                       value,               expected_reason)
    ("integer zero",                0,                   "sign"),
    ("negative float",              -1.5,                "sign"),
    ("positive infinity",           float("inf"),        "finiteness"),
    ("NaN",                         float("nan"),        "finiteness"),
    ("None",                        None,                "type"),
    ("string '50'",                 "50",                "type"),
    ("list [50]",                   [50],                "type"),
]

print(f"{'case':30}  {'PFN emits':70}  {'FN emits':70}")
print("-" * 175)
for label, val, expected in CASES:
    pfn = positive_finite_number_error(val, "hz", "teleoperate")
    fn = finite_number_error(val, "hz", "teleoperate")
    print(f"{label:30}  {(pfn or '(accepted)')[:70]:70}  {(fn or '(accepted)')[:70]:70}")

print()
print("Expected fix: positive_finite_number_error should emit")
print("  'must be a positive finite number, got <repr>'  (or split like finite_number_error)")
print("so the user is told which of the two invariants their value violated.")
print()
print("Currently 'hz=inf' reports 'must be > 0' -- but inf > 0 is True in Python;")
print("the actual reason for the refusal is 'not finite'. A user or LLM reading the")
print("message will not know which invariant is at stake.")
