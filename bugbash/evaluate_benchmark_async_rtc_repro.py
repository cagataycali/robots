"""
Defect: SimEngine.evaluate_benchmark is silently narrower than its three
siblings (run_policy / start_policy / eval_policy) by two RTC kwargs:
async_rtc and rtc_inference_timeout_s.

Three siblings expose these as named parameters and emit a structured
refusal when the combination is unsupported (eval_policy with a spec +
async_rtc=True returns {"status":"error","content":[{"text":...
"async_rtc is only supported on the success_fn eval path. The spec/
benchmark path stays synchronous for bit-stable reproducibility; use
run_policy(async_rtc=...)..."]}).

evaluate_benchmark can't accept them at all - the user who copy-pastes
the arg name from the sibling gets a bare Python TypeError that cites
nothing about reproducibility, nothing about the alternative, nothing
about the design intent (same intent sits inside policy_runner.py:4202,
unreachable from the signature).

Upstream source-of-truth:
  strands_robots/simulation/base.py:6192  (evaluate_benchmark signature, 16 args)
  strands_robots/simulation/base.py:3456  (run_policy signature, 24 args — has async_rtc, rtc_inference_timeout_s)
  strands_robots/simulation/base.py:5192  (start_policy, 24 args — has them)
  strands_robots/simulation/base.py:5707  (eval_policy, 20 args — has them)
  strands_robots/simulation/policy_runner.py:4202  (the good refusal, hidden)
  strands_robots/simulation/policy_runner.py:4682  (_evaluate_with_spec, no rtc_inference_timeout_s param)

Repro: pure signature inspection — no sim required.
"""

import inspect
from strands_robots.simulation.base import SimEngine

RTC_KWARGS = ("async_rtc", "rtc_inference_timeout_s")
METHODS = ("run_policy", "start_policy", "eval_policy", "evaluate_benchmark")

print("=== Which policy-rollout surfaces expose RTC kwargs? ===")
for m in METHODS:
    sig = inspect.signature(getattr(SimEngine, m))
    params = set(sig.parameters.keys())
    row = {k: (k in params) for k in RTC_KWARGS}
    print(f"  {m}: {row}")

# Bind check — what a user sees when they try to pass async_rtc:
print("\n=== bind_partial(evaluate_benchmark, async_rtc=True) ===")
try:
    inspect.signature(SimEngine.evaluate_benchmark).bind_partial(
        benchmark_name="go2_walk_forward", async_rtc=True
    )
    print("  (accepted — should have been TypeError)")
except TypeError as e:
    print(f"  TypeError: {e}")

print("\n=== Sibling refusal shape (what evaluate_benchmark SHOULD mirror) ===")
# eval_policy returns a structured error for the same unsupported combo.
# We don't run eval_policy here (needs a sim) but the message is at
# policy_runner.py:4204-4214:
expected_msg = (
    "async_rtc is only supported on the success_fn eval path. "
    "The spec/benchmark path stays synchronous for bit-stable "
    "reproducibility; use run_policy(async_rtc=...) for "
    "benchmark-style latency masking."
)
print(f"  structured text (hidden inside PolicyRunner.evaluate): {expected_msg!r}")

print("\n=== Assertion ===")
missing = []
for k in RTC_KWARGS:
    for m in ("run_policy", "start_policy", "eval_policy"):
        if k not in inspect.signature(getattr(SimEngine, m)).parameters:
            raise SystemExit(f"UNEXPECTED: {m} missing {k}, update defect scope")
    if k not in inspect.signature(SimEngine.evaluate_benchmark).parameters:
        missing.append(k)

if missing:
    raise SystemExit(
        f"\n*** DEFECT CONFIRMED: evaluate_benchmark is missing {missing} ***\n"
        "Three sibling surfaces expose them; evaluate_benchmark hides the\n"
        "intentional-refusal path behind a bare TypeError.\n"
    )
