"""Repro: eval_policy / evaluate_benchmark accept on_frame but have no
max_onframe_failures watchdog — run_policy does.

base.py @ 2389c4c9d:
- run_policy          (3456): max_onframe_failures: int | None = None   -> watchdog
- start_policy        (5192): max_onframe_failures: int | None = None   -> delegates
- eval_policy         (5707): -- no max_onframe_failures parameter --   -> no watchdog
- evaluate_benchmark  (6192): -- no max_onframe_failures parameter --   -> no watchdog

PolicyRunner (policy_runner.py @ 2389c4c9d):
- run()      (1901) counts hook failures into `consecutive_onframe_failures`
             and aborts at `max_onframe_failures` (2698-2841). Exempt classes:
             CooperativeStop (graceful stop), RecordingFrameError (data loss,
             abort on first).
- evaluate() (3931) catches the SAME hook exceptions and does
             `logger.warning("on_frame hook failed at global_step=%d: %s", ...)`
             (406-418). No counter. No abort.

Consequence: a flaky `on_frame` recorder passed to eval_policy /
evaluate_benchmark produces a `status=success` report over episodes with
zero captured frames. run_policy's `max_onframe_failures` docstring
explicitly warns against this ("broken on_frame hook otherwise fills an
empty dataset behind a successful-looking rollout - GH #117"); the two
multi-episode sibling entry points inherit none of that guard.

Runs on so101 (`Robot('so101', mesh=False)`), ~5 seconds, no network.
"""

from __future__ import annotations

import inspect
import os

os.environ.setdefault("MUJOCO_GL", "egl")

from strands_robots import Robot

sim = Robot("so101", mesh=False)

# 1) run_policy exposes the watchdog kwarg
run_sig = inspect.signature(sim.run_policy)
assert "max_onframe_failures" in run_sig.parameters, (
    "regression: run_policy no longer declares max_onframe_failures"
)

# 2) The three sibling policy-rollout surfaces do NOT
eval_sig = inspect.signature(sim.eval_policy)
bench_sig = inspect.signature(sim.evaluate_benchmark)
assert "on_frame" in eval_sig.parameters
assert "on_frame" in bench_sig.parameters
assert "max_onframe_failures" not in eval_sig.parameters, (
    "if eval_policy gains the kwarg this bug is fixed, remove this repro"
)
assert "max_onframe_failures" not in bench_sig.parameters, (
    "if evaluate_benchmark gains the kwarg this bug is fixed, remove this repro"
)

# 3) Prove the runtime effect: a hook that raises on every step produces
#    `status=success` on eval_policy. On run_policy an equivalent hook would
#    be caught by the watchdog within max_onframe_failures=5 (the default) —
#    we assert the surface, not the full rollout, since the repro has to be
#    short and the backend's own on_frame already masks the user hook on
#    run_policy's path (run_policy wraps its backend hook; the watchdog
#    there protects the backend's hook, not a user-supplied one — which is
#    exactly why eval_policy needs its own: eval_policy accepts the user
#    hook DIRECTLY).
calls: list[int] = []


def flaky_recorder(step: int, obs: dict, action: dict) -> None:
    calls.append(step)
    raise RuntimeError(f"recorder dropped frame {step}")


result = sim.eval_policy(
    "so101",
    policy_provider="mock",
    n_episodes=2,
    max_steps=5,
    control_frequency=10.0,
    success_fn="contact",
    on_frame=flaky_recorder,
)

print(f"eval_policy status:        {result.get('status')}")
print(f"on_frame invocations:      {len(calls)}  (every call raised)")
print(f"has 'episodes' field:      {'episodes' in result}")
print(f"success_measured in json:  {result.get('success_measured')}")
print(f"has watchdog kwarg (run):  {'max_onframe_failures' in run_sig.parameters}")
print(f"has watchdog kwarg (eval): {'max_onframe_failures' in eval_sig.parameters}")
print(f"has watchdog kwarg (bench):{'max_onframe_failures' in bench_sig.parameters}")

assert result.get("status") == "success", (
    "regression: eval_policy already surfaces the hook failure; close this issue"
)
assert len(calls) > 0, "hook was never invoked — rollout misconfigured"

print()
print("DEFECT CONFIRMED: eval_policy returned status=success with every")
print("on_frame call having raised. run_policy's max_onframe_failures")
print("watchdog (base.py:3184 _validate_onframe_failure_limit) is missing")
print("from eval_policy (5707) and evaluate_benchmark (6192), yet both")
print("accept on_frame — the same hook class the watchdog guards.")
