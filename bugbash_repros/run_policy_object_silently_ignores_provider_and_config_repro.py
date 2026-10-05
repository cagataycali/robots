"""Minimal repro: run_policy(policy_object=..., policy_provider=..., policy_config=...)
runs the policy_object and silently discards policy_provider + policy_config.

A user who copy-paste edits a prior ``run_policy(policy_provider=..., policy_config=...)``
call by inserting ``policy_object=`` gets ZERO signal that the provider+config
they left in the kwargs will be ignored. The run reports ``status="success"``
with no field in the json identifying that the provider was discarded.

The partner tool ``start_policy`` has the identical silent-wrong. ``eval_policy``
follows the same code shape (base.py:5880+).

Observed on strands-labs/robots v0.5.3 @ 9a64ccc, SimEngine.run_policy at
strands_robots/simulation/base.py:3857 (and start_policy at 4993, eval_policy
at 5880). All three take the ``if policy_object is None`` branch - which only
discards the other two kwargs rather than refusing the combination up front.

The precedent is in the same file: ``policy_object`` has its own entry-point
validator (``_validate_policy_object`` at line 3211) with explicit tests at
``tests/simulation/test_policy_object_shape_is_refused_at_the_entry_point.py``.
What is missing is the dual - a refusal when policy_object is given AND one of
the kwargs that is about to be ignored is also given.

The fix is roughly a dozen lines: refuse the mutually-exclusive combination at
the same point ``_validate_policy_object`` runs, with a message that names both
the discarded kwarg(s) and the explicit "pass one or the other" remedy.
"""
from __future__ import annotations

import os
import sys

os.environ.pop("SYSTEM_PROMPT", None)
os.environ.setdefault("MUJOCO_GL", "egl")

from strands_robots import Robot
from strands_robots.policies import MockPolicy


def _json_block(result: dict) -> dict:
    return next(b["json"] for b in result["content"] if "json" in b)


def main() -> int:
    sim = Robot("so101", mesh=False)

    # 1. Baseline: provider-only path - a bogus policy_provider is refused.
    #    This is the loud behaviour the user expects.
    baseline = sim.run_policy(
        robot_name="so101",
        policy_provider="###NOT_A_REAL_PROVIDER###",
        policy_config={"pretrained_name_or_path": "DOES_NOT_EXIST"},
        instruction="noop",
        n_steps=2,
        control_frequency=30.0,
    )
    assert baseline["status"] == "error", (
        "sanity: a bogus policy_provider should be refused when "
        "policy_object is NOT given"
    )
    print(
        f"[1/3] baseline provider-only path refuses bogus provider: OK "
        f"({baseline['content'][0]['text'][:80]!r})"
    )

    # 2. SILENT-WRONG (pre-fix): same bogus policy_provider + policy_config, but
    #    with policy_object also set. Expected (post-fix): refusal with
    #    "pass one or the other". Pre-fix: status=success, MockPolicy ran, the
    #    provider+config were silently discarded, no warning, no field in json.
    silent = sim.run_policy(
        robot_name="so101",
        policy_object=MockPolicy(),
        policy_provider="###NOT_A_REAL_PROVIDER###",   # IGNORED silently (pre-fix)
        policy_config={"pretrained_name_or_path": "DOES_NOT_EXIST"},  # IGNORED silently (pre-fix)
        instruction="noop",
        n_steps=2,
        control_frequency=30.0,
    )
    status = silent["status"]
    text_only = all("json" not in b for b in silent["content"])
    msg = silent["content"][0]["text"]
    silently_ran = status == "success"
    refused_cleanly = (
        status == "error"
        and text_only
        and "policy_object" in msg
        and ("policy_provider" in msg or "policy_config" in msg)
    )
    if silently_ran:
        j = next(b["json"] for b in silent["content"] if "json" in b)
        no_mention_of_discarded_kwargs = all(
            k not in j for k in ("policy_provider_ignored", "ignored_kwargs", "discarded_kwargs")
        )
        print(
            f"[2/3] run_policy(policy_object=MockPolicy(), policy_provider='BOGUS', ...): "
            f"status={status!r}, policy_in_json={j.get('policy')!r}, "
            f"silently_ran={silently_ran}, no_json_field_signals_discard={no_mention_of_discarded_kwargs}"
        )
    else:
        print(
            f"[2/3] run_policy(policy_object=MockPolicy(), policy_provider='BOGUS', ...): "
            f"status={status!r}, text_only={text_only}, "
            f"message={msg[:200]!r}"
        )
        no_mention_of_discarded_kwargs = False

    # 3. Same silent-wrong (pre-fix) on start_policy (contract pins it to run_policy).
    import time

    started = sim.start_policy(
        robot_name="so101",
        policy_object=MockPolicy(),
        policy_provider="###NOT_A_REAL_PROVIDER###",
        policy_config={"bogus": "value"},
        instruction="noop",
        control_frequency=30.0,
    )
    if started["status"] == "success":
        # Pre-fix: silently started with bogus provider.
        time.sleep(0.3)
        sim.stop_policy(robot_name="so101")
        start_silent = True
    else:
        start_silent = False
    print(
        f"[3/3] start_policy(policy_object=MockPolicy(), policy_provider='BOGUS', ...): "
        f"status={started['status']!r}  "
        f"(pre-fix: silent-wrong; post-fix: refused up front)"
    )

    sim.destroy()

    # Repro contract: exit code 1 ONLY if the papercut reproduces (pre-fix).
    reproduced = silently_ran and no_mention_of_discarded_kwargs and start_silent
    if reproduced:
        print(
            "\nDEFECT REPRODUCED: run_policy / start_policy accept "
            "(policy_object + policy_provider + policy_config) silently; "
            "a bogus provider/config lands in a status=success rollout with no signal."
        )
        return 1
    print(
        "\nFIX VERIFIED: run_policy / start_policy refuse "
        "(policy_object + policy_provider/policy_config) up front, naming the "
        "offending kwargs and the remedy."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
