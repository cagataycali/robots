"""Repro: create_policy(provider, policy_config={typo: ...}) silently swallows the typo.

Defect
======
create_policy("mock", policy_config={"amplitud": 0.5})
  -> returns MockPolicy(amplitude=0.5)  # DEFAULT value, user's intent lost

The identical typo passed either (a) as a top-level kwarg to create_policy,
or (b) inside policy_config= via sim.run_policy(), is correctly refused with
a 'did you mean "amplitude"?' TypeError by policy_kwargs_error() at
strands_robots/policies/factory.py:740.

Asymmetric guard: run_policy UNWRAPS policy_config before create_policy
(simulation/base.py:1903 `create_policy(policy_provider, **config)`), so the
near-miss screen sees the user's keys. The direct create_policy() path does
NOT unwrap: _resolve_policy_class returns resolved_kwargs = {"policy_config":
{"amplitud": 0.5}}, policy_kwargs_error sees ONE kwarg "policy_config" which
is not a near-miss for any MockPolicy ctor param, then MockPolicy(**{...})
runs with the whole dict going into **kwargs sink. The user's typo rides on
a nested dict and the amplitude field keeps the default - byte-identical to
the moveit2 silent-swallow bug the factory's own docstring (:741-:750) says
this helper was built to prevent.

docs/concepts/index.md:17 publishes `create_policy(provider, **config)` as
the canonical signature; `policy_config` is the dict every rollout entry point
takes (run_policy, start_policy, eval_policy, evaluate_benchmark). A user who
transfers that habit to the direct create_policy() call hits this.

Repro
=====
$ python bugbash_repros/create_policy_policy_config_swallow_repro.py

Expected (desired, byte-identical to the top-level and the run_policy forms):
  TypeError: MockPolicy (policy provider 'mock') does not accept 'amplitud'
  (did you mean 'amplitude'?). ...

Actual:
  MockPolicy returns with amplitude=0.5 (default), caller believes 0.5 came
  from them. If the user asked for amplitude=0.9 the policy would still run
  on 0.5 under status=success.

Upstream
========
strands_robots/policies/factory.py:502 (_resolve_policy_class returns kwargs verbatim)
strands_robots/policies/factory.py:740 (policy_kwargs_error, correctly detects typos when they are at the right nesting)
strands_robots/policies/factory.py:905 (create_policy calls policy_kwargs_error with the un-unwrapped resolved_kwargs)
strands_robots/simulation/base.py:1903 (sibling: run_policy unwraps correctly: `create_policy(policy_provider, **config)`)
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

from strands_robots.policies import create_policy


def main() -> int:
    print("=== create_policy policy_config= silent-swallow repro ===\n")

    print("[1] create_policy('mock', amplitud=0.5)  # top-level typo - correctly refused")
    try:
        create_policy("mock", amplitud=0.5)
        print("    UNEXPECTED: no TypeError raised")
        return 1
    except TypeError as e:
        text = str(e)
        print(f"    OK: TypeError raised ({text[:120]}...)")

    print("\n[2] create_policy('mock', policy_config={'amplitud': 0.5})  # NESTED typo")
    try:
        p = create_policy("mock", policy_config={"amplitud": 0.5})
    except TypeError as e:
        print(f"    OK: TypeError raised ({str(e)[:200]}...)")
        return 0

    actual_amplitude = getattr(p, "amplitude", None)
    print(f"    DEFECT: no TypeError. Returned MockPolicy(amplitude={actual_amplitude})")
    print(f"            User's intent ('amplitud': 0.5) was discarded into **kwargs sink.")
    print(f"            amplitude kept its DEFAULT value, not any user value.")

    print("\n[3] For contrast: the SAME typo via run_policy() is correctly refused.")
    print("    (Not run here to keep the repro flat; see simulation/base.py:1903")
    print("     where run_policy does `create_policy(policy_provider, **config)` - the unwrap)")

    print("\nFactory's own docstring at factory.py:741-750 says this helper was built to")
    print("prevent exactly this shape: 'a constructor with **kwargs dropped")
    print("create_policy(\"moveit2\", hots=\"x\") silently'. The screen works when the typo")
    print("is in the kwargs dict; it does not when the typo is one level deep via")
    print("policy_config=.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
