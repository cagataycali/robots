"""Repro: ``run_multi_policy(policies={rname: X})`` leaks ``AttributeError``.

End-user scenario: a working robotics engineer has read the ``run_multi_policy``
docstring, which calls each value "a pre-built Policy driven in one
lockstep physics loop". They pass a string, a provider name, a config dict,
``None``, or the ``Policy`` CLASS instead of an instance (every mistake
``policy_object`` guards against on ``run_policy``), and the tool-envelope
contract leaks a raw CPython ``AttributeError`` from a library internal.

Expected: structured ``{"status": "error", "content": [{"text": ...}]}``
naming the parameter the caller got wrong - the exact posture
``_validate_policy_object`` ships for ``run_policy`` / ``eval_policy`` /
``evaluate_benchmark`` (strands_robots/simulation/base.py:3256-3310,
tests/simulation/test_policy_object_shape_is_refused_at_the_entry_point.py).

Actual (before this patch):
    AttributeError: 'str' object has no attribute 'get_actions'
    AttributeError: 'NoneType' object has no attribute 'get_actions'
    AttributeError: 'int' object has no attribute 'get_actions'
    AttributeError: 'dict' object has no attribute 'get_actions'
    AttributeError: type object 'MockPolicy' has no attribute '...'

Raise site: strands_robots/simulation/mujoco/simulation.py:7841 (MuJoCo),
            strands_robots/simulation/isaac/simulation.py:7207 (Isaac).

The sibling helpers already pin exactly this contract:

* ``_normalize_multi_policy_instructions`` (base.py:4176-4195) refuses a
  non-string / non-mapping ``instructions`` "up front rather than reaching
  ``.get()`` and surfacing as a bare ``AttributeError`` past the
  tool-envelope contract" (its own docstring).
* ``_normalize_multi_policy_horizons`` refuses a non-positive-int
  ``action_horizon`` for the same reason.

The one value the base validator did NOT type-check was ``policies``'s own
values - which are exactly the pre-built policies ``policy_object`` guards.
Fix: ``_validate_multi_policies`` now routes each value through
``policy_object_error(value, param=f"policies[{rname!r}]")`` (reused: the
helper already parametrizes its message on the parameter name).

Run:
    MUJOCO_GL=egl python bugbash_repros/run_multi_policy_value_shape_repro.py
"""

from __future__ import annotations

import os
import sys
import traceback

os.environ.setdefault("MUJOCO_GL", "egl")

from strands_robots import Robot, create_policy

r = Robot("so101", mesh=False)
good = create_policy("mock")

BAD_VALUES = [
    ("scalar", 42),
    ("provider-name", "mock"),
    ("config-dict", {"provider": "mock"}),
    ("the-class", type(good)),
    ("none", None),
]


def _run(label: str, value: object) -> None:
    print(f"\n--- {label}: policies={{'so101': {value!r}}} ---")
    try:
        res = r.run_multi_policy(policies={"so101": value}, duration=0.1, control_frequency=10)
    except AttributeError:
        print("DEFECT: raw CPython AttributeError leaked past the tool envelope:")
        traceback.print_exc()
        return
    status = res.get("status")
    text = res["content"][0]["text"][:200].replace("\n", " | ")
    print(f"status={status!r} text={text!r}")


def _ok(label: str, policies: dict) -> None:
    print(f"\n--- {label}: policies={{k: Policy() for k in {list(policies)!r}}} ---")
    res = r.run_multi_policy(policies=policies, duration=0.1, control_frequency=10)
    status = res.get("status")
    text = res["content"][0]["text"][:200].replace("\n", " | ")
    print(f"status={status!r} text={text!r}")


if __name__ == "__main__":
    print("== BAD VALUES (want structured error for every one) ==")
    for label, val in BAD_VALUES:
        _run(label, val)
    print("\n== GOOD CASE (want status=success, guard must not cost a working call) ==")
    _ok("single-robot good policy", {"so101": good})
    sys.exit(0)
