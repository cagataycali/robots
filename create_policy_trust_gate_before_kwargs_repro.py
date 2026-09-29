"""Repro for `create_policy` running the trust-remote-code gate before kwargs validation.

Symptom
-------
For `kimodo` and `lerobot_local` (the two providers gated behind
`STRANDS_TRUST_REMOTE_CODE`), a misspelled or unknown kwarg is masked by the
security gate: the caller is told to opt in to remote code first, and only
learns about the typo on a second call - after having flipped a
security-sensitive environment variable.

The docstring of `create_policy` at policies/factory.py:660 promises the
TypeError arrives "before construction, so no model is downloaded and no
server dialled on a typo". On the two gated providers, that promise is
broken: `_check_trust_remote_code(canonical)` at line 673 (pre-fix) runs
before `policy_kwargs_error(...)` at line 674.

Every other provider (mock, groot, wbc, microduck, protomotions, remote,
moveit2, curobo, rl, cosmos3, ...) enforces the kwargs check first because
they have no trust gate, so the asymmetry is invisible unless you probe the
two gated ones.

Fix
---
Two lines swap in `strands_robots/policies/factory.py`:

    canonical, PolicyClass, resolved_kwargs = _resolve_policy_class(provider, **kwargs)
    ...
-   _check_trust_remote_code(canonical)
-   if (kwargs_error := policy_kwargs_error(...)) is not None:
-       raise TypeError(kwargs_error)
+   if (kwargs_error := policy_kwargs_error(...)) is not None:
+       raise TypeError(kwargs_error)
+   _check_trust_remote_code(canonical)
    return PolicyClass(**resolved_kwargs)

Kwargs validation is pure (no import beyond what `_resolve_policy_class`
already did, no network, no code execution), so it is safe to run first.
The trust gate still fires when a well-typed kwarg-bag would otherwise
proceed to actual model loading.

Run
---
    $ export -n STRANDS_TRUST_REMOTE_CODE
    $ python create_policy_trust_gate_before_kwargs_repro.py
"""

from __future__ import annotations

import os
import sys
import warnings


def main() -> int:
    os.environ.pop("STRANDS_TRUST_REMOTE_CODE", None)
    warnings.simplefilter("ignore")

    from strands_robots.policies.factory import create_policy

    # A kwarg no policy binds. `pretrained_path` is the shape lerobot 0.4
    # docs used; strands-robots picked `pretrained_name_or_path`, so a user
    # copying from the outer ecosystem hits this.
    BAD = {"pretrained_path": "hf://user/model"}

    print("=== create_policy('kimodo', pretrained_path='...') ===")
    print("Round 1 (fresh env, no STRANDS_TRUST_REMOTE_CODE):")
    try:
        create_policy("kimodo", **BAD)
    except Exception as exc:  # noqa: BLE001
        print(f"  {type(exc).__name__}: {str(exc).splitlines()[0]}")
    if os.environ.get("STRANDS_TRUST_REMOTE_CODE") is None:
        print("  ^ pre-fix: UntrustedRemoteCodeError (user is nudged to opt in)")

    print()
    print("Round 2 (user has now set STRANDS_TRUST_REMOTE_CODE=1):")
    os.environ["STRANDS_TRUST_REMOTE_CODE"] = "1"
    try:
        create_policy("kimodo", **BAD)
    except Exception as exc:  # noqa: BLE001
        print(f"  {type(exc).__name__}: {str(exc).splitlines()[0]}")
    print("  ^ NOW the typo is visible (pre-fix); with the fix, this is Round 1")

    print()
    print("=== Reference: non-gated provider reports the typo on Round 1 ===")
    os.environ.pop("STRANDS_TRUST_REMOTE_CODE", None)
    try:
        create_policy("microduck", **BAD)
    except Exception as exc:  # noqa: BLE001
        print(f"  {type(exc).__name__}: {str(exc).splitlines()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
