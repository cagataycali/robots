"""
Repro: `preflight_policy`, `preflight_reason`, `policy_overrides_preflight`
silently swallow a TypeError on non-string provider inputs.

Upstream (strands-labs/robots@92d1d5136):
  - strands_robots/policies/factory.py:938-949  (preflight_policy)
  - strands_robots/policies/factory.py:1049-1057 (policy_overrides_preflight)
  - preflight_reason delegates through policy_overrides_preflight

The three agent-tool companion helpers run `_resolve_policy_class(provider)`
inside a broad `except Exception: ... return False/None`. That branch is
documented as "swallow resolution failures" (unknown provider, missing optional
dep). But `_resolve_policy_class` ALSO raises `TypeError` from
`_provider_type_error` on non-string input (int/None/list/bytes/dict/Policy),
and the broad `except Exception` catches it too. Caller gets `False`/`None`
back as if the policy "has no preflight hook" — a silent-wrong verdict.

The sibling `policy_provider_error` (same file, line 1095) uses
`_provider_type_error` as an EXPLICIT pre-check BEFORE the resolution try/except
and returns the proper typed message:
    "policy_provider must be a string, got NoneType (None). Pass a provider
     name (list_providers() reports them), a HuggingFace model ID, or a server
     URL."

That same reusable helper (`_provider_type_error`, factory.py:481) is sitting
in the module. Fix is <10 LOC per function (3 locations).

Running this repro reproduces the asymmetry.
"""
from strands_robots.policies.factory import (
    preflight_policy,
    preflight_reason,
    policy_overrides_preflight,
    policy_provider_error,
    _resolve_policy_class,
)


def main() -> int:
    print("--- 1. sibling (policy_provider_error) refuses non-string with a typed message ---")
    for bad in [None, 123, b"wbc", [], {}]:
        msg = policy_provider_error(bad)
        print(f"  {bad!r} -> {msg!r}")
    print()

    print("--- 2. _resolve_policy_class ALSO refuses non-string with TypeError ---")
    for bad in [None, 123, b"wbc"]:
        try:
            _resolve_policy_class(bad)
        except TypeError as e:
            print(f"  {bad!r} -> TypeError: {str(e)[:80]}...")
    print()

    print("--- 3. preflight trio behaviour on non-string ---")
    print("    (pre-fix: silent False/None | post-fix: TypeError match sibling) ")
    silent_wrong = []
    propagated = []
    for bad in [None, 123, [], {}, b"wbc"]:
        row = {"input": repr(bad)}
        for name, fn, args in [
            ("overrides_preflight", policy_overrides_preflight, (bad,)),
            ("preflight_policy",    preflight_policy,          (bad, {"j1"})),
            ("preflight_reason",    preflight_reason,          (bad, lambda: {"j1"})),
        ]:
            try:
                row[name] = repr(fn(*args))
            except TypeError as e:
                row[name] = f"TypeError: {str(e)[:60]}..."
                propagated.append((bad, name))
        print(f"  {row}")
        if (
            row["overrides_preflight"] == "False"
            and row["preflight_policy"] == "None"
            and row["preflight_reason"] == "None"
        ):
            silent_wrong.append(bad)

    print()
    print("--- 4. verdict ---")
    if silent_wrong:
        print(f"REPRO (pre-fix): silent-wrong on {silent_wrong!r}")
        print("Caller cannot distinguish 'no preflight hook' from 'you handed me a non-string'.")
        return 1
    if propagated:
        print(f"POST-FIX: TypeError propagates cleanly for {sorted({str(i) for i,_ in propagated})}")
        print("Behaviour now matches the sibling policy_provider_error.")
        return 0
    print("Unexpected state.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
