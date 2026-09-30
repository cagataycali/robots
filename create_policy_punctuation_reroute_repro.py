"""Reproducer: create_policy() silently reroutes punctuation-shaped typos to lerobot_local.

Symptom
-------
`create_policy("wbc/")` (trailing slash typo of the real provider `wbc`) does
NOT produce the same "Unknown policy provider 'wbc/'. Did you mean: 'wbc'?"
error that `create_policy("wbcc")`, `create_policy("Wbc")`, and even
`create_policy("wbc ")` (trailing space) produce. Instead it:

  1. Passes the `_is_smart_string` check (has a `/`),
  2. Reaches stage 3 of `resolve_policy` ("has slash → HuggingFace repo id"),
  3. Silently rewrites the caller's request as `lerobot_local` with
     `pretrained_name_or_path="wbc/"`,
  4. Surfaces an `UntrustedRemoteCodeError` (or, with the opt-in set, an HF
     lookup failure) naming a provider the user never asked for.

Same class of bug applies to any user typo that happens to contain `/` or `:`:
`":"`, `"/"`, `"C:\\path"`, `"wbc/"`, `"protomotions:"`, etc. The
`provider_can_be_created(":")` preflight even returns True for `":"` — it
tells hardware entry points "yes, this string will build" for a bare colon.

Root cause: `strands_robots/policies/factory.py:189-197` — `_is_smart_string`
treats "any string containing / or non-alpha : or ws:// or grpc://" as a smart
string. It gates the smart-string resolver, which last-resorts to
`lerobot_local`. There's no shape validation on what an HF repo id / URL
actually looks like: `":"`, `"/"`, `"wbc/"`, `"C:\\path"` all pass.

The did-you-mean logic in `import_policy_class` (factory.py:317-340) is only
reached when `_is_smart_string` returns False. So the ONLY typos that get
suggestions are ones without `/` or `:`. Every typo that happens to contain
punctuation slips through to `lerobot_local` with no correction hint.

Upstream anchors (strands-labs/robots @ e6eaf86):
  - strands_robots/policies/factory.py:189-197  (`_is_smart_string`)
  - strands_robots/policies/factory.py:369-379  (smart-string dispatch)
  - strands_robots/policies/factory.py:222-232  (`provider_can_be_created`)
  - strands_robots/registry/policies.py:559-564 (stage-5 fallback)

Run:
    python create_policy_punctuation_reroute_repro.py
"""
from __future__ import annotations

import sys
import warnings

# Silence the "Unrecognised policy … falling back to lerobot_local" WARNING so
# the demonstration output is clean. The warning is issued by resolve_policy on
# stderr at WARNING level - not an exception the caller can trap.
import logging
logging.getLogger("strands_robots.registry.policies").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

from strands_robots.policies.factory import (  # noqa: E402
    _is_smart_string,
    create_policy,
    provider_can_be_created,
)


PROVIDER_TYPOS = [
    # (spelling, expected_behaviour)
    ("wbcc",       "did-you-mean 'wbc'"),   # baseline: no punctuation → suggestion
    ("Wbc",        "did-you-mean 'wbc'"),   # case fold → suggestion
    ("wbc ",       "did-you-mean 'wbc'"),   # trailing space → suggestion
    ("wbc/",       "did-you-mean 'wbc'"),   # trailing slash typo
    (":",          "did-you-mean or ValueError('not a policy provider')"),
    ("/",          "did-you-mean or ValueError('not a policy provider')"),
    ("protomotions:", "did-you-mean 'protomotions'"),  # trailing colon typo
    ("C:\\path",   "ValueError (Windows path is not a provider)"),
]


def summarise(exc: Exception) -> str:
    txt = str(exc).replace("\n", " ")
    return f"{type(exc).__name__}: {txt[:180]}"


def main() -> int:
    hits: list[str] = []
    for spelling, expected in PROVIDER_TYPOS:
        try:
            create_policy(spelling)
        except Exception as exc:
            got = summarise(exc)
        else:
            got = "returned a policy (silent-wrong!)"

        print(f"provider={spelling!r:16s} _is_smart_string={_is_smart_string(spelling)!s:5}  "
              f"provider_can_be_created={provider_can_be_created(spelling)!s:5}")
        print(f"    expected: {expected}")
        print(f"    got:      {got}")
        print()

        # Any typo that reroutes to lerobot_local without a did-you-mean is a hit.
        if "lerobot_local" in got.lower() and "did you mean" not in got.lower():
            hits.append(spelling)

    print("=" * 60)
    if hits:
        print(f"FAIL: {len(hits)} typos silently rerouted to lerobot_local:")
        for h in hits:
            print(f"  - {h!r}")
        print()
        print("Every one of these SHOULD have produced a did-you-mean suggestion")
        print("(or a shape-rejection ValueError), NOT a report naming a provider")
        print("the caller did not request.")
        return 1
    print("PASS: every typo produced a did-you-mean or shape-rejection.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
