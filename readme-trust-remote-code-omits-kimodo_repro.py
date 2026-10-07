"""
Repro: README.md:128 Security section names the trust_remote_code gate on
       `lerobot_local` only. The gate frozenset includes `kimodo` too
       (strands_robots/policies/factory.py:260-263).

Expected: README naming matches gate (both or neither — SECURITY.md has nothing,
          so README is the only root-prose source of truth on scope).
Actual:   README names 1 of 2 gated providers; kimodo is omitted from README
          root entirely while being an in-tree v0.5.3 security-gated provider.

Also:     docs/learn/policies/index.md:78 ("lerobot_local needs
          STRANDS_TRUST_REMOTE_CODE=1") carries the same single-provider framing.
          docs/learn/policies/kimodo.md:14 + lerobot-local.md:15 both describe
          the gate correctly on their own pages.
"""
import os, re, sys
from pathlib import Path

os.environ.pop("STRANDS_TRUST_REMOTE_CODE", None)

from strands_robots.policies.factory import (
    _HF_REMOTE_CODE_PROVIDERS,
    _check_trust_remote_code,
    UntrustedRemoteCodeError,
)

# 1. Gate truth
gated = sorted(_HF_REMOTE_CODE_PROVIDERS)
print(f"gate providers: {gated}")
assert gated == ["kimodo", "lerobot_local"], gated

# 2. README truth
readme = Path(__file__).parent / "README.md"
text = readme.read_text()
sec_line = next(l for l in text.splitlines() if "trust_remote_code" in l)
print(f"README security line: {sec_line.strip()}")

in_readme = {name: name in text for name in gated}
print(f"README mentions: {in_readme}")

# 3. The asymmetry
assert in_readme["lerobot_local"], "lerobot_local should be in README"
assert not in_readme["kimodo"], "kimodo is omitted in README (the defect)"

# 4. Both gates actually fire identically at runtime
for prov in gated:
    try:
        _check_trust_remote_code(prov)
        print(f"{prov}: did not raise (bug)")
    except UntrustedRemoteCodeError as e:
        first = str(e).splitlines()[0]
        print(f"{prov} raises: {first}")

# 5. docs/learn/policies/index.md L78 carries the same single-provider framing
pindex = Path(__file__).parent / "docs" / "learn" / "policies" / "index.md"
pindex_line = next(
    (l for l in pindex.read_text().splitlines() if "STRANDS_TRUST_REMOTE_CODE" in l),
    "<missing>",
)
print(f"docs/learn/policies/index.md: {pindex_line.strip()}")

print()
print("defect: README root (the only root-prose source of truth, since")
print("        SECURITY.md names neither gate nor providers) understates the")
print("        trust_remote_code gate to 1 of 2 providers.")
