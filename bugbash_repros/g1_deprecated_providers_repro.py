"""
Repro for cagataycali/robots-harness#<pending>:
docs/robots/unitree_g1.md:44 lists `kimodo` and `protomotions` under
"Providers written for this body" with NO deprecation marker. Both providers
are declared "deprecated, removed in 0.7" at docs/learn/policies/index.md:21
and carry `!!! warning "Deprecated, removed in 0.7"` banners on their own pages
(docs/learn/policies/kimodo.md:7, docs/learn/policies/protomotions.md:7).

A new user who follows the "Providers written for this body" sentence as a
recommendation builds against a provider that disappears next release, with
no in-tree marker on the page they read.

G1 is the ONLY robot page in docs/robots/*.md that mentions either name.
"""
import pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

g1 = (ROOT / "docs/robots/unitree_g1.md").read_text()
index = (ROOT / "docs/learn/policies/index.md").read_text()
kimodo = (ROOT / "docs/learn/policies/kimodo.md").read_text()
protomo = (ROOT / "docs/learn/policies/protomotions.md").read_text()

# 1. canonical deprecation prose in index.md:21
assert 'Deprecated providers (`moveit2`, `kimodo`, `protomotions`) stay in the table until 0.7' in index

# 2. canonical banners on provider pages
assert '!!! warning "Deprecated, removed in 0.7"' in kimodo
assert '!!! warning "Deprecated, removed in 0.7"' in protomo

# 3. find the recommendation line
line = next(l for l in g1.splitlines() if l.startswith("Providers written for this body"))
print("G1 recommendation line (upstream main):")
print(" ", line)
assert "kimodo" in line and "protomotions" in line

# 4. defect check: upstream main has no marker on this line
if "deprecated" not in line.lower() and "0.7" not in line:
    print("\nDEFECT REPRODUCED: unitree_g1.md:44 names kimodo+protomotions with no marker")
    print("  compare index.md:21: 'Deprecated providers (moveit2, kimodo, protomotions) stay in the table until 0.7'")
    print("  compare kimodo.md:7:        !!! warning Deprecated, removed in 0.7")
    print("  compare protomotions.md:7:  !!! warning Deprecated, removed in 0.7")
    sys.exit(1)
else:
    print("\nFIX APPLIED: recommendation line now carries the deprecation marker.")
    sys.exit(0)
