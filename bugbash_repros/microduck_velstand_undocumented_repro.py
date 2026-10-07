"""Repro: velstand.onnx is the upstream-default walking policy since microduck
weights v5, but strands-robots' docs, registry aliases, hint lists, and driver
docstring exclusively name alpha_walking.onnx. A user who follows docs runs a
legacy weight; a user who runs the correct weight (velstand.onnx) finds no
reference to it anywhere in strands-robots.

Run:
    python3 bugbash_repros/microduck_velstand_undocumented_repro.py

Preconditions:
    - huggingface_hub installed (comes with the `microduck` extra, or
      `pip install huggingface_hub`)
    - Network access to the Hugging Face Hub (read-only).

Expected (what docs claim):
    The microduck provider documents the complete shipped-weight set, and the
    default walk policy named by docs matches what Pollen Robotics ships as
    its default walking policy.

Actual:
    Pollen's manifest.json on `pollen-robotics/microduck-policies` ships 10
    .onnx weights; `velstand.onnx` is marked
    `kind: "perpetual", slot: "walk"` with description:
      "Walking gait that also stands still at zero command; the default walk
       policy since v5, with no separate standing network."

    Nothing in strands-robots knows this file name:
      - docs/learn/policies/microduck.md: lists 9 shipped weights (no velstand)
      - docs/learn/hardware/microduck.md: names alpha_walking.onnx only
      - docs/robots/microduck.md: cites verification against alpha_walking.onnx
      - strands_robots/registry/policies.py:110: microduck_walk alias -> alpha_walking.onnx
      - strands_robots/policies/microduck/__init__.py / policy.py /
        composite.py / drivers/microduck.py: all mention alpha_walking only.

    The user who follows every documented path gets alpha_walking (legacy,
    needs separate alpha_stand network to hold still). The user who reads the
    Pollen manifest and asks for velstand.onnx finds no mention of it in
    strands-robots at all. The _shipped_weights_hint() does surface velstand in
    the 404-only path, so the knowledge of this file lives in the Hub listing
    alone.

Fix sketch (not applied here):
    - Expand docs/learn/policies/microduck.md paragraph + skill table to list
      velstand and note it supersedes alpha_walking/alpha_stand as the default
      walk-and-stand policy since v5.
    - Add a `microduck_velstand` alias to strands_robots/registry/policies.py,
      beside the `microduck_walk`/`microduck_stand` aliases.
    - Update docs/learn/hardware/microduck.md so the first-touch example names
      velstand.onnx (while keeping alpha_walking as a secondary example).

This repro only asserts the three observable facts.
"""

from __future__ import annotations

import re
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files

MICRODUCK_REPO = "pollen-robotics/microduck-policies"
# Resolve the repo root the way `strands_robots` itself is laid out: this file
# ships at `<repo>/bugbash_repros/`, so the strands_robots root is one up.
REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    # 1. Hub carries 10 .onnx weights, including velstand.onnx.
    files = list_repo_files(MICRODUCK_REPO)
    shipped = sorted(f for f in files if f.endswith(".onnx"))
    print(f"[1] Hub `{MICRODUCK_REPO}` carries {len(shipped)} .onnx weights:")
    for f in shipped:
        marker = "  <-- the undocumented one" if f == "velstand.onnx" else ""
        print(f"      {f}{marker}")
    assert "velstand.onnx" in shipped, "velstand.onnx missing from the Hub listing"

    # 2. The manifest declares velstand as the default walk policy since v5.
    manifest_path = hf_hub_download(MICRODUCK_REPO, "manifest.json")
    import json

    manifest = json.loads(Path(manifest_path).read_text())
    policies = manifest.get("policies", [])
    velstand_entry = next((p for p in policies if p.get("file") == "velstand.onnx"), None)
    assert velstand_entry is not None, "velstand.onnx missing from manifest.json"
    print()
    print("[2] Pollen's manifest.json entry for velstand.onnx:")
    print(f"      kind: {velstand_entry.get('kind')!r}")
    print(f"      slot: {velstand_entry.get('slot')!r}")
    print(f"      entry_pose: {velstand_entry.get('entry_pose')!r}")
    print(f"      description: {velstand_entry.get('description', '')!r}")
    assert velstand_entry.get("slot") == "walk", "velstand no longer claims the walk slot"
    assert "default walk policy" in velstand_entry.get("description", ""), (
        "manifest description changed; this repro's premise needs re-reading"
    )

    # 3. strands-robots nowhere mentions velstand.
    docs_root = REPO_ROOT / "docs"
    code_root = REPO_ROOT / "strands_robots"
    hits = []
    for root in (docs_root, code_root):
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix in (".pyc",):
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if re.search(r"velstand", text):
                hits.append(p.relative_to(REPO_ROOT))
    print()
    print("[3] Files in docs/ or strands_robots/ that mention velstand:")
    if hits:
        for h in hits:
            print(f"      {h}")
    else:
        print("      (none)")

    # 4. meanwhile, alpha_walking is named everywhere.
    alpha_hits_docs = 0
    alpha_hits_code = 0
    for root, counter_name in ((docs_root, "docs"), (code_root, "code")):
        for p in root.rglob("*"):
            if not p.is_file() or p.suffix in (".pyc",):
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if "alpha_walking" in text:
                if counter_name == "docs":
                    alpha_hits_docs += 1
                else:
                    alpha_hits_code += 1
    print()
    print(
        f"[4] By contrast `alpha_walking` is named in "
        f"{alpha_hits_docs} docs/ files and {alpha_hits_code} strands_robots/ files."
    )

    # 5. Assert the mismatch.
    assert not hits, (
        f"velstand IS now mentioned in strands-robots ({len(hits)} files). "
        "This defect is fixed; retire the repro."
    )

    print()
    print("DEFECT CONFIRMED: velstand.onnx is on the Hub as the upstream-default")
    print("walk policy since v5, but strands-robots does not know its name.")
    print("A user who follows docs/learn/policies/microduck.md runs the legacy")
    print("alpha_walking.onnx; a user who reads Pollen's manifest and asks for")
    print("velstand finds no reference to it in strands-robots at all.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
