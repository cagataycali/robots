"""Repro: docs/concepts/backends.md's "Simulation backends" table omits mjlab
entirely and mislabels isaac as a "plugin `strands-robots-sim`" when it is in
fact a built-in backend installed through the in-tree extra `[sim-isaac]`.

Source of truth: ``strands_robots.simulation.factory._BUILTIN_BACKENDS``
and ``pyproject.toml`` ``[project.optional-dependencies]``.

A user who reads concepts/backends.md first (the entry-level backend page
linked from docs/concepts/architecture.md and the Simulation Lane in the
IA) is told:
    (a) there are three backends; and
    (b) isaac ships in a separate PyPI package ``strands-robots-sim``.

Both are false:
    (a) there are four built-in backends: mujoco, newton, isaac, mjlab.
        mjlab is also referenced throughout the rest of the docs tree:
          - docs/learn/simulation/mjlab.md (a full page with pip-install,
            benchmarks and a Robot(..., backend="mjlab") sketch);
          - docs/learn/simulation/index.md (the "You should see" block
            at line 28 literally prints 'mjlab' from list_backends());
          - docs/hooks/extras.py (sim-mjlab extra description);
          - docs/hooks/providers.py (rsl_rl_onnx -> sim-mjlab extra).
    (b) ``isaac`` is in ``_BUILTIN_BACKENDS`` and installs via the extra
        ``strands-robots[sim-isaac]`` (pyproject.toml:628). There is a
        separate out-of-tree plugin repo (strands-labs/robots-sim)
        referenced by factory.py:361 and robot.py:676, but it is not the
        documented install path for the built-in ``isaac`` backend.

Running this script with the pinned stack reproduces the mismatch without
needing any GPU or NVIDIA toolchain.
"""

from __future__ import annotations

import re
from pathlib import Path

from strands_robots.simulation.factory import _BUILTIN_BACKENDS, list_backends


def main() -> int:
    # --- Ground truth from the code ---
    actual_builtins = sorted(_BUILTIN_BACKENDS.keys())
    all_names = list_backends()

    print("=== Code: _BUILTIN_BACKENDS ===")
    print(actual_builtins)  # ['isaac', 'mjlab', 'mujoco', 'newton']
    print()
    print("=== Code: list_backends() ===")
    print(all_names)  # includes 'mjlab' + 'mujoco_warp' alias
    print()

    # --- What docs/concepts/backends.md claims ---
    doc = Path("docs/concepts/backends.md").read_text(encoding="utf-8")
    desc_line = next(line for line in doc.splitlines() if line.startswith("description:"))
    print("=== Docs: concepts/backends.md description frontmatter ===")
    print(desc_line)
    print()

    # Pull just the "Simulation backends" table.
    table_block = re.search(
        r"## Simulation backends\s+\|.*?\n\n", doc, flags=re.S
    )
    table_text = table_block.group(0) if table_block else ""
    print("=== Docs: Simulation backends table ===")
    print(table_text)

    # --- Reproduce the three defects ---
    docs_names_in_table = re.findall(r"\|\s*([A-Z][A-Za-z0-9 ]+?)\s*\|", table_text)
    docs_names_in_table = [n for n in docs_names_in_table if n not in {"backend", "selects with", "runs on", "for"}]

    assert actual_builtins == ["isaac", "mjlab", "mujoco", "newton"], actual_builtins
    assert "mjlab" in all_names
    assert "mjlab" not in doc.lower(), "If this starts to fail, the docs page added mjlab (good)"

    # Defect 1: the table (and its header prose) omits mjlab.
    print("DEFECT 1: 'Simulation backends' table omits mjlab.")
    print(f"  Table lists: {docs_names_in_table}")
    print(f"  Code built-ins: {actual_builtins}")
    print()

    # Defect 2: isaac row calls it a plugin shipped in strands-robots-sim.
    assert "plugin `strands-robots-sim`" in doc, "If this starts to fail, isaac row was corrected"
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    assert "sim-isaac = [" in pyproject
    print("DEFECT 2: isaac row claims 'plugin `strands-robots-sim`', but")
    print("  pyproject.toml declares a first-party extra `[sim-isaac]`;")
    print("  isaac is in _BUILTIN_BACKENDS (factory.py:57).")
    print()

    # Defect 3: the paragraph after the table says "All three build from
    # the same registry entry" — but there are four.
    assert "All three build from the same registry entry" in doc
    print("DEFECT 3: Paragraph after the table says 'All three build from the")
    print("  same registry entry'; there are four built-in backends.")
    print()

    print("All three assertions above are true against this commit "
          "(docs/concepts/backends.md disagrees with the code).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
