"""
Repro: `[all]` extra silently omits sim backends & robots the README advertises.

README promises (strands-labs/robots@main):
  - L95 "150+ robots across 8 categories - ... aerial ..."
  - L99 "Simulate with an agent-callable MuJoCo tool: ... Newton and Isaac backends"
  - L101 "ROS 2 - observe and command any graph (use_ros)"

But `pip install 'strands-robots[all]'` resolves without:
  - sim-newton (6 real deps: newton, warp-lang, mujoco pin, ...)
  - sim-isaac  (7 real deps: usd-core, imageio, ...)
  - sim-gs     (3 real deps: gsplat, plyfile, torch)
  - crazyflie  (1 real dep: cflib)  → the aerial robot the registry ships
  - ros2       (1 real dep: cyclonedds)
  - voice      (1 real dep: strands-agents[bidi])
  - cosmos3-service / cosmos3-diffusers / cosmos3-sim  (9 real deps combined)
  - ur         (1 real dep: ur-rtde)
  - sim        (1 real dep: robot_descriptions) ← nested helper

This is static-truth: no install needed, we read pyproject.toml.

Exit codes:
  0 - fixed (every non-empty omission is either in [all] or has an inline comment
      explaining why it is held back)
  1 - defect present (silent omission)
"""
from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
README    = REPO_ROOT / "README.md"

def main() -> int:
    with PYPROJECT.open("rb") as f:
        data = tomllib.load(f)
    od = data["project"]["optional-dependencies"]
    all_extras = od.get("all", [])

    # Which named extras does [all] cite via `strands-robots[<name>]`?
    cited: set[str] = set()
    for d in all_extras:
        m = re.match(r"strands-robots\[([^\]]+)\]", d)
        if m:
            cited.add(m.group(1))

    meta = {"all", "dev"}
    declared = set(od.keys()) - meta
    omitted  = declared - cited

    # Non-empty omissions = real functional capability missing
    non_empty_omissions = {name: od[name] for name in omitted if od[name]}

    # The README promises these capabilities (verify literal text is still there)
    readme_text = README.read_text(encoding="utf-8")
    readme_asserts = {
        "sim-newton": "Newton and Isaac backends",
        "sim-isaac":  "Newton and Isaac backends",
        "crazyflie":  "aerial",      # row L95
        "ros2":       "ROS 2",       # row L101
        "voice":      None,           # not README-promised; omission allowed if commented
        "sim-gs":     None,           # not README-promised
        "cosmos3-service":  None,
        "cosmos3-diffusers":None,
        "cosmos3-sim":      None,
        "ur":        "ur",            # 24 occurrences, UR robots are a documented category
        "sim":       None,            # internal helper
    }

    print(f"[all] cites {len(cited)} named extras.")
    print(f"Declared extras: {len(declared)}.")
    print(f"Non-empty omissions from [all]: {len(non_empty_omissions)}.\n")

    failures: list[str] = []

    for name, needs_readme_token in readme_asserts.items():
        if name not in non_empty_omissions:
            continue
        if needs_readme_token:
            hit = needs_readme_token in readme_text
            if hit:
                failures.append(
                    f"[all] omits [{name}] (deps: {non_empty_omissions[name]}) "
                    f"but README advertises {needs_readme_token!r}"
                )
            else:
                print(f"  skip {name!r}: no README promise {needs_readme_token!r}")
        else:
            # Omission allowed only if pyproject carries an inline comment on the all/extra
            # fragment that cites a reason. Quick heuristic: look for the extra name near
            # a '# ' line inside the [project.optional-dependencies] section for `all`.
            src = PYPROJECT.read_text(encoding="utf-8")
            # Grab the all = [...] block
            m = re.search(r"^all\s*=\s*\[(.*?)^\]", src, re.MULTILINE | re.DOTALL)
            block = m.group(1) if m else ""
            mentioned = name in block or re.search(rf"#\s*{re.escape(name)}", block)
            if not mentioned:
                failures.append(
                    f"[all] omits [{name}] (deps: {non_empty_omissions[name]}) "
                    f"with no inline comment citing why"
                )

    # Report
    if failures:
        print(f"DEFECT: {len(failures)} silent omissions in [all]:")
        for f in failures:
            print(f"  - {f}")
        print()
        print("Non-empty omissions dump:")
        for name, deps in sorted(non_empty_omissions.items()):
            print(f"  [{name}]: {deps}")
        return 1

    print("OK: every non-empty omission from [all] is either not README-promised or commented.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
