"""
Minimal repro: every documented quickstart in docs/learn/ teaches
``sim.add_robot("<name>")`` — the exact form that `add_robot` now flags
as DEPRECATED. A new user following docs gets scolded on their first call,
and the hint cannot be silenced via ``warnings.filterwarnings`` because
it is emitted via ``logger.info`` + an in-envelope text banner rather than
the ``DeprecationWarning`` the source comment on
``strands_robots/simulation/mujoco/simulation.py:2614-2619`` promises.

Scope:
  - docs/learn/simulation/index.md:15      sim.add_robot("so101")        # the hero quickstart
  - docs/learn/simulation/mujoco.md:23     sim.add_robot("so101")
  - docs/learn/simulation/mjlab.md:22      sim.add_robot("so101")
  - docs/learn/simulation/newton.md:22     sim.add_robot("so100")
  - docs/learn/simulation/isaac.md:33,62   sim.add_robot("so101") × 2
  - docs/learn/simulation/predicates-and-rollouts.md:18   sim.add_robot("so101")
  - docs/learn/simulation/worlds-and-objects.md:15        sim.add_robot("so101", position=…)
  - docs/learn/simulation/randomization.md:14             sim.add_robot("so101")
  - docs/learn/policies/{index,wbc,wbc-latent,kimodo,holosoma,protomotions,
      microduck,curobo,remote,moveit2,lerobot-local,rl}.md
  - docs/learn/training/rl.md
  - examples/ros2/sim_bridge_demo.py:33
  - examples/mjlab/vec_eval_bench.py:59, 108, 201
  (23 docs sites + 4 example-script sites — ``grep -rn 'add_robot("' docs examples``)

Also: the SAME page (docs/learn/simulation/index.md:72-73) literally advertises:

    `add_robot(name)` resolves a registry name (`so101`, `panda`, `g1`, `go2`, …)
    through `strands_robots.simulation.model_registry`;
    `add_robot(name="arm", data_config="franka")` gives the instance its own name.

So the quickstart teaches X, the deprecation hint accuses the user who did X,
and the deprecation comment (…"kept for one release with a DeprecationWarning")
claims the hint is a `DeprecationWarning` — but it never is.

Run with the project installed editable:

    cd $(git rev-parse --show-toplevel)
    python bugbash_repros/quickstart_docs_teach_deprecated_add_robot_form_repro.py

Expected (if docs & impl agreed, OR deprecation was emitted properly):
    - The documented quickstart form should not nag; OR
    - The nag should be a real `DeprecationWarning` the user can filter.

Actual (v0.5.3 HEAD = 2a0e84d98, 2026-10-07):
    - Documented form `sim.add_robot("so101")` → status=success BUT with a
      `Warning: Hint: … deprecated name-as-registry-key fallback …` banner
      appended to the content payload.
    - `warnings.simplefilter("always")` + a custom `warnings.showwarning`
      hook catches ZERO Python warnings from the deprecation path
      (only unrelated `ResourceWarning` on /proc/cpuinfo).
    - `logger.info("add_robot: resolved model via instance name …")` fires,
      so the hint is actually at INFO level — below the project's default
      WARNING threshold, invisible unless the user bumps logging.
    - Doc contradicts itself: `index.md:72-73` prose teaches the deprecated
      form; `index.md:15` demonstrates it; the engine now flags it.
"""
from __future__ import annotations

import os
import warnings

os.environ.setdefault("MUJOCO_GL", "egl")

from strands_robots.simulation import create_simulation


def _capture_warnings():
    caught: list[tuple[str, str]] = []

    def _hook(message, category, filename, lineno, file=None, line=None):
        caught.append((category.__name__, str(message)))

    warnings.simplefilter("always")
    warnings.showwarning = _hook
    return caught


def main() -> int:
    print("=" * 72)
    print("DOCUMENTED quickstart form (docs/learn/simulation/index.md:15)")
    print("    sim.add_robot('so101')")
    print("=" * 72)

    caught = _capture_warnings()

    sim = create_simulation("mujoco")
    sim.create_world(timestep=0.002)
    r = sim.add_robot("so101")  # <-- this is literally what the hero quickstart shows
    sim.cleanup()

    print(f"\nstatus: {r.get('status')!r}  (succeeded — but with a nag)")
    banner = None
    for c in r.get("content", []):
        text = c.get("text", "")
        if "deprecated name-as-registry-key" in text:
            banner = text.splitlines()[-1]
            break
    print(f"content banner: {banner!r}")

    py_dep_warnings = [c for c in caught if c[0] == "DeprecationWarning"]
    print(f"\nPython DeprecationWarnings caught: {len(py_dep_warnings)}")
    print("  (promised by simulation.py:2614-2619 comment, actually emitted via")
    print("   logger.info + an in-envelope text banner — unfilterable, invisible)")

    # Preferred form from the hint — same scene, no banner:
    print()
    print("=" * 72)
    print("Hint-preferred form:")
    print("    sim.add_robot(name='arm', data_config='so101')")
    print("=" * 72)
    sim2 = create_simulation("mujoco")
    sim2.create_world(timestep=0.002)
    r2 = sim2.add_robot(name="arm", data_config="so101")
    sim2.cleanup()
    banner2 = None
    for c in r2.get("content", []):
        text = c.get("text", "")
        if "deprecated name-as-registry-key" in text:
            banner2 = text.splitlines()[-1]
            break
    print(f"status: {r2.get('status')!r}, banner: {banner2!r}  (silent — expected)")

    # Prove it fires for g1 too (every humanoid/manipulation page is affected):
    print()
    print("=" * 72)
    print("Humanoid page example (docs/learn/policies/wbc.md:57, holosoma.md:54,")
    print("kimodo.md, wbc-latent.md:45, protomotions.md, microduck.md):")
    print("    sim.add_robot('g1')")
    print("=" * 72)
    sim3 = create_simulation("mujoco")
    sim3.create_world(timestep=0.002)
    r3 = sim3.add_robot("g1")
    sim3.cleanup()
    banner3 = None
    for c in r3.get("content", []):
        text = c.get("text", "")
        if "deprecated name-as-registry-key" in text:
            banner3 = text.splitlines()[-1]
            break
    print(f"status: {r3.get('status')!r}")
    print(f"content banner: {banner3!r}")

    print()
    print("=" * 72)
    print("Summary")
    print("=" * 72)
    print(" - Hero quickstart teaches the deprecated form. (docs/learn/simulation/index.md:15)")
    print(" - 23 docs pages + 4 examples scripts call `add_robot('<name>')`.")
    print(" - Deprecation is emitted via `logger.info` + text banner, NOT")
    print("   as a `DeprecationWarning` — unfilterable, invisible at default log level.")
    print(" - Source comment at simulation.py:2614-2619 claims it IS a DeprecationWarning.")

    # Non-zero exit so CI pipelines pick this up when linked.
    return 1 if banner else 0


if __name__ == "__main__":
    raise SystemExit(main())
