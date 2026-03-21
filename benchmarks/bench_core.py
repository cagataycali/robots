#!/usr/bin/env python3
"""Benchmark: strands_robots core (non-lerobot paths).

Profiles imports, registry, mock policy, data configs — everything
that does NOT require torch/lerobot/network.

Usage:
    python benchmarks/bench_core.py
    python benchmarks/bench_core.py --json          # JSON to stdout
    python benchmarks/bench_core.py --out results/   # write artifacts to dir
"""

import argparse
import gc
import json
import platform
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

_phases = []


class _Phase:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        gc.collect()
        tracemalloc.start()
        self.t0 = time.time()
        return self

    def __exit__(self, *_):
        elapsed = time.time() - self.t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rec = {
            "name": self.name,
            "ms": round(elapsed * 1000, 2),
            "peak_mb": round(peak / 1e6, 2),
        }
        _phases.append(rec)
        print(f"  {self.name:45s} {rec['ms']:>10.2f}ms  {rec['peak_mb']:>6.1f}MB")


def phase(name):
    return _Phase(name)


def _git_info():
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"], text=True
        ).strip()
        return branch, sha
    except Exception:
        return "unknown", "unknown"


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def run():
    branch, sha = _git_info()
    print(f"🔬 bench_core ({branch}@{sha})")
    print("=" * 65)

    with phase("import strands_robots"):
        import strands_robots  # noqa: F401

    with phase("import policies.factory"):
        from strands_robots.policies.factory import create_policy, list_providers  # noqa: F401

    with phase("import policies.base"):
        from strands_robots.policies.base import Policy  # noqa: F401

    with phase("import registry"):
        from strands_robots.registry import (  # noqa: F401
            format_robot_table,
            get_policy_provider,
            get_robot,
            list_aliases,
            list_policy_providers,
            list_robots,
            resolve_policy,
        )

    with phase("list_providers()"):
        providers = list_providers()

    with phase("list_robots()"):
        robots = list_robots()

    with phase("list_policy_providers()"):
        pprovs = list_policy_providers()

    with phase("list_aliases()"):
        aliases = list_aliases()

    # Robot config lookups
    for r in robots[:6]:
        rname = r["name"] if isinstance(r, dict) else str(r)
        with phase(f"get_robot({rname})"):
            get_robot(rname)

    # Policy provider lookups
    for pn in pprovs:
        with phase(f"get_policy_provider({pn})"):
            get_policy_provider(pn)

    # GR00T module imports (no server needed)
    with phase("import groot.client"):
        from strands_robots.policies.groot.client import Gr00tInferenceClient  # noqa: F401

    with phase("import groot.policy"):
        from strands_robots.policies.groot.policy import Gr00tPolicy  # noqa: F401

    with phase("import groot.data_config"):
        from strands_robots.policies.groot.data_config import load_data_config  # noqa: F401

    # Mock policy lifecycle
    with phase("import mock.MockPolicy"):
        from strands_robots.policies.mock import MockPolicy

    with phase("MockPolicy(6-DOF) create"):
        mock = MockPolicy(
            action_space={f"j{i}": 0.0 for i in range(6)}
        )

    obs = {"observation.state": np.zeros(6, dtype=np.float32)}

    with phase("mock cold inference"):
        mock.get_actions_sync(obs, "test")

    # Warm loop
    times = []
    for _ in range(100):
        t0 = time.time()
        mock.get_actions_sync(obs, "test")
        times.append((time.time() - t0) * 1000)
    _phases.append(
        {
            "name": "mock warm inference (100x)",
            "ms": round(np.mean(times), 4),
            "peak_mb": 0,
            "p50_ms": round(np.percentile(times, 50), 4),
            "p99_ms": round(np.percentile(times, 99), 4),
        }
    )
    print(
        f"  {'mock warm inference (100x)':45s} "
        f"mean={np.mean(times):.4f}ms  "
        f"p50={np.percentile(times, 50):.4f}ms  "
        f"p99={np.percentile(times, 99):.4f}ms"
    )

    # Data config
    with phase("load_data_config('so100')"):
        dc = load_data_config("so100")

    # Tools (lazy)
    with phase("import tools"):
        from strands_robots import tools  # noqa: F401

    # Policy resolution (no creation)
    for pol in providers:
        with phase(f"resolve_policy('{pol}')"):
            resolve_policy(pol)

    with phase("format_robot_table()"):
        format_robot_table()

    # Heavy import (torch transitive)
    with phase("import robot.Robot"):
        from strands_robots.robot import Robot  # noqa: F401

    # --- Summary ---
    print("\n" + "=" * 65)
    print("📊 SUMMARY")
    print("=" * 65)
    for p in _phases:
        extras = ""
        if p.get("p50_ms"):
            extras = f" (p50={p['p50_ms']}ms p99={p['p99_ms']}ms)"
        print(f"  {p['name']:45s} {p['ms']:>10.2f}ms  {p['peak_mb']:>6.1f}MB{extras}")

    return {
        "benchmark": "bench_core",
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "arch": platform.machine(),
            "branch": branch,
            "commit": sha,
        },
        "phases": _phases,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark strands_robots core paths")
    parser.add_argument("--json", action="store_true", help="Print JSON to stdout")
    parser.add_argument("--out", type=str, help="Output directory for artifacts")
    args = parser.parse_args()

    results = run()

    if args.json:
        print(json.dumps(results, indent=2))

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_file = out_dir / f"bench_core_{ts}.json"
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n📁 Saved: {out_file}")
