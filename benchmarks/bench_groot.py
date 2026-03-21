#!/usr/bin/env python3
"""Benchmark: GR00T policy via inference server.

Profiles the GR00T gRPC client path: connect → send observation → receive actions.
Requires a running GR00T inference server (gr00t-server).

Usage:
    python benchmarks/bench_groot.py
    python benchmarks/bench_groot.py --server 10.0.0.10:50051
    python benchmarks/bench_groot.py --data-config so100 --n-warm 50
    python benchmarks/bench_groot.py --json --out results/
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


def run(server: str, data_config: str, n_warm: int = 20):
    branch, sha = _git_info()

    print(f"🔬 bench_groot ({branch}@{sha})")
    print(f"   server: {server}")
    print(f"   data_config: {data_config}")
    print("=" * 65)

    with phase("import strands_robots"):
        import strands_robots  # noqa: F401

    with phase("import Gr00tPolicy"):
        from strands_robots.policies.groot.policy import Gr00tPolicy

    with phase("import Gr00tInferenceClient"):
        from strands_robots.policies.groot.client import Gr00tInferenceClient

    with phase("import load_data_config"):
        from strands_robots.policies.groot.data_config import load_data_config

    with phase(f"load_data_config('{data_config}')"):
        dc = load_data_config(data_config)
        print(f"    → state={dc.state_keys} action={dc.action_keys}")

    # Create policy (connects to server)
    with phase("create Gr00tPolicy"):
        policy = Gr00tPolicy(
            server_url=server,
            data_config=data_config,
        )

    # Build synthetic observation matching data_config
    with phase("build observation"):
        obs = {}
        # State keys → random float arrays
        for sk in dc.state_keys:
            obs[f"observation.{sk}"] = np.random.randn(6).astype(np.float32)
        # Video keys → random images
        for vk in dc.video_keys:
            obs[vk] = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Cold inference (includes potential gRPC channel warmup)
    with phase("inference cold"):
        try:
            actions_cold = policy.get_actions_sync(obs, "bench_cold")
            print(f"    → {len(actions_cold)} action chunks")
        except Exception as e:
            print(f"    → FAILED: {e}")
            _phases[-1]["error"] = str(e)

    # Warm inference loop
    warm_times = []
    errors = 0
    for i in range(n_warm):
        t0 = time.time()
        try:
            policy.get_actions_sync(obs, "bench_warm")
        except Exception:
            errors += 1
        warm_times.append((time.time() - t0) * 1000)

    if warm_times:
        warm_rec = {
            "name": f"inference warm ({n_warm}x)",
            "ms": round(np.mean(warm_times), 4),
            "peak_mb": 0,
            "p50_ms": round(np.percentile(warm_times, 50), 4),
            "p95_ms": round(np.percentile(warm_times, 95), 4),
            "p99_ms": round(np.percentile(warm_times, 99), 4),
            "errors": errors,
        }
        _phases.append(warm_rec)
        print(
            f"  {'inference warm (' + str(n_warm) + 'x)':45s} "
            f"mean={warm_rec['ms']:.2f}ms  "
            f"p50={warm_rec['p50_ms']:.2f}ms  "
            f"p99={warm_rec['p99_ms']:.2f}ms"
            f"{'  ⚠️ ' + str(errors) + ' errors' if errors else ''}"
        )

    # Summary
    print("\n" + "=" * 65)
    print("📊 SUMMARY")
    print("=" * 65)
    for p in _phases:
        extras = ""
        if p.get("p50_ms"):
            extras = f" (p50={p['p50_ms']}ms p99={p['p99_ms']}ms)"
        if p.get("error"):
            extras = f" ❌ {p['error'][:60]}"
        print(f"  {p['name']:45s} {p['ms']:>10.2f}ms  {p['peak_mb']:>6.1f}MB{extras}")

    return {
        "benchmark": "bench_groot",
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "arch": platform.machine(),
            "branch": branch,
            "commit": sha,
            "server": server,
            "data_config": data_config,
            "n_warm": n_warm,
        },
        "phases": _phases,
        "warm_inference_series_ms": [round(t, 4) for t in warm_times],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark GR00T policy via inference server"
    )
    parser.add_argument(
        "--server",
        type=str,
        default="localhost:50051",
        help="GR00T inference server address (default: localhost:50051)",
    )
    parser.add_argument(
        "--data-config",
        type=str,
        default="so100",
        help="Data config name (default: so100)",
    )
    parser.add_argument(
        "--n-warm",
        type=int,
        default=20,
        help="Number of warm inference iterations (default: 20)",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON to stdout")
    parser.add_argument("--out", type=str, help="Output directory for artifacts")
    args = parser.parse_args()

    results = run(
        server=args.server, data_config=args.data_config, n_warm=args.n_warm
    )

    if args.json:
        print(json.dumps(results, indent=2))

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")

        out_file = out_dir / f"bench_groot_{ts}.json"
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"📁 Saved: {out_file}")
