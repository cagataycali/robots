#!/usr/bin/env python3
"""Benchmark: LeRobot local policy (ACT) end-to-end.

Profiles the full lifecycle: import → resolve from HF Hub → create policy →
cold inference → warm inference loop. Also exports Chrome Trace Format for
visualization in ui.perfetto.dev.

Usage:
    python benchmarks/bench_lerobot.py
    python benchmarks/bench_lerobot.py --policy lerobot/act_aloha_sim_transfer_cube_human
    python benchmarks/bench_lerobot.py --device cpu
    python benchmarks/bench_lerobot.py --json --out results/
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
_trace_events = []  # Chrome Trace Format
_t0_global = None


class _Phase:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        global _t0_global
        if _t0_global is None:
            _t0_global = time.time()
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

        # Chrome Trace event
        _trace_events.append(
            {
                "name": self.name,
                "cat": "benchmark",
                "ph": "X",
                "ts": int((self.t0 - _t0_global) * 1e6),
                "dur": int(elapsed * 1e6),
                "pid": 1,
                "tid": 1,
                "args": {"peak_mb": rec["peak_mb"]},
            }
        )


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


def _detect_device():
    """Auto-detect best available device."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def run(policy_name: str, device: str = None, n_warm: int = 10):
    branch, sha = _git_info()
    if device is None:
        device = _detect_device()

    print(f"🔬 bench_lerobot ({branch}@{sha})")
    print(f"   policy: {policy_name}")
    print(f"   device: {device}")
    print("=" * 65)

    with phase("import strands_robots"):
        import strands_robots  # noqa: F401

    with phase("import create_policy"):
        from strands_robots.policies.factory import create_policy

    with phase("import LerobotLocalPolicy"):
        from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy  # noqa: F401

    # Resolve from Hub (config + metadata download)
    with phase("resolve_policy_from_hub"):
        from strands_robots.registry import resolve_policy

        provider, kwargs = resolve_policy(policy_name)
        print(f"    → provider={provider}")

    # Create policy (weight download/load + device placement)
    with phase("create_policy (full)"):
        policy = create_policy(provider, device=device, **kwargs)

    # Determine observation shape from policy
    with phase("build observation"):
        import torch

        # Standard ACT observation: image + state
        obs = {}
        if hasattr(policy, "policy") and hasattr(policy.policy, "config"):
            cfg = policy.policy.config
            # Get state dim from config
            state_dim = getattr(cfg, "input_features", {}).get(
                "observation.state", 14
            )
            if isinstance(state_dim, dict):
                state_dim = state_dim.get("shape", [14])
                state_dim = state_dim[0] if isinstance(state_dim, list) else state_dim
        else:
            state_dim = 14  # default ACT

        obs["observation.state"] = np.random.randn(state_dim).astype(np.float32)
        # 480x640x3 image
        obs["observation.images.top"] = np.random.randint(
            0, 255, (480, 640, 3), dtype=np.uint8
        )

    # Cold inference
    with phase("inference cold"):
        actions_cold = policy.get_actions_sync(obs, "bench_cold")

    # Warm inference loop
    warm_times = []
    for i in range(n_warm):
        t0 = time.time()
        policy.get_actions_sync(obs, "bench_warm")
        warm_times.append((time.time() - t0) * 1000)

    warm_rec = {
        "name": f"inference warm ({n_warm}x)",
        "ms": round(np.mean(warm_times), 4),
        "peak_mb": 0,
        "p50_ms": round(np.percentile(warm_times, 50), 4),
        "p95_ms": round(np.percentile(warm_times, 95), 4),
        "p99_ms": round(np.percentile(warm_times, 99), 4),
        "min_ms": round(np.min(warm_times), 4),
        "max_ms": round(np.max(warm_times), 4),
    }
    _phases.append(warm_rec)
    print(
        f"  {'inference warm (' + str(n_warm) + 'x)':45s} "
        f"mean={warm_rec['ms']:.2f}ms  "
        f"p50={warm_rec['p50_ms']:.2f}ms  "
        f"p99={warm_rec['p99_ms']:.2f}ms"
    )

    # Reset
    with phase("policy.reset()"):
        if hasattr(policy, "reset"):
            policy.reset()

    # --- Summary ---
    print("\n" + "=" * 65)
    print("📊 SUMMARY")
    print("=" * 65)
    for p in _phases:
        extras = ""
        if p.get("p50_ms"):
            extras = f" (p50={p['p50_ms']}ms p99={p['p99_ms']}ms)"
        print(f"  {p['name']:45s} {p['ms']:>10.2f}ms  {p['peak_mb']:>6.1f}MB{extras}")

    results = {
        "benchmark": "bench_lerobot",
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "arch": platform.machine(),
            "device": device,
            "branch": branch,
            "commit": sha,
            "policy": policy_name,
            "n_warm": n_warm,
        },
        "phases": _phases,
        "warm_inference_series_ms": [round(t, 4) for t in warm_times],
        "trace_events": _trace_events,
    }

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark LeRobot local policy end-to-end"
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="lerobot/act_aloha_sim_transfer_cube_human",
        help="HF Hub policy name or local path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: cuda, mps, cpu (auto-detected if omitted)",
    )
    parser.add_argument(
        "--n-warm",
        type=int,
        default=10,
        help="Number of warm inference iterations (default: 10)",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON to stdout")
    parser.add_argument("--out", type=str, help="Output directory for artifacts")
    args = parser.parse_args()

    results = run(
        policy_name=args.policy, device=args.device, n_warm=args.n_warm
    )

    if args.json:
        print(json.dumps(results, indent=2))

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")

        # Stats JSON
        stats_file = out_dir / f"bench_lerobot_{ts}.json"
        with open(stats_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"📁 Stats: {stats_file}")

        # Chrome Trace
        trace_file = out_dir / f"bench_lerobot_{ts}_trace.json"
        with open(trace_file, "w") as f:
            json.dump({"traceEvents": _trace_events}, f)
        print(f"📁 Trace: {trace_file}  →  ui.perfetto.dev")

        # Inference CSV
        csv_file = out_dir / f"bench_lerobot_{ts}_inference.csv"
        with open(csv_file, "w") as f:
            f.write("iteration,ms\n")
            for i, t in enumerate(results["warm_inference_series_ms"]):
                f.write(f"{i},{t}\n")
        print(f"📁 CSV:   {csv_file}")
