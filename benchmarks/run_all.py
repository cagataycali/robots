#!/usr/bin/env python3
"""Run all benchmark formats and export all artifacts.

Usage:
    python benchmarks/run_all.py                    # everything
    python benchmarks/run_all.py --skip-lerobot     # skip heavy lerobot tests
    python benchmarks/run_all.py --out results/     # custom output dir

Produces:
    results/
    ├── bench_core_<ts>.json          # Core benchmark stats
    ├── pytest_bench.json             # pytest-benchmark JSON
    ├── hist.svg                      # pytest-benchmark histogram
    ├── core_profile.prof             # cProfile raw data
    ├── core_speedscope.json          # Speedscope flame graph
    ├── core_memray.bin               # memray raw data
    ├── memray_flamegraph.html        # memray interactive flamegraph
    ├── memray_table.html             # memray table report
    ├── pyspy_flame.svg               # py-spy flame graph (Linux only)
    ├── bench_lerobot_<ts>.json       # LeRobot stats
    ├── bench_lerobot_<ts>_trace.json # LeRobot Perfetto trace
    └── bench_lerobot_<ts>_inference.csv
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def run_cmd(label, cmd, cwd=None, timeout=300):
    """Run a command and report result."""
    print(f"\n{'='*60}")
    print(f"🔬 {label}")
    print(f"   $ {' '.join(cmd)}")
    print(f"{'='*60}")
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            print(f"✅ {label}")
            # Print last 10 lines of output
            lines = result.stdout.strip().split("\n")
            for line in lines[-10:]:
                print(f"   {line}")
        else:
            print(f"❌ {label} (exit {result.returncode})")
            if result.stderr:
                for line in result.stderr.strip().split("\n")[-5:]:
                    print(f"   {line}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"⏰ {label} (timed out after {timeout}s)")
        return False
    except FileNotFoundError:
        print(f"⚠️  {label} (command not found)")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run all benchmark formats")
    parser.add_argument(
        "--out", type=str, default="benchmarks/results", help="Output directory"
    )
    parser.add_argument(
        "--skip-lerobot", action="store_true", help="Skip LeRobot benchmarks"
    )
    parser.add_argument(
        "--skip-groot", action="store_true", help="Skip GR00T benchmarks"
    )
    parser.add_argument(
        "--groot-server", type=str, default="localhost:50051", help="GR00T server"
    )
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Resolve paths
    repo = Path(__file__).parent.parent
    venv_python = repo / ".venv" / "bin" / "python"
    venv_pytest = repo / ".venv" / "bin" / "pytest"

    if not venv_python.exists():
        venv_python = Path(sys.executable)
    if not venv_pytest.exists():
        venv_pytest = Path("pytest")

    results = {}
    t0 = time.time()

    # ----------------------------------------------------------------
    # 1. bench_core.py (custom script)
    # ----------------------------------------------------------------
    results["bench_core"] = run_cmd(
        "bench_core.py → JSON",
        [str(venv_python), "benchmarks/bench_core.py", "--out", str(out)],
        cwd=str(repo),
    )

    # ----------------------------------------------------------------
    # 2. pytest-benchmark → JSON + histogram SVG
    # ----------------------------------------------------------------
    results["pytest_benchmark"] = run_cmd(
        "pytest-benchmark → JSON + histogram",
        [
            str(venv_pytest),
            "benchmarks/test_bench.py",
            "--benchmark-only",
            f"--benchmark-json={out}/pytest_bench.json",
            f"--benchmark-histogram={out}/hist",
            "-q",
            "--no-header",
            "--override-ini=addopts=",
        ],
        cwd=str(repo),
    )

    # ----------------------------------------------------------------
    # 3. cProfile → .prof → Speedscope JSON
    # ----------------------------------------------------------------
    prof_file = out / "core_profile.prof"
    results["cprofile"] = run_cmd(
        "cProfile → .prof",
        [
            str(venv_python),
            "-c",
            f"""
import cProfile
pr = cProfile.Profile()
pr.enable()
import strands_robots
from strands_robots.registry import list_robots, list_aliases, get_robot, resolve_policy, format_robot_table
from strands_robots.policies.factory import list_providers
from strands_robots.policies.mock import MockPolicy
from strands_robots.policies.groot.data_config import load_data_config
import numpy as np
list_robots(); list_aliases(); list_providers()
get_robot('so100'); resolve_policy('groot')
load_data_config('so100'); format_robot_table()
mock = MockPolicy(action_space={{f'j{{i}}': 0.0 for i in range(6)}})
obs = {{'observation.state': np.zeros(6, dtype=np.float32)}}
for _ in range(100): mock.get_actions_sync(obs, 'test')
pr.disable()
pr.dump_stats('{prof_file}')
print('saved {prof_file}')
""",
        ],
        cwd=str(repo),
    )

    speedscope_file = out / "core_speedscope.json"
    results["speedscope"] = run_cmd(
        "Speedscope → JSON (open at speedscope.app)",
        [
            str(venv_python),
            "-c",
            f"""
import pstats, json
stats = pstats.Stats('{prof_file}')
stats.sort_stats('cumulative')
frames, frame_map, samples, weights = [], {{}}, [], []
def gfid(name, file, line):
    key = f'{{file}}:{{name}}:{{line}}'
    if key not in frame_map:
        frame_map[key] = len(frames)
        frames.append({{'name': name, 'file': file, 'line': line}})
    return frame_map[key]
for (file, line, name), (cc, nc, tt, ct, callers) in stats.stats.items():
    fid = gfid(name, file, line)
    samples.append([fid])
    weights.append(ct * 1e6)
speedscope = {{
    '$schema': 'https://www.speedscope.app/file-format-schema.json',
    'shared': {{'frames': frames}},
    'profiles': [{{
        'type': 'sampled', 'name': 'strands_robots',
        'unit': 'microseconds', 'startValue': 0,
        'endValue': sum(weights), 'samples': samples, 'weights': weights,
    }}],
    'name': 'strands_robots', 'activeProfileIndex': 0,
}}
with open('{speedscope_file}', 'w') as f: json.dump(speedscope, f)
print(f'Saved {{len(frames)}} frames to {speedscope_file}')
""",
        ],
        cwd=str(repo),
    )

    # ----------------------------------------------------------------
    # 4. memray → flamegraph HTML + table HTML
    # ----------------------------------------------------------------
    memray_bin = out / "core_memray.bin"
    results["memray_record"] = run_cmd(
        "memray → record",
        [
            str(venv_python),
            "-m",
            "memray",
            "run",
            "-o",
            str(memray_bin),
            "--force",
            "benchmarks/_memray_target.py",
        ],
        cwd=str(repo),
    )

    results["memray_flamegraph"] = run_cmd(
        "memray → flamegraph HTML",
        [
            str(venv_python),
            "-m",
            "memray",
            "flamegraph",
            str(memray_bin),
            "-o",
            str(out / "memray_flamegraph.html"),
            "--force",
        ],
        cwd=str(repo),
    )

    results["memray_table"] = run_cmd(
        "memray → table HTML",
        [
            str(venv_python),
            "-m",
            "memray",
            "table",
            str(memray_bin),
            "-o",
            str(out / "memray_table.html"),
            "--force",
        ],
        cwd=str(repo),
    )

    # ----------------------------------------------------------------
    # 5. py-spy → flame SVG (Linux only, needs root on macOS)
    # ----------------------------------------------------------------
    import platform

    if platform.system() == "Linux":
        results["pyspy"] = run_cmd(
            "py-spy → flame SVG",
            [
                str(repo / ".venv" / "bin" / "py-spy"),
                "record",
                "-o",
                str(out / "pyspy_flame.svg"),
                "--",
                str(venv_python),
                "benchmarks/_memray_target.py",
            ],
            cwd=str(repo),
        )
    else:
        print(f"\n⚠️  py-spy: skipped (needs root on {platform.system()})")
        results["pyspy"] = None

    # ----------------------------------------------------------------
    # 6. bench_lerobot.py (optional)
    # ----------------------------------------------------------------
    if not args.skip_lerobot:
        results["bench_lerobot"] = run_cmd(
            "bench_lerobot.py → JSON + trace + CSV",
            [
                str(venv_python),
                "benchmarks/bench_lerobot.py",
                "--out",
                str(out),
            ],
            cwd=str(repo),
            timeout=600,
        )
    else:
        print("\n⚠️  bench_lerobot: skipped (--skip-lerobot)")
        results["bench_lerobot"] = None

    # ----------------------------------------------------------------
    # 7. bench_groot.py (optional)
    # ----------------------------------------------------------------
    if not args.skip_groot:
        results["bench_groot"] = run_cmd(
            "bench_groot.py → JSON",
            [
                str(venv_python),
                "benchmarks/bench_groot.py",
                "--server",
                args.groot_server,
                "--out",
                str(out),
            ],
            cwd=str(repo),
            timeout=120,
        )
    else:
        print("\n⚠️  bench_groot: skipped (--skip-groot)")
        results["bench_groot"] = None

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"📊 ALL BENCHMARKS COMPLETE ({elapsed:.1f}s)")
    print(f"{'='*60}")

    for name, ok in results.items():
        icon = "✅" if ok else ("⚠️ " if ok is None else "❌")
        print(f"  {icon} {name}")

    print(f"\n📁 Artifacts in: {out.resolve()}")
    print(f"\nVisualize:")
    print(f"  Perfetto:   open ui.perfetto.dev → drag *_trace.json")
    print(f"  Speedscope: open speedscope.app → drag core_speedscope.json")
    print(f"  Memray:     open memray_flamegraph.html in browser")
    print(f"  Histogram:  open hist.svg in browser")

    # List all files
    print(f"\n📦 Generated files:")
    for f in sorted(out.iterdir()):
        if f.is_file():
            size = f.stat().st_size
            unit = "KB" if size > 1024 else "B"
            size_fmt = f"{size/1024:.1f}" if size > 1024 else str(size)
            print(f"  {f.name:45s} {size_fmt:>8s} {unit}")


if __name__ == "__main__":
    main()
