# BENCHMARKS.md — strands_robots Performance Engineering Memo

> **Purpose**: Living document tracking performance across devices, branches, and policies.
> Every entry records timestamp, commit, device, and full profiling data.
>
> **Source of truth**: The JSON files in `benchmarks/results_*/` directories.
> This document is a human-readable summary generated from those artifacts.
>
> **How to reproduce**: Run `python benchmarks/run_all.py --out benchmarks/results_<device>/`
> on each device, then update this file.

---

## Table of Contents

- [Methodology](#methodology)
- [Devices Under Test](#devices-under-test)
- [Cross-Device Comparison](#cross-device-comparison)
- [Detailed Results](#detailed-results)
  - [MacBook Pro M3 Max (MPS)](#macbook-pro-m3-max-mps)
  - [EC2 L40S (CUDA)](#ec2-l40s-cuda)
  - [Thor (Jetson / CUDA)](#thor-jetson--cuda)
- [Key Findings](#key-findings)
- [Performance Targets](#performance-targets)
- [Benchmark Artifacts](#benchmark-artifacts)

---

## Methodology

Each benchmark run profiles **checkpoints** through the code:

1. **Import phase** — `import strands_robots`, sub-modules, policies
2. **Registry phase** — `list_robots()`, `resolve_policy()`, config lookups
3. **Policy creation** — model download, weight loading, device placement
4. **Inference** — cold (first call) and warm (subsequent calls, 10-100 iterations)
5. **Memory** — peak RSS via `tracemalloc` at each checkpoint

All times are wall-clock. Memory is Python-heap only (does not include GPU VRAM).

### Output formats per run

| Format | File | Visualize with |
|--------|------|---------------|
| **Perfetto** | `*_trace.json` | [ui.perfetto.dev](https://ui.perfetto.dev) |
| **Speedscope** | `core_speedscope.json` | [speedscope.app](https://speedscope.app) |
| **memray** | `memray_flamegraph.html` | Browser |
| **pytest-benchmark** | `pytest_bench.json` | `--benchmark-compare` |
| **Histogram** | `hist.svg` | Browser |
| **JSON stats** | `bench_*.json` | jq, Python |

---

## Devices Under Test

| Device | CPU | GPU | RAM | OS | Python |
|--------|-----|-----|-----|----|--------|
| MacBook Pro M3 Max | Apple M3 Max (arm64) | MPS (Metal) | 64GB unified | macOS 15.x | 3.13.12 |
| EC2 L40S | x86_64 (AWS) | NVIDIA L40S 48GB | 192GB | Ubuntu 24.04 | 3.12.3 |
| Thor (Jetson) | ARM Cortex-A78AE | Jetson Orin (CUDA) | 32GB | JetPack 6.x | TBD |

---

## Cross-Device Comparison

> All numbers from commit `07b5d68` on branch `benchmarks`, 2026-03-21.

### Core Operations (no GPU needed)

| Phase | MacBook M3 Max | EC2 L40S | Notes |
|-------|---------------:|---------:|-------|
| `import strands_robots` | 92ms | 165ms | Lazy imports ✅ |
| `list_robots()` | 0.8ms | 6.3ms | 38 robots |
| `list_aliases()` | 0.1ms | 0.2ms | 82 aliases |
| `get_robot(*)` | 0.1ms | 0.2ms | Per lookup |
| `resolve_policy(*)` | 0.2ms | 0.3ms | Per lookup |
| `import groot.client` | 11.5ms | 60.7ms | zmq + msgpack |
| `load_data_config('so100')` | 0.1ms | 0.01ms | |
| `format_robot_table()` | 0.9ms | 1.4ms | |
| Mock warm inference (p50) | 0.12ms | 0.12ms | Identical |
| Mock warm inference (p99) | 0.17ms | 0.15ms | |
| **`import robot.Robot`** | **19,535ms** | **20,278ms** | ⚠️ Eagerly imports torch |

### LeRobot ACT Policy (GPU-accelerated)

| Phase | MacBook M3 Max (MPS) | EC2 L40S (CUDA) | Notes |
|-------|---------------------:|------------------:|-------|
| `import strands_robots` | 112ms | 4ms | (cached on 2nd run) |
| `import LerobotLocalPolicy` | 3,141ms | 24ms | (torch already loaded) |
| `resolve_policy_from_hub` | 16,328ms | 0.6ms | HF Hub config (cached) |
| `create_policy (full)` | 1,325ms | 15,087ms | Weight download + load |
| **Inference cold** | **170ms** | **348ms** | First forward pass |
| **Inference warm (mean)** | **6.0ms** | **1.9ms** | 🏆 CUDA wins |
| Inference warm (p50) | 6.0ms | **1.7ms** | |
| Inference warm (p99) | 6.3ms | **3.0ms** | |
| Peak memory | 165.8MB | 148.2MB | |

### Key insight: CUDA warm inference is **3.2x faster** than MPS (1.7ms vs 6.0ms)

---

## Detailed Results

### MacBook Pro M3 Max (MPS)

#### Core Paths (bench_core)

- **Date**: 2026-03-21
- **Branch**: `benchmarks` @ `07b5d68`
- **Python**: 3.13.12
- **Artifacts**: `benchmarks/results_macbook/`

| Phase | Time (ms) | Peak Mem (MB) |
|-------|----------:|-------------:|
| `import strands_robots` | 92.0 | 3.8 |
| `import policies.factory` | 0.0 | 0.0 |
| `import policies.base` | 0.0 | 0.0 |
| `import registry` | 0.0 | 0.0 |
| `list_providers()` | 0.5 | 0.0 |
| `list_robots()` | 0.8 | 0.1 |
| `list_policy_providers()` | 0.1 | 0.0 |
| `list_aliases()` | 0.1 | 0.0 |
| `get_robot(aloha)` | 0.1 | 0.0 |
| `get_policy_provider(groot)` | 0.1 | 0.0 |
| `import groot.client` | 11.5 | 0.5 |
| `MockPolicy(6-DOF)` create | 0.0 | 0.0 |
| Mock cold inference | 0.6 | 0.0 |
| Mock warm (100x) | **0.12** | 0.0 |
| `load_data_config("so100")` | 0.1 | 0.0 |
| `import tools` | 0.9 | 0.0 |
| `resolve_policy("groot")` | 0.2 | 0.0 |
| `format_robot_table()` | 0.9 | 0.0 |
| **`import robot.Robot`** | **19,535** | **219.1** |

#### LeRobot ACT (bench_lerobot, MPS)

| Phase | Time (ms) | Peak Mem (MB) |
|-------|----------:|-------------:|
| `import strands_robots` | 112 | 3.8 |
| `import LerobotLocalPolicy` | 3,141 | 55.8 |
| `resolve_policy_from_hub` | 16,328 | 165.8 |
| `create_policy (full)` | 1,325 | 2.3 |
| Inference cold | 170 | 0.9 |
| **Inference warm (mean)** | **6.0** | 0.0 |
| Inference warm (p50) | 6.0 | — |
| Inference warm (p99) | 6.3 | — |

---

### EC2 L40S (CUDA)

#### Core Paths (bench_core)

- **Date**: 2026-03-21
- **Branch**: `benchmarks` @ `07b5d68`
- **Python**: 3.12.3
- **GPU**: NVIDIA L40S 48GB
- **Artifacts**: `benchmarks/results_ec2/`

| Phase | Time (ms) | Peak Mem (MB) |
|-------|----------:|-------------:|
| `import strands_robots` | 164.5 | 4.2 |
| `import policies.factory` | 0.0 | 0.0 |
| `import registry` | 0.0 | 0.0 |
| `list_providers()` | 6.0 | 0.0 |
| `list_robots()` | 6.3 | 0.1 |
| `list_policy_providers()` | 0.1 | 0.0 |
| `list_aliases()` | 0.2 | 0.0 |
| `get_robot(aloha)` | 0.2 | 0.0 |
| `get_policy_provider(groot)` | 0.1 | 0.0 |
| `import groot.client` | 60.7 | 2.1 |
| `MockPolicy(6-DOF)` create | 0.1 | 0.0 |
| Mock cold inference | 0.7 | 0.0 |
| Mock warm (100x) | **0.12** | 0.0 |
| `load_data_config("so100")` | 0.0 | 0.0 |
| `import tools` | 8.6 | 0.1 |
| `resolve_policy("groot")` | 0.3 | 0.0 |
| `format_robot_table()` | 1.4 | 0.0 |
| **`import robot.Robot`** | **20,278** | **200.6** |

#### LeRobot ACT (bench_lerobot, CUDA)

| Phase | Time (ms) | Peak Mem (MB) |
|-------|----------:|-------------:|
| `import strands_robots` | 4.4 | 0.1 |
| `import create_policy` | 0.0 | 0.0 |
| `import LerobotLocalPolicy` | 24.1 | 2.0 |
| `resolve_policy_from_hub` | 0.6 | 0.0 |
| `create_policy (full)` | 15,087 | 148.2 |
| Build observation | 3.2 | 0.9 |
| Inference cold | 347.6 | 0.9 |
| **Inference warm (mean)** | **1.9** | 0.0 |
| Inference warm (p50) | **1.7** | — |
| Inference warm (p95) | 2.5 | — |
| Inference warm (p99) | **3.0** | — |
| `policy.reset()` | 0.2 | 0.0 |

**Warm inference series (ms)**: 3.17, 1.74, 1.72, 1.71, 1.75, 1.74, 1.71, 1.69, 1.73, 1.73

---

### Thor (Jetson / CUDA)

> 🔲 **TODO**: Thor was unreachable during this benchmark run (2026-03-21).
>
> To run when available:
> ```bash
> git clone -b benchmarks https://github.com/cagataycali/robots.git
> cd robots && pip install -e . && pip install pytest-benchmark memray numpy
> python benchmarks/run_all.py --out benchmarks/results_thor/
> ```

---

## Key Findings

### 2026-03-21 (benchmarks @ 07b5d68)

1. **`import strands_robots` is fast on both devices**: 92ms (Mac) / 165ms (EC2) — lazy imports work ✅
2. **Registry operations are instant everywhere**: All <7ms, near-zero memory
3. **`import robot.Robot` is the universal bottleneck**: ~20s on both devices (torch import)
   - **Action item**: Make `Robot` class lazy-load torch
4. **CUDA warm inference is 3.2x faster than MPS**: 1.7ms vs 6.0ms
5. **Both devices comfortably meet 50Hz**: 1.7ms (CUDA) and 6.0ms (MPS) << 20ms budget
6. **`create_policy` is slow on EC2 (15s)**: This is weight download + CUDA compilation
   - First run; subsequent runs with cached weights are much faster
7. **Mock policy is identical across devices**: 0.12ms — pure Python, no GPU

### Performance by policy provider (cross-device)

| Provider | Create | Cold | Warm (p50) | Mac | EC2 |
|----------|--------|------|-----------|-----|-----|
| **mock** | 0.03ms | 0.6ms | 0.12ms | ✅ | ✅ |
| **lerobot ACT (MPS)** | 1,325ms | 170ms | 6.0ms | ✅ | — |
| **lerobot ACT (CUDA)** | 15,087ms | 348ms | **1.7ms** | — | ✅ |
| **groot** | — | — | — | 🔲 | 🔲 |

---

## Performance Targets

| Metric | Target | Mac (MPS) | EC2 (CUDA) | Status |
|--------|--------|-----------|-----------|--------|
| `import strands_robots` | <200ms | 92ms | 165ms | ✅ |
| Registry lookup | <5ms | 0.1ms | 0.3ms | ✅ |
| Policy warm inference (ACT) | <20ms (50Hz) | 6.0ms | **1.7ms** | ✅ |
| Policy warm inference (GR00T) | <20ms (50Hz) | TBD | TBD | 🔲 |
| Cold start (cached) | <5s | ~4.5s | ~15s | ⚠️ EC2 slow |
| Peak memory (inference) | <500MB | 165.8MB | 148.2MB | ✅ |
| `import robot.Robot` | <500ms | 19,535ms | 20,278ms | ❌ needs lazy torch |

---

## Benchmark Artifacts

All raw data lives in the repo under `benchmarks/`:

```
benchmarks/
├── run_all.py                    # Unified runner (all formats)
├── bench_core.py                 # Core paths profiler
├── bench_lerobot.py              # LeRobot E2E profiler
├── bench_groot.py                # GR00T profiler
├── test_bench.py                 # pytest-benchmark suite
├── _memray_target.py             # memray profiling target
├── README.md                     # Usage guide
├── results_macbook/              # MacBook M3 Max artifacts
│   ├── bench_core_*.json         # Core stats
│   ├── core_speedscope.json      # → speedscope.app
│   ├── core_profile.prof         # cProfile raw
│   ├── core_memray.bin           # memray raw
│   ├── memray_flamegraph.html    # Memory flame graph
│   ├── memray_table.html         # Top allocators
│   ├── pytest_bench.json         # Statistical benchmarks
│   └── hist.svg                  # Histogram
└── results_ec2/                  # EC2 L40S artifacts
    ├── bench_core_*.json
    ├── bench_lerobot_*.json      # LeRobot stats
    ├── bench_lerobot_*_trace.json # → ui.perfetto.dev
    ├── bench_lerobot_*_inference.csv
    ├── core_speedscope.json
    ├── core_profile.prof
    ├── core_memray.bin
    ├── memray_flamegraph.html
    ├── memray_table.html
    ├── pytest_bench.json
    └── hist.svg
```

### How to visualize

1. **Perfetto**: Drag `*_trace.json` into [ui.perfetto.dev](https://ui.perfetto.dev) — timeline + flame chart
2. **Speedscope**: Drag `core_speedscope.json` into [speedscope.app](https://speedscope.app) — call flame graph
3. **memray**: Open `memray_flamegraph.html` in browser — memory allocation flame graph
4. **Histogram**: Open `hist.svg` in browser — distribution of benchmark times
5. **pytest-benchmark**: `pytest benchmarks/test_bench.py --benchmark-compare` — compare runs

---

## Changelog

| Date | Commit | Change | Impact |
|------|--------|--------|--------|
| 2026-03-21 | `07b5d68` | Initial multi-device benchmarks | Baseline established |
| 2026-03-21 | `07b5d68` | EC2 L40S CUDA profiling | 1.7ms warm inference confirmed |
| 2026-03-21 | `2c1907c` | Lazy imports for strands_robots | `import` 60s → 92ms |
| 2026-03-21 | `07b5d68` | 6 visualization formats | Perfetto, Speedscope, memray, pytest-benchmark, SVG, CSV |
