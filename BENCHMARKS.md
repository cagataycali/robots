# BENCHMARKS.md — strands_robots Engineering Memo

> **Purpose**: Living document tracking performance across devices, branches, and policies.
> Every entry records timestamp, commit, device, and full profiling data so we can
> detect regressions and validate optimizations over time.
>
> **How to contribute**: Run the benchmark scripts (linked below), paste the summary
> table into a new entry under the appropriate device section. Always include the
> commit hash and branch name.

---

## Table of Contents

- [Methodology](#methodology)
- [Benchmark Scripts](#benchmark-scripts)
- [Devices Under Test](#devices-under-test)
- [Results](#results)
  - [MacBook Pro M3 Max (MPS)](#macbook-pro-m3-max-mps)
  - [EC2 L40S (CUDA)](#ec2-l40s-cuda)
  - [Thor (Jetson / CUDA)](#thor-jetson--cuda)
- [Key Findings](#key-findings)
- [Performance Targets](#performance-targets)
- [Artifact Formats](#artifact-formats)

---

## Methodology

Each benchmark run profiles **checkpoints** through the code:

1. **Import phase** — `import strands_robots`, sub-modules, policies
2. **Registry phase** — `list_robots()`, `resolve_policy()`, config lookups
3. **Policy creation** — model download, weight loading, device placement
4. **Inference** — cold (first call) and warm (subsequent calls, 10-100 iterations)
5. **Memory** — peak RSS via `tracemalloc` at each checkpoint

All times are wall-clock. Memory is Python-heap only (does not include GPU VRAM).
Each run uses `gc.collect()` + fresh `tracemalloc.start()` per phase to isolate.

### What we measure per checkpoint

| Field | Description |
|-------|-------------|
| `elapsed_ms` | Wall-clock time for the phase |
| `peak_mb` | Peak Python heap allocation during the phase |
| `p50_ms` / `p99_ms` | Latency percentiles (for inference loops) |

---

## Benchmark Scripts

Stored in `/Users/cagatay/strands-labs/` (not in repo, run locally per device):

| Script | What it profiles |
|--------|-----------------|
| Run inline Python (see entries below) | Full pipeline per policy |

### Export formats per run

| File | Format | Open with |
|------|--------|-----------|
| `benchmark_trace.json` | Chrome Trace Format | [chrome://tracing](chrome://tracing) or [ui.perfetto.dev](https://ui.perfetto.dev) |
| `benchmark_stats.json` | Full JSON stats | Any editor, jq, Python |
| `benchmark_phases.csv` | CSV phase table | Excel, Google Sheets |
| `benchmark_inference.csv` | CSV warm inference series | Excel, Sheets |
| `benchmark_non_lerobot.json` | Non-lerobot paths | jq, Python |

---

## Devices Under Test

| Device | CPU | GPU | RAM | OS | Notes |
|--------|-----|-----|-----|----|-------|
| MacBook Pro M3 Max | Apple M3 Max (arm64) | MPS (Metal) | 64GB unified | macOS 15.x | Primary dev machine |
| EC2 L40S | x86_64 | NVIDIA L40S 48GB | 192GB | Ubuntu 22.04 | GR00T inference server |
| Thor (Jetson) | ARM Cortex-A78AE | Jetson (CUDA) | 32GB | JetPack 6.x | On-robot compute |

---

## Results

### MacBook Pro M3 Max (MPS)

#### Run 1 — Non-LeRobot Paths

- **Date**: 2026-03-21T17:15
- **Branch**: `feat/lerobot-local`
- **Commit**: `287e707`
- **Python**: 3.13.12
- **strands_robots**: 0.3.9.dev32

| Phase | Time (ms) | Peak Mem (MB) | Notes |
|-------|----------:|-------------:|-------|
| `import strands_robots` | 92.0 | 3.8 | Lazy imports working ✅ |
| `import policies.factory` | 0.0 | 0.0 | |
| `import policies.base` | 0.0 | 0.0 | |
| `import registry` | 0.0 | 0.0 | |
| `list_providers()` | 0.5 | 0.0 | → `['groot', 'lerobot_local', 'mock']` |
| `list_robots()` | 0.8 | 0.1 | → 38 robots |
| `list_policy_providers()` | 0.1 | 0.0 | |
| `list_aliases()` | 0.1 | 0.0 | → 82 aliases |
| `get_robot(aloha)` | 0.1 | 0.0 | |
| `get_robot(apollo)` | 0.1 | 0.0 | |
| `get_robot(arx_l5)` | 0.1 | 0.0 | |
| `get_policy_provider(groot)` | 0.1 | 0.0 | |
| `get_policy_provider(lerobot_local)` | 0.1 | 0.0 | |
| `get_policy_provider(mock)` | 0.1 | 0.0 | |
| `import groot.client` | 11.5 | 0.5 | msgpack + zmq |
| `import groot.policy` | 0.0 | 0.0 | |
| `import groot.data_config` | 0.0 | 0.0 | |
| `import mock.MockPolicy` | 0.0 | 0.0 | |
| `MockPolicy(6-DOF)` create | 0.0 | 0.0 | |
| `mock cold inference` | 0.6 | 0.0 | 8 actions returned |
| `mock warm (100x)` | **0.14** | 0.0 | p50=0.14ms p99=0.21ms |
| `load_data_config("so100")` | 0.1 | 0.0 | |
| `import tools` | 0.9 | 0.0 | Lazy ✅ |
| **`import robot.Robot`** | **19,535** | **219.6** | ⚠️ Eagerly imports torch! |
| `resolve_policy("groot")` | 0.3 | 0.0 | |
| `resolve_policy("lerobot_local")` | 0.3 | 0.0 | |
| `format_robot_table()` | 0.9 | 0.0 | 3815 chars |

**Key takeaway**: Everything is <1ms except `import robot.Robot` which pulls in
torch/lerobot transitively → **19.5s, 220MB**. This is the main optimization target
for users who only need registry/config operations.

---

#### Run 2 — LeRobot Local (ACT policy, 14-DOF)

- **Date**: 2026-03-21T17:05
- **Branch**: `feat/lerobot-local`
- **Commit**: `287e707`
- **Model**: `lerobot/act_aloha_sim_transfer_cube_human`
- **Device**: MPS (Apple Metal)

| Phase | Time (ms) | Peak Mem (MB) | Notes |
|-------|----------:|-------------:|-------|
| `import strands_robots` | 112 | 3.8 | |
| `import create_policy` | 0.0 | 0.0 | |
| `import LerobotLocalPolicy` | 3,141 | 55.8 | lerobot + torch + transformers |
| `resolve_policy_from_hub` | 16,328 | 165.8 | HF Hub config download + parse |
| `create_policy (full)` | 1,325 | 2.3 | Weight loading (cached on disk) |
| `build_observation` | 1.3 | 0.9 | 480×640×3 image + 14-state |
| **`inference cold`** | **170** | 0.9 | First forward pass (MPS compile) |
| **`inference warm (mean)`** | **6.0** | 0.0 | 10 iterations |
| `inference warm p50` | 6.0 | — | |
| `inference warm p95` | 6.2 | — | |
| `inference warm p99` | 6.3 | — | |
| `policy.reset()` | 0.1 | 0.0 | |
| `create_policy (smart-string)` | 1,294 | 2.3 | 2nd load, configs cached |
| `inference (smart-string)` | 50 | 0.9 | Semi-warm (new policy instance) |

**Summary**:
- **Import cost**: ~3.1s (lerobot) + 16.3s (HF hub resolution) = ~19.4s first time
- **Inference**: 170ms cold → **6ms warm** (p99=6.3ms) — well within 50Hz control loop budget
- **Peak memory**: 165.8MB (during HF hub resolution)

---

### EC2 L40S (CUDA)

> 🔲 **TODO**: Run benchmarks on EC2 with CUDA.
>
> Expected tests:
> - GR00T policy (requires gr00t inference server running)
> - LeRobot local with CUDA acceleration
> - `import strands_robots` without torch installed (pure registry path)
>
> Command to run on EC2:
> ```bash
> ssh -i ~/strands-robots.pem ubuntu@<EC2_IP>
> cd /path/to/robots
> source .venv/bin/activate
> python3 benchmark_lerobot.py   # (copy script from local)
> python3 benchmark_groot.py     # (with gr00t server running)
> ```

---

### Thor (Jetson / CUDA)

> 🔲 **TODO**: Run benchmarks on Thor (Jetson).
>
> Expected tests:
> - GR00T policy end-to-end (robot connected)
> - LeRobot local on Jetson GPU
> - Camera → policy → action latency (full loop)
> - Serial communication overhead
>
> This is the real-world deployment target. Numbers here matter most for
> control loop frequency targets.

---

## Key Findings

### 2026-03-21 (feat/lerobot-local @ 287e707)

1. **`import strands_robots` is fast**: 86-112ms, 3.8MB — lazy imports work ✅
2. **Registry operations are instant**: All <1ms, 0 memory overhead
3. **`import robot.Robot` is the bottleneck**: 19.5s, 220MB — eagerly imports torch
   - **Action item**: Make `Robot` class also use lazy imports for torch/lerobot
   - Users doing `from strands_robots.robot import Robot` pay full torch cost
4. **LeRobot warm inference**: 6ms on MPS — excellent for 50Hz control (20ms budget)
5. **HF Hub resolution**: 16.3s on first call — dominated by network + config parsing
   - Subsequent calls use disk cache → 1.3s
6. **Mock policy**: 0.14ms per inference — useful for testing without GPU

### Performance by policy provider

| Provider | Create (ms) | Cold Inference (ms) | Warm Inference (ms) | Memory (MB) |
|----------|------------:|-------------------:|-------------------:|------------:|
| **mock** | 0.03 | 0.6 | 0.14 (p99=0.21) | 0 |
| **lerobot_local** (ACT) | 1,325 | 170 | 6.0 (p99=6.3) | 165.8 peak |
| **groot** | — | — | — | — (needs server) |

---

## Performance Targets

| Metric | Target | Current (MPS) | Status |
|--------|--------|--------------|--------|
| `import strands_robots` | <200ms | 92ms | ✅ |
| Registry lookup | <5ms | 0.1ms | ✅ |
| Policy warm inference (ACT) | <20ms (50Hz) | 6.0ms | ✅ |
| Policy warm inference (GR00T) | <20ms (50Hz) | TBD | 🔲 |
| Cold start (no cache) | <30s | ~21s | ⚠️ |
| Cold start (cached) | <5s | ~4.5s | ✅ |
| Peak memory (inference) | <500MB | 165.8MB | ✅ |
| `import robot.Robot` | <500ms | 19,535ms | ❌ needs lazy torch |

---

## Artifact Formats

Every benchmark run should produce these files (saved locally, not in git):

```
benchmark_trace.json        # Chrome Trace → ui.perfetto.dev
benchmark_stats.json        # Full JSON with metadata
benchmark_phases.csv        # Phase summary → Excel/Sheets
benchmark_inference.csv     # Warm inference time series
benchmark_non_lerobot.json  # Non-lerobot path profiling
```

### How to visualize

1. **Perfetto** (best): Drag `benchmark_trace.json` into [ui.perfetto.dev](https://ui.perfetto.dev)
2. **Chrome**: Open `chrome://tracing`, click Load, select `benchmark_trace.json`
3. **Spreadsheets**: Import CSV files into Excel/Sheets for charts
4. **CLI**: `jq '.summary' benchmark_stats.json`

---

## Changelog

| Date | Commit | Change | Impact |
|------|--------|--------|--------|
| 2026-03-21 | `287e707` | Initial benchmarks, lazy imports for strands_robots | `import` 60s → 92ms |
| 2026-03-21 | `287e707` | LeRobot local E2E on MPS | 6ms warm inference confirmed |
