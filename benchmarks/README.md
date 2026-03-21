# benchmarks

Profiling scripts for `strands_robots` — measure import times, registry operations,
policy creation, and inference latency across devices.

## Quick Start

```bash
# Run everything (core + all formats)
python benchmarks/run_all.py --skip-lerobot --skip-groot

# Or individual scripts
python benchmarks/bench_core.py
python benchmarks/bench_lerobot.py
python benchmarks/bench_groot.py --server 10.0.0.10:50051
```

## Scripts

| Script | What it profiles | Requirements |
|--------|-----------------|--------------|
| `run_all.py` | **All formats** — one command | numpy, pytest-benchmark, memray |
| `bench_core.py` | Imports, registry, mock policy, data configs | numpy only |
| `bench_lerobot.py` | LeRobot ACT policy end-to-end | torch, lerobot |
| `bench_groot.py` | GR00T inference via gRPC | running gr00t-server |
| `test_bench.py` | pytest-benchmark suite (statistical) | pytest-benchmark |

## Output Formats

| Format | File | Open with | What you see |
|--------|------|-----------|-------------|
| **Perfetto** | `*_trace.json` | [ui.perfetto.dev](https://ui.perfetto.dev) | Timeline, flame chart, zoom |
| **Speedscope** | `core_speedscope.json` | [speedscope.app](https://speedscope.app) | Call flame graph (left-heavy, sandwich) |
| **memray** | `memray_flamegraph.html` | Browser | Memory allocation flame graph |
| **memray table** | `memray_table.html` | Browser | Top allocators by size |
| **pytest-benchmark** | `pytest_bench.json` | CLI or JSON viewer | Statistical comparison across runs |
| **Histogram** | `hist.svg` | Browser | Distribution of benchmark times |
| **cProfile** | `core_profile.prof` | `snakeviz`, `gprof2dot` | Raw profiling data |
| **py-spy** | `pyspy_flame.svg` | Browser | Wall-clock flame graph (Linux only) |
| **JSON stats** | `bench_*.json` | jq, Python | Full stats + metadata |
| **CSV** | `*_inference.csv` | Excel, Sheets | Warm inference time series |

## Usage Examples

```bash
# Full run (skip policies that need GPU/server)
python benchmarks/run_all.py --skip-lerobot --skip-groot

# With LeRobot (needs torch + lerobot installed)
python benchmarks/run_all.py --skip-groot

# With GR00T server
python benchmarks/run_all.py --groot-server 10.0.0.10:50051

# Custom output directory
python benchmarks/run_all.py --out /tmp/bench_results/

# Just pytest-benchmark with comparison
pytest benchmarks/test_bench.py --benchmark-only --benchmark-compare

# Just bench_core with JSON output
python benchmarks/bench_core.py --json --out results/

# LeRobot with custom policy and device
python benchmarks/bench_lerobot.py \
  --policy lerobot/act_aloha_sim_transfer_cube_human \
  --device cuda \
  --n-warm 50 \
  --out results/
```

## Install Dependencies

```bash
pip install pytest-benchmark[histogram] memray py-spy
```

`py-spy` needs root on macOS, works without sudo on Linux.

## Device Matrix

| Device | run_all | bench_lerobot | bench_groot |
|--------|---------|---------------|-------------|
| MacBook M3 Max (MPS) | ✅ | ✅ | — |
| EC2 L40S (CUDA) | 🔲 | 🔲 | 🔲 |
| Thor Jetson (CUDA) | 🔲 | 🔲 | 🔲 |

See [BENCHMARKS.md](../BENCHMARKS.md) for collected results.
