# benchmarks

Profiling scripts for `strands_robots` — measure import times, registry operations,
policy creation, and inference latency across devices.

## Quick Start

```bash
# Core paths (no GPU/network needed)
python benchmarks/bench_core.py

# LeRobot local policy (needs torch + lerobot)
python benchmarks/bench_lerobot.py

# GR00T policy (needs running inference server)
python benchmarks/bench_groot.py --server 10.0.0.10:50051
```

## Scripts

| Script | What it profiles | Requirements |
|--------|-----------------|--------------|
| `bench_core.py` | Imports, registry, mock policy, data configs | numpy only |
| `bench_lerobot.py` | LeRobot ACT policy end-to-end | torch, lerobot |
| `bench_groot.py` | GR00T inference via gRPC | running gr00t-server |

## Output Options

```bash
# Print JSON to stdout
python benchmarks/bench_core.py --json

# Save artifacts to a directory
python benchmarks/bench_lerobot.py --out results/

# Custom policy / device
python benchmarks/bench_lerobot.py --policy lerobot/act_aloha_sim --device cuda --n-warm 50
```

## Output Artifacts

Each `--out` run creates:

| File | Format | Visualize with |
|------|--------|---------------|
| `bench_*_<ts>.json` | Full stats + metadata | jq, Python |
| `bench_*_<ts>_trace.json` | Chrome Trace | [ui.perfetto.dev](https://ui.perfetto.dev) |
| `bench_*_<ts>_inference.csv` | Warm inference series | Excel, Sheets |

## Device Matrix

| Device | bench_core | bench_lerobot | bench_groot |
|--------|-----------|---------------|-------------|
| MacBook M3 Max (MPS) | ✅ | ✅ | — |
| EC2 L40S (CUDA) | 🔲 | 🔲 | 🔲 |
| Thor Jetson (CUDA) | 🔲 | 🔲 | 🔲 |

See [BENCHMARKS.md](../BENCHMARKS.md) for collected results.
