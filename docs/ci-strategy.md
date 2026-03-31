# CI Strategy

This document describes the testing tiers and CI strategy for `strands-robots`.

## Testing Tiers

| Tier | Directory | Runner | Trigger | What It Tests |
|------|-----------|--------|---------|---------------|
| **Unit** | `tests/` | `ubuntu-latest` | Every PR push | Mocked inference, registry, config parsing, error handling |
| **Integration** | `tests_integ/` | `ubuntu-latest` (CPU gate) | Label `run-integ-tests`, weekly schedule, manual | Import validation, test collection, CPU-compatible subset |
| **Integration (GPU)** | `tests_integ/` | Self-hosted GPU | Manual dispatch, weekly schedule | Real model downloads, CUDA inference, full e2e pipeline |

## Workflows

### `pr-and-push.yml` → `test-lint.yml` (existing)

Runs on every PR and push to `main`/`dev`:
- Lint (ruff check + format + mypy)
- Unit tests (`pytest tests/ -x --strict-markers`)

### `test-integ.yml` (new)

Runs integration tests on three triggers:
1. **Weekly schedule** (Mondays 06:00 UTC) — regression detection
2. **Manual dispatch** — ad-hoc runs with configurable Python version and timeout
3. **PR label** `run-integ-tests` — on-demand for PRs touching inference code

#### CPU Gate

On `ubuntu-latest`, integration tests are collected and validated but GPU-marked tests
are skipped. This catches import errors, fixture issues, and any CPU-compatible tests
without requiring expensive GPU runners.

#### GPU Runners (future)

When self-hosted GPU runners are configured, uncomment the `integ-tests-gpu` job in
`test-integ.yml`. The job uses the `[self-hosted, gpu]` label to target GPU-capable
machines.

**Proven hardware** (from cross-repo CI on [cagataycali/strands-gtc-nvidia](https://github.com/cagataycali/strands-gtc-nvidia)):

| Runner | Hardware | GPU | Unit Tests | Integ Tests |
|--------|----------|-----|------------|-------------|
| Thor | NVIDIA Jetson AGX Thor | Blackwell sm_110, 132GB unified | 263/263 ✅ | 3/26 (env issue) |
| EC2 | g6e.4xlarge | NVIDIA L40S 46GB, CUDA 13.0 | 263/263 ✅ | 13/26 ✅ |

## Test Markers

- `@pytest.mark.gpu` — requires CUDA GPU and real model weights. Registered in `pyproject.toml`.
- Use `pytest -m gpu` to run only GPU tests, or `pytest -m "not gpu"` to skip them.

## Integration Test Suites

### `tests_integ/lerobot_local/`

Tests ACT and Diffusion policy full pipelines: load real model weights from HuggingFace Hub →
build observation → run inference → validate output shape, dtype, and value ranges.

**Requirements**: `lerobot>=0.5.0`, internet access, ~2GB disk for model weights.

**Environment variables**:
- `LEROBOT_ACT_MODEL` — override ACT model (default: `lerobot/act_aloha_sim_transfer_cube_human`)
- `LEROBOT_DIFFUSION_MODEL` — override Diffusion model (default: `lerobot/diffusion_pusht`)
- `LEROBOT_RTC_MODEL` — flow-matching model for RTC tests
- `LEROBOT_DOWNLOAD_TIMEOUT` — timeout for HF model downloads (default: 300s)

### `tests_integ/groot/`

Tests GR00T N1.6 service and local inference modes. Starts a ZMQ inference server,
sends observations, validates action shapes and dtypes.

**Requirements**: CUDA GPU, Isaac-GR00T N1.6 SDK, `nvidia/GR00T-N1.6-3B` model.

**Environment variables**:
- `GROOT_MODEL_PATH` — model path (default: `nvidia/GR00T-N1.6-3B`)
- `GROOT_EMBODIMENT_TAG` — robot embodiment (default: `GR1`)
- `GROOT_SERVER_TIMEOUT` — server startup timeout (default: 180s)

## Running Locally

```bash
# Unit tests only (fast, no GPU needed)
hatch run test

# Integration tests (requires lerobot + internet)
hatch run test-integ

# GPU-only integration tests
pytest tests_integ/ -v --timeout=300 -m gpu

# Skip GPU tests
pytest tests_integ/ -v --timeout=300 -m "not gpu"
```
