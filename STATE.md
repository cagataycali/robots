# PR #56 State — feat/lerobot-local

## Branch: `feat/lerobot-local`
## PR: strands-labs/robots#56 (from cagataycali/robots)

## Changes Summary

### Commit 1: `ba8e8f8` — Initial lerobot_local implementation
- Full `LerobotLocalPolicy` with ACT, Diffusion, Pi0, SmolVLA support
- `ProcessorBridge` for pre/post-processing pipelines
- Smart-string resolution (`lerobot/act_aloha_sim_*` → auto-detect)
- Registry integration with HuggingFace org patterns

### Commit 2: `15c30f5` — PR #55 review fixes
- Simplified processor bridge
- Added integration tests (ACT + Diffusion real models)

### Commit 3 (current) — Test coverage + CI fixes
- **Unit tests**: 48 → **125 tests** (+77 new)
- **Coverage**: policy.py 70%→96%, processor.py 53%→100%, resolution.py 58%→79%
- **CI-safe**: All tests mock torch/lerobot — no GPU or heavy deps needed
- **Lint clean**: black ✅, isort ✅, flake8 ✅
- **conftest.py**: Shared torch mock fixtures for all test files
- Fixed `_resolve_tokenizer` to handle missing config gracefully
- Fixed `_needs_language_tokens` to handle missing config gracefully

### Integration Tests (23 tests — `tests_integ/`)
- ACT policy: load, inference, state keys, strands format, stability
- Diffusion policy: load, inference, action range validation
- Factory resolution: smart-string + explicit provider
- ProcessorBridge: real model configs
- Error handling: invalid models, consecutive failures

## Test Results
```
Unit:   125 passed, 0 failed (6s, no GPU needed)
Integ:   23 passed, 0 failed (23s, requires lerobot + models)
Lint:    0 errors (black + isort + flake8)
```

## Files Changed (this commit)
- `strands_robots/policies/lerobot_local/policy.py` — 2 defensive fixes
- `tests/conftest.py` — NEW shared torch/lerobot mock fixtures
- `tests/test_lerobot_local.py` — 48→125 tests, full coverage push
