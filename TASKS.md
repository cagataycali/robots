# PR #56 — Remaining Tasks

**PR**: https://github.com/strands-labs/robots/pull/56
**Branch**: `feat/lerobot-local` on `cagataycali/robots`
**Reviewer**: @awsarron
**Status**: 30 of 82 threads unresolved

## 🔴 Must Fix (code changes required)

### T1: `pyproject.toml` — mypy per-module overrides instead of global
- **Thread**: Global `warn_return_any`, `disallow_untyped_defs`, `ignore_missing_imports` should be scoped
- **Fix**: Keep strict defaults, override only for specific modules
- **Status**: ❌ TODO

### T2: `gr00t_inference.py` — port default mismatch
- **Thread**: "this wasn't updated. The actual defaults should be what is in the docstring"
- **Fix**: port default is already 5555, docstring says 5555 — verify alignment throughout
- **Status**: ✅ Already correct (port=5555 matches docstring=5555)

### T3: `gr00t_inference.py` — auth tokens via env var tracking
- **Thread**: "Where is this tracked?"
- **Fix**: Already done (GROOT_API_TOKEN env fallback exists). Need to create tracking issue.
- **Status**: ❌ Need to create issue on strands-labs/robots

### T4: `robot.py` — cast once instead of repeatedly
- **Thread**: "this wasn't resolved"
- **Fix**: Extract `_make_tool_result()` helper
- **Status**: ❌ TODO

### T5: `tests/groot/test_policy.py` — unified skip style with pytestmark
- **Thread**: "we're using different styles for skipping"
- **Fix**: Use class-level `pytestmark` for all groot tests
- **Status**: ❌ TODO

### T6: `tests/test_policies.py` — module-level pytestmark
- **Thread**: "is this resolved?" (2 threads)
- **Fix**: Replace per-test `@pytest.mark.skipif` with module-level `pytestmark`
- **Status**: ❌ TODO

### T7: `tests/test_lerobot_local.py` — imports at top + remove superfluous tests
- **Thread**: "is this resolved?" + "Where is this tracked?"
- **Fix**: Already at top. Remove trivial attribute-checking tests, create tracking issue.
- **Status**: ❌ Need to create issue + remove superfluous tests

### T8: `tests_integ/` — imports at top of file
- **Thread**: "is this resolved?"
- **Fix**: Already done — verify all imports are at top
- **Status**: ✅ Already correct

### T9: `tests_integ/` — consolidate to e2e behavioral tests
- **Thread**: "is this resolved?" — integ tests are unit-style, need e2e behavioral
- **Fix**: Consolidate into fewer e2e scenarios, remove resolution test class
- **Status**: ❌ TODO

### T10: `tests_integ/` — RTC e2e test
- **Thread**: "is this resolved?"
- **Fix**: Add RTC e2e test with SmolVLA or Pi0
- **Status**: ❌ TODO (deferred — needs model download)

### T11: `tests_integ/` — ProcessorBridge e2e test
- **Thread**: "do we have e2e tests for processor bridge supported models?"
- **Status**: ✅ TestProcessorBridgeIntegration exists

### T12: `tests/groot/test_client.py` — CI `[all]` + `--strict-markers`
- **Thread**: "is this resolved?"
- **Fix**: Already in `.github/workflows/test-lint.yml` + pyproject.toml has `--strict-markers`
- **Status**: ✅ Already done in fd5f8fe

### T13: `policy.py` — fail-fast audit (PR-wide)
- **Thread**: 4 locations where catch+warn should be raise
- **Fix**: Audit all exception handlers, convert to fail-fast
- **Status**: ❌ TODO

### T14: `policy.py` — `_load_model()` in error messages
- **Thread**: "Should we be referencing private internal methods in error messages?"
- **Fix**: Use user-facing language instead
- **Status**: ❌ TODO

### T15: `policy.py` — debug log flooding at 50Hz
- **Thread**: "for 50hz devices that's still 5 times per second" + "could this flood logs?"
- **Fix**: Rate-limit RTC debug log to every 100th call (~2s at 50Hz). Check L694 too.
- **Status**: ❌ TODO

### T16: `policy.py` — state padding should fail-fast
- **Thread**: "what effect does padding have? Should this be a fail fast case?"
- **Fix**: Raise ValueError instead of silently padding
- **Status**: ❌ TODO

### T17: `processor.py:113` — `_pipeline_cls` usage question
- **Thread**: "why do we not use `self._pipeline_cls`?"
- **Fix**: Clarify or remove `_pipeline_cls` if unused
- **Status**: ❌ TODO

### T18: `processor.py` — fail-fast (2 locations)
- **Thread**: "should this be a fail fast and early case instead?"
- **Fix**: Convert warnings to raises
- **Status**: ❌ TODO

### T19: `policy.py:499` — select_action vs predict_action_chunk
- **Thread**: Arron highlighted temporal ensemble + action queue logic in `select_action`
- **Fix**: Add code comment explaining the design. Non-RTC always uses `select_action`.
- **Status**: ❌ TODO (code comment)

### T20: `pyproject.toml` — Python <3.14 justification
- **Thread**: "gotcha. What are the issues?"
- **Fix**: Document draccus + argparse incompatibility with 3.14's typing.Dict
- **Status**: ❌ TODO (comment response)

### T21: `pyproject.toml` — Python 3.10/3.11 support strategy
- **Thread**: Design discussion — lerobot needs 3.12+ but base SDK should support 3.10+
- **Fix**: Already `>=3.10` with lerobot as optional dep. Respond to thread.
- **Status**: ✅ Already correct (requires-python = ">=3.10")

### T22: `__init__.py` — lazy imports for faster startup
- **Thread**: Arron: "Robot class imports lerobot.*, consider lazy loading"
- **Fix**: Use `__getattr__` pattern in `__init__.py` for heavy imports
- **Status**: ❌ TODO (separate PR recommended)

### T23: `test_lerobot_local.py` — superfluous comment about tests_integ
- **Thread**: "Where is this tracked?"
- **Fix**: Remove the comment + create tracking issue
- **Status**: ❌ TODO

## 📋 Tracking Issues Needed

- [ ] Auth tokens via env vars (gr00t_inference.py) → create issue
- [ ] Superfluous tests cleanup → create issue
- [ ] Lazy imports for startup perf → create issue

## 🧪 Test Plan

```bash
# Local (macOS arm64, Python 3.13) — 281 passed ✅
pytest tests/ -v --timeout=30

# EC2 (L40S, CUDA, Python 3.12) — 281 passed ✅
ssh ubuntu@13.218.34.208 "cd robots && python3 -m pytest tests/ -v --timeout=30"

# Thor (Jetson, CUDA) — via GitHub Actions workflow
```

## 📝 Commit Plan

1. **Commit 1**: Mechanical fixes (pyproject mypy, pytestmark, robot.py cast, port fix, log throttle)
2. **Commit 2**: Fail-fast audit (policy.py + processor.py)
3. **Commit 3**: Test improvements (remove superfluous, add code comments)
4. **Commit 4**: Create tracking issues for deferred items
