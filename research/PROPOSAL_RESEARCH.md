# Proposal: Cross-Dependency AST Testing for Robotics Safety

> **Date**: 2026-03-11  
> **Author**: DevDuck Research  
> **Status**: Prototype implemented and validated

## The Insight

Python robotics codebases have a unique problem: **a renamed function parameter in an upstream library can cause a real robot to behave unpredictably**. Standard `pytest` catches this at runtime — after the robot has already started moving. We need compile-time-like validation for a dynamic language.

## What We Built

### `tests/test_cross_deps.py` — Three-layer validation

**Layer 1: Import Resolution** (catches removed/renamed modules)
```
strands_robots/**/*.py → AST extract → from lerobot.X import Y → importlib.import_module() → ✅/❌
```

**Layer 2: Signature Validation** (catches renamed parameters)
```
our_code.py:42 → LeRobotDataset.create(serial_number=...) → inspect.signature() → ❌ renamed to serial_number_or_name
```

**Layer 3: Version Compatibility** (catches version drift)
```
pyproject.toml says lerobot>=0.5.0 → installed is 0.5.0 → API surface check → ✅
```

### Results on First Run

```
tests/test_cross_deps.py::TestImportResolution::test_all_critical_imports_resolve     FAILED  (3 breaks)
tests/test_cross_deps.py::TestImportResolution::test_optional_imports_are_guarded     PASSED
tests/test_cross_deps.py::TestSignatureValidation::test_lerobot_signatures            FAILED  (1 break)
tests/test_cross_deps.py::TestSignatureValidation::test_mujoco_signatures             PASSED
tests/test_cross_deps.py::TestVersionCompatibility::test_dependency_versions           PASSED
tests/test_cross_deps.py::TestVersionCompatibility::test_lerobot_api_surface           PASSED
tests/test_cross_deps.py::TestVersionCompatibility::test_lerobot_dataset_create_signature PASSED
tests/test_cross_deps.py::TestVersionCompatibility::test_lerobot_robot_configs_discoverable PASSED
```

**6/8 passed, 2 failed — catching 4 real breaks that would have reached production.**

## Why This Matters for Hardware

| Scenario | Without AST testing | With AST testing |
|---|---|---|
| lerobot bumps 0.5 → 0.6 | Robot fails mid-task on first `get_policy_class()` call | CI catches removed import before merge |
| transformers renames Florence2 config | Silent import error, fallback policy runs wrong model | Test catches `MISSING_ATTR` immediately |
| RealSense kwarg renamed | Camera init fails, robot runs blind | Signature check catches `serial_number` → `serial_number_or_name` |
| New lerobot Robot config added | We don't know it exists | `test_lerobot_robot_configs_discoverable` finds it |

## Architecture for CI

```yaml
# .github/workflows/cross-deps.yml
name: Cross-Dependency Validation
on: [push, pull_request, schedule]

jobs:
  ast-validation:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.12", "3.13"]
        lerobot-version: ["0.5.0", "latest"]
    steps:
      - uses: actions/checkout@v4
      - name: Install deps
        run: pip install -e ".[ml,sim]" lerobot==${{ matrix.lerobot-version }}
      - name: Run AST validation
        run: pytest tests/test_cross_deps.py -v
```

This runs against **multiple lerobot versions** to catch forward-compatibility issues.

## Future Extensions

1. **Snapshot tests**: Serialize 3rd-party signatures to JSON, fail when they change
2. **Deprecation tracking**: Parse `warnings.warn("deprecated")` in upstream code
3. **Type annotation validation**: Check our type hints match upstream signatures
4. **Protocol conformance**: Verify our `Policy` ABC matches what lerobot expects
5. **Bidirectional validation**: Check that lerobot code calling OUR exports also resolves

## Files Created

```
research/
├── CROSS_DEPENDENCY_AST_RESEARCH.md     ← This proposal + findings
├── LEROBOT_MIGRATION_RESEARCH.md        ← Detailed lerobot 0.5 break analysis
├── E2E_DEVX_FLOW_RESEARCH.md            ← DevX hot path validation
├── cross_dependency_ast_results.json    ← Full import validation data
└── lerobot_signature_validation.json    ← Full signature validation data

tests/
└── test_cross_deps.py                   ← The test suite (ready for CI)
```

## Conclusion

Cross-dependency AST testing is not just "nice to have" for robotics — it's a safety mechanism. The prototype caught 4 real breaks in strands-robots that would have caused runtime failures on hardware. The cost is ~2 seconds of CI time per run. The benefit is knowing your robot won't crash because a library renamed a parameter.
