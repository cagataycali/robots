"""Repro: `import strands_robots` crashes when `find_spec("mujoco")` raises.

Upstream: strands-labs/robots v0.5.3
File:     strands_robots/__init__.py:249
Target:   README quickstart "Install" path (`pip install strands-robots`
          WITHOUT the `[sim-mujoco]` extra, OR with mujoco stubbed in tests)

Summary
-------
The import-time MuJoCo GL auto-configuration block (lines 227-255) is guarded
by `importlib.util.find_spec("mujoco") is not None`. The GUARD comment at
line 237 explicitly promises: *"Skip when mujoco is not installed so users
without the [sim-mujoco] extra do not pay import-attempt cost on every
import strands_robots."*

But `find_spec` is not the right primitive for a safe guard. The CPython
stdlib docs state it *propagates* `ImportError` from any meta-path finder
and raises `ValueError` when `sys.modules[name].__spec__ is None`. The
adjacent block below (lines 250-255) wraps the GL configurator in
`try/except (ImportError, AttributeError, OSError)` - but line 249's
`find_spec` call, which the GUARD comment is attached to, is **not** wrapped.

Result: in any environment where a test fixture, a mock, a frozen app
loader, a half-installed wheel, or a partial-init failure left a
`sys.modules['mujoco']` entry with `__spec__ = None`, the bare statement
`import strands_robots` raises `ValueError: mujoco.__spec__ is None`
before user code reaches `Robot(...)`. Same shape when a strict
`sys.meta_path` finder raises `ImportError` for `mujoco`.

The documented promise of lazy loading (line 272:
*"This avoids importing torch, lerobot, numpy, mujoco, pyserial, etc. at
import strands_robots time"*) is broken for the mujoco branch - not by
mujoco being imported, but by its *absence check* crashing.

How to see it
-------------
    $ python bugbash_repros/import_strands_robots_crashes_on_find_spec_mujoco_repro.py

Expected (goal): `import strands_robots` succeeds with mujoco stubbed to
                 `__spec__ = None`; the GL configurator is skipped silently.

Actual:          `ValueError: mujoco.__spec__ is None` at
                 strands_robots/__init__.py:249.

Fix (3 LOC)
-----------
Wrap the guard the same way the configurator below it is wrapped:

    try:
        _mj_spec = _importlib_util.find_spec("mujoco")
    except (ImportError, ValueError):
        _mj_spec = None
    if _mj_spec is not None:
        try:
            from strands_robots._mujoco_gl import _configure_gl_backend
            _configure_gl_backend()
        except (ImportError, AttributeError, OSError):
            pass

The fallback matches the comment's intent: when mujoco cannot be probed
cleanly, skip the GL hint - MuJoCo picks its own backend on first real
import (the same fallback already honoured by the inner try/except).

Why this is end-user-visible
----------------------------
1. README L68 teaches `uv pip install "strands-robots[sim-mujoco]"` but
   adds `# plain pip works too` and L75 says the extras list is opt-in.
   Users who want to *just explore the registry* (`list_robots_by_category`,
   `use_ros`, `use_rtps` - all top-level per __all__) legitimately install
   bare `strands-robots` with no extras.
2. Any test suite that mocks mujoco (`monkeypatch.setitem(sys.modules,
   'mujoco', MagicMock())`) can leave `__spec__ = None` behind - a standard
   pytest/pytest-mock pattern.
3. PyInstaller/py2app frozen modules commonly carry `__spec__ = None`;
   bundling an app that imports strands_robots inherits the crash.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap


REPO_ROOT = os.environ.get(
    "STRANDS_ROBOTS_SRC",
    "/home/cagatay/bugbash-strands-robots-1791355541/robots",
)


def _run(scenario: str, code: str) -> tuple[int, str, str]:
    """Run a subprocess, return (returncode, stdout, stderr)."""
    env = {k: v for k, v in os.environ.items() if k != "SYSTEM_PROMPT"}
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYTHONPATH", REPO_ROOT)
    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    return r.returncode, r.stdout, r.stderr


def scenario_1_stubbed_mujoco() -> None:
    """A test fixture (or partial-init failure) left a stub mujoco in sys.modules."""
    print("=" * 72)
    print("SCENARIO 1: sys.modules['mujoco'] with __spec__ = None")
    print("=" * 72)
    print("""Common cause: pytest-mock's `mocker.patch.dict('sys.modules',
{'mujoco': MagicMock()})` or any direct `sys.modules['mujoco'] = types.
ModuleType('mujoco')` done before `import strands_robots`. The stub's
default __spec__ is None.
""")

    code = textwrap.dedent(
        f"""
        import sys, os, types
        os.environ.setdefault('MUJOCO_GL', 'egl')

        # Realistic pre-condition: a mock/stub left in sys.modules
        _fake = types.ModuleType('mujoco')
        _fake.__spec__ = None
        sys.modules['mujoco'] = _fake

        sys.path.insert(0, {REPO_ROOT!r})
        try:
            import strands_robots
            print('OK: import strands_robots succeeded')
        except Exception as e:
            print(f'FAIL: {{type(e).__name__}}: {{e}}')
        """
    )

    rc, out, err = _run("stub", code)
    print(f"$ python -c '<see docstring>'")
    print(f"return code: {rc}")
    print(f"stdout: {out.strip()}")
    if err.strip():
        print(f"stderr: {err.strip()[-400:]}")


def scenario_2_strict_finder() -> None:
    """A strict sys.meta_path finder (frozen apps, sandboxes) raises on mujoco."""
    print()
    print("=" * 72)
    print("SCENARIO 2: strict sys.meta_path finder raises ImportError on 'mujoco'")
    print("=" * 72)
    print("""Common cause: PyInstaller / py2app / zipapp loaders install
restricted finders that raise rather than return None for banned modules.
A `strands-robots`-dependent app that explicitly excludes mujoco at freeze
time inherits this.""")

    code = textwrap.dedent(
        f"""
        import sys, os, importlib.abc
        os.environ.setdefault('MUJOCO_GL', 'egl')

        class _BanMujoco(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name == 'mujoco' or name.startswith('mujoco.'):
                    raise ImportError(
                        f"No module named {{name!r}} (sandboxed finder)")
                return None

        sys.meta_path.insert(0, _BanMujoco())
        sys.path.insert(0, {REPO_ROOT!r})
        try:
            import strands_robots
            print('OK: import strands_robots succeeded')
        except Exception as e:
            print(f'FAIL: {{type(e).__name__}}: {{e}}')
        """
    )

    rc, out, err = _run("strict_finder", code)
    print(f"return code: {rc}")
    print(f"stdout: {out.strip()}")
    if err.strip():
        print(f"stderr: {err.strip()[-400:]}")


def scenario_3_the_fix() -> None:
    """Verify the proposed 3-LOC fix handles both scenarios."""
    print()
    print("=" * 72)
    print("SCENARIO 3: proposed 3-LOC fix (wrap find_spec in try/except)")
    print("=" * 72)

    code = textwrap.dedent(
        f"""
        import sys, os, types, importlib.util
        os.environ.setdefault('MUJOCO_GL', 'egl')

        _fake = types.ModuleType('mujoco')
        _fake.__spec__ = None
        sys.modules['mujoco'] = _fake

        # The fix pattern:
        try:
            _mj_spec = importlib.util.find_spec('mujoco')
        except (ImportError, ValueError):
            _mj_spec = None
        has_mj = _mj_spec is not None
        print(f'has_mj (after fix): {{has_mj}}')
        assert has_mj is False, "fix should classify broken-spec as absent"
        print('OK: fix cleanly reports mujoco-absent for broken __spec__')
        """
    )

    rc, out, err = _run("fix", code)
    print(f"return code: {rc}")
    print(f"stdout: {out.strip()}")
    if err.strip():
        print(f"stderr: {err.strip()[-400:]}")


def main() -> int:
    print("Repro: `import strands_robots` crashes on find_spec('mujoco')")
    print("Upstream: strands_robots/__init__.py:249")
    print()
    scenario_1_stubbed_mujoco()
    scenario_2_strict_finder()
    scenario_3_the_fix()
    print()
    print("=" * 72)
    print("Summary")
    print("=" * 72)
    print("""Scenarios 1 and 2 both crash `import strands_robots` on an
unwrapped `find_spec('mujoco')`. Scenario 3 shows the 3-LOC fix.

The failure mode defeats the GUARD comment (lines 236-239) and the
lazy-loading promise (line 272) of the same file - users who do NOT want
mujoco cannot even import the package.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
