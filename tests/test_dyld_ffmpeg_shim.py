"""Unit tests for the macOS dyld ffmpeg shim helpers in ``strands_robots._dyld``.

These cover the internal probing/safety helpers that decide whether (and how)
``ensure_ffmpeg_on_dyld_path`` puts Homebrew's ffmpeg on the dyld search path so
torchcodec can decode video. The logic is pure and platform-gated, so every
branch is exercised here with ``monkeypatch`` rather than a real macOS host:

  - ``_find_ffmpeg_lib_dir`` prefers ``HOMEBREW_PREFIX`` and requires a
    versioned ``libavutil.*.dylib``.
  - ``_torchcodec_installed`` reflects import availability.
  - ``_is_safe_to_reexec`` refuses to re-exec inside REPLs / Jupyter / pytest /
    ``python -c`` and allows it only for a plain script invocation.
  - ``ensure_ffmpeg_on_dyld_path`` re-execs exactly once when safe, swallows an
    ``execv`` failure, and warns (without re-exec) when unsafe.
"""

import warnings

import pytest

from strands_robots import _dyld


def test_find_ffmpeg_lib_dir_prefers_homebrew_prefix(monkeypatch, tmp_path):
    """A versioned libavutil under ``HOMEBREW_PREFIX/lib`` is returned first."""
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "libavutil.59.dylib").write_bytes(b"")
    monkeypatch.setenv("HOMEBREW_PREFIX", str(tmp_path))

    assert _dyld._find_ffmpeg_lib_dir() == str(lib)


def test_find_ffmpeg_lib_dir_requires_versioned_soname(monkeypatch, tmp_path):
    """A bare ``libavutil.dylib`` (no version) does not satisfy the probe."""
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "libavutil.dylib").write_bytes(b"")  # unversioned -> ignored
    monkeypatch.setenv("HOMEBREW_PREFIX", str(tmp_path))
    # Neutralize the canonical fallbacks so only our temp dir is considered.
    monkeypatch.setattr(_dyld, "_CANDIDATE_LIB_DIRS", ())

    assert _dyld._find_ffmpeg_lib_dir() is None


def test_find_ffmpeg_lib_dir_none_when_absent(monkeypatch):
    """No Homebrew prefix and no candidate dirs -> ``None``."""
    monkeypatch.delenv("HOMEBREW_PREFIX", raising=False)
    monkeypatch.setattr(_dyld, "_CANDIDATE_LIB_DIRS", ())

    assert _dyld._find_ffmpeg_lib_dir() is None


def test_torchcodec_installed_reflects_find_spec(monkeypatch):
    """``_torchcodec_installed`` is True iff importlib can locate the module."""
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    assert _dyld._torchcodec_installed() is True

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert _dyld._torchcodec_installed() is False


def _clear_reexec_blockers(monkeypatch):
    """Make ``_is_safe_to_reexec`` see a plain-script environment."""
    # No REPL: during a pytest run sys.ps1 is absent and sys.flags.interactive
    # is already 0, so we only need to neutralize the module-table checks below.
    monkeypatch.delattr(_dyld.sys, "ps1", raising=False)
    # No Jupyter / IPython / pytest in the module table this function inspects.
    monkeypatch.setattr(_dyld.sys, "modules", {})
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)


def test_is_safe_to_reexec_true_for_plain_script(monkeypatch):
    """A normal ``python script.py`` invocation is safe to re-exec."""
    _clear_reexec_blockers(monkeypatch)
    monkeypatch.setattr(_dyld.sys, "argv", ["train.py", "--epochs", "1"])

    assert _dyld._is_safe_to_reexec() is True


def test_is_safe_to_reexec_false_inside_pytest(monkeypatch):
    """Presence of pytest in ``sys.modules`` blocks re-exec."""
    _clear_reexec_blockers(monkeypatch)
    monkeypatch.setattr(_dyld.sys, "modules", {"pytest": object()})
    monkeypatch.setattr(_dyld.sys, "argv", ["train.py"])

    assert _dyld._is_safe_to_reexec() is False


def test_is_safe_to_reexec_false_inside_ipykernel(monkeypatch):
    """A Jupyter/IPython kernel must never have its process image replaced."""
    _clear_reexec_blockers(monkeypatch)
    monkeypatch.setattr(_dyld.sys, "modules", {"ipykernel": object()})
    monkeypatch.setattr(_dyld.sys, "argv", ["train.py"])

    assert _dyld._is_safe_to_reexec() is False


def test_is_safe_to_reexec_false_for_dash_c(monkeypatch):
    """``python -c '...'`` (argv[0] == '-c') has nothing safe to re-run."""
    _clear_reexec_blockers(monkeypatch)
    monkeypatch.setattr(_dyld.sys, "argv", ["-c"])

    assert _dyld._is_safe_to_reexec() is False


def _arm_macos_ffmpeg(monkeypatch, tmp_path):
    """Set up the happy path: macOS + torchcodec + ffmpeg dir, no opt-out/guard."""
    monkeypatch.setattr(_dyld.sys, "platform", "darwin")
    monkeypatch.delenv(_dyld._OPT_OUT_ENV, raising=False)
    monkeypatch.delenv(_dyld._GUARD_ENV, raising=False)
    monkeypatch.delenv(_dyld._DYLD_VAR, raising=False)
    monkeypatch.setattr(_dyld, "_torchcodec_installed", lambda: True)
    monkeypatch.setattr(_dyld, "_find_ffmpeg_lib_dir", lambda: str(tmp_path))


def test_ensure_reexecs_once_when_safe(monkeypatch, tmp_path):
    """When safe, the env var is exported and ``execv`` is invoked exactly once."""
    _arm_macos_ffmpeg(monkeypatch, tmp_path)
    monkeypatch.setattr(_dyld, "_is_safe_to_reexec", lambda: True)
    calls: list[tuple] = []
    monkeypatch.setattr(_dyld.os, "execv", lambda exe, argv: calls.append((exe, argv)))

    _dyld.ensure_ffmpeg_on_dyld_path()

    assert len(calls) == 1
    # Guard set so a re-exec'd process won't loop, and child env carries the dir.
    assert _dyld.os.environ[_dyld._GUARD_ENV] == "1"
    assert str(tmp_path) in _dyld.os.environ[_dyld._DYLD_VAR]


def test_ensure_swallows_execv_failure(monkeypatch, tmp_path):
    """If ``execv`` raises, the call falls through and returns False, not crashes."""
    _arm_macos_ffmpeg(monkeypatch, tmp_path)
    monkeypatch.setattr(_dyld, "_is_safe_to_reexec", lambda: True)

    def _boom(exe, argv):
        raise OSError("execv denied")

    monkeypatch.setattr(_dyld.os, "execv", _boom)

    assert _dyld.ensure_ffmpeg_on_dyld_path() is False


def test_ensure_warns_without_reexec_when_unsafe(monkeypatch, tmp_path):
    """Unsafe context (e.g. Jupyter): warn with the export hint, do not re-exec."""
    _arm_macos_ffmpeg(monkeypatch, tmp_path)
    monkeypatch.setattr(_dyld, "_is_safe_to_reexec", lambda: False)
    monkeypatch.setattr(_dyld.os, "execv", lambda *a: pytest.fail("execv must not run when unsafe"))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _dyld.ensure_ffmpeg_on_dyld_path()

    assert result is False
    assert any(issubclass(w.category, RuntimeWarning) for w in caught)
    assert any(_dyld._DYLD_VAR in str(w.message) for w in caught)


def test_is_safe_to_reexec_false_in_interactive_repl(monkeypatch):
    """An interactive REPL (``sys.ps1`` present) is never safe to re-exec."""
    monkeypatch.setattr(_dyld.sys, "ps1", ">>> ", raising=False)
    monkeypatch.setattr(_dyld.sys, "modules", {})
    monkeypatch.setattr(_dyld.sys, "argv", ["train.py"])

    assert _dyld._is_safe_to_reexec() is False


def test_ensure_noop_when_no_ffmpeg_dir(monkeypatch):
    """macOS + torchcodec but no Homebrew ffmpeg dir -> return False, no env set."""
    monkeypatch.setattr(_dyld.sys, "platform", "darwin")
    monkeypatch.delenv(_dyld._OPT_OUT_ENV, raising=False)
    monkeypatch.delenv(_dyld._DYLD_VAR, raising=False)
    monkeypatch.setattr(_dyld, "_torchcodec_installed", lambda: True)
    monkeypatch.setattr(_dyld, "_find_ffmpeg_lib_dir", lambda: None)

    assert _dyld.ensure_ffmpeg_on_dyld_path() is False
    assert _dyld._DYLD_VAR not in _dyld.os.environ
