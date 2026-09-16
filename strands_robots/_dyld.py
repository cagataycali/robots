"""macOS dyld shim so torchcodec finds Homebrew's ffmpeg with zero user setup.

The problem
-----------
torchcodec ships ``libtorchcodec_coreN.dylib`` linked against ffmpeg via
``@rpath/libavutil.NN.dylib`` etc. On macOS those ffmpeg dylibs live in the
Homebrew prefix (``/opt/homebrew/lib`` on Apple Silicon), which is NOT on the
default dyld search path. So ``import torchcodec`` fails with
``Library not loaded: @rpath/libavutil.59.dylib`` and
``StreamingLeRobotDataset`` cannot decode video frames.

Why ``os.environ`` mid-process does NOT fix it
----------------------------------------------
dyld snapshots ``DYLD_*`` env vars at process launch. Setting
``os.environ["DYLD_FALLBACK_LIBRARY_PATH"]`` after the interpreter has started
has no effect on subsequent ``dlopen`` of torchcodec (verified). Preloading the
ffmpeg dylibs with ``ctypes.CDLL(..., RTLD_GLOBAL)`` also does not satisfy
torchcodec's ``@rpath`` lookups on macOS.

The fix (zero-touch, idempotent)
--------------------------------
``ensure_ffmpeg_on_dyld_path()`` runs eagerly at ``import strands_robots``:

1. No-op unless we're on macOS arm64/x86_64 AND a Homebrew ffmpeg lib dir
   containing ``libavutil.*.dylib`` exists AND torchcodec is installed.
2. Set ``DYLD_FALLBACK_LIBRARY_PATH`` to include the ffmpeg lib dir. This makes
   child processes (DataLoader workers with ``num_workers>0``, subprocess
   training) inherit a correct environment immediately.
3. For the CURRENT process, dyld already snapshotted its env - so if (and only
   if) the env var was not already correct, re-exec the interpreter ONCE with
   the augmented environment. A guard env var prevents an exec loop.

Re-exec is gated tightly: it only fires when torchcodec is importable, ffmpeg is
present, and the var was missing - i.e. exactly the case where video streaming
would otherwise crash. Headless/Linux/Jetson and torchcodec-less installs never
re-exec. Opt out entirely with ``STRANDS_ROBOTS_NO_DYLD_SHIM=1``.

Where re-exec is unsafe (``python -c``, the REPL, Jupyter, pytest) the import
stays silent: the remedy is kept and :func:`video_decode_hint` hands it to the
first video-decoding open (``stream_dataset`` with video keys), which warns
there. A new user's ``import strands_robots`` is not the place to talk about
decoding video they have not asked for.
"""

from __future__ import annotations

import glob
import os
import sys

_GUARD_ENV = "_STRANDS_ROBOTS_DYLD_REEXEC"
_OPT_OUT_ENV = "STRANDS_ROBOTS_NO_DYLD_SHIM"
_DYLD_VAR = "DYLD_FALLBACK_LIBRARY_PATH"

# Homebrew lib dirs to probe, in priority order (Apple Silicon, then Intel).
_CANDIDATE_LIB_DIRS = ("/opt/homebrew/lib", "/usr/local/lib")

#: The remedy for a process the shim could not fix (an interactive host it
#: refused to re-exec); ``None`` when the process is fine or the question never
#: arose. Read through :func:`video_decode_hint`.
_pending_hint: str | None = None


def _find_ffmpeg_lib_dir() -> str | None:
    """Return a dir containing ffmpeg's ``libavutil.*.dylib``, or ``None``.

    Honors ``HOMEBREW_PREFIX`` (so non-standard Homebrew installs work) before
    falling back to the canonical Apple-Silicon / Intel prefixes.
    """
    dirs: list[str] = []
    brew_prefix = os.environ.get("HOMEBREW_PREFIX")
    if brew_prefix:
        dirs.append(os.path.join(brew_prefix, "lib"))
    dirs.extend(_CANDIDATE_LIB_DIRS)

    for d in dirs:
        # The versioned soname (libavutil.59.dylib) is what torchcodec's
        # @rpath entry resolves against; a bare libavutil.dylib alone is not
        # enough, so require at least one versioned match.
        if glob.glob(os.path.join(d, "libavutil.*.dylib")):
            return d
    return None


def _torchcodec_installed() -> bool:
    import importlib.util

    return importlib.util.find_spec("torchcodec") is not None


def _is_safe_to_reexec() -> bool:
    """True only for plain ``python script.py`` / ``python -m`` execution.

    Re-exec'ing replaces the process image - fine for a script, catastrophic
    inside a Jupyter kernel, an IPython REPL, a pytest run, or an embedding
    host. Detect those and refuse to re-exec there (we still export the env var
    for child processes and warn the user how to fix the current one).
    """
    # Interactive interpreter (python -i, plain REPL).
    if hasattr(sys, "ps1") or sys.flags.interactive:
        return False
    # Jupyter / IPython kernels.
    if "ipykernel" in sys.modules or "IPython" in sys.modules:
        return False
    # Test runners - re-exec would detach from the collector.
    if "pytest" in sys.modules or "PYTEST_CURRENT_TEST" in os.environ:
        return False
    # ``python -c '...'`` sets argv[0] to '-c' and ``python -`` / ``cmd |
    # python`` / ``python - <<EOF`` set it to '-' (program read from stdin).
    # Re-exec'ing either replays a stdin that has already been consumed, so the
    # fresh interpreter reads EOF and exits 0 having run nothing - a silent
    # no-op. Neither is safe to re-run; export the env var for children instead.
    argv0 = sys.argv[0] if sys.argv else ""
    if argv0 in ("", "-c", "-"):
        return False
    return True


def ensure_ffmpeg_on_dyld_path() -> bool:
    """Ensure Homebrew ffmpeg is on the dyld search path for torchcodec.

    Returns:
        True if the environment was already correct (or we just set it for
        child processes without needing a re-exec); the function may not return
        at all if it re-execs the current process.
    """
    # Opt-out / non-macOS / already-guarded fast paths.
    if os.environ.get(_OPT_OUT_ENV):
        return False
    if sys.platform != "darwin":
        return False
    if _torchcodec_installed() is False:
        # No torchcodec → nothing to fix (proprio-only streaming still works).
        return False

    ffmpeg_dir = _find_ffmpeg_lib_dir()
    if ffmpeg_dir is None:
        # No Homebrew ffmpeg found; let torchcodec raise its own clear error
        # if/when the user tries to decode video.
        return False

    current = os.environ.get(_DYLD_VAR, "")
    parts = [p for p in current.split(":") if p]
    already = ffmpeg_dir in parts

    # Always export for CHILD processes (DataLoader workers, subprocess train).
    if not already:
        os.environ[_DYLD_VAR] = ":".join([*parts, ffmpeg_dir])

    if already or os.environ.get(_GUARD_ENV):
        return True

    # The CURRENT process needs the var set at launch (dyld snapshot). Re-exec
    # once - but ONLY when it's safe (plain script run, not Jupyter/REPL/pytest).
    if _is_safe_to_reexec():
        os.environ[_GUARD_ENV] = "1"
        try:
            # sys.argv drops the interpreter options: under ``python -m pkg``
            # argv is ['/path/to/pkg/__main__.py', ...], so re-execing with
            # [sys.executable, *sys.argv] runs the __main__.py as a SCRIPT and
            # ``-m``-style module resolution (and any -X/-W flags) is lost -
            # observed as "No module named <subcommand>" for
            # ``python -m strands_robots dashboard``. sys.orig_argv (3.10+)
            # preserves the exact original command line.
            argv = list(getattr(sys, "orig_argv", None) or [sys.executable, *sys.argv])
            os.execv(sys.executable, argv)
        except Exception:
            return False  # fall through; children still benefit
    else:
        # Interactive/embedded: don't nuke the host. Remember the fix and say it
        # when video decode is actually asked for (:func:`video_decode_hint`),
        # not at import - ``python -c "import strands_robots"`` is the first
        # thing a new user types, and a warning about decoding video they never
        # requested is the wrong first impression. The hint is still logged at
        # debug level for anyone tracing the shim.
        global _pending_hint
        _pending_hint = (
            "strands_robots: torchcodec needs Homebrew ffmpeg on the dyld path "
            f"to decode video in this process. Set it before launching Python:\n"
            f"    export {_DYLD_VAR}={ffmpeg_dir}\n"
            "(child processes already inherit it; proprio-only streaming via "
            "drop_videos=True needs no ffmpeg)."
        )
        import logging

        logging.getLogger(__name__).debug(_pending_hint)
    return False


def video_decode_hint() -> str | None:
    """The dyld remedy for THIS process if it will not be able to decode video, else ``None``.

    Callers about to decode video (a streaming dataset with video keys) surface
    this as a warning at that point, so the person who asked for video hears
    about ffmpeg and the person who only imported the package does not.
    """
    return _pending_hint


#: Memo for :func:`quiet_video_backend` - the answer cannot change within a
#: process (dyld reads its search path once, at launch), and the probe that
#: produces it is the expensive import this function exists to do only once.
_quiet_backend: str | None = None
_quiet_backend_probed: bool = False


def quiet_video_backend() -> str | None:
    """The LeRobot ``video_backend`` to pass when the caller named none.

    LeRobot's own default resolver (``get_safe_default_video_backend``) tries to
    import torchcodec and, when the wheel is present but its native library
    cannot load, logs the WHOLE loader exception - five FFmpeg-version
    tracebacks, ~150 lines - before quietly falling back to pyav. Measured in
    this process class (``python -``, IPython, Jupyter, ``python -c``: the hosts
    :func:`ensure_ffmpeg_on_dyld_path` refuses to re-exec) on the first
    ``stop_recording`` of a 20-frame episode: the wall lands between
    ``run_policy``'s success and ``stop_recording``'s "Episode saved", for a
    recording that then reads back fine.

    This probe asks that same question - importing the same module LeRobot
    imports - with stderr captured, and answers in one line: ``"pyav"`` plus a
    single warning naming the remedy (the dyld export when Homebrew ffmpeg is
    installed but invisible to this process, otherwise the install), or ``None``
    - "let LeRobot choose" - when torchcodec loads or is not installed at all
    (LeRobot's message for an absent wheel is already one line). Probed once per
    process.

    Returns:
        ``"pyav"`` when torchcodec is installed but cannot load, else ``None``.
    """
    global _quiet_backend, _quiet_backend_probed
    if _quiet_backend_probed:
        return _quiet_backend
    _quiet_backend_probed = True
    if not _torchcodec_installed():
        return None
    import contextlib
    import importlib
    import io
    import logging
    import warnings

    try:
        with contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # The module name is LeRobot's, deliberately: its resolver imports
            # plain ``torchcodec``, and a probe that asked a NARROWER question
            # than the decision it is standing in for would answer "pyav" for a
            # torchcodec LeRobot would have used. ``torchcodec/__init__``
            # imports ``._core``, which imports ``.ops`` - the module that
            # dlopen()s libtorchcodec - so this name already covers the native
            # load, while naming a private submodule would turn any torchcodec
            # refactor into a silent pyav downgrade.
            importlib.import_module("torchcodec")
    except (ImportError, OSError, RuntimeError) as e:
        text = str(e)
        if _pending_hint:
            remedy = _pending_hint
        elif "libav" in text:
            remedy = (
                "torchcodec's FFmpeg shared libraries were not found - "
                "brew install ffmpeg (macOS) / apt install ffmpeg (Linux); torchcodec supports ffmpeg 4-8."
            )
        else:
            remedy = (
                "torchcodec was built for a different torch - install the matching one: "
                "https://github.com/pytorch/torchcodec#installing-torchcodec"
            )
        logging.getLogger(__name__).warning(
            "torchcodec is installed but cannot load in this process; decoding video with pyav instead "
            "(recording is unaffected; `strands-robots doctor` has the full diagnosis). %s",
            remedy,
        )
        _quiet_backend = "pyav"
        return _quiet_backend
    return None
