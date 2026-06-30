"""MuJoCo lazy import and GL backend configuration."""

import contextlib
import ctypes
import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
import warnings
from typing import Any

logger = logging.getLogger(__name__)

# Canonical "no live world" error message shared by every world-touching
# MuJoCo facade/mixin method. Defined in this low-level module so the
# Simulation facade and its mixins (physics, randomization, rendering,
# recording) can all source the single string without a circular import -
# an agent that learns the error from one action recognises it identically
# from every other.
_NO_WORLD_MSG = "No world. Call create_world (or load_scene) first."

_mujoco = None
_mujoco_viewer = None


def _is_headless() -> bool:
    """Detect if running in a headless environment (no display server).

    Returns True on Linux when no DISPLAY or WAYLAND_DISPLAY is set,
    which means GLFW-based rendering will fail.

    Windows and macOS are always False because MuJoCo uses native
    windowing backends (WGL on Windows, CGL on macOS) that support
    offscreen rendering without X11/Wayland. The EGL/OSMesa fallback
    is Linux-specific.
    """
    if sys.platform != "linux":
        return False
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return False
    return True


def _configure_gl_backend() -> None:  # noqa: C901
    """Auto-configure MuJoCo's OpenGL backend for headless environments.

    MuJoCo reads MUJOCO_GL at import time to select the OpenGL backend:
    - "egl"    - EGL (GPU-accelerated offscreen, requires libEGL + NVIDIA driver)
    - "osmesa" - OSMesa (CPU software rendering, slower but always works)
    - "glfw"   - GLFW (default, requires X11/Wayland display server)

    This function MUST be called before `import mujoco`. Setting MUJOCO_GL
    after import has no effect - the backend is locked at import time.

    Never overrides a user-set MUJOCO_GL value.
    """
    if os.environ.get("MUJOCO_GL"):
        logger.debug(f"MUJOCO_GL already set to '{os.environ['MUJOCO_GL']}', respecting user config")
        return

    if not _is_headless():
        return

    # Headless Linux - probe for EGL first (GPU-accelerated), then fall back to OSMesa (CPU)
    try:
        ctypes.cdll.LoadLibrary("libEGL.so.1")
        os.environ["MUJOCO_GL"] = "egl"
        logger.info("Headless environment detected - using MUJOCO_GL=egl (GPU-accelerated offscreen)")
        return
    except OSError:
        pass

    try:
        ctypes.cdll.LoadLibrary("libOSMesa.so")
        os.environ["MUJOCO_GL"] = "osmesa"
        logger.info("Headless environment detected - using MUJOCO_GL=osmesa (CPU software rendering)")
        return
    except OSError:
        pass

    logger.warning(
        "Headless environment detected but neither EGL nor OSMesa found. "
        "MuJoCo rendering will likely fail. Install one of:\n"
        "  GPU: apt-get install libegl1-mesa-dev  (or NVIDIA driver provides libEGL)\n"
        "  CPU: apt-get install libosmesa6-dev\n"
        "Then set: export MUJOCO_GL=egl  (or osmesa)"
    )


def _ensure_mujoco() -> "Any":
    """Lazy import MuJoCo to avoid hard dependency.

    Auto-configures the OpenGL backend for headless environments before
    importing mujoco, since MUJOCO_GL must be set at import time.

    Uses require_optional() for consistent dependency management across
    the strands-robots package.
    """
    global _mujoco, _mujoco_viewer
    if _mujoco is None:
        _configure_gl_backend()
        from strands_robots.utils import require_optional

        _mujoco = require_optional(
            "mujoco",
            pip_install="mujoco",
            extra="sim-mujoco",
            purpose="MuJoCo simulation",
        )
    if _mujoco_viewer is None and not _is_headless():
        try:
            import mujoco.viewer as viewer

            _mujoco_viewer = viewer
        except ImportError:
            pass
    return _mujoco


_rendering_available: bool | None = None


def _can_render() -> bool:
    """Check if MuJoCo offscreen rendering is available.

    Probes once by creating a minimal Renderer in a subprocess. Result is cached.
    Returns False on headless environments without EGL/OSMesa.

    On headless Linux, if MUJOCO_GL is not set after _configure_gl_backend()
    ran, it means neither EGL nor OSMesa is available. In that case the
    default GLFW backend would be used, which calls glfw.init() - abort()
    at the C level (SIGABRT), killing the entire process before Python can
    catch the error. We short-circuit to False to avoid the fatal probe.

    When MUJOCO_GL IS set (e.g. "egl"), the library may still be dysfunctional
    (libEGL.so.1 loadable but no GPU/driver). In that case mj.Renderer() aborts
    at the C level too. We run the probe in a subprocess so a SIGABRT in the
    child doesn't kill the host process.
    """
    global _rendering_available
    if _rendering_available is not None:
        return _rendering_available

    # Guard: on headless systems without an offscreen GL backend configured,
    # mj.Renderer() will use GLFW which triggers a C-level abort (SIGABRT).
    # Skip the probe entirely - rendering is impossible anyway.
    if _is_headless() and not os.environ.get("MUJOCO_GL"):
        _rendering_available = False
        logger.warning(
            "Headless environment without EGL/OSMesa - rendering disabled. "
            "Physics and joint observations will still work. "
            "Install libegl1-mesa-dev or libosmesa6-dev for camera rendering."
        )
        return False

    # Probe rendering in a subprocess to survive C-level aborts (SIGABRT).
    # On some CI environments, libEGL.so.1 is loadable but non-functional -
    # mj.Renderer() triggers a fatal abort that kills the entire process.
    # By running the probe in a child process, we detect the failure safely.
    probe_script = (
        "import mujoco;"
        "m=mujoco.MjModel.from_xml_string('<mujoco><worldbody/></mujoco>');"
        "r=mujoco.Renderer(m,height=1,width=1);"
        "r.close()"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", probe_script],
            capture_output=True,
            timeout=10,
            env=os.environ.copy(),
        )
        if result.returncode == 0:
            _rendering_available = True
            logger.info("MuJoCo rendering available (subprocess probe passed)")
        else:
            _rendering_available = False
            stderr = result.stderr.decode(errors="replace").strip()
            # Truncate for readability
            if len(stderr) > 200:
                stderr = stderr[:200] + "..."
            logger.warning(
                "MuJoCo rendering unavailable (subprocess probe failed, rc=%d): %s. "
                "Physics/policy will work, but render/camera observations will be skipped.",
                result.returncode,
                stderr,
            )
    except (subprocess.TimeoutExpired, OSError) as e:
        _rendering_available = False
        logger.warning(
            "MuJoCo rendering probe timed out or failed to run: %s. Rendering disabled.",
            e,
        )

    return _rendering_available  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# MuJoCo stderr-noise suppression
# ---------------------------------------------------------------------------
# When two scenes are merged via ``spec.attach()`` / ``spec.compile()`` MuJoCo
# prints a block of benign "compiler option conflict" lines straight to the C
# stderr (fd 2) -- e.g.::
#
#     WARNING: Attach conflict when attaching 'scene' to 'strands_sim', ...
#     timestep: parent has 0.002, child has 0.005, keeping parent value
#     iterations: parent has 100, child has 10, keeping parent value
#     ...
#
# These are not actionable by the user (we intentionally keep the parent's
# compiler options) and they spam the console on every add_robot / world
# build. Because they originate in C, Python's ``warnings`` / ``logging`` can't
# intercept them -- we have to filter the raw file descriptor.
#
# ``filter_mujoco_attach_noise()`` is a context manager that captures fd 2 for
# the duration of the wrapped call, drops the known-benign lines, forwards
# everything else through unchanged, and re-emits any dropped lines at DEBUG
# level so they are still recoverable with STRANDS_ROBOTS_VERBOSE_MUJOCO=1.


# Lines we consider benign attach/compile chatter. Matched case-insensitively
# against each captured stderr line.
_MUJOCO_NOISE_PATTERNS = (
    re.compile(r"Attach conflict when attaching", re.IGNORECASE),
    re.compile(
        r"^(timestep|iterations|ls_iterations|impratio|integrator|cone|"
        r"jacobian|solver|tolerance|ls_tolerance|noslip_iterations|"
        r"noslip_tolerance|ccd_iterations|ccd_tolerance|sdf_iterations|"
        r"sdf_initpoints|gravity|wind|magnetic|density|viscosity):"
        r".*(parent has|keeping parent value)",
        re.IGNORECASE,
    ),
    re.compile(r"parent has .* child has .* keeping parent value", re.IGNORECASE),
)

_noise_lock = threading.Lock()


def _is_mujoco_noise(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    return any(p.search(stripped) for p in _MUJOCO_NOISE_PATTERNS)


@contextlib.contextmanager
def filter_mujoco_attach_noise():
    """Suppress MuJoCo's benign attach/compile spam.

    MuJoCo emits "Attach conflict ... keeping parent value" chatter when two
    scenes are merged. Depending on the MuJoCo build this surfaces either as a
    Python ``UserWarning`` (from ``spec.to_xml()`` / ``compile()``) OR as raw
    C-level writes to stderr (fd 2). We suppress BOTH:

    * a ``warnings.catch_warnings`` filter drops the matching ``UserWarning``;
    * an fd-2 capture drops the matching raw lines and forwards the rest.

    No-op (yields immediately) when STRANDS_ROBOTS_VERBOSE_MUJOCO is truthy,
    when fd 2 can't be captured (e.g. already redirected, no real stderr in
    some test/Jupyter setups), or on any failure -- never let log hygiene
    break a working sim.
    """
    if os.getenv("STRANDS_ROBOTS_VERBOSE_MUJOCO", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        yield
        return

    # Layer 1: drop the matching Python UserWarning regardless of fd capture.
    _wctx = warnings.catch_warnings()
    _wctx.__enter__()
    warnings.filterwarnings(
        "ignore",
        message=r".*Attach conflict when attaching.*",
        category=UserWarning,
    )

    # Layer 2: capture raw fd-2 writes. Need a real, dup-able stderr fd.
    try:
        orig_fd = sys.stderr.fileno()
        saved_fd = os.dup(orig_fd)
    except (AttributeError, OSError, ValueError):
        # No real fd (captured stderr / pytest capsys / Jupyter) -> warning
        # filter still applies; just skip the fd capture.
        try:
            yield
        finally:
            _wctx.__exit__(None, None, None)
        return

    tmp = tempfile.TemporaryFile(mode="w+b")
    try:
        with _noise_lock:
            os.dup2(tmp.fileno(), orig_fd)
            try:
                yield
            finally:
                # Flush C-side then restore the real stderr.
                try:
                    sys.stderr.flush()
                except (ValueError, OSError):
                    # Best-effort flush; stderr may already be closed or
                    # detached during interpreter teardown. Nothing to recover.
                    pass
                os.dup2(saved_fd, orig_fd)
    finally:
        # Replay captured output, dropping the known-benign noise lines.
        try:
            tmp.seek(0)
            captured = tmp.read().decode(errors="replace")
        except OSError:
            # Capture file unreadable (closed/truncated); treat as no output
            # rather than masking the original yielded body with an I/O error.
            captured = ""
        finally:
            tmp.close()
            try:
                os.close(saved_fd)
            except OSError:
                # saved_fd may already be closed during teardown; cleanup is
                # best-effort, so a double-close is safe to ignore.
                pass

        if captured:
            kept, dropped = [], []
            for line in captured.splitlines(keepends=True):
                (dropped if _is_mujoco_noise(line) else kept).append(line)
            if kept:
                try:
                    sys.stderr.write("".join(kept))
                    sys.stderr.flush()
                except (ValueError, OSError):
                    # Best-effort replay of kept lines; if stderr is gone the
                    # captured noise is simply dropped. Nothing to recover.
                    pass
            if dropped:
                logger.debug(
                    "Suppressed %d benign MuJoCo attach/compile line(s) "
                    "(set STRANDS_ROBOTS_VERBOSE_MUJOCO=1 to see them):\n%s",
                    len(dropped),
                    "".join(dropped).rstrip(),
                )
        # Tear down the UserWarning filter last.
        _wctx.__exit__(None, None, None)


@contextlib.contextmanager
def capture_stderr_fd():
    """Capture C-level (fd 2) stderr writes into a list-wrapped string.

    Unlike ``contextlib.redirect_stderr`` -- which only swaps Python's
    ``sys.stderr`` object and is blind to writes coming from C extensions
    such as MuJoCo's OpenGL backend -- this redirects the underlying file
    descriptor 2, so warnings like the one-time ``ARB_clip_control`` notice
    emitted by ``renderer.render()`` are actually captured.

    Yields a single-element list; after the block exits, ``box[0]`` holds the
    captured text (possibly empty).

    No-op-ish fallback: if fd 2 can't be duped (captured stderr, pytest
    capsys, Jupyter), it yields an empty box and lets writes pass through
    rather than breaking the wrapped call.
    """
    box = [""]
    try:
        orig_fd = sys.stderr.fileno()
        saved_fd = os.dup(orig_fd)
    except (AttributeError, OSError, ValueError):
        # No real fd to capture; yield empty and pass through.
        yield box
        return

    tmp = tempfile.TemporaryFile(mode="w+b")
    try:
        with _noise_lock:
            os.dup2(tmp.fileno(), orig_fd)
            try:
                yield box
            finally:
                try:
                    sys.stderr.flush()
                except (ValueError, OSError):
                    pass  # stderr may be closed/detached (pytest capsys, Jupyter)
                os.dup2(saved_fd, orig_fd)
    finally:
        try:
            tmp.seek(0)
            box[0] = tmp.read().decode(errors="replace")
        except OSError:
            box[0] = ""
        finally:
            tmp.close()
            try:
                os.close(saved_fd)
            except OSError:
                pass  # fd may already be closed during interpreter shutdown
