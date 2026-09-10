"""MuJoCo GL backend selection - stdlib only, imported by ``strands_robots/__init__``.

MuJoCo reads ``MUJOCO_GL`` once, on ``import mujoco``, so the package
configures it at import time. These helpers used to live in
``strands_robots.simulation.mujoco.backend``; reaching that module runs
``strands_robots/simulation/__init__.py`` and its eager imports, which loaded
14 simulation modules on every ``import strands_robots`` with mujoco
installed. This leaf module has no parent package initialiser to pay for, and
``backend`` re-exports every name from it, so callers and tests that patch
``backend._library_loads`` etc. keep working - but patch THIS module when the
behaviour under test runs from ``strands_robots/__init__`` or ``doctor``.
"""

import ctypes
import logging
import os
import platform
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


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


# MuJoCo's own ``MUJOCO_GL`` vocabulary, transcribed from ``mujoco.gl_context``:
# it folds the value with ``.lower().strip()``, reads one family as "build no GL
# context at all", accepts a platform-dependent set of backend names, and raises
# ``RuntimeError`` at import time for anything else.
#
# Transcribed rather than asked. ``mujoco.gl_context`` computes its copy of this
# at module import, so it is already stale relative to a caller's environment;
# importing it is itself what raises for exactly the values a caller needs told
# about; and mujoco may not be installed at all, which is a separate question.
# Same reasoning as ``doctor._hf_token_path``, which transcribes the Hub's
# token-path rule for a Hub that may not be installed either.
_MUJOCO_GL_DISABLE: frozenset[str] = frozenset({"disable", "disabled", "off", "false", "0"})
_MUJOCO_GL_ANY_PLATFORM: frozenset[str] = frozenset({"enable", "enabled", "on", "true", "1", "glfw", ""})
_MUJOCO_GL_PLATFORM_ONLY: dict[str, frozenset[str]] = {
    "Linux": frozenset({"glx", "egl", "osmesa"}),
    "Windows": frozenset({"wgl"}),
    "Darwin": frozenset({"cgl"}),
}
# Backends that render with no display server: MuJoCo routes Linux ``egl`` and
# ``osmesa`` to offscreen contexts, while every other backend it selects (GLFW on
# Linux and Windows, CGL on macOS) draws through the platform's window server.
_MUJOCO_GL_OFFSCREEN: frozenset[str] = frozenset({"egl", "osmesa"})


def _mujoco_gl_value() -> str:
    """The ``MUJOCO_GL`` value MuJoCo will read, folded the way MuJoCo folds it.

    MuJoCo reads ``os.environ.get("MUJOCO_GL", "").lower().strip()``, so ``EGL``,
    ``egl`` and ``" egl "`` all select the same backend. Comparing the raw string
    instead answers about a spelling rather than about a backend: ``MUJOCO_GL=EGL``
    renders through EGL, while a case-sensitive test reads it as a value nobody
    recognises.

    Returns:
        The folded value. Empty both for an unset variable and for one holding
        only whitespace, which MuJoCo also reads as "no preference".
    """
    return os.environ.get("MUJOCO_GL", "").lower().strip()


def _mujoco_gl_valid_values(system: str | None = None) -> frozenset[str]:
    """Values MuJoCo accepts for ``MUJOCO_GL`` on a platform.

    Args:
        system: Platform name as :func:`platform.system` reports it. Defaults to
            the running platform.

    Returns:
        The platform-independent names together with that platform's own backend
        names. A folded value outside this set and outside
        :data:`_MUJOCO_GL_DISABLE` makes ``import mujoco`` raise ``RuntimeError``.
    """
    name = platform.system() if system is None else system
    return _MUJOCO_GL_ANY_PLATFORM | _MUJOCO_GL_PLATFORM_ONLY.get(name, frozenset())


def _mujoco_gl_disables_rendering(value: str) -> bool:
    """Whether a folded ``MUJOCO_GL`` value builds no GL context at all.

    MuJoCo accepts this family and then defines no ``GLContext``, so rendering is
    unavailable rather than misconfigured. It is a different answer from a value
    MuJoCo refuses at import, and naming the wrong one sends a caller after the
    wrong remedy.

    Args:
        value: A value already folded by :func:`_mujoco_gl_value`.

    Returns:
        True when MuJoCo will skip GL context creation entirely.
    """
    return value in _MUJOCO_GL_DISABLE


def _mujoco_gl_offscreen_values(system: str | None = None) -> frozenset[str]:
    """Accepted values that render on a platform with no display server.

    Empty on a platform whose only backend draws through the window server, which
    is what a caller needs in order not to recommend a value MuJoCo refuses there.

    Args:
        system: Platform name as :func:`platform.system` reports it. Defaults to
            the running platform.

    Returns:
        The offscreen backends that platform accepts.
    """
    return _MUJOCO_GL_OFFSCREEN & _mujoco_gl_valid_values(system)


# The system library each offscreen backend loads, in the order
# ``_configure_gl_backend`` probes them. A caller deciding what to *recommend*
# reads this rather than the platform vocabulary above: the two answer different
# questions, and only this one says whether a host can reach the backend.
_MUJOCO_GL_OFFSCREEN_LIBRARIES: tuple[tuple[str, str], ...] = (
    ("egl", "libEGL.so.1"),
    ("osmesa", "libOSMesa.so"),
)


def _library_loads(name: str) -> bool:
    """Whether a shared library can be loaded on this host.

    Args:
        name: Library soname, as :func:`ctypes.cdll.LoadLibrary` takes it.

    Returns:
        Whether the loader found it.
    """
    try:
        ctypes.cdll.LoadLibrary(name)
    except OSError:
        return False
    return True


def _mujoco_gl_loadable_offscreen_values(system: str | None = None) -> frozenset[str]:
    """Offscreen backends this host can reach, not merely the ones it accepts.

    :func:`_mujoco_gl_offscreen_values` answers a platform question: which
    offscreen values MuJoCo accepts on this operating system. Whether one of them
    can *render* is a different question, because each loads a system library that
    may not be installed. Recommending a value whose library is absent sends the
    reader after an export that changes nothing - which is why a verdict offering
    an offscreen backend reads this set and keeps the platform set only to tell
    "this platform has no offscreen backend" apart from "its libraries are
    missing".

    Args:
        system: Platform name as :func:`platform.system` reports it. Defaults to
            the running platform.

    Returns:
        The platform's offscreen backends whose library loads here.
    """
    accepted = _mujoco_gl_offscreen_values(system)
    return frozenset(
        value for value, library in _MUJOCO_GL_OFFSCREEN_LIBRARIES if value in accepted and _library_loads(library)
    )


# glvnd EGL vendor ICD payload that points at the NVIDIA EGL library, plus the
# standard directories glvnd scans for vendor ICD JSON files. When MUJOCO_GL is
# "egl", libglvnd loads the first vendor whose ICD JSON is registered here; an
# NVIDIA host missing 10_nvidia.json silently falls through to Mesa llvmpipe.
_NVIDIA_EGL_ICD_JSON = '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}'
_GLVND_EGL_VENDOR_DIRS: tuple[str, ...] = (
    "/usr/share/glvnd/egl_vendor.d",
    "/etc/glvnd/egl_vendor.d",
)
# Directories that may hold an installed NVIDIA EGL library (x86_64 + aarch64).
_NVIDIA_EGL_LIB_DIRS: tuple[str, ...] = (
    "/usr/lib",
    "/usr/lib/x86_64-linux-gnu",
    "/usr/lib/aarch64-linux-gnu",
    "/usr/lib64",
)


def _nvidia_egl_library_present() -> bool:
    """Return True when an NVIDIA EGL library (libEGL_nvidia.so.*) is installed."""
    for root in _NVIDIA_EGL_LIB_DIRS:
        try:
            if any(Path(root).glob("libEGL_nvidia.so.*")):
                return True
        except OSError:
            continue
    return False


def _nvidia_egl_icd_registered() -> bool:
    """Return True when a glvnd EGL vendor ICD already references NVIDIA."""
    for vendor_dir in _GLVND_EGL_VENDOR_DIRS:
        try:
            entries = sorted(Path(vendor_dir).glob("*.json"))
        except OSError:
            continue
        for entry in entries:
            try:
                if "nvidia" in entry.read_text(encoding="utf-8", errors="replace").lower():
                    return True
            except OSError:
                continue
    return False


def _ensure_nvidia_egl_vendor_icd() -> None:
    """Route MuJoCo's EGL backend to the NVIDIA GPU when the vendor ICD is missing.

    glvnd routes ``MUJOCO_GL=egl`` to whichever EGL vendor ICD JSON is registered
    under ``/usr/share/glvnd/egl_vendor.d/``. On an NVIDIA host or container where
    ``libEGL_nvidia`` is installed but the NVIDIA ICD (``10_nvidia.json``) is
    absent - common in CUDA base images that do not request the ``graphics``
    driver capability - glvnd silently falls back to Mesa ``llvmpipe`` (CPU
    software rasterization), roughly two orders of magnitude slower. That
    throttles every policy observation, rollout video, and dataset recording with
    no signal (see :func:`_warn_if_software_rendering`, which makes the symptom
    loud; this makes it go away).

    Rather than require root to write the system ICD, this stages a vendor ICD
    JSON in the user-writable strands-robots base dir and points glvnd at it via
    the ``__EGL_VENDOR_LIBRARY_FILENAMES`` env var (the documented libglvnd
    override) - the NVIDIA ICD first, then any already-registered system ICDs as
    fallback so non-NVIDIA setups are unaffected. Must run before ``import
    mujoco``; best-effort and never raises.

    No-op when: not Linux; the user already set ``__EGL_VENDOR_LIBRARY_FILENAMES``
    or ``__EGL_VENDOR_LIBRARY_DIRS`` (an explicit override is always respected); an
    NVIDIA ICD is already registered system-wide; or no NVIDIA EGL library is
    installed (not an NVIDIA host, so Mesa is the correct backend).
    """
    if sys.platform != "linux":
        return
    # Respect an explicit user/glvnd vendor override - never second-guess it.
    if os.environ.get("__EGL_VENDOR_LIBRARY_FILENAMES") or os.environ.get("__EGL_VENDOR_LIBRARY_DIRS"):
        return
    if _nvidia_egl_icd_registered():
        return
    if not _nvidia_egl_library_present():
        return

    try:
        from strands_robots.utils import get_base_dir

        icd_dir = get_base_dir() / "egl_vendor.d"
        icd_dir.mkdir(parents=True, exist_ok=True)
        nvidia_icd = icd_dir / "10_nvidia.json"
        nvidia_icd.write_text(_NVIDIA_EGL_ICD_JSON, encoding="utf-8")
    except OSError as e:
        logger.debug("Could not stage NVIDIA EGL vendor ICD, leaving glvnd default: %s", e)
        return

    # NVIDIA first, then any system vendor ICDs (Mesa etc.) as fallback.
    filenames = [str(nvidia_icd)]
    for vendor_dir in _GLVND_EGL_VENDOR_DIRS:
        try:
            filenames.extend(str(p) for p in sorted(Path(vendor_dir).glob("*.json")))
        except OSError:
            continue
    os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = ":".join(filenames)
    logger.info(
        "Registered NVIDIA EGL vendor ICD for GPU-accelerated MuJoCo offscreen rendering "
        "via __EGL_VENDOR_LIBRARY_FILENAMES (%s)",
        nvidia_icd,
    )


def _configure_gl_backend() -> None:  # noqa: C901
    """Auto-configure MuJoCo's OpenGL backend for headless environments.

    MuJoCo reads MUJOCO_GL at import time to select the OpenGL backend:
    - "egl"    - EGL (GPU-accelerated offscreen, requires libEGL + NVIDIA driver)
    - "osmesa" - OSMesa (CPU software rendering, slower but always works)
    - "glfw"   - GLFW (default, requires X11/Wayland display server)

    MuJoCo folds the value with ``.lower().strip()`` before reading it, so
    ``EGL`` and ``" egl "`` select EGL as well (see :func:`_mujoco_gl_value`).

    This function MUST be called before `import mujoco`. Setting MUJOCO_GL
    after import has no effect - the backend is locked at import time.

    Never overrides a user-set MUJOCO_GL value.
    """
    existing = os.environ.get("MUJOCO_GL")
    # Read the folded value, because that is the backend MuJoCo will select.
    # ``MUJOCO_GL=EGL`` renders through EGL and so has to reach the vendor-ICD
    # guarantee too; a case-sensitive test here left glvnd on its default, which
    # on an NVIDIA host missing the NVIDIA ICD is Mesa llvmpipe - the silent
    # software-rasterizer fallback ``_ensure_nvidia_egl_vendor_icd`` exists to
    # prevent. A whitespace-only value is not a preference: MuJoCo reads it as
    # unset, so auto-configuration is what respects it.
    value = _mujoco_gl_value()
    if value:
        logger.debug(f"MUJOCO_GL already set to '{existing}', respecting user config")
        if value == "egl":
            _ensure_nvidia_egl_vendor_icd()
        return

    if not _is_headless():
        return

    # Headless Linux - probe for EGL first (GPU-accelerated), then fall back to OSMesa (CPU)
    try:
        ctypes.cdll.LoadLibrary("libEGL.so.1")
        os.environ["MUJOCO_GL"] = "egl"
        _ensure_nvidia_egl_vendor_icd()
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
