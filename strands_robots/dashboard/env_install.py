"""The dashboard installs the extras a spawn needs, into its own environment.

The owner's SO-101 spawn failed three ways in one afternoon, and two of them
were the environment: the lerobot path wanted the ``[lerobot]`` extra (torch),
and nothing in the dashboard could say so or do anything about it beyond
printing the child's ``ImportError``. This module is the dashboard's answer:

* :func:`snapshot` - which of the package's declared extras are installed, from
  ``importlib.metadata`` (the extra list is the package's own
  ``Provides-Extra``; no name a client sends is ever installed verbatim);
* :func:`spawn_preflight` - before a child is started, the driver the factory
  WILL pick and the modules it needs, so a spawn that would die on an import is
  refused with the extra that fixes it instead of started;
* :func:`missing_extra_in` - the extra a dead child's log names, so the Devices
  sheet can offer the install next to the refusal;
* :class:`InstallRun` / :func:`start` - ONE install at a time, as a subprocess
  of the interpreter the dashboard runs in (``uv pip install -p <python>`` when
  ``uv`` is on PATH, ``<python> -m pip install`` otherwise), from the package's
  own source when this is an editable install, with its output in a ring buffer
  that passes through :func:`~strands_robots.dashboard.log_redaction.redact_secrets`
  before a client reads it.

The allow-list is the point. An install endpoint that took a package name would
be ``pip install <anything>`` behind a session cookie; this one takes an extra
name, checks it against what the package declares, and spells the install line
itself.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import json
import logging
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

from strands_robots.dashboard.log_redaction import redact_secrets
from strands_robots.utils import refusal_repr

logger = logging.getLogger(__name__)

#: The distribution whose extras are the allow-list.
DISTRIBUTION = "strands-robots"

#: Lines of install output kept for the client.
_LINE_CAP = 400

#: An install that prints nothing for this long is still alive; the cap is on
#: the whole run so a hung resolver cannot hold the one slot forever.
INSTALL_TIMEOUT_S = 1800.0

#: A metadata requirement row: ``name[extras]<specifier>; marker``. The name is
#: the leading distribution token; the extra the row belongs to is the
#: ``extra == '<name>'`` clause of its marker. Parsed with two regexes rather
#: than ``packaging`` because that is not one of this package's dependencies.
_REQ_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")
_REQ_EXTRA = re.compile(r"""extra\s*==\s*['"]([A-Za-z0-9_.-]+)['"]""")

#: The install hint :func:`~strands_robots.utils.require_optional` prints, and
#: the children's ImportErrors therefore carry: the extra is the bracketed name.
_EXTRA_HINT = re.compile(r"strands-robots\[([A-Za-z0-9_.-]+)\]")

#: What each hardware driver family imports at connect time, by the module the
#: interpreter would fail on. Keyed by the driver name :func:`resolve_driver`
#: answers, then by whether cameras are configured.
_DRIVER_MODULES: dict[str, tuple[tuple[str, str], ...]] = {
    # (module, extra that supplies it)
    "lerobot": (("lerobot", "lerobot"),),
    "strands": (("serial", "dashboard"),),
}
_CAMERA_MODULES: tuple[tuple[str, str], ...] = (("cv2", "dashboard"),)


# ---------------------------------------------------------------------------
# What is declared, what is installed.
# ---------------------------------------------------------------------------


def _distribution() -> importlib.metadata.Distribution | None:
    try:
        return importlib.metadata.distribution(DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        return None


def declared_extras() -> dict[str, list[str]]:
    """Every extra the installed package declares, with the distributions it needs.

    Returns:
        Extra name -> distribution names whose requirement row carries that
        extra's marker, in declaration order. Empty when the package is not
        installed as a distribution (a bare source checkout on ``sys.path``).
    """
    dist = _distribution()
    if dist is None:
        return {}
    names = list(dist.metadata.get_all("Provides-Extra") or [])
    out: dict[str, list[str]] = {name: [] for name in names}
    for raw in dist.requires or []:
        head, _, marker = raw.partition(";")
        extra = _REQ_EXTRA.search(marker)
        name_match = _REQ_NAME.match(head)
        if extra is None or name_match is None:
            continue
        if extra.group(1) in out:
            out[extra.group(1)].append(name_match.group(1))
    return out


def _installed_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def extra_status(name: str, requirements: list[str]) -> dict[str, Any]:
    """One extra's row: installed when every distribution it names is present.

    Presence, not version: the question the Devices sheet asks is "is the
    module there", and a version mismatch is pip's to report in the install log.

    Args:
        name: The extra.
        requirements: Its distribution names from :func:`declared_extras`.

    Returns:
        ``{"name", "installed", "missing": [distribution names], "count"}``.
    """
    missing: list[str] = []
    for dist_name in requirements:
        if dist_name.lower() == DISTRIBUTION:
            # ``all = ["strands-robots[lerobot]", ...]``: a self reference is
            # satisfied by the extras it names, which have their own rows.
            continue
        if _installed_version(dist_name) is None:
            missing.append(dist_name)
    return {"name": name, "installed": not missing, "missing": missing, "count": len(requirements)}


def editable_source() -> Path | None:
    """The source tree this install is editable against, or ``None`` for a wheel install."""
    dist = _distribution()
    if dist is None:
        return None
    # Read through the RECORD entry (``dist.files``) with an explicit encoding
    # rather than ``Distribution.read_text``, which decodes with the locale.
    entry = next((f for f in dist.files or [] if f.name == "direct_url.json"), None)
    if entry is None:
        return None
    try:
        raw = Path(str(entry.locate())).read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        info = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(info, dict) or not (info.get("dir_info") or {}).get("editable"):
        return None
    url = str(info.get("url") or "")
    if not url.startswith("file://"):
        return None
    path = Path(url[len("file://") :])
    return path if path.is_dir() else None


def installer() -> str:
    """``"uv"`` when uv is on PATH, else ``"pip"``."""
    return "uv" if shutil.which("uv") else "pip"


def install_command(extra: str) -> list[str]:
    """The argv that installs ``extra`` into THIS interpreter.

    Args:
        extra: A declared extra (callers check the allow-list first).

    Returns:
        ``uv pip install -p <python> <spec>`` or ``<python> -m pip install <spec>``,
        where ``spec`` is ``-e <source>[<extra>]`` for an editable install and
        ``strands-robots[<extra>]`` otherwise.
    """
    source = editable_source()
    spec = [f"{DISTRIBUTION}[{extra}]"] if source is None else ["-e", f"{source}[{extra}]"]
    if installer() == "uv":
        return ["uv", "pip", "install", "-p", sys.executable, *spec]
    return [sys.executable, "-m", "pip", "install", *spec]


def snapshot() -> dict[str, Any]:
    """The environment the dashboard runs in, extra by extra."""
    extras = declared_extras()
    source = editable_source()
    return {
        "python": sys.executable,
        "version": sys.version.split()[0],
        "prefix": sys.prefix,
        "installer": installer(),
        "editable_source": None if source is None else str(source),
        "extras": [extra_status(name, reqs) for name, reqs in sorted(extras.items())],
        "install": None if current is None else current.status(),
    }


# ---------------------------------------------------------------------------
# Before a spawn: what the child would import.
# ---------------------------------------------------------------------------


def _importable(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def missing_extra_in(text: str | None) -> str | None:
    """The declared extra a child's refusal names, or ``None``.

    Reads the ``pip install 'strands-robots[<extra>]'`` hint
    :func:`~strands_robots.utils.require_optional` prints; a name outside the
    declared extras is ignored, so a log line cannot steer the install.
    """
    if not text:
        return None
    match = _EXTRA_HINT.search(text)
    if match is None:
        return None
    extra = match.group(1)
    return extra if extra in declared_extras() else None


def spawn_preflight(robot_name: str, cameras: Any = None, driver: str | None = None) -> dict[str, Any] | None:
    """What a real-mode spawn of ``robot_name`` would fail to import, or ``None``.

    Resolves the driver the way the factory will
    (:func:`~strands_robots.drivers.resolve_driver`) and checks the modules
    that driver imports at connect, plus OpenCV when cameras are configured.

    Args:
        robot_name: The registry robot the spawn names.
        cameras: The spawn's camera dict, or ``None``.
        driver: An explicit ``driver=`` when the spawn carries one.

    Returns:
        ``None`` when the child would import everything it needs, else
        ``{"missing": [modules], "missing_extra": <extra>, "remedy": <line>,
        "driver": <resolved>}`` naming the first extra that supplies a missing
        module.
    """
    from strands_robots.drivers import resolve_driver
    from strands_robots.registry import resolve_name

    try:
        resolved = resolve_driver(resolve_name(robot_name), driver)
    except ValueError as e:
        return {"missing": [], "missing_extra": None, "remedy": str(e), "driver": None}
    needed = list(_DRIVER_MODULES.get(resolved, ()))
    if cameras:
        needed.extend(_CAMERA_MODULES)
    missing = [(module, extra) for module, extra in needed if not _importable(module)]
    if not missing:
        return None
    extras = declared_extras()
    first_extra = next((extra for _, extra in missing if extra in extras), missing[0][1])
    return {
        "driver": resolved,
        "missing": [module for module, _ in missing],
        "missing_extra": first_extra,
        "remedy": f"pip install '{DISTRIBUTION}[{first_extra}]'",
    }


# ---------------------------------------------------------------------------
# One install at a time.
# ---------------------------------------------------------------------------


class InstallRun:
    """One ``pip install`` of a declared extra, with its output in a ring buffer."""

    def __init__(self, extra: str, *, argv: list[str] | None = None) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.extra = extra
        self.started_at = time.time()
        self.command = argv if argv is not None else install_command(extra)
        self._lines: deque[str] = deque(maxlen=_LINE_CAP)
        self._lock = threading.Lock()
        self.proc = subprocess.Popen(
            self.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
        self._reader = threading.Thread(target=self._read, name=f"env-install-{self.id}", daemon=True)
        self._reader.start()

    def _read(self) -> None:
        assert self.proc.stdout is not None
        deadline = time.monotonic() + INSTALL_TIMEOUT_S
        try:
            for raw in iter(self.proc.stdout.readline, b""):
                line = redact_secrets(raw.decode("utf-8", "replace").rstrip())
                if line:
                    with self._lock:
                        self._lines.append(line)
                if time.monotonic() > deadline and self.alive():
                    self.cancel()
                    with self._lock:
                        self._lines.append(f"install stopped: no result after {int(INSTALL_TIMEOUT_S)} s")
        except Exception as e:  # noqa: BLE001 - the reader must never take the server down
            with self._lock:
                self._lines.append(f"[reader error: {redact_secrets(str(e))}]")

    def alive(self) -> bool:
        """Whether the installer is still running."""
        return self.proc.poll() is None

    def cancel(self) -> None:
        """Stop the installer's process group."""
        if self.alive():
            try:
                self.proc.terminate()
            except (ProcessLookupError, PermissionError):
                pass  # already gone, or not ours to signal

    def lines(self) -> list[str]:
        """The redacted output so far (last :data:`_LINE_CAP` lines)."""
        with self._lock:
            return list(self._lines)

    def status(self) -> dict[str, Any]:
        """The run as the Devices sheet and the Settings drawer read it."""
        alive = self.alive()
        code = self.proc.returncode
        if alive:
            state = "running"
        elif code == 0:
            state = "done"
        else:
            state = "failed"
        return {
            "id": self.id,
            "extra": self.extra,
            "status": state,
            "alive": alive,
            "exit_code": code,
            "started_at": self.started_at,
            "command": list(self.command),
            "lines": self.lines(),
        }


#: The one install this process runs at a time; finished runs stay readable
#: until the next one starts.
current: InstallRun | None = None
_start_lock = threading.Lock()


def extra_name_error(extra: object) -> str | None:
    """Why ``extra`` is not an installable extra, or ``None`` when it is one."""
    if not isinstance(extra, str) or not extra.strip():
        return f"extra must be the name of a declared extra, got {refusal_repr(extra)}"
    declared = declared_extras()
    if extra not in declared:
        return f"extra {refusal_repr(extra)} is not one {DISTRIBUTION} declares; choose from {sorted(declared)}"
    return None


def start(extra: object, *, argv: list[str] | None = None) -> InstallRun:
    """Begin installing ``extra``, refusing while another install is running.

    Args:
        extra: A declared extra name (anything else is refused by name).
        argv: The command to run instead of :func:`install_command`; tests only.

    Returns:
        The run; poll :meth:`InstallRun.status`.

    Raises:
        ValueError: ``extra`` is not a declared extra.
        RuntimeError: An install is already running.
    """
    global current
    if (reason := extra_name_error(extra)) is not None:
        raise ValueError(reason)
    with _start_lock:
        if current is not None and current.alive():
            raise RuntimeError(f"an install of [{current.extra}] is already running (session {current.id})")
        current = InstallRun(str(extra), argv=argv)
        return current


def get(run_id: str) -> InstallRun | None:
    """The run with this id, or ``None``."""
    if current is not None and current.id == run_id:
        return current
    return None


def refresh_import_caches() -> None:
    """Let a module installed a moment ago be found without a restart."""
    importlib.invalidate_caches()
