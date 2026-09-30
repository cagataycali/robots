"""Locate and drive the Isaac Lab interpreter that runs outside this process.

Isaac Lab is never imported by strands-robots. It pins its own torch, warp,
newton and numpy (Isaac Lab 3.0.0rc1: torch 2.11, warp 1.16, newton 1.5), which
conflict with the versions the other extras resolve, and Kit hard-exits the
process that closes it. So it lives in its own virtual environment and this
module talks to that environment's interpreter only through ``argv``, a log
file and an exit-code file.

The interpreter is named by the operator, never by the agent: the
:data:`ISAACLAB_PYTHON_ENV` environment variable, or the ``python=`` argument of
:class:`~strands_robots.training.isaaclab.IsaacLabTrainer`. An agent
that could choose the interpreter could choose any program on the machine.

The NVIDIA Omniverse EULA is not accepted on the operator's behalf either.
With Isaac Sim installed, ``isaaclab train`` stops on an interactive EULA prompt
unless :data:`EULA_ENV` is ``YES``; the child runs with no stdin, so an
unaccepted EULA would end the run at the prompt. :func:`runtime_problems`
reports it before anything launches.
"""

from __future__ import annotations

import logging
import math
import os
import re
import signal
import subprocess
import time
from pathlib import Path

#: Environment variable naming the Python interpreter of the Isaac Lab venv.
ISAACLAB_PYTHON_ENV = "ISAACLAB_PYTHON"

#: Environment variable naming the directory job records are written under.
JOBS_DIR_ENV = "STRANDS_ISAACLAB_JOBS"

#: Environment variable through which the operator accepts the Omniverse EULA.
EULA_ENV = "OMNI_KIT_ACCEPT_EULA"

_EULA_ACCEPTED = frozenset({"yes", "y"})

#: Variables that would point the Isaac Lab interpreter at this process's
#: packages. The two stacks must not share an import path.
_LEAKED_ENV = ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "PYTHONSTARTUP")

#: File the launch wrapper writes the child's exit status into.
EXIT_CODE_FILE = "exit_code"

#: Environment variable the launch wrapper reads the exit-code path from, so
#: the path never enters the shell text.
_EXIT_FILE_ENV = "STRANDS_ISAACLAB_EXIT_FILE"

#: Environment variables the launch wrapper reads its deadline from: the
#: seconds the run may take, and the marker file it writes when they pass.
_DEADLINE_ENV = "STRANDS_ISAACLAB_DEADLINE_S"
_TIMED_OUT_FILE_ENV = "STRANDS_ISAACLAB_TIMED_OUT_FILE"

# ``"$@"`` runs the argv the wrapper was given, unquoted by no shell: every
# token reaches the child verbatim. The status lands in the file only after the
# child exits, so its absence while the process group is gone means the run was
# killed.
#
# With a deadline, a watchdog in the same process group enforces it whether or
# not anyone polls: it writes the timed-out marker, sends SIGTERM to the whole
# group (the wrapper leads it, see ``start_new_session``), gives the run up to
# 15 s to exit, then sends SIGKILL to whatever is left - itself included. It
# ignores SIGTERM so it survives to do that, and it checks every second that
# the run is still alive, so it leaves within a second of a run that ends on
# its own. Before, the deadline was checked only inside ``status()``: an agent
# that died left a ``timeout_s=30`` run holding the GPU for as long as it liked.
_WRAPPER = (
    '"$@" & child=$!; '
    'if [ -n "$STRANDS_ISAACLAB_DEADLINE_S" ]; then ( trap "" TERM; left="$STRANDS_ISAACLAB_DEADLINE_S"; '
    'while [ "$left" -gt 0 ]; do sleep 1; kill -0 "$child" 2>/dev/null || exit 0; left=$((left - 1)); done; '
    'date +%s > "$STRANDS_ISAACLAB_TIMED_OUT_FILE"; kill -TERM -$$ 2>/dev/null; grace=15; '
    'while [ "$grace" -gt 0 ] && kill -0 "$child" 2>/dev/null; do sleep 1; grace=$((grace - 1)); done; '
    "kill -KILL -$$ 2>/dev/null ) & fi; "
    'status=0; wait "$child" || status=$?; echo "$status" > "$STRANDS_ISAACLAB_EXIT_FILE"'
)


def resolve_python(explicit: str | None = None) -> str | None:
    """Return the Isaac Lab interpreter path, or ``None`` when none is configured.

    Args:
        explicit: A path given to the trainer's constructor; wins over the
            environment.

    Returns:
        The configured path (not yet checked to exist), or ``None``.
    """
    return explicit or os.environ.get(ISAACLAB_PYTHON_ENV) or None


def default_jobs_dir(explicit: str | None = None) -> Path:
    """Return the directory job records live under.

    Args:
        explicit: A directory given to the trainer's constructor.

    Returns:
        ``explicit``, else ``$STRANDS_ISAACLAB_JOBS``, else
        ``~/.cache/strands_robots/isaaclab/jobs``.
    """
    configured = explicit or os.environ.get(JOBS_DIR_ENV)
    if configured:
        # Absolute, now: the launch wrapper runs in the run's output_dir, so a
        # relative jobs dir named a different directory there and every run
        # ended "killed" with no exit status to read.
        return Path(configured).expanduser().resolve()
    return Path.home() / ".cache" / "strands_robots" / "isaaclab" / "jobs"


def runtime_problems(python: str | None, *, context: str) -> list[str]:
    """Report why the Isaac Lab runtime cannot be launched; empty when it can.

    Read-only: stats the interpreter and reads the environment.

    Args:
        python: The resolved interpreter path, or ``None``.
        context: Prefix naming the provider in each message.

    Returns:
        One message per missing prerequisite.
    """
    problems: list[str] = []
    if not python:
        problems.append(
            f"{context}: no Isaac Lab interpreter is configured - install Isaac Lab in its own virtual "
            f"environment and set {ISAACLAB_PYTHON_ENV} to that environment's python "
            "(see docs/learn/training/isaaclab.md); strands-robots never imports Isaac Lab itself"
        )
    else:
        path = Path(python)
        if not path.is_file():
            problems.append(f"{context}: {ISAACLAB_PYTHON_ENV}={python!r} does not exist or is not a file")
        elif not os.access(path, os.X_OK):
            problems.append(f"{context}: {ISAACLAB_PYTHON_ENV}={python!r} is not executable")
    if os.environ.get(EULA_ENV, "").strip().lower() not in _EULA_ACCEPTED:
        problems.append(
            f"{context}: the NVIDIA Omniverse EULA is not accepted - set {EULA_ENV}=YES in this process's "
            "environment to accept it (Isaac Lab otherwise stops on an interactive prompt the headless "
            "child cannot answer; strands-robots does not accept it for you)"
        )
    return problems


def child_env() -> dict[str, str]:
    """Return this process's environment without the variables that leak its packages.

    ``PYTHONUNBUFFERED=1`` keeps the log in the order things happened: with a
    pipe for stdout, a crash's traceback (stderr, unbuffered) otherwise lands
    above a block of iteration metrics flushed after it.
    """
    env = dict(os.environ)
    for name in _LEAKED_ENV:
        env.pop(name, None)
    env["PYTHONUNBUFFERED"] = "1"
    return env


# ``gym.register(id="Isaac-Cartpole", ...)`` in an Isaac Lab task package.
_REGISTER_ID_RE = re.compile(r"""\bid\s*=\s*["']([A-Za-z][A-Za-z0-9_.:-]{0,127})["']""")

#: Task packages Isaac Lab registers its gym ids in.
_TASK_PACKAGES = ("isaaclab_tasks", "isaaclab_tasks_experimental")

#: Environment variable through which the OPERATOR names their own task
#: packages: ``"acme_tasks:register_tasks,other_tasks"`` - each a module
#: importable in the Isaac Lab venv (installed, or on a ``.pth`` there),
#: optionally with the function Isaac Lab's ``--external_callback`` calls to
#: register its gym ids. Operator-owned like ``ISAACLAB_PYTHON``: an agent can
#: train these tasks but cannot point the Isaac Lab process at other code.
TASK_PACKAGES_ENV = "STRANDS_ISAACLAB_TASK_PACKAGES"

_TASK_PACKAGE_ENTRY_RE = re.compile(
    r"^([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)(?::([A-Za-z_][A-Za-z0-9_]*))?\Z"
)


def operator_task_packages() -> dict[str, str | None]:
    """``{module: callback-or-None}`` from :data:`TASK_PACKAGES_ENV`; malformed entries are skipped (logged)."""
    out: dict[str, str | None] = {}
    for raw in os.environ.get(TASK_PACKAGES_ENV, "").split(","):
        entry = raw.strip()
        if not entry:
            continue
        match = _TASK_PACKAGE_ENTRY_RE.match(entry)
        if match is None:
            logging.getLogger(__name__).warning("%s: ignoring malformed entry %r", TASK_PACKAGES_ENV, entry)
            continue
        out[match.group(1)] = match.group(2)
    return out


_TASK_CACHE: dict[str, tuple[float, frozenset[str]]] = {}


def registered_tasks(python: str) -> frozenset[str] | None:
    """Return the task ids the Isaac Lab install behind *python* registers, or ``None``.

    Read-only and without starting Isaac Lab: scans the ``gym.register(id=...)``
    calls in the task packages of that interpreter's virtual environment
    (``site-packages``, or a source checkout an editable install points at).
    ``None`` when no task package is found, so the caller launches as before
    and the run reports an unknown id itself.
    """
    roots = _task_package_roots(Path(python)) + [root for root, _ in _operator_package_roots(Path(python))]
    if not roots:
        return None
    stamp = max(root.stat().st_mtime for root in roots)
    key = "|".join(str(r) for r in roots)
    cached = _TASK_CACHE.get(key)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    ids: set[str] = set()
    for root in roots:
        for source in root.rglob("__init__.py"):
            try:
                text = source.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if "register(" in text:
                ids.update(_REGISTER_ID_RE.findall(text))
    tasks = frozenset(ids)
    _TASK_CACHE[key] = (stamp, tasks)
    return tasks or None


def task_package_of(python: str, task: str) -> tuple[str, str | None] | None:
    """The operator package (and its callback) whose sources register *task*, or ``None``.

    ``None`` too when *task* is Isaac Lab's own: those need no callback.
    """
    for root, module in _operator_package_roots(Path(python)):
        for source in root.rglob("*.py"):
            try:
                text = source.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if task in _REGISTER_ID_RE.findall(text):
                return module, operator_task_packages().get(module)
    return None


def _operator_package_roots(python: Path) -> list[tuple[Path, str]]:
    """``(directory, module)`` of each :data:`TASK_PACKAGES_ENV` package found in the venv."""
    found: list[tuple[Path, str]] = []
    for module in operator_task_packages():
        relative = Path(*module.split("."))
        for base in _site_search_paths(python):
            candidate = base / relative
            if (candidate / "__init__.py").is_file():
                found.append((candidate, module))
                break
    return found


def _site_search_paths(python: Path) -> list[Path]:
    """The venv's ``site-packages`` plus every absolute path its ``.pth`` files add."""
    venv = python.expanduser().absolute().parent.parent
    search: list[Path] = []
    for site in sorted(venv.glob("lib/python3*/site-packages")):
        search.append(site)
        for pth in site.glob("*.pth"):
            try:
                lines = pth.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            search.extend(Path(line.strip()) for line in lines if line.strip().startswith("/"))
    return search


def _task_package_roots(python: Path) -> list[Path]:
    """Directories of Isaac Lab's task packages in the venv that owns *python*."""
    venv = python.expanduser().absolute().parent.parent
    roots: list[Path] = []
    for site in sorted(venv.glob("lib/python3*/site-packages")):
        search = [site]
        for pth in site.glob("*.pth"):
            try:
                lines = pth.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            search.extend(Path(line.strip()) for line in lines if line.strip().startswith("/"))
        for base in search:
            for name in _TASK_PACKAGES:
                candidate = base / name
                if (candidate / "__init__.py").is_file() and candidate not in roots:
                    roots.append(candidate)
    return roots


def launch(
    cmd: list[str],
    *,
    cwd: Path,
    log_path: Path,
    exit_file: Path,
    timeout_s: float | None = None,
    timed_out_file: Path | None = None,
) -> subprocess.Popen[bytes]:
    """Start *cmd* detached in its own session, logging to *log_path*.

    The child gets no stdin (a prompt fails instead of hanging), and a
    ``/bin/sh`` wrapper records its exit status in *exit_file* once it ends.

    Args:
        cmd: The argv to run; passed to the wrapper as positional arguments.
        cwd: Working directory (Isaac Lab writes ``logs/`` under it).
        log_path: File receiving stdout and stderr.
        exit_file: File the wrapper writes the exit status to.
        timeout_s: Wall-clock limit the wrapper itself enforces on the whole
            process group, polled or not. ``None`` = no limit.
        timed_out_file: Marker the wrapper writes when *timeout_s* passes;
            required with *timeout_s*.

    Returns:
        The wrapper process, leader of a new process group.
    """
    env = child_env()
    env[_EXIT_FILE_ENV] = str(Path(exit_file).resolve())
    if timeout_s is not None:
        if timed_out_file is None:
            raise ValueError("launch: timeout_s needs a timed_out_file to mark the run with")
        env[_DEADLINE_ENV] = str(max(1, math.ceil(float(timeout_s))))
        env[_TIMED_OUT_FILE_ENV] = str(Path(timed_out_file).resolve())
    with open(log_path, "wb") as log:
        return subprocess.Popen(  # noqa: S603 - fixed argv, no shell interpolation of any token
            ["/bin/sh", "-c", _WRAPPER, "isaaclab-run", *cmd],
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )


def process_alive(pid: int) -> bool:
    """Whether *pid* is a live, non-zombie process."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
    except OSError:
        return True
    return "State:\tZ" not in status


def terminate(pid: int, *, grace_s: float = 10.0) -> None:
    """Stop the process group led by *pid*: SIGTERM, then SIGKILL after *grace_s*."""
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        if not process_alive(pid):
            return
        time.sleep(0.1)
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        return
