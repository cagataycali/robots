"""``strands-robots dev`` — start | stop | restart | status | logs for the dashboard.

The dashboard server itself is ``strands-robots dashboard`` (foreground). This command
manages it as a background process: detached spawn with a log file, health/auth-guard
wait, clean stop that also frees the ports and arm buses a dead server can leave held.

Two hard-won rules are baked in rather than remembered:
- The child runs with ``sys.executable`` and ``cwd`` = the imported package's parent, so
  the server is always the SAME code this command was invoked from — never a second venv
  or a stale site-packages copy.
- ``start`` wants a real TTY: macOS grants camera access only to terminal-started
  processes; a daemon-started dashboard is blind fleet-wide. Elsewhere the same
  refusal stands as a conservative default. ``--no-tty`` overrides.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

DEFAULT_PORT = 8090
ZENOH_PORT = 7447
PROCESS_PATTERN = "strands_robots dashboard"
LOG_DIR = Path(os.path.expanduser("~/.strands_dashboard/logs"))
TOKEN_FILE = Path(os.path.expanduser("~/.strands_dashboard/local_api_token.txt"))
CALIBRATION_DIR = Path(os.path.expanduser("~/.cache/huggingface/lerobot/calibration/robots"))
PROFILES_FILE = Path(os.path.expanduser("~/.strands_dashboard/profiles.json"))


# ---------------------------------------------------------------- pure rules

def server_argv(port: int, token_file: str | None) -> list[str]:
    """The exact child argv. --force: port-guard false-positives on CLOSE_WAIT."""
    argv = [sys.executable, "-m", "strands_robots", "dashboard", "--port", str(port), "--force"]
    if token_file:
        argv += ["--auth-token-file", token_file]
    return argv


def package_root() -> Path:
    """Directory whose ``strands_robots/`` is the package WE were imported from."""
    import strands_robots

    return Path(strands_robots.__file__).resolve().parent.parent


def guard_ok(unauth_status: int, auth_status: int) -> bool:
    """The auth guard is up when anonymous is refused AND the token is accepted."""
    return unauth_status == 401 and auth_status == 200


def logs_to_prune(names: list[str], keep: int = 5) -> list[str]:
    """Oldest-first names beyond the newest ``keep`` (names sort by timestamp)."""
    return sorted(names)[:-keep] if len(names) > keep else []


ARM_BUS_MARKERS = ("cu.usbmodem", "ttyACM", "ttyUSB")
"""Arm serial buses, by platform device naming.

macOS calls a USB CDC device ``/dev/cu.usbmodemXXXX``; Linux calls the same arm
``/dev/ttyACM*`` (CDC-ACM) or ``/dev/ttyUSB*`` (a USB-serial bridge such as the
CH341 in an SO-arm). The macOS marker is unchanged, so a mac keeps matching
exactly what it matched before -- the Linux names are additional, not a swap.
"""


def arm_bus_holder_pids(lsof_output: str) -> set[int]:
    """PIDs holding an arm serial bus, read from ``lsof -nP`` output.

    A marker is matched as a device path (``/dev/`` + marker), not anywhere on
    the line: a regular file named ``ttyUSB0.log`` is not a bus, and killing
    whoever holds it is not something a stop is entitled to do. ``lsof`` reports
    the resolved device node, so an arm opened through ``/dev/serial/by-id/...``
    still reads as ``/dev/ttyACM*`` here.
    """
    pids: set[int] = set()
    for line in lsof_output.splitlines():
        if not any(f"/dev/{marker}" in line for marker in ARM_BUS_MARKERS):
            continue
        fields = line.split()
        if len(fields) < 2 or not fields[1].isdigit():
            continue  # not an lsof file row: a header, or a truncated capture
        pids.add(int(fields[1]))
    return pids


def tty_refusal_reason(platform_name: str) -> str:
    """Why a TTY-less start is refused, in terms that hold on THIS platform.

    macOS has a mechanism to name: TCC grants camera access only to a
    terminal-started process. Elsewhere the refusal is a conservative default,
    and says so rather than borrowing a reason from another operating system.
    """
    if platform_name == "darwin":
        return "macOS never grants camera access to a daemon-started process."
    return "a daemon-started start is refused by default rather than assumed camera-capable."


def calibration_verdicts(profiles: dict, has_calibration) -> list[str]:
    """One line per real-mode profile: a robot_id with no robot-side calibration file
    joins the mesh, streams cameras, and reports NO JOINTS — say it before the wait."""
    lines = []
    for p in profiles.values():
        rid, name = p.get("robot_id"), p.get("name", "?")
        if p.get("mode") != "real" or not rid:
            continue
        if has_calibration(rid):
            lines.append(f"  ok: {name} (robot_id={rid})")
        else:
            lines.append(
                f"  MISSING: {name} robot_id={rid} has no robot-side calibration "
                f"-> it will spawn with NO JOINTS"
            )
    return lines


# ------------------------------------------------------------- process query

def _pgrep() -> list[int]:
    out = subprocess.run(
        ["pgrep", "-f", PROCESS_PATTERN], capture_output=True, text=True
    ).stdout.split()
    me = os.getpid()
    return [int(p) for p in out if p.isdigit() and int(p) != me]


def _lsof() -> str | None:
    return "/usr/sbin/lsof" if os.path.exists("/usr/sbin/lsof") else shutil.which("lsof")


def _port_holder(port: int) -> int | None:
    tool = _lsof()
    if not tool:
        return None
    out = subprocess.run(
        [tool, "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"], capture_output=True, text=True
    ).stdout.split()
    return int(out[0]) if out else None


def _http(url: str, token: str | None = None, timeout: float = 3.0) -> int:
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return 0


def _token() -> str | None:
    try:
        return TOKEN_FILE.read_text().strip() or None
    except OSError:
        return None


# ------------------------------------------------------------------- actions

def status(port: int) -> int:
    pids = _pgrep()
    print(f"running: pid {pids[0]}" if pids else "not running")
    code = _http(f"http://127.0.0.1:{port}/api/health")
    print(f"http :{port} -> {code}")
    token = _token()
    if code == 200 and token:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/fleet", headers={"Authorization": f"Bearer {token}"}
        )
        try:
            with urllib.request.urlopen(req, timeout=4) as r:
                peers = json.load(r).get("peers", {})
            for pid_, p in sorted(peers.items()):
                joints = len((p.get("state") or {}).get("joints") or {})
                cams = list((p.get("cameras") or {}).keys())
                print(f"  {pid_}: joints={joints} cams={cams} role={p.get('role')}")
        except Exception:
            pass
    return 0


def stop(port: int) -> int:
    pids = _pgrep()
    if not pids:
        print("already stopped")
    else:
        print(f"stopping pid {pids[0]}")
        os.kill(pids[0], signal.SIGTERM)
        for _ in range(15):
            if not _pgrep():
                break
            time.sleep(1)
        for p in _pgrep():
            print(f"no exit in 15s -> SIGKILL {p} (the ghost that keeps :{ZENOH_PORT})")
            os.kill(p, signal.SIGKILL)
            time.sleep(2)
    # Ports must be free — the ghost hides here.
    for pt in (port, ZENOH_PORT):
        holder = _port_holder(pt)
        if holder:
            print(f"port {pt} still held by {holder} -> SIGKILL")
            os.kill(holder, signal.SIGKILL)
            time.sleep(2)
    # Arm buses: orphaned spawn children only READ registers; killing a reader sends no
    # torque and cannot move an arm. Only safe here, after the stop.
    tool = _lsof()
    if tool:
        out = subprocess.run([tool, "-nP"], capture_output=True, text=True).stdout
        for h in arm_bus_holder_pids(out):
            print(f"arm bus held by orphan pid {h} -> SIGKILL (reader, moves nothing)")
            try:
                os.kill(h, signal.SIGKILL)
            except ProcessLookupError:
                pass
        out = subprocess.run([tool, "-nP"], capture_output=True, text=True).stdout
        if arm_bus_holder_pids(out):
            print("WARNING: a bus is STILL held — unplug/replug the arm before spawning")
        else:
            print("ports and arm buses free")
    else:
        print("note: lsof unavailable — port/bus check is permission-limited, not clean")
    return 0


def start(port: int, allow_no_tty: bool, wait: bool = True) -> int:
    if _pgrep():
        print(f"already running (pid {_pgrep()[0]}) — use restart")
        return 1
    if not os.isatty(0) and not allow_no_tty:
        print(f"REFUSED: no TTY. {tty_refusal_reason(sys.platform)}", file=sys.stderr)
        print("Run from a terminal, or pass --no-tty for a camera-less start.", file=sys.stderr)
        return 3

    print("== calibration preflight ==")
    try:
        profiles = json.loads(PROFILES_FILE.read_text())
    except Exception:
        profiles = {}
    for line in calibration_verdicts(
        profiles, lambda rid: bool(glob.glob(str(CALIBRATION_DIR / "*" / f"{rid}.json")))
    ):
        print(line)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    for name in logs_to_prune([p.name for p in LOG_DIR.glob("dashboard_*.log")]):
        (LOG_DIR / name).unlink(missing_ok=True)
    log_path = LOG_DIR / time.strftime("dashboard_%Y%m%d_%H%M%S.log")
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)

    token_file = str(TOKEN_FILE) if TOKEN_FILE.exists() else None
    child = subprocess.Popen(
        server_argv(port, token_file),
        cwd=package_root(),
        stdout=fd, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    os.close(fd)
    print(f"started pid {child.pid}  log: {log_path}")
    if not wait:
        return 0

    print("== waiting for the auth guard ==")
    token = _token()
    for _ in range(40):
        un = _http(f"http://127.0.0.1:{port}/api/fleet")
        au = _http(f"http://127.0.0.1:{port}/api/fleet", token=token) if token else 0
        if guard_ok(un, au):
            print("guard ok (unauth 401 / auth 200)")
            break
        if child.poll() is not None:
            print(f"server exited during startup (code {child.returncode}) — read the log:")
            print(f"  tail -40 {log_path}")
            return 1
        time.sleep(2)
    return status(port)


def logs() -> int:
    files = sorted(LOG_DIR.glob("dashboard_*.log"))
    if not files:
        print(f"no logs under {LOG_DIR}")
        return 1
    os.execvp("tail", ["tail", "-f", str(files[-1])])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="strands-robots dev", description="Manage the dashboard as a background dev server."
    )
    ap.add_argument("action", choices=["start", "stop", "restart", "status", "logs"])
    ap.add_argument("--port", type=int, default=int(os.getenv("STRANDS_DASHBOARD_PORT", DEFAULT_PORT)))
    ap.add_argument("--no-tty", action="store_true", help="allow a TTY-less (camera-less) start")
    ap.add_argument("--no-wait", action="store_true", help="don't wait for the auth guard")
    args = ap.parse_args(argv)

    allow_no_tty = args.no_tty or os.getenv("DASH_ALLOW_NO_TTY", "") == "1"
    if args.action == "status":
        return status(args.port)
    if args.action == "stop":
        return stop(args.port)
    if args.action == "start":
        return start(args.port, allow_no_tty, wait=not args.no_wait)
    if args.action == "restart":
        stop(args.port)
        return start(args.port, allow_no_tty, wait=not args.no_wait)
    return logs()


if __name__ == "__main__":
    sys.exit(main())
