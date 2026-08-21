"""U16: UI selection -> a Python file that recreates the exact peer elsewhere.

The dashboard's spawn form and the profile store both describe a rig as a
payload dict (robot_name/mode/port/cameras/robot_id/peer_id). This renders
that payload as a copy-pasteable ``strands_robots`` script whose factory call
mirrors the dashboard's own child spawner (device_manager._SPAWNER) line for
line - same env posture, same eager connect, same keep-alive - so "runs as a
dashboard child" and "runs on an edge device" are the same object, not two
programs that drift apart.

Pure module: no I/O, no imports from the dashboard app. The route feeds it
either a saved profile or the live form state; tests feed it dicts.
"""

from __future__ import annotations

import ipaddress

import time
from typing import Any, Mapping

from . import camera_liveness

__all__ = ["render_snippet", "snippet_filename"]

#: Env the dashboard's own spawner sets. Rendered as setdefault so an edge
#: box with a deliberate different posture (real ACL file, mTLS) wins.
_MESH_ENV: tuple[tuple[str, str], ...] = (
    ("STRANDS_ROBOTS_NO_DYLD_SHIM", "1"),
    ("STRANDS_MESH_LOCAL_DEV", "1"),
    ("STRANDS_MESH_MULTICAST", "true"),
    ("STRANDS_MESH", "true"),
    ("STRANDS_MESH_CAMERA_HZ", "5"),
)

#: Default zenoh port, mirroring ``mesh/session.py``.
DEFAULT_HUB_PORT = 7447

#: Q53: keys whose value must come from THIS dashboard's live posture rather than the frozen
#: table above, because the file's whole promise is "recreates this exact rig".
_LIVE_KEYS: frozenset[str] = frozenset({"STRANDS_MESH_CAMERA_HZ", "STRANDS_MESH_MULTICAST"})

#: Q53: a key that only ever LOOSENS security. STRANDS_MESH_LOCAL_DEV=1 disables mesh wire
#: security, so it may be rendered only when this dashboard is itself running that way - never
#: as a hardcoded default. Writing it into a file an operator runs on their LAN would disable
#: encryption on a box they never chose to expose, and the peer would then fail to join a
#: secured desk for a reason no message explains.
_SECURITY_LOOSENING_KEYS: frozenset[str] = frozenset({"STRANDS_MESH_LOCAL_DEV"})


def resolve_mesh_env(env: Mapping[str, str] | None) -> list[tuple[str, str]]:
    """The env block to render, taking live values over the frozen defaults.

    Pure: the route passes ``os.environ`` (already carrying ``settings.apply_mesh_env()``'s
    mesh keys), tests pass dicts. A key that loosens security is emitted ONLY when the live env
    says so; every other key falls back to the default the dashboard's own spawner uses.
    """
    live = env or {}
    out: list[tuple[str, str]] = []
    for key, default in _MESH_ENV:
        value = str(live.get(key, "")).strip()
        if key in _SECURITY_LOOSENING_KEYS:
            if value:
                out.append((key, value))
            continue
        if key in _LIVE_KEYS and value:
            out.append((key, value))
        else:
            out.append((key, default))
    return out


def snippet_filename(peer_id: str) -> str:
    """A filename safe to offer as a download, derived from the peer id."""
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in peer_id)
    return f"{safe or 'robot'}.py"


def _fmt(value: Any, indent: int = 0) -> str:
    """Render a payload value as Python source (JSON types only).

    ``repr`` is almost right but renders dict one-line; camera configs are
    nested and read better expanded. Only handles what a spawn payload can
    legally contain - str/int/float/bool/None/dict - because the payload was
    validated by the same rules the spawn route applies.
    """
    if isinstance(value, Mapping):
        if not value:
            return "{}"
        pad = " " * (indent + 4)
        items = ",\n".join(f"{pad}{k!r}: {_fmt(v, indent + 4)}" for k, v in value.items())
        return "{\n" + items + ",\n" + " " * indent + "}"
    return repr(value)


def hub_host_from_reached(reached_on: str | None) -> tuple[str | None, str | None]:
    """Is the host the browser reached this dashboard on usable as a ZENOH hub address?

    Q122. The snippet is copied onto ANOTHER machine, so the address in it must be reachable from
    there - and the only address the server had was the one the browser used, which answers a
    different question. Two ways that goes wrong, one already handled and one not:

    * loopback - ``127.0.0.1`` on the edge device means the EDGE DEVICE. Already refused.
    * a public hostname - the interesting case, and it is cagatay's own setup. This dashboard is
      published through a Cloudflare tunnel (``robots.cagatay.my``) that proxies HTTP to :8090 and
      NOTHING else. The zenoh hub is a raw TCP port; ``tcp/robots.cagatay.my:7447`` resolves to a
      Cloudflare edge address that does not carry it, so the peer starts, connects to nothing and
      never appears - the precise failure this file's port comment already warns about, arriving
      through a different door. A snippet that states a wrong address is worse than one that states
      none: it looks authoritative, and it is the file the operator runs on a machine they cannot see.

    Returns ``(host, note)``: the host to use, or None plus a note explaining what was rejected.
    A note is returned WITH a host only when something is worth saying about it - never silently.
    """
    reached = (reached_on or "").strip().strip("[]").lower()
    if not reached:
        return None, None
    if reached in ("localhost", "127.0.0.1", "::1") or reached.endswith(".localhost"):
        return None, (
            "you opened this dashboard on localhost, and on the edge device that name means the "
            "edge device itself"
        )
    try:
        ip = ipaddress.ip_address(reached)
    except ValueError:
        # A NAME. Treat the LAN-shaped ones as usable; anything else is an entry point, not a host.
        if reached.endswith((".local", ".lan", ".internal", ".home", ".arpa")) or "." not in reached:
            return reached, None
        return None, (
            f"you reached this dashboard at {reached}, which is a public name - usually a tunnel or "
            "reverse proxy that forwards HTTP only. The zenoh hub is a raw TCP port and is almost "
            "certainly not published there, so a peer pointed at it would connect to nothing"
        )
    if ip.is_loopback:
        return None, "that address is loopback: on the edge device it would mean the edge device"
    if ip.is_private or ip.is_link_local:
        return reached, None
    return None, (
        f"{reached} is a public address; the zenoh hub should be reached over your own network, not "
        "the internet. Use this machine's LAN address (the Mesh tab shows it)"
    )


def _stamped_names(cameras: Any) -> dict[str, str]:
    """{camera: the roster name this index carried when it was configured}, for the cameras that have one."""
    if not isinstance(cameras, Mapping):
        return {}
    out: dict[str, str] = {}
    for cam, cfg in cameras.items():
        if isinstance(cfg, Mapping):
            was = cfg.get("device_name")
            if isinstance(was, str) and was.strip():
                out[str(cam)] = was.strip()
    return out


def render_snippet(
    payload: Mapping[str, Any],
    *,
    hub_host: str | None = None,
    hub_note: str | None = None,
    mesh_env: Mapping[str, str] | None = None,
    hub_port: int | str | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Render a spawn payload/profile as a deployable Python script.

    Refuses (``{"error": ...}``) instead of guessing when the payload cannot
    describe a runnable robot - a generated file that starts the WRONG rig is
    worse than no file.

    ``hub_host`` is the address edge devices reach THIS dashboard's zenoh hub
    on; rendered into ``ZENOH_CONNECT``. None means same-machine deploy and
    the line is emitted commented out, showing where the host goes.
    """
    robot_name = str(payload.get("robot_name") or "").strip()
    if not robot_name:
        return {"error": "payload has no robot_name"}
    mode = str(payload.get("mode") or "").strip()
    if mode not in ("real", "sim"):
        return {"error": f"mode must be real or sim, got {mode!r}"}
    port = payload.get("port")
    if mode == "real" and not port:
        return {"error": "port required for mode=real"}
    peer_id = str(payload.get("peer_id") or payload.get("name") or "").strip()
    if not peer_id:
        return {"error": "payload has no peer_id"}

    stamp = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(now if now is not None else time.time()))
    name = payload.get("name") or peer_id

    lines: list[str] = ["#!/usr/bin/env python3", f'"""{name} - generated by the strands-robots dashboard, {stamp}.']
    lines += [
        "",
        "Recreates this exact rig on any machine with strands-robots installed:",
        "",
        "    pip install strands-robots",
        f"    python {snippet_filename(peer_id)}",
        "",
    ]
    role = payload.get("role")
    if role in ("leader", "follower") and payload.get("role_source") == "measured":
        volts = payload.get("role_volts")
        volts_txt = f" at {volts}V" if volts is not None else ""
        lines.append(f"This arm measured as the {role.upper()}{volts_txt} on its servo bus.")
        lines.append("")
    if payload.get("cameras"):
        lines.append("Camera indices are PER-MACHINE: index 1 here may be index 0 on the")
        lines.append("edge device. Re-check them there (lerobot-find-cameras opencv).")
        # The roster name this index carried when it was configured is the ONE thing that makes that
        # re-check possible: "index 0" is a position in a list that closes up when a device is
        # unplugged, while "USB2.0_CAM1" is a camera. The dashboard stamps it into the config
        # (camera_liveness.stamp_device_names) and it is stripped from the CODE below, because
        # hardware_robot refuses an unknown camera option and would kill every camera on the arm --
        # so it is said HERE, where a human reads it, instead of being thrown away.
        for cam, was in sorted(_stamped_names(payload.get("cameras")).items()):
            lines.append(f'  {cam}: was "{was}" on the dashboard machine - check that camera is still that index.')
        lines.append("")
    if mode == "real":
        # Q47: the file warned that camera INDICES are per-machine and said nothing about the
        # port, which is the same class of identifier and the more certain to be wrong: this
        # dashboard runs on macOS, where the arm is /dev/cu.usbmodem5AB0181806, and the edge
        # device it is being deployed to is almost always Linux, where the same arm is
        # /dev/ttyACM0. Left unsaid, the generated file fails on an open() of a path that never
        # existed there - the exact failure Q46 removed from the home screen's snippet.
        serial = payload.get("serial_number")
        lines.append(f"The port below ({port}) is how THIS machine names that USB device.")
        lines.append("On Linux the same arm is usually /dev/ttyACM0 - check with")
        lines.append("`lerobot-find-port`, or list /dev/serial/by-id/ where the name is stable.")
        if serial and serial != port:
            # The serial IS the identity the dashboard itself keys profiles by, so it is the one
            # string that survives the move. Saying so turns a reboot-shuffled port from a
            # mystery into a lookup.
            lines.append(f"That device's USB serial is {serial} - unlike the port, it does not")
            lines.append("change when the device is replugged or the machine reboots.")
        lines.append("")
    lines[-1] = lines[-1] if lines[-1] else '"""'
    if lines[-1] != '"""':
        lines.append('"""')
    lines += ["", "import os", "import time", ""]

    lines.append("# The mesh posture this dashboard runs with (setdefault: your own env wins).")
    for key, val in resolve_mesh_env(mesh_env):
        lines.append(f'os.environ.setdefault("{key}", "{val}")')
    # Q53: the port comes from THIS dashboard's mesh settings. Hardcoding 7447 while the desk
    # ran on another port produced a peer that starts, logs nothing wrong and never appears -
    # the exact failure the Mesh tab warns about with "every robot on the desk must agree on it".
    try:
        port_txt = str(int(str(hub_port))) if hub_port not in (None, "") else str(DEFAULT_HUB_PORT)
    except (TypeError, ValueError):
        port_txt = str(DEFAULT_HUB_PORT)
    if hub_host:
        lines.append("# Reach this dashboard's zenoh hub from the edge device:")
        lines.append(f'os.environ.setdefault("ZENOH_CONNECT", "tcp/{hub_host}:{port_txt}")')
    else:
        if hub_note:
            # The operator asked for a deployable file and got a commented-out line; without the
            # reason, the only reading is "the dashboard forgot".
            lines.append(f"# NOTE: {hub_note}.")
        lines.append("# Deploying to ANOTHER machine? Point the peer at this dashboard's hub:")
        lines.append(
            f'# os.environ.setdefault("ZENOH_CONNECT", "tcp/<dashboard-host>:{port_txt}")'
        )
    lines += ["", "from strands_robots import Robot", "", "robot = Robot("]
    lines.append(f"    {robot_name!r},")
    lines.append(f"    mode={mode!r},")
    if mode == "real":
        serial = payload.get("serial_number")
        comment = f"  # USB serial {serial}" if serial and serial != port else ""
        lines.append(f"    port={str(port)!r},{comment}")
    if payload.get("robot_id"):
        lines.append(f"    id={str(payload['robot_id'])!r},  # lerobot calibration identity")
    cameras = payload.get("cameras")
    if cameras:
        # STRIPPED, not rendered verbatim: a spawn payload or remembered profile carries the
        # dashboard's own bookkeeping key (device_name), the child never sees it (the spawner strips
        # it before Popen for exactly this reason), and hardware_robot refuses an unknown camera
        # option BY NAME -- so a snippet generated for a camera-stamped arm produced a file that
        # died at connect with "Unknown option(s) for camera 'main': ['device_name']". A generated
        # file that starts the wrong rig is worse than no file, and this one started no rig at all.
        # The name itself is not lost: it is in the docstring above, which is where it helps.
        lines.append(f"    cameras={_fmt(camera_liveness.without_annotations(cameras), 4)},")
    lines.append("    mesh=True,")
    lines.append(f"    peer_id={peer_id!r},")
    lines.append(")")
    if mode == "real":
        lines += [
            "",
            "# Connect eagerly so joints + camera frames publish immediately (a camera",
            "# this machine will not open costs the camera, not the whole arm).",
            "ok, degraded, err = robot.connect_eagerly()",
            "for cam, reason in (degraded or {}).items():",
            '    print(f"camera {cam!r} unavailable, dropped: {reason}")',
            "if not ok:",
            '    print(f"eager connect failed (will retry on first task): {err}")',
        ]
    lines += [
        "",
        f'print("{peer_id} online - visible on the dashboard fleet")',
        "while True:",
        "    time.sleep(1)",
        "",
    ]
    return {
        "snippet": "\n".join(lines),
        "filename": snippet_filename(peer_id),
        "peer_id": peer_id,
    }
