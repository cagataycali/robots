"""U16: the deploy snippet must recreate the EXACT rig, or refuse.

The generated file is the contract that the dashboard's spawn form and a
plain `python robot.py` on an edge device describe the same object. The
strongest test therefore EXECUTES the generated source (with strands_robots
stubbed) and asserts the factory receives exactly the profile's arguments -
string comparison on generated code rots; a call-capture does not.

Run with --no-cov.
"""

from __future__ import annotations

import sys
import types
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from strands_robots.dashboard.deploy import render_snippet, snippet_filename

ARM_1 = {
    "cameras": {
        "top": {"fps": 30, "height": 1080, "index_or_path": 2, "type": "opencv", "width": 1920},
        "wrist": {"fps": 30, "height": 1080, "index_or_path": 1, "type": "opencv", "width": 1920},
    },
    "mode": "real",
    "name": "so101-arm-1",
    "peer_id": "so101-arm-1",
    "port": "/dev/cu.usbmodem5AB01818061",
    "robot_id": "follower_arm",
    "robot_name": "so101",
    "serial_number": "5AB0181806",
}


def _run_generated(source: str) -> tuple[tuple, dict, list[str]]:
    """Execute a generated snippet with Robot + time.sleep stubbed.

    Returns (args, kwargs, printed). The snippet's `while True: sleep(1)`
    keep-alive is broken by making sleep raise.
    """
    calls: list[tuple[tuple, dict]] = []
    printed: list[str] = []

    class _Robot:
        def __init__(self, *a, **kw):
            calls.append((a, kw))

        def connect_eagerly(self):
            return True, {}, None

    fake = types.ModuleType("strands_robots")
    fake.Robot = _Robot

    class _Stop(Exception):
        pass

    def _sleep(_s):
        raise _Stop

    import time as _time

    with mock.patch.dict(sys.modules, {"strands_robots": fake}), \
         mock.patch.object(_time, "sleep", _sleep), \
         mock.patch("builtins.print", lambda *a, **k: printed.append(" ".join(map(str, a)))):
        try:
            exec(compile(source, "generated.py", "exec"), {"__name__": "__main__"})
        except _Stop:
            pass
    assert calls, "generated snippet never constructed a Robot"
    return calls[0][0], calls[0][1], printed


def test_real_arm_snippet_reconstructs_the_exact_profile():
    result = render_snippet(ARM_1, hub_host="192.168.1.42", now=0)
    args, kwargs, printed = _run_generated(result["snippet"])
    assert args == ("so101",)
    assert kwargs == {
        "mode": "real",
        "port": "/dev/cu.usbmodem5AB01818061",
        "id": "follower_arm",
        "cameras": ARM_1["cameras"],
        "mesh": True,
        "peer_id": "so101-arm-1",
    }
    assert any("so101-arm-1 online" in line for line in printed)


def test_hub_host_lands_in_zenoh_connect_and_env_is_setdefault():
    import os

    src = render_snippet(ARM_1, hub_host="192.168.1.42")["snippet"]
    env_before = dict(os.environ)
    try:
        os.environ.pop("ZENOH_CONNECT", None)
        os.environ["STRANDS_MESH_CAMERA_HZ"] = "2"  # edge box's own choice
        _run_generated(src)
        assert os.environ["ZENOH_CONNECT"] == "tcp/192.168.1.42:7447"
        assert os.environ["STRANDS_MESH_CAMERA_HZ"] == "2", "setdefault must not clobber"
    finally:
        os.environ.clear()
        os.environ.update(env_before)


def test_no_hub_host_leaves_zenoh_connect_commented():
    src = render_snippet(ARM_1)["snippet"]
    assert '# os.environ.setdefault("ZENOH_CONNECT"' in src
    live = [line for line in src.splitlines() if line.startswith('os.environ.setdefault("ZENOH_CONNECT"')]
    assert not live, "same-machine deploy must not point the peer at itself"


def test_measured_role_and_camera_warning_reach_the_header():
    profile = {**ARM_1, "role": "follower", "role_source": "measured", "role_volts": 12.6}
    src = render_snippet(profile)["snippet"]
    assert "FOLLOWER at 12.6V" in src
    assert "Camera indices are PER-MACHINE" in src
    # a NAMED role (no measurement) must not be presented as measured
    named = {**ARM_1, "role": "leader", "role_source": "name"}
    assert "LEADER" not in render_snippet(named)["snippet"]


def test_sim_payload_renders_without_port_or_connect_block():
    result = render_snippet({"robot_name": "so101", "mode": "sim", "peer_id": "sim-1"})
    args, kwargs, _ = _run_generated(result["snippet"])
    assert kwargs == {"mode": "sim", "mesh": True, "peer_id": "sim-1"}
    assert "connect_eagerly" not in result["snippet"]


@pytest.mark.parametrize(
    "payload, missing",
    [
        ({"mode": "real", "port": "/dev/x", "peer_id": "p"}, "robot_name"),
        ({"robot_name": "so101", "mode": "real", "peer_id": "p"}, "port"),
        ({"robot_name": "so101", "mode": "hover", "peer_id": "p"}, "mode"),
        ({"robot_name": "so101", "mode": "sim"}, "peer_id"),
    ],
)
def test_unrunnable_payload_is_refused_not_guessed(payload, missing):
    result = render_snippet(payload)
    assert missing in result["error"]


def test_filename_is_shell_safe():
    assert snippet_filename("so101 arm/1!") == "so101-arm-1-.py"
    assert snippet_filename("") == "robot.py"


# ---------------------------------------------------------------- route ----


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    from strands_robots.dashboard import auth
    from strands_robots.dashboard import settings as dsettings

    monkeypatch.setenv("STRANDS_DASH_AUTH_STORE", str(tmp_path / "auth.json"))
    monkeypatch.setattr(dsettings, "SETTINGS_FILE", tmp_path / "settings.json")
    dsettings._cache = None
    auth._cache_key = None
    auth._cache = {}


def _client():
    from strands_robots.dashboard.server import create_app

    app = create_app()
    app.state.devices.profiles.get = mock.Mock(
        side_effect=lambda k: dict(ARM_1) if k == "5AB0181806" else None
    )
    return TestClient(app)


def test_route_renders_a_remembered_profile():
    body = _client().post("/api/deploy/snippet", json={"serial": "5AB0181806"}).json()
    assert body["filename"] == "so101-arm-1.py"
    assert "so101-arm-1" in body["snippet"]


def test_route_404s_unknown_serial_and_422s_no_payload():
    client = _client()
    assert client.post("/api/deploy/snippet", json={"serial": "nope"}).status_code == 404
    assert client.post("/api/deploy/snippet", json={}).status_code == 422


def test_route_withholds_loopback_hub_host():
    # TestClient reaches the app on http://testserver - a real hostname, so it
    # IS offered; loopback must not be.
    body = _client().post("/api/deploy/snippet", json={"serial": "5AB0181806"}).json()
    assert "tcp/testserver:7447" in body["snippet"]
    body2 = _client().post(
        "/api/deploy/snippet",
        json={"serial": "5AB0181806", "hub_host": "robots.example.com"},
    ).json()
    assert "tcp/robots.example.com:7447" in body2["snippet"]


def test_real_snippet_warns_the_port_is_per_machine():
    """Q47: a generated file whose port cannot exist on the target is a silent failure.

    The docstring already said camera indices are per-machine. The PORT is the same class of
    identifier and the more certain to be wrong - this dashboard runs on macOS
    (/dev/cu.usbmodem...) and edge devices are Linux (/dev/ttyACM0) - so the file must say so
    itself, where the person reading it is.
    """
    out = render_snippet(
        {
            "robot_name": "so101",
            "mode": "real",
            "port": "/dev/cu.usbmodem5AB0181806",
            "serial_number": "5AB0181806",
            "peer_id": "so101-arm-1",
        },
        now=0,
    )
    src = out["snippet"]
    assert "/dev/cu.usbmodem5AB0181806) is how THIS machine names" in src
    assert "/dev/ttyACM0" in src, "name the Linux form: that is where these files get deployed"
    assert "lerobot-find-port" in src
    # The serial is the identity that survives the move - the dashboard keys profiles by it.
    assert "USB serial 5AB0181806" in src
    assert "does not" in src and "replugged" in src
    # It stays a comment in the docstring: the code itself must still be runnable as-is.
    body = src.split('"""')[2]
    assert "ttyACM0" not in body, "the advice must not leak into executable code"


def test_sim_snippet_says_nothing_about_ports():
    """A sim rig has no port, so port advice would be noise that trains people to skim."""
    out = render_snippet(
        {"robot_name": "so101", "mode": "sim", "peer_id": "sim-a"}, now=0,
    )
    assert "ttyACM0" not in out["snippet"]
    assert "per-machine" not in out["snippet"].lower()


def test_port_advice_without_a_serial_stays_silent_about_serials():
    """Never claim a stable identity we were not given."""
    out = render_snippet(
        {"robot_name": "so101", "mode": "real", "port": "/dev/ttyUSB0", "peer_id": "p"}, now=0,
    )
    src = out["snippet"]
    assert "how THIS machine names" in src
    assert "USB serial" not in src


# --- Q53: the snippet must mirror the LIVE posture, not a frozen table -------------------

from strands_robots.dashboard import deploy  # noqa: E402

_SIM_PAYLOAD = {"robot_name": "so101", "mode": "sim", "peer_id": "sim-1"}


def test_configured_mesh_port_reaches_the_snippet():
    """The Mesh tab says "every robot on the desk must agree on it" -- and then the generated
    file said 7447 regardless. A peer pointed at the wrong port starts, logs nothing wrong and
    never appears in the fleet."""
    out = deploy.render_snippet(_SIM_PAYLOAD, hub_host="10.0.0.5", hub_port=7448)
    assert 'ZENOH_CONNECT", "tcp/10.0.0.5:7448"' in out["snippet"]
    assert "7447" not in out["snippet"]


def test_the_commented_line_carries_the_port_too():
    """Same-machine deploy renders the line commented out; it is a template the operator edits,
    so the wrong port there is just as misleading."""
    out = deploy.render_snippet(_SIM_PAYLOAD, hub_port="7500")
    assert '# os.environ.setdefault("ZENOH_CONNECT", "tcp/<dashboard-host>:7500")' in out["snippet"]


def test_a_nonsense_port_falls_back_to_the_default():
    for bad in ("", None, "abc", {}):
        out = deploy.render_snippet(_SIM_PAYLOAD, hub_host="h", hub_port=bad)
        assert "tcp/h:7447" in out["snippet"], bad


def test_camera_rate_mirrors_the_desk():
    out = deploy.render_snippet(_SIM_PAYLOAD, mesh_env={"STRANDS_MESH_CAMERA_HZ": "12"})
    assert 'setdefault("STRANDS_MESH_CAMERA_HZ", "12")' in out["snippet"]


def test_wire_security_is_never_disabled_by_a_hardcoded_default():
    """STRANDS_MESH_LOCAL_DEV=1 disables mesh wire security. It was in the frozen table, so
    every generated file turned it off on a machine the operator never chose to expose -- and a
    peer with security off cannot join a secured desk either."""
    out = deploy.render_snippet(_SIM_PAYLOAD, mesh_env={})
    assert "STRANDS_MESH_LOCAL_DEV" not in out["snippet"]


def test_a_dashboard_running_local_dev_still_says_so():
    """Mirroring the desk is the point: if wire security IS off here, a peer that leaves it on
    will not join, and silence would be its own trap."""
    out = deploy.render_snippet(_SIM_PAYLOAD, mesh_env={"STRANDS_MESH_LOCAL_DEV": "1"})
    assert 'setdefault("STRANDS_MESH_LOCAL_DEV", "1")' in out["snippet"]


def test_defaults_survive_for_the_keys_that_are_not_posture():
    out = deploy.render_snippet(_SIM_PAYLOAD, mesh_env={})
    for line in (
        'setdefault("STRANDS_MESH", "true")',
        'setdefault("STRANDS_MESH_MULTICAST", "true")',
        'setdefault("STRANDS_MESH_CAMERA_HZ", "5")',
        'setdefault("STRANDS_ROBOTS_NO_DYLD_SHIM", "1")',
    ):
        assert line in out["snippet"], line


def test_resolve_mesh_env_is_pure_and_ordered():
    rows = deploy.resolve_mesh_env({"STRANDS_MESH_CAMERA_HZ": " 9 "})
    assert dict(rows)["STRANDS_MESH_CAMERA_HZ"] == "9", "whitespace is the operator's typo, not a value"
    assert [k for k, _ in rows] == [
        k for k, _ in deploy._MESH_ENV if k != "STRANDS_MESH_LOCAL_DEV"
    ]


# ── Q122: the address in the snippet must be reachable from the machine that RUNS it ──
# Added as a section (this file already had tests that all still matter). The snippet is copied onto
# an edge device, so "the host the browser used" is an answer to a different question.
import pytest  # noqa: E402

from strands_robots.dashboard.deploy import hub_host_from_reached  # noqa: E402


@pytest.mark.parametrize(
    "reached,expected",
    [
        ("192.168.1.151", "192.168.1.151"),  # LAN literal: exactly what an edge device needs
        ("10.0.0.5", "10.0.0.5"),
        ("fe80::1", "fe80::1"),
        ("mac.local", "mac.local"),  # mDNS name, resolvable on the same network
        ("MAC.LOCAL", "mac.local"),
        ("thor", "thor"),  # a bare hostname is a LAN name by shape
        ("robots.cagatay.my", None),  # the tunnel: HTTP in, no zenoh port behind it
        ("8.8.8.8", None),  # public literal
        ("localhost", None),
        ("127.0.0.1", None),
        ("::1", None),
        ("", None),
        (None, None),
    ],
)
def test_only_an_address_another_machine_could_use_is_offered(reached, expected):
    host, _ = hub_host_from_reached(reached)
    assert host == expected


def test_every_refusal_explains_itself_and_no_acceptance_needs_to():
    """A commented-out line with no reason reads as "the dashboard forgot"."""
    for reached in ("localhost", "127.0.0.1", "robots.cagatay.my", "8.8.8.8"):
        host, note = hub_host_from_reached(reached)
        assert host is None and note, reached
    # the tunnel case must say what it actually suspects, since the operator can see the site working
    _, note = hub_host_from_reached("robots.cagatay.my")
    assert "tunnel or reverse proxy" in note and "HTTP only" in note
    # and an accepted host says nothing: a note beside a working value is noise that trains
    # operators to ignore notes
    for reached in ("192.168.1.151", "mac.local", "thor"):
        host, note = hub_host_from_reached(reached)
        assert host and note is None, reached
    # nothing at all is not a refusal to explain
    assert hub_host_from_reached("") == (None, None)


def test_the_snippet_carries_the_reason_and_never_a_rejected_address():
    from strands_robots.dashboard.deploy import render_snippet

    host, note = hub_host_from_reached("robots.cagatay.my")
    # ARM_1 is this file's own fixture - reused rather than invented, so this test breaks if the
    # payload contract moves.
    text = render_snippet(ARM_1, hub_host=host, hub_note=note, mesh_env={}, hub_port=7447)["snippet"]
    # Scoped to what "must not appear" MEANS: not as a hub address. The note names the host on
    # purpose - a warning that will not say which address it is about cannot be acted on - so the
    # first draft of this assertion (the whole file) failed on the explanation itself.
    assert "tcp/robots.cagatay.my" not in text, "a rejected address must not appear as a hub"
    for line in text.splitlines():
        if "robots.cagatay.my" in line:
            assert line.lstrip().startswith("#"), f"only a comment may name it: {line!r}"
    assert "NOTE:" in text and "tunnel or reverse proxy" in text
    # "no ZENOH_CONNECT" means no ACTIVE one: the commented <dashboard-host> example is the whole
    # point of the fallback branch, and it contains the same string.
    active = [line for line in text.splitlines() if "ZENOH_CONNECT" in line and not line.lstrip().startswith("#")]
    assert active == [], f"nothing may SET the hub when the address was rejected: {active}"
    assert any("<dashboard-host>" in line for line in text.splitlines()), "the example line still guides"

    lan = render_snippet(ARM_1, hub_host="192.168.1.151", mesh_env={}, hub_port=7447)["snippet"]
    assert 'ZENOH_CONNECT", "tcp/192.168.1.151:7447"' in lan
    assert "NOTE:" not in lan


# --- the dashboard's own bookkeeping key must not reach the generated code -------------------------
# Added 2026-08-22. camera_liveness.stamp_device_names writes `device_name` into the camera config at
# spawn, and a remembered profile / spawn payload is exactly what this renderer is handed. The code
# path that hands a config to a CHILD strips it (hardware_robot refuses an unknown camera option by
# name, which kills every camera on the arm); this renderer did not, so every snippet generated for a
# camera-stamped arm since that landed was a file that died at connect on the edge device.


def _stamped_payload():
    return {
        "robot_name": "so101", "mode": "real", "port": "/dev/cu.usbmodem5AB0181806",
        "peer_id": "so101-follower",
        "cameras": {
            "main": {"index_or_path": 0, "fps": 30, "device_name": "USB2.0_CAM1"},
            "wrist": {"index_or_path": 2, "fps": 30},
        },
    }


def test_device_name_never_reaches_the_generated_code():
    snippet = render_snippet(_stamped_payload())["snippet"]
    code = snippet.split('"""', 2)[2]  # everything after the docstring: the runnable half
    assert "device_name" not in code
    # and the options the child DOES accept are still there, unchanged
    assert "'index_or_path': 0," in code and "'fps': 30," in code
    assert "'index_or_path': 2," in code


def test_the_stamped_name_survives_where_a_human_reads_it():
    """Stripping it must not DISCARD it: 'index 0' is a position, 'USB2.0_CAM1' is a camera, and the
    file's own advice is to re-check the indices on the edge device -- which needs the name."""
    snippet = render_snippet(_stamped_payload())["snippet"]
    docstring = snippet.split('"""')[1]
    assert 'main: was "USB2.0_CAM1" on the dashboard machine' in docstring
    assert "wrist" not in docstring.split("Camera indices are PER-MACHINE")[1].split("\n\n")[0]


def test_a_payload_with_no_stamp_renders_exactly_as_before():
    """No annotation anywhere: no note line, and the cameras dict is untouched."""
    payload = _stamped_payload()
    payload["cameras"] = {"main": {"index_or_path": 0, "fps": 30}}
    snippet = render_snippet(payload)["snippet"]
    assert "on the dashboard machine" not in snippet
    assert "'index_or_path': 0," in snippet
