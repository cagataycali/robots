"""Camera surface of the operator dashboard (consolidated).

Consolidated verbatim from: test_dashboard_camera_annotation_boundaries.py, test_dashboard_camera_index_beyond_roster.py, test_dashboard_camera_liveness.py, test_dashboard_camera_modes.py, test_dashboard_camera_option_values.py, test_dashboard_camera_probe_serialisation.py, test_dashboard_camera_reconfigure.py, test_dashboard_camera_visibility.py.
Each section keeps its original tests unchanged.
"""

from __future__ import annotations

import re
import threading
import time
from pathlib import Path

import pytest

from strands_robots.dashboard import camera_liveness
from strands_robots.dashboard import cameras as cam
from strands_robots.dashboard import cameras as camera_facts
from strands_robots.dashboard.device_manager import (
    _CAMERA_ENUM_VALUES,
    CAMERA_FPS_CANDIDATES,
    CAMERA_MODE_CANDIDATES,
    DeviceManager,
    _camera_option_values,
    camera_option_value_problem,
    indices_beyond_roster,
    modes_from_readbacks,
    validate_cameras,
)

# ============================================================================
# from tests/test_dashboard_camera_annotation_boundaries.py
# No dashboard bookkeeping key may cross into a child's argv or a generated file.
# ============================================================================

DASHBOARD = Path(__file__).resolve().parents[1] / "strands_robots" / "dashboard"

#: A line that hands `cameras` to a child process or writes it into generated code.
_SINK_PATTERNS = (
    re.compile(r"dumps\(.*[\"']cameras[\"']"),  # subprocess payload
    re.compile(r"cameras=\{?_fmt\("),  # generated python
)


def _sink_lines() -> list[tuple[str, int, str]]:
    found: list[tuple[str, int, str]] = []
    for path in sorted(DASHBOARD.glob("*.py")):
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if any(p.search(line) for p in _SINK_PATTERNS):
                found.append((path.name, n, line.strip()))
    return found


def test_every_outward_camera_sink_strips_the_annotation():
    sinks = _sink_lines()
    # If this is zero the census has stopped measuring anything (a rename, a refactor) and a green
    # here would be a false one - the same "narrowed to nothing" failure the audit runner reports.
    assert sinks, "no camera sink found at all: this guard has stopped measuring, fix the patterns"
    leaking = [(f, n, t) for f, n, t in sinks if "without_annotations" not in t]
    assert not leaking, (
        f"{len(leaking)} of {len(sinks)} camera sink(s) hand a config outward WITHOUT stripping the "
        f"dashboard's own device_name; hardware_robot refuses it and every camera on the arm dies: "
        + "; ".join(f"{f}:{n} {t}" for f, n, t in leaking)
    )


def test_the_stripper_still_removes_what_the_child_refuses():
    """The census above is worthless if the function it insists on calling stops working."""
    from strands_robots.dashboard.camera_liveness import ANNOTATION_KEYS, without_annotations

    assert ANNOTATION_KEYS, "an empty annotation set would make every strip a no-op"
    stamped = {"main": {"index_or_path": 0, "fps": 30, **{k: "x" for k in ANNOTATION_KEYS}}}
    out = without_annotations(stamped)
    assert out is not None
    for key in ANNOTATION_KEYS:
        assert key not in out["main"]
    assert out["main"] == {"index_or_path": 0, "fps": 30}
    # An unstamped mapping is returned UNCHANGED (identity), so the common path allocates nothing.
    plain = {"main": {"index_or_path": 0}}
    assert without_annotations(plain) is plain


# ============================================================================
# from tests/test_dashboard_camera_index_beyond_roster.py
# A camera index this machine cannot have is refused BEFORE the arm is despawned.
# ============================================================================


def test_an_index_past_the_roster_count_is_named_with_its_camera() -> None:
    assert indices_beyond_roster({"wrist": {"index_or_path": 7}}, 3) == {"wrist": 7}


def test_indices_inside_the_count_are_allowed_even_though_the_order_is_unknown() -> None:
    # The roster's ORDER is not evidence about which camera an index is (Continuity cameras renumber),
    # so an in-range index must never be refused here on the strength of a name.
    assert indices_beyond_roster({"top": {"index_or_path": 0}, "wrist": {"index_or_path": 2}}, 3) == {}


def test_a_negative_index_is_impossible_too() -> None:
    assert indices_beyond_roster({"top": {"index_or_path": -1}}, 3) == {"top": -1}


def test_an_empty_roster_refuses_nothing() -> None:
    # Enumeration failing (no ffmpeg, unsupported platform) is not evidence that a camera is absent.
    assert indices_beyond_roster({"wrist": {"index_or_path": 9}}, 0) == {}
    assert indices_beyond_roster({"wrist": {"index_or_path": 9}}, -1) == {}


def test_paths_and_bare_ints_are_handled_the_way_the_config_allows_them() -> None:
    # lerobot's shape allows a bare value as well as a mapping; a PATH cannot be compared to a count.
    assert indices_beyond_roster({"a": 5, "b": "/dev/video0", "c": True}, 3) == {"a": 5}


def test_reconfigure_refuses_an_impossible_index_without_touching_the_running_peer(monkeypatch) -> None:
    """The whole point: the arm keeps running, and no despawn is attempted."""
    dm = DeviceManager.__new__(DeviceManager)
    monkeypatch.setattr(DeviceManager, "_camera_names", lambda self, refresh=False: [{"index": 0}, {"index": 1}])

    def _never(*_a, **_k):  # pragma: no cover - the assertion is that this is unreachable
        raise AssertionError("despawn must not happen when the config cannot work")

    monkeypatch.setattr(DeviceManager, "despawn", _never)
    monkeypatch.setattr(DeviceManager, "spawn", _never)

    result = dm.reconfigure_cameras("so101-arm-1", {"wrist": {"index_or_path": 5}})

    assert "error" in result and not result.get("reconfigured")
    assert "index 5" in result["error"]
    assert "2 capture device" in result["error"]  # says what it counted
    assert "left running and untouched" in result["error"]  # says what it did NOT do


# ============================================================================
# from tests/test_dashboard_camera_liveness.py
# The second liveness rail: an index the machine does not list at all.
# ============================================================================

_CFG = {"top": {"index_or_path": 0}, "wrist": {"index_or_path": 1}}


def test_an_index_the_machine_does_not_list_is_positive_evidence() -> None:
    """dead_cameras cannot catch this case, by design, and it is the worst one.

    A camera unplugged BEFORE the arm subscribed has no frame history, so it has no age and cannot
    be judged stale - the session would start with an image stream that never existed. The OS
    enumeration is independent evidence: if the index is not listed, nothing is there to open.
    """
    missing = camera_liveness.missing_cameras(_CFG, [0, 2])
    assert missing == [{"camera": "wrist", "index": 1}]


def test_an_empty_or_absent_scan_is_not_evidence_of_an_empty_machine() -> None:
    """A scan that found nothing is far more often a failed scan than a camera-less Mac.

    Refusing a session on that would make this gate the thing that blocks work, which is how a
    safety check gets switched off permanently.
    """
    assert camera_liveness.missing_cameras(_CFG, []) == []
    assert camera_liveness.missing_cameras(_CFG, None) == []


def test_a_camera_configured_by_path_is_not_judged_by_an_index_list() -> None:
    """Absence from an index list says nothing about /dev/video0 or a named device."""
    cfg = {"top": {"index_or_path": "/dev/video9"}, "side": {"index_or_path": True}}
    assert camera_liveness.missing_cameras(cfg, [0, 1]) == []


def test_the_refusal_names_the_renumbering_trap_not_just_the_gap() -> None:
    """ "Put it back and press record" is the wrong fix and the tempting one.

    Removing a camera renumbers the rest, so after a replug the same index may be a different
    camera - recording then captures the wrong view while every surface looks healthy. The refusal
    has to say RESCAN, and it has to stay continuable like every other gate here.
    """
    text = camera_liveness.missing_refusal([{"camera": "wrist", "index": 1}], peer_id="so101-arm-1")
    assert "not listed by this machine at all" in text
    assert "wrist (index 1)" in text
    assert "RESCAN before recording" in text
    assert "renumbers" in text and "wrong view" in text
    assert "ignore_missing_cameras" in text, "a gate with no way past it gets disabled wholesale"


def test_both_rails_stay_independent() -> None:
    """A camera can be listed and dead, or unlisted and never-seen - one must not mask the other."""
    # t must be a real stamp: camera_age reads t<=0 as UNKNOWN, not as ancient - a zero is a missing
    # clock, and treating it as a two-day-old frame would refuse a session on a formatting quirk.
    listed_but_dead = camera_liveness.dead_cameras(_CFG, {"top": {"t": 1_000.0}}, now=100_000.0)
    assert [d["camera"] for d in listed_but_dead] == ["top"]
    assert camera_liveness.missing_cameras(_CFG, [0, 1]) == [], "top is listed: not this rail's fault"


# ---------------------------------------------------------------------------
# The third rail: the index is listed, opens, streams - and is a different camera
# ---------------------------------------------------------------------------


_ROSTER = [
    {"listing_index": 0, "name": "Logi 4K Pro"},
    {"listing_index": 1, "name": "neon_net Camera"},
    {"listing_index": 2, "name": "USB2.0_CAM1"},
]


def test_a_renumbered_index_is_reported_with_where_the_camera_went() -> None:
    """The failure the other two rails cannot see, and the only one that produces a BAD dataset.

    Pull the camera at index 1 and the list closes up: index 1 now opens, streams and looks perfect
    while showing what used to be index 2. The remembered device name is the only thing that can
    say so - and the operator's real question is "where is my camera now", so the answer carries it.
    """
    drift = camera_liveness.identity_drift({"wrist": {"index_or_path": 1, "device_name": "USB2.0_CAM1"}}, _ROSTER)
    assert drift == [
        {"camera": "wrist", "index": 1, "remembered": "USB2.0_CAM1", "now": "neon_net Camera", "moved_to": 2}
    ]


def test_two_cameras_with_ONE_name_is_admitted_as_a_guess_not_offered_as_a_fix() -> None:
    """This machine really has two devices both called USB2.0_CAM1.

    Names cannot tell them apart, so proposing an index would be a coin flip dressed as advice.
    """
    roster = [*_ROSTER, {"listing_index": 3, "name": "USB2.0_CAM1"}]
    drift = camera_liveness.identity_drift({"wrist": {"index_or_path": 1, "device_name": "USB2.0_CAM1"}}, roster)
    assert drift[0]["ambiguous"] is True
    assert "moved_to" not in drift[0], "no guessed index"
    text = camera_liveness.drift_refusal(drift, peer_id="arm-1")
    assert "more than one camera is called USB2.0_CAM1" in text


def test_the_same_device_at_the_same_index_is_not_drift() -> None:
    cfg = {"top": {"index_or_path": 0, "device_name": "Logi 4K Pro"}}
    assert camera_liveness.identity_drift(cfg, _ROSTER) == []
    # Whitespace and case are formatting, not a different camera.
    cfg2 = {"top": {"index_or_path": 0, "device_name": "  logi  4K   pro "}}
    assert camera_liveness.identity_drift(cfg2, _ROSTER) == []


def test_no_remembered_name_means_nothing_to_compare() -> None:
    """Most profiles predate the field: a missing memory is not a change."""
    assert camera_liveness.identity_drift({"top": {"index_or_path": 0}}, _ROSTER) == []
    assert camera_liveness.identity_drift({"top": {"index_or_path": 0, "device_name": " "}}, _ROSTER) == []


def test_an_index_the_roster_does_not_list_belongs_to_the_OTHER_rail() -> None:
    """Absence is missing_cameras' verdict. Two rails must not both shout about one fact."""
    cfg = {"wrist": {"index_or_path": 9, "device_name": "USB2.0_CAM1"}}
    assert camera_liveness.identity_drift(cfg, _ROSTER) == []
    assert camera_liveness.missing_cameras(cfg, [0, 1, 2]) == [{"camera": "wrist", "index": 9}]


def test_an_empty_roster_or_nameless_entries_are_not_evidence() -> None:
    cfg = {"wrist": {"index_or_path": 1, "device_name": "USB2.0_CAM1"}}
    assert camera_liveness.identity_drift(cfg, []) == []
    assert camera_liveness.identity_drift(cfg, None) == []
    assert camera_liveness.identity_drift(cfg, [{"listing_index": 1, "name": "  "}]) == []


def test_the_drift_refusal_explains_why_a_working_camera_is_being_refused() -> None:
    """Refusing something that visibly works needs the reason stated, or it reads as a bug."""
    drift = camera_liveness.identity_drift({"wrist": {"index_or_path": 1, "device_name": "USB2.0_CAM1"}}, _ROSTER)
    text = camera_liveness.drift_refusal(drift, peer_id="so101-arm-1")
    assert "changed hands" in text
    assert "still opens and still streams" in text and "wrong view" in text
    assert "look perfectly healthy and be unusable" in text
    assert "index 2 now" in text, "say where the camera went"
    assert "ignore_camera_identity" in text


# ============================================================================
# from tests/test_dashboard_camera_modes.py
# U19: the reconfigure sheet's fps/resolution selects offer REAL modes only.
# ============================================================================


def _rb(req_w, req_h, req_fps, got_w, got_h, got_fps):
    return {
        "requested": {"width": req_w, "height": req_h, "fps": req_fps},
        "got": {"width": got_w, "height": got_h, "fps": got_fps},
    }


NATIVE = {"width": 1280, "height": 720, "fps": 30.0}


class TestModesFromReadbacks:
    def test_agreed_mode_is_kept(self) -> None:
        modes = modes_from_readbacks(NATIVE, [_rb(640, 480, 30, 640, 480, 30.0)])
        assert {"width": 640, "height": 480, "fps": 30} in modes

    def test_ignored_set_contributes_nothing(self) -> None:
        # Driver answers its native mode for every request: only native survives.
        rbs = [
            _rb(w, h, fps, 1280, 720, 30.0)
            for w, h in CAMERA_MODE_CANDIDATES
            for fps in CAMERA_FPS_CANDIDATES
            if (w, h) != (1280, 720)
        ]
        modes = modes_from_readbacks(NATIVE, rbs)
        assert modes == [{"width": 1280, "height": 720, "fps": 30}]

    def test_native_mode_always_included_even_with_zero_readbacks(self) -> None:
        assert modes_from_readbacks(NATIVE, []) == [{"width": 1280, "height": 720, "fps": 30}]

    def test_fps_within_one_counts_as_agreement(self) -> None:
        # NTSC drivers report 29.97 for 30 - that IS the 30fps mode.
        modes = modes_from_readbacks(NATIVE, [_rb(640, 480, 30, 640, 480, 29.97)])
        assert {"width": 640, "height": 480, "fps": 30} in modes

    def test_fps_off_by_more_than_one_is_refused(self) -> None:
        modes = modes_from_readbacks(NATIVE, [_rb(640, 480, 60, 640, 480, 30.0)])
        assert {"width": 640, "height": 480, "fps": 60} not in modes

    def test_deduped_and_sorted_by_area_then_fps(self) -> None:
        modes = modes_from_readbacks(
            NATIVE,
            [
                _rb(1920, 1080, 30, 1920, 1080, 30),
                _rb(640, 480, 60, 640, 480, 60),
                _rb(640, 480, 15, 640, 480, 15),
                _rb(640, 480, 15, 640, 480, 15),  # duplicate probe
            ],
        )
        keys = [(m["width"], m["height"], m["fps"]) for m in modes]
        assert keys == sorted(set(keys), key=lambda k: (k[0] * k[1], k[2]))
        assert len(keys) == len(set(keys))

    def test_garbage_native_yields_no_phantom_mode(self) -> None:
        # A broken driver reporting 0x0@0 must not put an unusable row in the select.
        modes = modes_from_readbacks({"width": 0, "height": 0, "fps": 0.0}, [_rb(640, 480, 30, 640, 480, 30)])
        assert modes == [{"width": 640, "height": 480, "fps": 30}]

    def test_non_numeric_readback_is_skipped_not_fatal(self) -> None:
        modes = modes_from_readbacks(
            NATIVE,
            [
                {"requested": {"width": "x"}, "got": None},
                _rb(640, 480, 30, 640, 480, 30),
            ],
        )
        assert {"width": 640, "height": 480, "fps": 30} in modes


class TestProbeModesGuard:
    def test_streaming_index_is_refused_before_any_open(self, tmp_path) -> None:
        dm = DeviceManager(profiles_path=str(tmp_path / "profiles.json"))
        dm._claimed_camera_indices = lambda: {0: "so101-arm-1"}  # type: ignore[method-assign]
        dm._streaming_indices = lambda live: {0}  # type: ignore[method-assign, assignment]
        with pytest.raises(PermissionError, match="so101-arm-1"):
            dm.probe_modes(0, {"so101-arm-1": ["top"]})


# ============================================================================
# from tests/test_dashboard_camera_option_values.py
# An enumerated camera option's VALUE is refused before the arm is despawned.
# ============================================================================


def test_the_admitted_spellings_are_read_from_lerobots_own_enums_not_copied():
    """Drift, not assumption: wherever lerobot is importable its enums ARE the domain."""
    try:
        from lerobot.cameras.configs import ColorMode, Cv2Rotation
    except Exception:  # pragma: no cover - exercised on machines with no robot stack
        assert _camera_option_values() == dict(_CAMERA_ENUM_VALUES)
        return
    table = _camera_option_values()
    assert set(table["color_mode"]) == {str(m.value) for m in ColorMode}
    assert set(table["rotation"]) == {str(r.value) for r in Cv2Rotation}
    # And the frozen fallback must not have drifted from them either, or a machine without lerobot
    # would refuse a spelling the child accepts.
    assert set(_CAMERA_ENUM_VALUES["color_mode"]) == set(table["color_mode"])
    assert set(_CAMERA_ENUM_VALUES["rotation"]) == set(table["rotation"])


def test_rotation_admits_minus_90_and_refuses_the_obvious_270():
    """MEASURED: lerobot's Cv2Rotation is {0, 90, 180, -90}. 270 is the intuitive answer and wrong."""
    assert camera_option_value_problem("rotation", -90) is None
    assert camera_option_value_problem("rotation", "-90") is None
    problem = camera_option_value_problem("rotation", 270)
    assert problem and "270" in problem and "-90" in problem


def test_an_int_and_its_string_spelling_are_both_admitted():
    """The form sends strings and a remembered profile round-trips through JSON."""
    assert camera_option_value_problem("rotation", 90) is None
    assert camera_option_value_problem("rotation", "90") is None
    assert camera_option_value_problem("color_mode", "rgb") is None


def test_the_uppercase_spelling_is_refused_with_the_lowercase_hint():
    """ColorMode('RGB') raises in lerobot, so admitting it here would only move the death later."""
    problem = camera_option_value_problem("color_mode", "RGB")
    assert problem and "rgb" in problem
    assert "Did you mean 'rgb'?" in problem


def test_an_option_with_no_published_enumeration_is_not_this_functions_business():
    """fps/width/backend are graded by the numeric ranges beside it; silence here is deliberate."""
    for option, value in (("fps", 30), ("width", 640), ("backend", "avfoundation"), ("nonsense", "x")):
        assert camera_option_value_problem(option, value) is None


def test_a_structural_value_says_the_option_is_enumerated():
    problem = camera_option_value_problem("color_mode", {"rgb": True})
    assert problem and "enumerated" in problem
    assert camera_option_value_problem("rotation", True) is not None  # bool is not a rotation


def test_the_refusal_reaches_the_config_validator_before_any_despawn():
    """The wiring, not just the helper: validate_cameras must refuse the whole spawn."""
    from strands_robots.dashboard.device_manager import validate_cameras

    ok = validate_cameras({"main": {"index_or_path": 0, "color_mode": "rgb", "rotation": 90}})
    assert ok is None
    bad = validate_cameras({"main": {"index_or_path": 0, "color_mode": "RGB"}})
    assert bad and "main" in bad["error"] and "rgb" in bad["error"]


# ============================================================================
# from tests/test_dashboard_camera_probe_serialisation.py
# Two /api/devices requests may not probe the cameras at the same time.
# ============================================================================


def test_probe_needed_refresh_probes_when_cache_predates_the_request() -> None:
    # The operator pressed rescan to learn about a cable they just plugged in;
    # an answer measured BEFORE they asked cannot contain it.
    assert camera_facts.probe_needed(refresh=True, requested_at=100.0, cache_t=99.0, ttl_s=30.0, now=100.0)


def test_probe_needed_answer_that_landed_after_the_request_is_enough() -> None:
    # A probe finished while we waited for the lock: its result is at least as new
    # as the question, so re-probing would only fight it for the devices.
    assert not camera_facts.probe_needed(refresh=True, requested_at=100.0, cache_t=100.5, ttl_s=30.0, now=101.0)
    assert not camera_facts.probe_needed(refresh=False, requested_at=100.0, cache_t=100.0, ttl_s=30.0, now=101.0)


def test_probe_needed_plain_poll_still_honours_the_ttl() -> None:
    assert not camera_facts.probe_needed(refresh=False, requested_at=100.0, cache_t=80.0, ttl_s=30.0, now=100.0)
    assert camera_facts.probe_needed(refresh=False, requested_at=100.0, cache_t=60.0, ttl_s=30.0, now=100.0)


# ------------------------------------------------------------- the real manager
def test_concurrent_refresh_requests_probe_the_hardware_once(monkeypatch) -> None:
    mgr = DeviceManager()
    overlap = []
    running = threading.Event()
    calls = []

    def slow_scan(skip=None):
        calls.append(time.time())
        # If another thread is already inside the probe, record it: this is the
        # double-open that makes a healthy camera report "unavailable".
        if running.is_set():
            overlap.append(True)
        running.set()
        time.sleep(0.25)
        running.clear()
        return ([{"index": 0, "width": 640, "height": 480}], {})

    monkeypatch.setattr("strands_robots.dashboard.device_manager.scan_cameras_with_failures", slow_scan)
    monkeypatch.setattr(mgr, "_camera_names", lambda refresh=False: [])
    monkeypatch.setattr(mgr, "_claimed_camera_indices", lambda: {})
    monkeypatch.setattr(mgr, "_streaming_indices", lambda live: set())

    threads = [threading.Thread(target=lambda: mgr._cameras(refresh=True)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert overlap == [], "two probes held the cameras open at the same time"
    assert len(calls) == 1, f"the hardware was probed {len(calls)} times for 4 concurrent requests"


def test_a_later_refresh_still_gets_a_fresh_probe(monkeypatch) -> None:
    """Serialising must not turn rescan into a no-op for the NEXT press."""
    mgr = DeviceManager()
    calls = []

    def scan(skip=None):
        calls.append(1)
        return ([{"index": len(calls) - 1, "width": 640, "height": 480}], {})

    monkeypatch.setattr("strands_robots.dashboard.device_manager.scan_cameras_with_failures", scan)
    monkeypatch.setattr(mgr, "_camera_names", lambda refresh=False: [])
    monkeypatch.setattr(mgr, "_claimed_camera_indices", lambda: {})
    monkeypatch.setattr(mgr, "_streaming_indices", lambda live: set())

    mgr._cameras(refresh=True)
    time.sleep(0.01)
    mgr._cameras(refresh=True)
    assert len(calls) == 2


# ============================================================================
# from tests/test_dashboard_camera_reconfigure.py
# U19 v1: changing a robot's cameras is a respawn - one named, refuse-first operation.
# ============================================================================


class TestValidateCameras:
    def test_none_detaches_everything_and_is_legal(self) -> None:
        assert validate_cameras(None) is None

    def test_a_lerobot_shaped_config_passes(self) -> None:
        assert (
            validate_cameras(
                {
                    "top": {"index_or_path": 0, "fps": 30, "width": 1280, "height": 720},
                    "wrist": {"index_or_path": "/dev/video1"},
                }
            )
            is None
        )

    def test_the_live_crash_shape_is_refused_with_the_example(self) -> None:
        # The exact config that killed a child after 200+pid: a bare int.
        bad = validate_cameras({"main": 3})
        assert bad is not None
        assert "mapping" in bad["error"] and "index_or_path" in bad["error"]

    def test_a_non_dict_config_is_refused(self) -> None:
        assert validate_cameras([0, 1]) is not None

    def test_missing_index_or_path_is_refused_naming_the_camera(self) -> None:
        bad = validate_cameras({"top": {"fps": 30}})
        assert bad is not None and "top" in bad["error"] and "index_or_path" in bad["error"]

    @pytest.mark.parametrize("iop", [True, -1, 1.5, None])
    def test_index_or_path_must_be_an_index_or_a_path(self, iop: object) -> None:
        assert validate_cameras({"top": {"index_or_path": iop}}) is not None

    @pytest.mark.parametrize(
        ("field", "value"),
        [("fps", 0), ("fps", 241), ("fps", "30"), ("fps", True), ("width", 8), ("width", 100000), ("height", 5000)],
    )
    def test_fantasy_settings_are_refused_by_bound_not_by_driver(self, field: str, value: object) -> None:
        bad = validate_cameras({"top": {"index_or_path": 0, field: value}})
        assert bad is not None and field in bad["error"]

    def test_omitted_fields_mean_driver_defaults(self) -> None:
        assert validate_cameras({"top": {"index_or_path": 0}}) is None


class TestSpawnRefusesBadCamerasBeforePopen:
    def test_the_child_never_sees_the_bare_int_config(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        dm = DeviceManager(profiles_path=str(tmp_path / "profiles.json"))
        import strands_robots.dashboard.device_manager as mod

        monkeypatch.setattr(
            mod.subprocess,
            "Popen",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("Popen reached")),
        )
        result = dm.spawn("so101", "sim", cameras={"main": 3})
        assert "error" in result and "pid" not in result
        assert dm.robots == {}


class TestReconfigureCameras:
    def _fake_managed(self, dm: DeviceManager, peer_id: str = "so101-sim-1") -> None:
        import strands_robots.dashboard.device_manager as mod

        m = mod.ManagedRobot(peer_id=peer_id, robot_name="so101", mode="sim")
        dm.robots[peer_id] = m

    def test_an_unknown_peer_is_a_refusal_not_a_spawn(self, tmp_path) -> None:
        dm = DeviceManager(profiles_path=str(tmp_path / "profiles.json"))
        result = dm.reconfigure_cameras("ghost-1", {"top": {"index_or_path": 0}})
        assert "error" in result and "unknown managed peer" in result["error"]

    def test_an_invalid_config_never_touches_the_running_peer(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        # THE law of this feature: refusal before destruction.
        dm = DeviceManager(profiles_path=str(tmp_path / "profiles.json"))
        self._fake_managed(dm)
        monkeypatch.setattr(
            dm,
            "despawn",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("despawn reached for an invalid config")),
        )
        result = dm.reconfigure_cameras("so101-sim-1", {"main": 3})
        assert "error" in result
        assert "so101-sim-1" in dm.robots, "the running peer must survive a refused reconfigure"

    def test_a_replay_job_is_not_a_respawnable_robot(self, tmp_path) -> None:
        import strands_robots.dashboard.device_manager as mod

        dm = DeviceManager(profiles_path=str(tmp_path / "profiles.json"))
        dm.robots["replay-1"] = mod.ManagedRobot(peer_id="replay-1", robot_name="so101", mode="replay")
        result = dm.reconfigure_cameras("replay-1", None)
        assert "error" in result and "replay" in result["error"]

    def test_a_valid_reconfigure_despawns_then_respawns_under_the_same_id(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        dm = DeviceManager(profiles_path=str(tmp_path / "profiles.json"))
        self._fake_managed(dm)
        calls: list[tuple] = []

        def fake_despawn(peer_id):
            calls.append(("despawn", peer_id))
            dm.robots.pop(peer_id, None)
            return {"peer_id": peer_id, "stopped": True}

        def fake_spawn(robot_name, mode, peer_id=None, port=None, cameras=None, robot_id=None, remember=True):
            calls.append(("spawn", robot_name, mode, peer_id, cameras, remember))
            return {"peer_id": peer_id, "pid": 4242, "mode": mode}

        monkeypatch.setattr(dm, "despawn", fake_despawn)
        monkeypatch.setattr(dm, "spawn", fake_spawn)
        new_cams = {"wrist": {"index_or_path": 1, "fps": 60}}
        result = dm.reconfigure_cameras("so101-sim-1", new_cams)
        assert result.get("reconfigured") is True and result.get("pid") == 4242
        assert calls[0] == ("despawn", "so101-sim-1")
        assert calls[1] == ("spawn", "so101", "sim", "so101-sim-1", new_cams, True), (
            "the identity of the spawn must be the OLD peer's, only the cameras change, "
            "and remember=True so the profile keeps the change across replugs"
        )


class TestUnknownOptionsAreRefusedBeforeAnythingStops:
    """An unknown camera option cost the operator a WORKING arm (U19 backend verify, 2026-08-20).

    validate_cameras bounds-checked index_or_path/fps/width/height and let every other key through.
    hardware_robot._build_camera_config refuses unknown keys (deliberately -- a silently dropped option
    reports success while the camera streams at the default), but it only speaks inside the CHILD, and
    reconfigure_cameras despawns the running robot BEFORE spawning the replacement. So "framerate" instead
    of "fps" meant: arm killed, respawn dead with a ValueError in a log ring, and a 200 from the route.

    This class pins the promise validate_cameras' own docstring makes: everything the child would refuse
    is refused here, before a process exists.
    """

    def test_a_wrong_option_name_is_refused_and_named(self) -> None:
        bad = validate_cameras({"wrist": {"index_or_path": 1, "framerate": 60}})
        assert bad is not None
        assert "framerate" in bad["error"], "name the option the operator actually typed"
        assert "fps" in bad["error"], "and the accepted set, which is what tells them the right word"
        assert "despawn" in bad["error"], "say why refusing early matters: a reconfigure stops the robot first"
        # Deliberately NOT asserting a "did you mean 'fps'" here: difflib does not consider 'framerate'
        # close to 'fps' (measured), and writing the assertion first is what caught me claiming a
        # suggestion the code cannot make. The accepted list carries the answer instead.

    def test_a_near_miss_does_get_a_suggestion(self) -> None:
        """Where difflib CAN help, it should -- a one-character slip is the common case."""
        bad = validate_cameras({"wrist": {"index_or_path": 1, "widht": 640}})
        assert bad is not None and "Did you mean" in bad["error"] and "'width'" in bad["error"]

    def test_an_unknown_option_with_no_near_match_still_names_the_accepted_set(self) -> None:
        bad = validate_cameras({"top": {"index_or_path": 0, "zoom_factor": 3}})
        assert bad is not None and "zoom_factor" in bad["error"]
        for field in ("fps", "width", "height", "index_or_path"):
            assert field in bad["error"]

    def test_every_real_lerobot_option_is_accepted(self) -> None:
        """The refusal must not become a whitelist that fights the driver it wraps."""
        full = {
            "top": {
                "index_or_path": 0,
                "fps": 30,
                "width": 640,
                "height": 480,
                "color_mode": "rgb",
                "rotation": 90,
                "warmup_s": 1,
                "backend": "any",
                "type": "opencv",
            }
        }
        assert validate_cameras(full) is None

    def test_the_frozen_fallback_field_list_matches_lerobot(self) -> None:
        """The fallback exists for a machine with no robot stack; a stale list would refuse a legal option."""
        dataclasses = pytest.importorskip("dataclasses")
        cfgmod = pytest.importorskip("lerobot.cameras.opencv.configuration_opencv")
        real = tuple(sorted(f.name for f in dataclasses.fields(cfgmod.OpenCVCameraConfig)))
        from strands_robots.dashboard.device_manager import _CAMERA_OPTION_FIELDS

        assert tuple(sorted(_CAMERA_OPTION_FIELDS)) == real, (
            "lerobot's camera options changed: update _CAMERA_OPTION_FIELDS, the list used when "
            "lerobot is not importable"
        )


# ============================================================================
# from tests/test_dashboard_camera_visibility.py
# A camera is never omitted just because it could not be opened (U14).
# ============================================================================

BLOCKED_STDERR = (
    "OpenCV: not authorized to capture video (status 0), requesting...\nOpenCV: camera failed to properly initialize!\n"
)
DEAD_STDERR = "OpenCV: camera failed to properly initialize!\n"
BUSY_STDERR = "VIDEOIO ERROR: V4L: can't open camera by index 0: Device or resource busy\n"

ROSTER = [
    {"listing_index": 0, "name": "USB2.0_CAM1"},
    {"listing_index": 1, "name": "USB2.0_CAM1"},
    {"listing_index": 2, "name": "Logi 4K Pro"},
    {"listing_index": 3, "name": "neon_net Camera"},
]


# ------------------------------------------------------------- classification


def test_permission_denial_is_not_a_missing_camera():
    state, reason, remedy = cam.classify_probe_stderr(BLOCKED_STDERR)
    assert state == "blocked"
    assert "camera access" in reason
    # The remedy has to name the actual place, or it is not a remedy - and it
    # has to be true for THIS case: a daemon-parented dashboard never gets a
    # prompt at all (tccd: "Policy disallows prompt"), so advice that waits for
    # one is worse than none.
    assert "Privacy" in remedy and "Terminal" in remedy
    assert "background daemon" in remedy


def test_busy_device_is_in_use_not_absent():
    state, reason, remedy = cam.classify_probe_stderr(BUSY_STDERR)
    assert state == "in_use"
    assert "holding the camera" in reason and remedy


def test_initialise_failure_is_unreadable():
    assert cam.classify_probe_stderr(DEAD_STDERR)[0] == "unreadable"


def test_silence_means_absent():
    state, reason, remedy = cam.classify_probe_stderr("")
    assert (state, remedy) == ("absent", None)
    assert "no camera answered" in reason


def test_classification_is_case_insensitive_and_none_safe():
    assert cam.classify_probe_stderr("NOT AUTHORIZED to capture")[0] == "blocked"
    assert cam.classify_probe_stderr(None)[0] == "absent"  # type: ignore[arg-type]


# --------------------------------------------------------------------- merge


def test_a_claimed_camera_keeps_the_size_we_measured_before():
    rows = cam.merge_cameras(
        probed=[],
        claimed={1: "so101-arm-1"},
        roster=ROSTER,
        remembered={1: {"index": 1, "width": 1920, "height": 1080, "fps": 30.0}},
    )
    (row,) = [r for r in rows if r["index"] == 1]
    assert row["state"] == "in_use" and row["claimed_by"] == "so101-arm-1"
    assert (row["width"], row["height"]) == (1920, 1080)
    # Honest about provenance: it is what we last saw, not a fresh measurement.
    assert row["geometry_from"] == "remembered"
    assert "despawn so101-arm-1" in row["remedy"]


def test_every_roster_camera_gets_a_row_even_when_all_probes_fail():
    rows = cam.merge_cameras(
        probed=[],
        claimed={},
        roster=ROSTER,
        failures={i: BLOCKED_STDERR for i in range(4)},
    )
    assert [r["index"] for r in rows] == [0, 1, 2, 3]
    assert {r["state"] for r in rows} == {"blocked"}
    assert all(r["available"] is False for r in rows)


def test_the_live_shape_of_this_machine_lists_four_not_two():
    """Regression for the reported symptom, with the real numbers."""
    rows = cam.merge_cameras(
        probed=[{"index": 0, "width": 640, "height": 480, "fps": 30.0}],
        claimed={1: "so101-arm-1", 2: "so101-arm-1"},
        roster=ROSTER,
        failures={3: DEAD_STDERR},
    )
    assert [r["index"] for r in rows] == [0, 1, 2, 3]
    assert [r["state"] for r in rows] == ["ready", "in_use", "in_use", "unreadable"]
    assert rows[0]["available"] is True and rows[0]["reason"].startswith("opened")


def test_a_name_is_labelled_as_a_guess():
    rows = cam.merge_cameras(probed=[{"index": 2}], claimed={}, roster=ROSTER)
    # Every roster index gets a row - that is the fix - so pick the one probed.
    (row,) = [r for r in rows if r["index"] == 2]
    assert row["name_hint"] == "Logi 4K Pro"
    # Listing order != OpenCV order, and this machine has two identical names.
    assert row["name_is_guess"] is True


def test_an_index_nobody_probed_says_unknown_not_absent():
    (row,) = cam.merge_cameras(probed=[], claimed={}, roster=[ROSTER[3]])
    assert row["state"] == "unknown" and "not probed" in row["reason"]


def test_claimed_indices_are_never_reported_as_failures():
    rows = cam.merge_cameras(
        probed=[],
        claimed={1: "arm"},
        roster=ROSTER[:2],
        failures={1: BLOCKED_STDERR},
    )
    (row,) = [r for r in rows if r["index"] == 1]
    assert row["state"] == "in_use"  # ownership outranks a stale probe verdict


def test_rows_are_sorted_and_unique():
    rows = cam.merge_cameras(
        probed=[{"index": 2}],
        claimed={0: "a"},
        roster=ROSTER,
        failures={2: ""},
        remembered={9: {"width": 1, "height": 1}},
    )
    indices = [r["index"] for r in rows]
    assert indices == sorted(set(indices)) and 9 in indices


def test_empty_everything_is_an_empty_list_not_a_crash():
    assert cam.merge_cameras(probed=[], claimed={}, roster=[]) == []


# ------------------------------------------------------------------- verdict


def test_blocked_verdict_speaks_once_when_the_machine_is_blocked():
    rows = cam.merge_cameras(
        probed=[],
        claimed={},
        roster=ROSTER,
        failures={i: BLOCKED_STDERR for i in range(4)},
    )
    verdict = cam.blocked_verdict(rows)
    assert verdict and verdict["kind"] == "camera_permission"
    assert verdict["indices"] == [0, 1, 2, 3] and verdict["remedy"]


def test_one_working_camera_disproves_a_permission_problem():
    rows = cam.merge_cameras(
        probed=[{"index": 0}],
        claimed={},
        roster=ROSTER,
        failures={1: BLOCKED_STDERR},
    )
    assert cam.blocked_verdict(rows) is None


def test_no_verdict_without_evidence():
    rows = cam.merge_cameras(probed=[], claimed={}, roster=ROSTER, failures={0: DEAD_STDERR})
    assert cam.blocked_verdict(rows) is None
    assert cam.blocked_verdict([]) is None


# ----------------------------------------------------- configured vs streaming


def test_a_configured_camera_with_no_frames_is_assigned_not_streaming():
    """Exactly the live state on this Mac: both arm cameras were configured and
    neither opened, so the arm dropped them and published nothing."""
    rows = cam.merge_cameras(
        probed=[],
        claimed={1: "so101-arm-1", 2: "so101-arm-1"},
        roster=ROSTER,
        streaming=set(),
    )
    states = {r["index"]: r for r in rows if r["index"] in (1, 2)}
    assert [states[1]["state"], states[2]["state"]] == ["assigned", "assigned"]
    assert "no frames are arriving" in states[1]["reason"]
    # The remedy points at the evidence, not at a picture that will never load.
    assert "log" in states[1]["remedy"] and "so101-arm-1" in states[1]["remedy"]


def test_frames_arriving_still_reads_as_in_use():
    (row,) = [
        r
        for r in cam.merge_cameras(
            probed=[],
            claimed={1: "arm"},
            roster=ROSTER[:2],
            streaming={1},
        )
        if r["index"] == 1
    ]
    assert row["state"] == "in_use" and "streaming for arm" in row["reason"]


def test_no_streaming_evidence_keeps_the_kinder_reading():
    """None means nobody told us - absence of evidence is not evidence."""
    (row,) = [r for r in cam.merge_cameras(probed=[], claimed={1: "arm"}, roster=ROSTER[:2]) if r["index"] == 1]
    assert row["state"] == "in_use"


# ---------------------------------------------------------------------------
# A camera that GOES AWAY mid-session is not an index that was always empty
# ---------------------------------------------------------------------------


def test_an_index_we_measured_before_reports_vanished_not_absent() -> None:
    """ "no camera answered at this index" is true here and useless.

    An index that was always empty needs no action. An index where we measured a real camera and now
    get nothing is an EVENT - unplugged, asleep, or dropped by its hub - and it was reading as the
    harmless case while quietly carrying its remembered 1920x1080, so a gap looked like a healthy
    camera with no advice attached.
    """
    rows = cam.merge_cameras(
        probed=[],
        claimed={},
        remembered={1: {"width": 1920, "height": 1080, "fps": 30}},
    )
    row = next(r for r in rows if r["index"] == 1)
    assert row["state"] == "vanished"
    assert "answered at this index earlier" in row["reason"]
    assert "unplugged" in row["reason"] and "hub" in row["reason"]
    assert row["available"] is False, "a vanished camera is not available"
    assert row["geometry_from"] == "remembered", "keep the measurement, keep the tag that dates it"


def test_the_vanished_remedy_names_the_index_shift_danger() -> None:
    """The consequence an operator cannot deduce, and must not learn from a ruined dataset.

    macOS camera indices are positions in a list that closes up when a device is removed, so pulling
    one camera can hand its number to another. A remembered resolution is then not proof of identity -
    it is the last thing seen at that POSITION.
    """
    rows = cam.merge_cameras(probed=[], claimed={}, remembered={2: {"width": 640, "height": 480}})
    remedy = next(r for r in rows if r["index"] == 2)["remedy"]
    assert "replug" in remedy
    assert "renumbers" in remedy, "say WHY the index cannot be trusted afterwards"
    assert "preview an index before assigning" in remedy, "give the check that makes it safe"
    assert "wrong view while everything on screen looks healthy" in remedy


def test_an_index_never_measured_still_reports_absent() -> None:
    """The distinction only exists where there is evidence: no memory, no event, no escalation."""
    rows = cam.merge_cameras(probed=[], claimed={}, failures={3: "nothing there"}, max_index=3)
    row = next(r for r in rows if r["index"] == 3)
    assert row["state"] == "absent"
    assert "no camera answered" in row["reason"]
    assert "remedy" not in row, "an empty index asks nothing of the operator"


def test_a_remembered_index_that_opens_now_is_simply_ready() -> None:
    """Memory must not haunt a working camera: a frame right now beats any history."""
    rows = cam.merge_cameras(
        probed=[{"index": 1, "width": 1280, "height": 720, "fps": 30}],
        claimed={},
        remembered={1: {"width": 1920, "height": 1080}},
    )
    row = next(r for r in rows if r["index"] == 1)
    assert row["state"] == "ready" and row["available"] is True
    assert row.get("geometry_from") != "remembered"
    assert row["width"] == 1280, "the fresh measurement wins"


def test_a_remembered_index_that_is_BLOCKED_keeps_the_permission_verdict() -> None:
    """A stderr that explains itself always beats an inference from memory.

    Permission denial has a cure that has nothing to do with cables; calling it "vanished" would send
    the operator to replug a camera that is plugged in and working.
    """
    rows = cam.merge_cameras(
        probed=[],
        claimed={},
        remembered={0: {"width": 1920, "height": 1080}},
        failures={0: "OpenCV: not authorized to capture video (status 0)"},
    )
    row = next(r for r in rows if r["index"] == 0)
    assert row["state"] == "blocked"
    assert "not granted camera access" in row["reason"]
