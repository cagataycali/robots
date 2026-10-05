"""A camera that did not open at connect is named on the robot's card, with the robot's reason.

A native driver records every configured camera it could not open under
``presence.camera_failures`` (name -> reason). The card and the detail screen
used to read only ``presence.cameras`` - the cameras that DID open - so a dropped
camera was an absence with no explanation. ``lib/cameraEvidence.cameraFailures``
turns the map into rows, ``cameraEvidence`` prefers it over every inference, and
``components/CameraFailures`` renders each row with a reconfigure button on both
surfaces. The Node cells run the TypeScript (tests/_dashboard_frontend); the
static cells keep both surfaces and the shipped bundle wired to it.
"""

from __future__ import annotations

import pathlib

from tests._dashboard_frontend import FRONTEND_SRC, requires_node, run_frontend

_REASON = "opened 0 asking for 640x480 at 5 fps, the device answers 640x480 at 30 fps and sent no frame"


@requires_node
def test_the_robots_own_reason_is_what_the_card_says() -> None:
    got = run_frontend(
        f"""
const m = await import('./cameraEvidence.ts')
const failures = {{ wrist: {_REASON!r}, top: '  ' }}
const ev = m.cameraEvidence('arm', [], [], ['wrist', 'top'], failures)
out({{
  rows: m.cameraFailures(failures, []),
  liveIsNotDropped: m.cameraFailures(failures, ['wrist']).map(f => f.name),
  none: m.cameraFailures(undefined, []),
  kind: ev.kind,
  message: ev.message,
  head: m.cameraPlaceholder(ev).head,
  oneHead: m.cameraPlaceholder(m.cameraEvidence('arm', [], [], undefined, {{ wrist: 'x' }})).head,
  withoutFailures: m.cameraEvidence('arm', [], [], ['wrist']).message,
  framesWin: m.cameraEvidence('arm', ['top'], ['top'], undefined, failures).kind,
}})
"""
    )
    assert got["rows"] == [{"name": "wrist", "reason": _REASON}, {"name": "top", "reason": "no reason given"}]
    assert got["liveIsNotDropped"] == ["top"] and got["none"] == []
    assert got["kind"] == "dropped" and _REASON in got["message"]
    assert "blocked by macOS" not in got["message"], "the robot's own reason replaces the guesses"
    assert got["head"] == "cameras dropped" and got["oneHead"] == "wrist dropped"
    assert "blocked by macOS" in got["withoutFailures"], "no reason known: the honest list of causes stays"
    assert got["framesWin"] == "ok"


def test_both_robot_surfaces_render_the_failures_with_a_reconfigure_affordance() -> None:
    components = FRONTEND_SRC / "components"
    rows = (components / "CameraFailures.tsx").read_text(encoding="utf-8")
    assert "cameraFailures(failures, arrived)" in rows and "onReconfigure(f.name)" in rows
    for surface, opener in (("RobotCard.tsx", "setCamSheet"), ("RobotDetail.tsx", "setCamConfig")):
        src = (components / surface).read_text(encoding="utf-8")
        assert "<CameraFailures failures={p?.camera_failures} arrived={cams}" in src, surface
        assert f"c => {opener}({{ cam: c, add: false }})" in src, surface
    bundle = pathlib.Path(FRONTEND_SRC).parent.parent / "static" / "app.js"
    assert "camera_failures" in bundle.read_text(encoding="utf-8"), "static/app.js is rebuilt from this source"
