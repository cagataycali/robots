"""The robot viewer's arrival is a preference, and its two halves agree on the wire.

``docs/assets/viewer/robot-viewer.js`` (the page) and ``mujoco-worker.js`` (the
engine) give a robot an arrival instead of a pop: the thumbnail is revealed
bottom-up while the meshes stream, the camera eases in from a fifth farther out
on the first load, the joints wake from the default pose into the rest pose one
after another, the environment sheen fades up, the stage orbits until the first
touch, the code panel flashes the line a slider just changed, and a clock reads
the simulated time while physics runs. Nothing in MkDocs or the browser checks
three things about that until a reader hits them:

* the two files are one protocol: the page posts ``setQposAll`` and reads
  ``nkey`` / ``qpos0`` / ``key_qpos`` / ``time`` that only the worker produces,
  and a rename on either side leaves the robot asleep with no error;
* the page's ``prefers-reduced-motion`` override stops at the shadow boundary,
  so the stage needs its own, and every motion the script drives has to read
  the same media query before it starts;
* the streaming reveal is a join between a JavaScript-set ``--p`` and a
  ``clip-path`` that reads it.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VIEWER_DIR = REPO_ROOT / "docs" / "assets" / "viewer"
VIEWER = VIEWER_DIR / "robot-viewer.js"
WORKER = VIEWER_DIR / "mujoco-worker.js"

_SHADOW_STYLE = re.compile(r"<style>(.*?)</style>", re.S)
_REDUCED_MOTION_BLOCK = re.compile(r"@media \(prefers-reduced-motion: reduce\)\s*\{(.*?)\}\s*\}?", re.S)


def _viewer() -> str:
    return VIEWER.read_text(encoding="utf-8")


def _worker() -> str:
    return WORKER.read_text(encoding="utf-8")


def _method(source: str, name: str) -> str:
    """The body of one two-space-indented class method, up to its closing brace."""
    match = re.search(rf"\n  {re.escape(name)}\([^)]*\) \{{\n(.*?)\n  \}}", source, re.S)
    assert match, f"robot-viewer.js has no method {name}()"
    return match.group(1)


def test_page_and_worker_speak_one_protocol() -> None:
    """Every message and field the page relies on is produced by the worker, and vice versa."""
    viewer, worker = _viewer(), _worker()
    assert 'type: "setQposAll"' in viewer, (
        "the page never posts setQposAll; the wake-up tween has no way to move the joints"
    )
    assert 'case "setQposAll"' in worker, "the worker does not handle setQposAll; the page's tween would be ignored"
    for field in ("nkey", "qpos0", "key_qpos"):
        assert re.search(rf"\b{field}:", worker), f"the worker's snapshot does not carry {field}"
    assert "m.nkey" in viewer and "m.qpos0" in viewer, "the page does not read nkey / qpos0 from the snapshot"
    assert "time: data.time" in worker, "the worker's pose message carries no simulated time"
    assert "pose.time" in viewer, "the page never reads the pose's time; the clock would stay at zero"
    for name in ("HINGE", "SLIDE", "BALL", "FREE"):
        assert f"{name}:" in worker, f"the worker's joint enums lack {name}; the tween cannot tell quaternions apart"


def test_worker_rests_on_the_first_keyframe() -> None:
    """Compile and reset land on keyframe 0 when the model declares one, so a humanoid stands."""
    worker = _worker()
    assert "function rest()" in worker, "the worker has no rest() helper"
    rest = re.search(r"function rest\(\) \{(.*?)\n\}", worker, re.S)
    assert rest and "key_qpos" in rest.group(1) and "nkey" in rest.group(1), "rest() ignores the model's keyframes"
    assert worker.count("rest();") >= 2, "compile and reset must both call rest(), else Reset lands on a different pose"


def test_stage_carries_its_own_reduced_motion_override() -> None:
    """The page's universal override cannot cross the shadow boundary; the stage repeats it."""
    style = _SHADOW_STYLE.search(_viewer())
    assert style, "robot-viewer.js has no <style> in its template"
    blocks = _REDUCED_MOTION_BLOCK.findall(style.group(1))
    universal = [b for b in blocks if re.search(r"\*,\s*\*::before,\s*\*::after\s*\{", b)]
    assert universal, "the viewer's shadow stylesheet has no universal `*, *::before, *::after` reduced-motion override"
    for prop in ("animation-duration", "transition-duration"):
        assert f"{prop}: 0.01ms !important" in universal[0], f"the stage's reduced-motion override does not pin {prop}"


def test_every_viewer_motion_reads_the_preference() -> None:
    """Wake-up, idle orbit, the camera flight and the exposure ramp all ask prefers-reduced-motion."""
    viewer = _viewer()
    assert 'matchMedia("(prefers-reduced-motion: reduce)")' in viewer
    for method in ("_wake", "_startOrbit"):
        assert "_reducedMotion()" in _method(viewer, method), (
            f"{method}() animates without reading the motion preference"
        )
    build_scene = _method(viewer, "async _buildScene")
    flight = re.search(r"if \((.*?)\) \{[^{}]*_flight = \{", build_scene, re.S)
    assert flight and "_reducedMotion()" in flight.group(1), "the camera flight is not guarded by the motion preference"
    loop = _method(viewer, "_loop")
    ramp = re.search(r"if \((.*?)\) \{[^{}]*_exposureRamp = \{", loop, re.S)
    assert ramp and "_reducedMotion()" in ramp.group(1), "the exposure ramp is not guarded by the motion preference"


def test_the_reveal_joins_script_and_stylesheet() -> None:
    """The script writes --p as the download fraction; the stylesheet clips the thumbnail by it."""
    viewer = _viewer()
    assert 'setProperty("--p"' in viewer, "the progress reveal never sets --p"
    style = _SHADOW_STYLE.search(viewer)
    assert style and re.search(r"clip-path:\s*inset\(calc\(100% - var\(--p", style.group(1)), (
        "the shadow stylesheet does not clip the reveal by var(--p); the thumbnail would never be revealed"
    )
    assert "_stopOrbit" in viewer and '"pointerdown"' in viewer, "the idle orbit has no first-touch stop"
