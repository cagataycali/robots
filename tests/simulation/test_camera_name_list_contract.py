"""``cameras=`` is an ordered list of distinct camera names on every surface.

Seven public methods accept a ``cameras`` subset - MuJoCo's ``render_all``, its
two plain-MP4 recorders, and every backend's ``start_recording`` - and each one
used to read whatever it was handed. Nothing validated the shape, so one
parameter had four different failure modes and two of them escaped the
structured ``{"status": "error"}`` contract these methods document:

* A single name passed as a bare string is iterable per character, so
  ``cameras="wrist"`` was read as five cameras, one per letter, and reported as
  five unknown cameras rather than as one mis-typed parameter.
* A ``Mapping`` is iterable over its keys, so ``cameras={"wrist": ...}`` was
  accepted with its values silently discarded.
* A repeated name failed in opposite directions depending on the surface:
  ``render_all(["wrist", "wrist"])`` returned two image blocks for one camera,
  ``start_cameras_recording`` reported "2 camera(s)" and opened a second encoder
  on the one output path (two artifacts for one file, the camera rendered and
  appended twice per capture tick), while ``start_recording`` silently collapsed
  it and declared one camera column for the two requested.
* A non-string element or a non-sequence raised a bare
  ``TypeError: can only concatenate str (not "int") to str`` /
  ``TypeError: 'int' object is not iterable`` out of the rendering surfaces, and
  dead-ended in a generic "Dataset init failed" on the dataset surfaces.

The rule these need already exists as
:func:`strands_robots.utils.name_list_error`, the shared domain for a parameter
carrying an ordered list of key names; it was wired only to the policy
``image_keys`` consumers. These tests pin that every ``cameras`` surface now
resolves through it, that the surfaces agree, and that a distinct list is
untouched.

``cameras=None`` keeps its "record/render every camera" meaning and an empty
sequence keeps each surface's existing verdict: like every other consumer of the
shared domain, the check is gated on a truthy value.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
from collections.abc import Callable
from typing import Any

import pytest

from strands_robots.simulation import Simulation
from strands_robots.simulation.models import SimWorld

_ARM_XML = """<mujoco model="arm">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base">
      <geom type="box" size="0.05 0.05 0.05"/>
      <body name="link1" pos="0 0 0.1">
        <joint name="pan" type="hinge" axis="0 0 1" damping="2" range="-1.5 1.5" limited="true"/>
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02"/>
      </body>
    </body>
  </worldbody>
  <actuator><position name="pan_act" joint="pan" kp="30" ctrlrange="-1.5 1.5"/></actuator>
</mujoco>
"""


def _call(surface: Callable[..., Any], **kwargs: Any) -> dict[str, Any]:
    """Invoke ``surface`` with keywords mypy must not narrow.

    Several cases here deliberately pass a value the parameter's annotation
    forbids - that is the caller mistake under test. Splatting a
    ``dict[str, Any]`` keeps the type checker out of the way without a
    suppression at each call site.
    """
    return surface(**kwargs)


@pytest.fixture
def cam_sim(tmp_path):
    """A MuJoCo world with one robot and one named camera.

    Deliberately un-annotated: the concrete ``Simulation`` symbol is a lazy
    module attribute rather than a class, so annotating it makes every
    ``sim._world`` read a union for mypy.
    """
    arm = tmp_path / "arm.xml"
    arm.write_text(_ARM_XML)
    sim = Simulation(backend="mujoco", tool_name="cameras_contract", mesh=False)
    sim.create_world()
    sim.add_robot(name="arm", urdf_path=str(arm))
    sim.add_camera(name="wrist", position=[0.4, 0.4, 0.4], target=[0.0, 0.0, 0.1])
    try:
        yield sim
    finally:
        sim.cleanup()


# Every value class that cannot be read as a list of distinct camera names.
# Parametrized over FACTORIES so a one-shot iterator is not consumed by
# whichever surface reads it first.
_UNUSABLE: list[tuple[str, Callable[[], Any]]] = [
    ("bare_string", lambda: "wrist"),
    ("mapping", lambda: {"wrist": {"width": 64}}),
    ("duplicate_name", lambda: ["wrist", "wrist"]),
    ("non_string_element", lambda: [3]),
    ("not_a_sequence", lambda: 3),
    ("blank_name", lambda: [" "]),
    ("one_shot_iterator", lambda: iter(["wrist"])),
]

_SURFACES: list[str] = [
    "render_all",
    "start_cameras_recording",
    "start_cameras_recording_synchronous",
    "start_recording",
]


def _invoke(sim: Any, surface: str, tmp_path: pathlib.Path, cameras: Any) -> dict[str, Any]:
    """Call one ``cameras``-taking surface with the supplied value."""
    if surface == "render_all":
        return _call(sim.render_all, cameras=cameras, width=64, height=48)
    if surface == "start_cameras_recording":
        return _call(sim.start_cameras_recording, cameras=cameras, output_dir=str(tmp_path / "mp4"), fps=10)
    if surface == "start_cameras_recording_synchronous":
        return _call(
            sim.start_cameras_recording_synchronous, cameras=cameras, output_dir=str(tmp_path / "mp4s"), fps=10
        )
    return _call(
        sim.start_recording,
        repo_id="local/cameras_contract",
        task="t",
        fps=30,
        root=str(tmp_path / "ds"),
        overwrite=True,
        cameras=cameras,
    )


class TestAValueThatIsNotCameraNamesIsRefused:
    """Each surface reports the parameter instead of guessing or raising."""

    @pytest.mark.parametrize("surface", _SURFACES)
    @pytest.mark.parametrize(("label", "make"), _UNUSABLE, ids=[c[0] for c in _UNUSABLE])
    def test_the_parameter_is_named_in_a_structured_error(
        self, cam_sim, tmp_path, surface: str, label: str, make: Callable[[], Any]
    ) -> None:
        result = _invoke(cam_sim, surface, tmp_path, make())
        assert result["status"] == "error", (surface, label, result)
        text = " ".join(block["text"] for block in result["content"] if "text" in block)
        assert "cameras" in text, (surface, label, text)
        assert surface in text, (surface, label, text)

    @pytest.mark.parametrize("surface", _SURFACES)
    @pytest.mark.parametrize(("label", "make"), [("non_string_element", lambda: [3]), ("not_a_sequence", lambda: 3)])
    def test_a_non_name_value_does_not_escape_the_tool_envelope(
        self, cam_sim, tmp_path, surface: str, label: str, make: Callable[[], Any]
    ) -> None:
        """These two used to raise a bare ``TypeError`` past the dispatch layer.

        The rendering surfaces reached ``"tag__" + 3`` / ``for c in 3`` with the
        caller's value, so a ``TypeError`` naming neither the parameter nor the
        method left a method documented to return a result dict.
        """
        try:
            result = _invoke(cam_sim, surface, tmp_path, make())
        except Exception as exc:  # noqa: BLE001 - an escape is the regression
            pytest.fail(f"{surface} raised {type(exc).__name__}: {exc}")
        assert result["status"] == "error", (surface, label, result)


class TestARefusedCallStartsNothing:
    """The refusal precedes the work each surface would otherwise begin."""

    def test_a_refused_capture_leaves_no_recording_running(self, cam_sim, tmp_path) -> None:
        result = _call(
            cam_sim.start_cameras_recording,
            cameras=["wrist", "wrist"],
            output_dir=str(tmp_path / "mp4"),
            fps=10,
        )
        assert result["status"] == "error", result
        status = cam_sim.get_cameras_recording_status()
        text = " ".join(block["text"] for block in status["content"] if "text" in block)
        assert "idle" in text.lower(), text
        assert not (tmp_path / "mp4").exists(), sorted((tmp_path / "mp4").glob("*"))

    def test_a_refused_dataset_call_creates_no_dataset(self, cam_sim, tmp_path) -> None:
        root = tmp_path / "ds"
        result = _call(
            cam_sim.start_recording,
            repo_id="local/cameras_contract",
            task="t",
            fps=30,
            root=str(root),
            overwrite=True,
            cameras=["wrist", "wrist"],
        )
        assert result["status"] == "error", result
        assert cam_sim._is_recording() is False
        assert not root.exists(), sorted(root.glob("*"))

    def test_a_refused_render_returns_no_image_block(self, cam_sim, tmp_path) -> None:
        """Pre-fix a duplicate rendered the same view twice and returned both."""
        result = _call(cam_sim.render_all, cameras=["wrist", "wrist"], width=64, height=48)
        assert result["status"] == "error", result
        assert [block for block in result["content"] if "image" in block] == []


class TestDistinctNamesAreStillHonored:
    """The guard is additive: a usable subset behaves exactly as before."""

    def test_render_all_renders_one_block_per_requested_camera(self, cam_sim) -> None:
        result = _call(cam_sim.render_all, cameras=["wrist"], width=64, height=48)
        assert result["status"] == "success", result
        assert len([block for block in result["content"] if "image" in block]) == 1

    def test_render_all_still_defaults_to_every_camera(self, cam_sim) -> None:
        result = _call(cam_sim.render_all, cameras=None, width=64, height=48)
        assert result["status"] == "success", result
        assert len([block for block in result["content"] if "image" in block]) >= 1

    def test_a_plain_mp4_capture_still_starts(self, cam_sim, tmp_path) -> None:
        result = _call(cam_sim.start_cameras_recording, cameras=["wrist"], output_dir=str(tmp_path / "mp4"), fps=10)
        assert result["status"] == "success", result
        cam_sim.stop_cameras_recording()

    def test_a_scoped_dataset_recording_still_starts(self, cam_sim, tmp_path) -> None:
        pytest.importorskip("lerobot")
        result = _call(
            cam_sim.start_recording,
            repo_id="local/cameras_contract",
            task="t",
            fps=30,
            root=str(tmp_path / "ds"),
            overwrite=True,
            cameras=["wrist"],
        )
        assert result["status"] == "success", result
        cam_sim.stop_recording()

    @pytest.mark.parametrize("surface", _SURFACES)
    def test_an_empty_subset_keeps_each_surfaces_own_verdict(self, cam_sim, tmp_path, surface: str) -> None:
        """An empty sequence already means "not supplied" to these callers.

        Like every other consumer of the shared domain the check is gated on a
        truthy value, so ``[]`` is not this change's to reinterpret.
        """
        result = _invoke(cam_sim, surface, tmp_path, [])
        text = " ".join(block.get("text", "") for block in result["content"])
        assert "must be a list of names" not in text, (surface, text)


class TestEverySurfaceAgrees:
    """One parameter, one verdict - the divergence this change removes."""

    @pytest.mark.parametrize(("label", "make"), _UNUSABLE, ids=[c[0] for c in _UNUSABLE])
    def test_the_surfaces_reach_the_same_verdict(self, cam_sim, tmp_path, label: str, make: Callable[[], Any]) -> None:
        verdicts = {}
        for index, surface in enumerate(_SURFACES):
            scope = tmp_path / f"parity{index}"
            scope.mkdir()
            try:
                verdicts[surface] = _invoke(cam_sim, surface, scope, make())["status"]
            except Exception as exc:  # noqa: BLE001 - a raise is one of the verdicts
                verdicts[surface] = f"raised {type(exc).__name__}"
        assert len(set(verdicts.values())) == 1, (label, verdicts)


def _backend_root() -> pathlib.Path:
    """The simulation package directory, derived from a symbol it exports."""
    return pathlib.Path(inspect.getfile(SimWorld)).parent


def _cameras_surfaces(source: str) -> list[tuple[str, bool]]:
    """Public methods in ``source`` taking ``cameras``, and whether each guards it."""
    found: list[tuple[str, bool]] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.FunctionDef) or node.name.startswith("_"):
            continue
        argnames = [arg.arg for arg in node.args.args + node.args.kwonlyargs]
        if "cameras" not in argnames:
            continue
        guarded = any(
            isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "name_list_error"
            for call in ast.walk(node)
        )
        found.append((node.name, guarded))
    return found


class TestGuardIsWiredAtEveryBackendSurface:
    """A backend cannot add a ``cameras`` subset without the shared domain.

    Wired, not returned: this sweep is satisfied by a surface that calls the
    domain and drops the refusal, so each backend's returned ``cameras``
    refusal is driven in ``test_recording_preflight_refusals_across_backends.py``.
    """

    def test_every_public_cameras_surface_resolves_the_shared_domain(self) -> None:
        unguarded: list[str] = []
        seen: list[str] = []
        for backend in ("mujoco", "newton", "isaac"):
            for module in sorted((_backend_root() / backend).glob("*.py")):
                for name, guarded in _cameras_surfaces(module.read_text()):
                    seen.append(f"{backend}/{module.name}::{name}")
                    if not guarded:
                        unguarded.append(f"{backend}/{module.name}::{name}")
        assert seen, f"no cameras= surface found under {_backend_root()}"
        assert unguarded == [], (
            f"these public methods accept a cameras= subset without calling name_list_error: {unguarded}"
        )

    def test_the_scan_covers_the_known_surfaces(self) -> None:
        """Non-vacuity: a scan root that resolved elsewhere would find nothing."""
        seen = {
            f"{backend}::{name}"
            for backend in ("mujoco", "newton", "isaac")
            for module in sorted((_backend_root() / backend).glob("*.py"))
            for name, _ in _cameras_surfaces(module.read_text())
        }
        assert seen == {
            "mujoco::render_all",
            "mujoco::start_cameras_recording",
            "mujoco::start_cameras_recording_synchronous",
            "mujoco::start_recording",
            "newton::start_recording",
            "isaac::start_cameras_recording",
            "isaac::start_recording",
        }, seen

    def test_the_scanner_detects_a_surface_that_drops_the_guard(self) -> None:
        """A scanner that silently matched nothing would look like a clean tree."""
        planted = (
            "class Mixin:\n    def start_cameras_recording(self, cameras=None):\n        return {'status': 'success'}\n"
        )
        assert _cameras_surfaces(planted) == [("start_cameras_recording", False)]
