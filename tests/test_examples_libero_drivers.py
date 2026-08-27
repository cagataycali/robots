"""Import + API-drift smoke tests for the LIBERO example drivers.

The ``examples/libero`` driver scripts are not part of the installed
package, so nothing else in CI imports them - an API rename in
``strands_robots`` can silently break the documented smoke path
(``python examples/libero/run.py mujoco --policy mock --n-episodes 5``)
without any test going red. These tests pin the contract:

1. The merged driver (``run.py``) imports cleanly with no backend
   installed at all - it defers every backend import until after
   subcommand parsing - and the MuJoCo agent driver imports cleanly
   against the *current* library (module-level imports are the first
   thing that breaks on drift).
2. The library symbols the drivers call still exist with the expected
   surface (``evaluate_benchmark`` kwargs, camera-recording pair,
   ``LiberoAdapter.ensure_scene``, ``gr00t_inference``).
3. The grep-stable result line ``run.py`` prints (built by
   ``_format_result_lines``, the single formatter both subcommands
   share) stays byte-compatible with ``libero_backend_matrix.py``'s
   ``_RE_RESULT`` parser - drift there produces empty ``success_rate``
   matrix cells rather than a crash, so only a test catches it.
4. The backend subcommands scope their flags: Isaac-only flags do not
   exist on the ``mujoco`` subcommand, and the shared flag base is
   identical across the two.
5. The examples do not depend on private (``_``-prefixed) LiberoAdapter
   methods.

The Isaac agent driver (``run_isaac_agent.py``) is excluded from the
import smoke: it intentionally guards its heavy imports behind
``IsaacSimulation.is_available`` at runtime, but its module docstring +
argparse setup still get covered by the repo-wide example lint tests.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import re
import sys
import textwrap
import types
from pathlib import Path

import pytest

_EXAMPLES_LIBERO = Path(__file__).resolve().parent.parent / "examples" / "libero"


def _load_example(filename: str):
    """Import an example script by path under a test-unique module name."""
    path = _EXAMPLES_LIBERO / filename
    assert path.is_file(), f"expected example driver at {path}"
    module_name = f"_example_smoke_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module


def test_run_imports_cleanly_without_any_backend() -> None:
    """``run.py`` must not import a simulation backend at module level.

    The subcommand design promises "import the chosen backend only
    after parsing" - a module-level ``strands_robots.simulation``
    import would make ``run.py isaac`` pay for MuJoCo (and break the
    matrix's ``skip``-row detection on hosts without it).
    """
    before = set(sys.modules)
    module = _load_example("run.py")
    assert callable(module.main)
    leaked = {m for m in set(sys.modules) - before if m.startswith(("mujoco", "omni", "isaacsim"))}
    assert not leaked, f"run.py imported backend modules at import time: {sorted(leaked)}"


@pytest.mark.parametrize("filename", ["run_mujoco_agent.py", "libero_backend_matrix.py"])
def test_driver_imports_cleanly(filename: str) -> None:
    # The MuJoCo agent driver imports `Simulation` (the MuJoCo backend)
    # at module level, so its import smoke needs mujoco installed.
    if filename == "run_mujoco_agent.py":
        pytest.importorskip("mujoco")
    module = _load_example(filename)
    assert callable(module.main)


def test_run_helper_surface() -> None:
    """Sibling agent-driver docstrings cross-reference these helpers by
    name; keep the canonical merged driver exposing them."""
    module = _load_example("run.py")
    assert callable(module._date_dir)
    assert callable(module._suite_for_task)
    assert callable(module._default_checkpoint_dir)
    assert callable(module._explain_lifecycle_failure)
    assert callable(module._configure_gr00t_image)
    assert callable(module._orchestrate_groot_server)
    assert callable(module._resolve_robot_asset)
    assert module._suite_for_task("libero-spatial-pick_up_the_red_cube") == "libero_spatial"
    assert module._suite_for_task("libero-10-LIVING_ROOM_SCENE5_x") == "libero_10"
    with pytest.raises(ValueError, match="libero-<suite>-<task_stem>"):
        module._suite_for_task("not-a-libero-task")
    with pytest.raises(ValueError, match="libero-<suite>-<task_stem>"):
        module._suite_for_task("libero-spatial")


@pytest.mark.parametrize("backend", ["mujoco", "isaac"])
def test_result_line_matches_backend_matrix_parser(backend: str) -> None:
    """The line run.py prints must parse with _RE_RESULT.

    Build the sample line with the driver's own formatter
    (``_format_result_lines`` - the exact function both subcommands
    print through); if either side changes shape, this goes red before
    the matrix silently prints empty cells.
    """
    run = _load_example("run.py")
    matrix = _load_example("libero_backend_matrix.py")
    sr, wt = 0.8, 123.4
    requested = "libero-spatial-pick_up_the_red_cube"
    resolved = "libero-spatial-pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"
    bench_line, result_line = run._format_result_lines(
        policy="mock",
        requested_task=requested,
        resolved_task=resolved,
        success_rate=sr,
        wall_time=wt,
        video_path="rollouts/x.mp4",
        backend=backend,
    )
    assert bench_line == f"benchmark_name={requested}"
    match = matrix._RE_RESULT.search(result_line)
    assert match is not None, f"_RE_RESULT failed to parse the driver's result-line shape: {result_line!r}"
    assert float(match.group("sr")) == pytest.approx(sr)
    assert float(match.group("wt")) == pytest.approx(wt)
    # The replay contract: task= echoes the CLI-requested task,
    # resolved_task= what actually ran, backend= the subcommand.
    assert f"task={requested}  " in result_line
    assert f"resolved_task={resolved}  " in result_line
    assert result_line.endswith(f"backend={backend}")


def test_backend_matrix_rows_invoke_run_py_subcommands() -> None:
    """The mujoco / isaac matrix rows must point at run.py with the
    matching subcommand, and the file they reference must exist."""
    matrix = _load_example("libero_backend_matrix.py")
    rows = {label: (filename, subcommand) for label, filename, subcommand, _ in matrix._BACKEND_ROWS}
    assert rows.get("mujoco") == ("run.py", ["mujoco"])
    assert rows.get("isaac-1") == ("run.py", ["isaac"])
    assert (_EXAMPLES_LIBERO / "run.py").is_file()


def test_isaac_only_flags_do_not_exist_on_the_mujoco_subcommand() -> None:
    """Subcommand flag scoping: the merge's whole point.

    Isaac-only flags must be rejected by the ``mujoco`` subcommand at
    parse time (not silently dropped), accepted by ``isaac``, and the
    shared base must parse identically on both.
    """
    run = _load_example("run.py")
    parser = run._build_parser()

    shared = ["--policy", "mock", "--n-episodes", "3", "--seed", "7", "--port", "8123"]
    mj = parser.parse_args(["mujoco", *shared])
    isc = parser.parse_args(["isaac", *shared, "--robot-usd", "/x/robot.usd", "--eef-body-name", "hand"])
    assert mj.backend == "mujoco"
    assert isc.backend == "isaac"
    for attr in ("policy", "n_episodes", "seed", "port", "task", "auto_server", "image", "deterministic"):
        assert hasattr(mj, attr) and hasattr(isc, attr), f"shared flag {attr!r} missing from a subcommand"
    assert isc.robot_usd == "/x/robot.usd"
    assert isc.eef_body_name == "hand"

    for flag in ("--robot-usd=/x.usd", "--robot-urdf=/x.urdf", "--eef-body-name=hand"):
        with pytest.raises(SystemExit):
            parser.parse_args(["mujoco", flag])
    assert not hasattr(mj, "robot_usd"), "mujoco subcommand should not carry Isaac-only namespace attrs"


@pytest.mark.parametrize(
    ("backend", "expected"),
    [("mujoco", "gr00t-libero-mujoco"), ("isaac", "gr00t-libero-isaac")],
)
def test_container_default_derives_from_the_backend_subcommand(backend: str, expected: str) -> None:
    """The ``--container`` default is computed per subcommand.

    The merge replaced two hardcoded container names with one f-string
    derived from the subparser's ``dest="backend"`` - the only
    behavioral mechanism the refactor changed. If either the f-string
    or the dest drifts, both subcommands resolve to one container name,
    which is exactly what the flag's help text promises cannot happen
    ("don't clobber each other's containers when run side-by-side").
    That failure surfaces as a GR00T lifecycle collision on a GPU host
    CI does not have, so this is the pin - it also documents that the
    two pre-merge drivers' defaults were preserved deliberately.
    """
    run = _load_example("run.py")
    args = run._build_parser().parse_args([backend])
    # Parse-time sentinel: None keeps an explicit --container
    # distinguishable from the derived default.
    assert args.container is None
    run._resolve_container_name(args)
    assert args.container == expected


def test_explicit_container_survives_default_resolution() -> None:
    """An explicit ``--container`` passes through resolution unchanged.

    The resolver guards on ``is None``, not truthiness - the half a
    naive ``args.container or f"..."`` rewrite would break.
    """
    run = _load_example("run.py")
    args = run._build_parser().parse_args(["mujoco", "--container", "my-name"])
    run._resolve_container_name(args)
    assert args.container == "my-name"


@pytest.mark.parametrize("filename", ["run.py", "run_mujoco_agent.py"])
def test_drivers_do_not_call_private_scene_generation(filename: str) -> None:
    """Public-API hygiene: the drivers must use ``LiberoAdapter.ensure_scene``
    (public), never the private ``_generate_scene_from_bddl``."""
    source = (_EXAMPLES_LIBERO / filename).read_text(encoding="utf-8")
    assert "_generate_scene_from_bddl" not in source, (
        f"{filename} references the private LiberoAdapter._generate_scene_from_bddl; "
        "use the public ensure_scene() instead."
    )
    assert "ensure_scene" in source


def test_library_surface_the_drivers_depend_on() -> None:
    """Pin the exact library API the drivers call, so a rename fails here
    with a readable message instead of at example runtime."""
    pytest.importorskip("mujoco")
    from strands_robots.benchmarks.libero import LiberoAdapter, load_libero_suite
    from strands_robots.simulation import Simulation, get_benchmark

    # Import the tool function from its submodule, not via the package's
    # lazy `strands_robots.tools.__getattr__`: when another test has already
    # imported the `strands_robots.tools.gr00t_inference` *module*, Python
    # binds the module object as the package attribute, shadowing the lazy
    # function resolution - so the package-level import is order-dependent
    # under the full test suite. (The drivers themselves run in fresh
    # interpreters, where the lazy path deterministically yields the
    # function.)
    from strands_robots.tools.gr00t_inference import gr00t_inference

    assert callable(load_libero_suite)
    assert callable(get_benchmark)
    assert callable(gr00t_inference)

    # `_resolve_task` forwards adapter_kwargs (the Isaac EEF state-source
    # configuration, #1802) through load_libero_suite.
    assert "adapter_kwargs" in inspect.signature(load_libero_suite).parameters

    # LiberoAdapter public pre-warm surface used by the scene pre-warm block.
    assert callable(LiberoAdapter.ensure_scene)
    assert callable(LiberoAdapter.prewarm)

    # Simulation methods the drivers call.
    for method in (
        "create_world",
        "add_robot",
        "load_scene",
        "list_robots",
        "start_cameras_recording",
        "stop_cameras_recording",
        "evaluate_benchmark",
        "destroy",
    ):
        assert callable(getattr(Simulation, method)), f"Simulation.{method} missing"

    # evaluate_benchmark kwargs the drivers pass.
    params = inspect.signature(Simulation.evaluate_benchmark).parameters
    for kwarg in ("benchmark_name", "n_episodes", "seed", "policy_provider", "policy_config", "robot_name"):
        assert kwarg in params, f"evaluate_benchmark lost the {kwarg!r} kwarg the LIBERO drivers pass"

    # start_cameras_recording kwargs the drivers pass.
    rec_params = inspect.signature(Simulation.start_cameras_recording).parameters
    for kwarg in ("cameras", "output_dir", "name"):
        assert kwarg in rec_params, f"start_cameras_recording lost the {kwarg!r} kwarg the LIBERO drivers pass"

    # gr00t_inference lifecycle kwargs the drivers pass. The @tool
    # decorator preserves the underlying signature via functools.wraps;
    # fall back to the raw function if not.
    tool_fn = inspect.unwrap(gr00t_inference)
    tool_params = inspect.signature(tool_fn).parameters
    for kwarg in (
        "action",
        "lifecycle",
        "hf_repo",
        "hf_subfolder",
        "hf_local_dir",
        "container_name",
        "hf_token",
        "checkpoint_path",
        "embodiment_tag",
        "protocol",
        "use_sim_policy_wrapper",
        "deterministic",
        "port",
    ):
        assert kwarg in tool_params, f"gr00t_inference lost the {kwarg!r} kwarg the LIBERO drivers pass"


# ---------------------------------------------------------------------------
# The zero-frame verdict is shared, because the videos= line is shared
# ---------------------------------------------------------------------------
#
# `stop_cameras_recording` answers status="success" whether the recorder
# captured every frame or none, and imageio drops a 0-frame mp4 rather than
# writing an empty file. Measured on the MuJoCo backend with a GL context the
# host cannot create (a `MUJOCO_GL` naming a windowed backend on a headless
# box): status="success", `frames: 0`, `errors: 46`, and nothing on disk.
# The result line prints `videos=<path>` for both subcommands, so the verdict
# has to hold for both - it is the shared loop's, not a per-backend hook's.

_ZERO_FRAME_REFUSAL = "Rendering unavailable (no OpenGL context). Install EGL or OSMesa"


def _stop_payload(*, camera: str = "default", frames: int = 0, refused: str | None = None) -> dict:
    """A ``stop_cameras_recording`` envelope in the shape both backends emit."""
    artifact: dict = {
        "camera": camera,
        "path": f"/tmp/rollout__{camera}.mp4",
        "frames": frames,
        "errors": 0 if frames else 46,
        "size_kb": 16.2 if frames else 0.0,
    }
    if refused is not None:
        artifact["render_refused"] = refused
    return {
        "status": "success",
        "content": [
            {"text": f"Stopped 'rollout' after 1.9s\n   {camera}  {frames} frames"},
            {"json": {"recording": "rollout", "artifacts": [artifact]}},
        ],
    }


class _FakeSim:
    """Minimal ``evaluate_benchmark`` / ``stop_cameras_recording`` pair."""

    def __init__(self, stop: dict) -> None:
        self._stop = stop
        self.evaluated = 0

    def evaluate_benchmark(self, **_kwargs: object) -> dict:
        self.evaluated += 1
        return {"status": "success", "content": [{"json": {"success_rate": 0.4}}]}

    def stop_cameras_recording(self) -> dict:
        return self._stop


def _report_args(run, backend: str):
    """The namespace ``_evaluate_and_report`` reads, for either subcommand."""
    return run.argparse.Namespace(
        backend=backend, policy="mock", n_episodes=2, seed=7, task="libero_spatial", port=5555
    )


def _plan_for(run, backend: str, calls: list[str]):
    """A plan in each subcommand's real shape.

    MuJoCo ships no ``check_recording`` at all; Isaac ships one that reports
    its ``on_frame`` retry counters. Neither may decide whether a rollout
    recorded anything, so the Isaac shape here reports without raising.
    """
    hook = None
    if backend == "isaac":

        def hook(_stop: dict) -> None:
            calls.append("check_recording")

    return run._EvalPlan(
        requested_task="libero_spatial",
        resolved_task="libero_spatial_task_0",
        recording_camera="default",
        arm_recording=lambda _video_dir, _rec_name: {},
        check_recording=hook,
    )


@pytest.mark.parametrize("backend", ["mujoco", "isaac"])
def test_a_rollout_that_recorded_no_frames_is_refused_on_either_subcommand(backend, tmp_path, monkeypatch):
    """Neither subcommand may print a videos= path for a file that never landed.

    Fails pre-fix on the ``mujoco`` shape: with no ``check_recording`` the
    loop printed the result lines and returned, so a blank rollout shipped a
    ``success_rate`` line naming a dropped mp4. Fails pre-fix on the
    ``isaac`` shape too once the hook only reports - which is the point: the
    verdict must not depend on a per-backend hook supplying it.
    """
    run = _load_example("run.py")
    monkeypatch.setattr(run, "_date_dir", lambda: str(tmp_path))
    calls: list[str] = []
    sim = _FakeSim(_stop_payload(frames=0))

    with pytest.raises(RuntimeError, match="0 frames"):
        run._evaluate_and_report(sim, _report_args(run, backend), _plan_for(run, backend, calls))

    assert sim.evaluated == 1, "premise: the eval itself ran and reported success"


@pytest.mark.parametrize("backend", ["mujoco", "isaac"])
def test_a_rollout_that_recorded_frames_still_reports_normally(backend, tmp_path, monkeypatch, capsys):
    """Control: a real recording is untouched, and the hook still runs."""
    run = _load_example("run.py")
    monkeypatch.setattr(run, "_date_dir", lambda: str(tmp_path))
    calls: list[str] = []
    sim = _FakeSim(_stop_payload(frames=120))

    run._evaluate_and_report(sim, _report_args(run, backend), _plan_for(run, backend, calls))

    printed = capsys.readouterr().out
    assert "benchmark_name=libero_spatial" in printed
    assert "success_rate=0.40" in printed and f"backend={backend}" in printed
    assert calls == (["check_recording"] if backend == "isaac" else [])


def test_the_backend_hook_reports_before_the_shared_verdict_raises(tmp_path, monkeypatch):
    """A backend's recorder diagnostics are context for the refusal, so they precede it."""
    run = _load_example("run.py")
    monkeypatch.setattr(run, "_date_dir", lambda: str(tmp_path))
    calls: list[str] = []

    with pytest.raises(RuntimeError, match="0 frames"):
        run._evaluate_and_report(
            _FakeSim(_stop_payload(frames=0)), _report_args(run, "isaac"), _plan_for(run, "isaac", calls)
        )

    assert calls == ["check_recording"], "the hook must have run before the verdict raised"


def test_the_refusal_carries_the_reason_the_recorder_recorded():
    """The MuJoCo payload names the remedy; the refusal must not discard it."""
    run = _load_example("run.py")

    with pytest.raises(RuntimeError) as excinfo:
        run._require_recorded_frames(_stop_payload(frames=0, refused=_ZERO_FRAME_REFUSAL), "default")

    message = str(excinfo.value)
    assert _ZERO_FRAME_REFUSAL in message, message
    assert "46 per-frame render errors" in message, message


def test_a_missing_artifact_for_the_named_camera_counts_as_no_video():
    """The result line names that camera's file whether or not an artifact exists."""
    run = _load_example("run.py")

    with pytest.raises(RuntimeError, match="'image'"):
        run._require_recorded_frames(_stop_payload(camera="default", frames=120), "image")


def test_the_zero_frame_verdict_is_applied_unconditionally() -> None:
    """Root cause: the verdict is the shared loop's, not a per-backend opt-in.

    Guarding the call on ``plan.check_recording is None`` would restore the
    split this closes - one backend deciding, the other not - so pin that the
    call is not inside any conditional.
    """
    run = _load_example("run.py")
    tree = ast.parse(textwrap.dedent(inspect.getsource(run._evaluate_and_report)))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "_require_recorded_frames"
    ]
    assert len(calls) == 1, "the shared loop applies the verdict exactly once"
    conditional = {id(n) for branch in ast.walk(tree) if isinstance(branch, ast.If) for n in ast.walk(branch)}
    assert id(calls[0]) not in conditional, "the zero-frame verdict must not be conditional"


# ---------------------------------------------------------------------------
# A refused setup step is reported where it refuses, on either subcommand
# ---------------------------------------------------------------------------
#
# Every `Simulation` / `IsaacSimulation` setup call answers a tool envelope
# rather than raising. `_setup_isaac` read the verdict of all five of its
# (`create_world`, `add_robot`, `add_camera` x2, `start_cameras_recording`);
# `_setup_mujoco` discarded all five of its. Measured by driving the real
# `_setup_mujoco` with one refused step at a time (the rest succeeding):
#
#   refused step             | before                          | after
#   -------------------------|---------------------------------|--------------
#   create_world             | printed a success_rate= line    | refused at setup
#   add_robot                | printed a success_rate= line    | refused at setup
#   load_scene               | printed a success_rate= line    | refused at setup
#   start_cameras_recording  | RuntimeError after the rollout  | refused pre-rollout
#
# Three of the four shipped a full result line, which `libero_backend_matrix.py`
# parses as a completed cell; the fourth surfaced only after the whole rollout
# as `rollout recorded 0 frames for camera 'image' (0 per-frame render errors)`
# - a render diagnosis for a recorder that never started. Before: 4 of 4 ran the
# rollout and 0 of 4 named the step. After: 0 of 4 and 4 of 4.

_SETUP_REFUSALS = {
    "create_world": "create_world: a world already exists (robots: robot; objects: none).",
    "add_robot": "Failed to load: asset for 'panda' is not on disk.",
    "load_scene": "Scene file not found: /var/cache/libero/spatial_task_0.xml",
    "start_cameras_recording": "Camera(s) not found: ['image', 'wrist_image']. Available: ['default']",
}


class _RefusingSetupSim:
    """A ``Simulation`` stand-in that refuses exactly one setup step.

    Refusing one at a time is what isolates *where* that step's refusal
    surfaces; in practice several cascade.
    """

    def __init__(self, refuse: str | None, *, has_robot: bool = True) -> None:
        self.refuse = refuse
        self.calls: list[str] = []
        self.recording = False
        self._has_robot = has_robot

    def _env(self, name: str) -> dict:
        self.calls.append(name)
        if name == self.refuse:
            return {"status": "error", "content": [{"text": _SETUP_REFUSALS[name]}]}
        return {"status": "success", "content": [{"text": f"{name} ok"}]}

    def create_world(self, **_kw: object) -> dict:
        return self._env("create_world")

    def add_robot(self, *_a: object, **_kw: object) -> dict:
        return self._env("add_robot")

    def load_scene(self, *_a: object, **_kw: object) -> dict:
        return self._env("load_scene")

    def list_robots(self) -> list[str]:
        return ["robot"] if self._has_robot else []

    def start_cameras_recording(self, **_kw: object) -> dict:
        result = self._env("start_cameras_recording")
        self.recording = result["status"] == "success"
        return result

    def stop_cameras_recording(self) -> dict:
        self.calls.append("stop_cameras_recording")
        if not self.recording:
            # Measured verbatim on a real MuJoCo sim: success, and no json block.
            return {"status": "success", "content": [{"text": "Was not recording cameras."}]}
        return {
            "status": "success",
            "content": [
                {"text": "Stopped 'r' after 1.9s"},
                {
                    "json": {
                        "artifacts": [
                            {
                                "camera": "image",
                                "path": "/tmp/r__image.mp4",
                                "frames": 120,
                                "errors": 0,
                                "size_kb": 16.2,
                            }
                        ]
                    }
                },
            ],
        }

    def evaluate_benchmark(self, **_kw: object) -> dict:
        self.calls.append("evaluate_benchmark")
        return {"status": "success", "content": [{"json": {"success_rate": 0.4}}]}

    def destroy(self) -> dict:
        self.calls.append("destroy")
        return {"status": "success", "content": [{"text": "ok"}]}


class _PrewarmSpec:
    """A benchmark spec whose scene the ``groot`` path pre-warms."""

    scene_path = "/var/cache/libero/spatial_task_0.xml"

    def ensure_scene(self) -> None:
        return None

    def prewarm(self, _sim: object) -> None:
        return None


def _drive_mujoco_setup(run, monkeypatch, tmp_path, refuse, *, has_robot=True):
    """Run the real ``_setup_mujoco`` (and the recording arm) against a fake sim.

    Returns ``(sim, raised)`` - ``raised`` is the exception the caller saw, or
    ``None`` if the whole path reported success.
    """
    sim = _RefusingSetupSim(refuse, has_robot=has_robot)
    sim_module = types.ModuleType("strands_robots.simulation")
    sim_module.Simulation = lambda **_kw: sim  # type: ignore[attr-defined]
    bench_module = types.ModuleType("strands_robots.simulation.benchmark")
    bench_module.get_benchmark = lambda _name: _PrewarmSpec()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "strands_robots.simulation", sim_module)
    monkeypatch.setitem(sys.modules, "strands_robots.simulation.benchmark", bench_module)
    monkeypatch.setattr(run, "_resolve_task", lambda *_a, **_k: "libero_spatial_task_0")
    monkeypatch.setattr(run, "_date_dir", lambda: str(tmp_path))

    args = run.argparse.Namespace(
        backend="mujoco", policy="groot", n_episodes=2, seed=7, task="libero_spatial_task_0", port=5555
    )
    try:
        built_sim, plan = run._setup_mujoco(args, "libero_spatial")
        run._evaluate_and_report(built_sim, args, plan)
    except Exception as exc:  # noqa: BLE001 - the caller's own verdict is the subject
        return sim, exc
    return sim, None


@pytest.mark.parametrize("refuse", sorted(_SETUP_REFUSALS))
def test_a_refused_setup_step_is_reported_where_it_refuses(refuse, tmp_path, monkeypatch, capsys):
    """A setup step the backend declined must not be discarded.

    Fails pre-fix for all four: ``create_world`` / ``add_robot`` /
    ``load_scene`` printed a full ``success_rate=`` line, and
    ``start_cameras_recording`` raised only after the rollout with a render
    diagnosis naming neither the step nor its camera list.
    """
    run = _load_example("run.py")
    sim, raised = _drive_mujoco_setup(run, monkeypatch, tmp_path, refuse)

    printed = capsys.readouterr().out
    assert raised is not None, (
        f"a refused {refuse} was discarded: the run reported success and printed {printed.strip()[:200]!r}"
    )
    assert "success_rate=" not in printed, f"a refused {refuse} still printed a result line: {printed[:200]!r}"
    message = str(raised)
    assert refuse in message, f"the refusal must name the step that declined; got: {message[:200]}"
    assert _SETUP_REFUSALS[refuse] in message, f"the backend's own reason must survive; got: {message[:200]}"
    assert "evaluate_benchmark" not in sim.calls, (
        f"a refused {refuse} must not spend a whole rollout first (calls: {sim.calls})"
    )


def test_the_fallback_add_robot_is_checked_too(tmp_path, monkeypatch):
    """The defensive ``add_robot`` behind the redundant-Panda check is a site too.

    It only runs for a scene that ships no Panda, so a fake reporting an empty
    ``list_robots()`` is what reaches it.
    """
    run = _load_example("run.py")
    sim, raised = _drive_mujoco_setup(run, monkeypatch, tmp_path, "add_robot", has_robot=False)

    assert raised is not None, "the fallback add_robot's refusal was discarded"
    assert "add_robot" in str(raised)
    assert sim.calls.count("add_robot") >= 1, f"premise: the fallback site ran (calls: {sim.calls})"


# ---------------------------------------------------------------------------
# Root cause: one owner for the rule, and no shim may drift off it
# ---------------------------------------------------------------------------

_SETUP_SHIMS = ("_setup_mujoco", "_setup_isaac")


def _is_status_guard(test: ast.expr) -> bool:
    """Is ``test`` the inline ``<envelope>.get("status") != "success"`` comparison?

    Matched with the quotes stripped. ``ast.unparse`` normalises every string
    literal to single quotes, so a double-quoted needle silently matches
    nothing - and both rules below would then pass without grading anything.
    """
    rendered = ast.unparse(test).replace('"', "").replace("'", "")
    return "get(status) !=" in rendered and "success" in rendered


def _setup_shim_sim_calls(source: str) -> dict[str, dict[str, list[tuple[str, int]]]]:
    """Per ``sim`` method, its call sites in either setup shim, split by check.

    A site counts as checked when its envelope reaches a verdict: either it
    flows into ``_require_ok`` (directly, or through a local that
    ``_require_ok`` is later called on), or the local it was assigned to is
    guarded by an explicit ``if <name>.get("status") != "success"``. Both forms
    are recognised so the rule reads the same before and after the shared owner
    exists.
    """
    tree = ast.parse(source)
    shims = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in _SETUP_SHIMS]
    assert len(shims) == len(_SETUP_SHIMS), f"expected both setup shims, found {[s.name for s in shims]}"

    out: dict[str, dict[str, list[tuple[str, int]]]] = {}
    for shim in shims:
        checked_calls: set[int] = set()
        checked_names: set[str] = set()
        for node in ast.walk(shim):
            # _require_ok(<call>, "what") / _require_ok(<name>, "what")
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "_require_ok" and node.args:
                first = node.args[0]
                if isinstance(first, ast.Call):
                    checked_calls.add(id(first))
                elif isinstance(first, ast.Name):
                    checked_names.add(first.id)
            # if <name>.get("status") != "success"
            if isinstance(node, ast.If) and _is_status_guard(node.test):
                for sub in ast.walk(node.test):
                    if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name):
                        checked_names.add(sub.value.id)
        # a call assigned to a checked local is itself checked
        for node in ast.walk(shim):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                if any(isinstance(t, ast.Name) and t.id in checked_names for t in node.targets):
                    checked_calls.add(id(node.value))

        for node in ast.walk(shim):
            if not isinstance(node, ast.Call):
                continue
            func = ast.unparse(node.func)
            if not func.startswith("sim."):
                continue
            method = func.split(".", 1)[1]
            bucket = out.setdefault(method, {"checked": [], "unchecked": []})
            key = "checked" if id(node) in checked_calls else "unchecked"
            bucket[key].append((shim.name, node.lineno))
    return out


def test_every_setup_step_checked_in_one_shim_is_checked_in_the_other() -> None:
    """A setup step whose verdict one shim reads must be read wherever it is called.

    Derived, so it needs no list of method names: ``destroy`` and ``step`` are
    unchecked at every site (cleanup, and a retry loop whose observable is the
    frame count) and are therefore consistent; ``create_world``, ``add_robot``
    and ``start_cameras_recording`` were read in ``_setup_isaac`` and discarded
    in ``_setup_mujoco``, which is exactly the drift this reports.
    """
    calls = _setup_shim_sim_calls((_EXAMPLES_LIBERO / "run.py").read_text(encoding="utf-8"))
    assert len(calls) >= 6, f"the scan reached too few sim methods to mean anything: {sorted(calls)}"

    offenders = {
        method: sites["unchecked"] for method, sites in calls.items() if sites["checked"] and sites["unchecked"]
    }
    assert not offenders, (
        "a setup step's verdict is read at one call site and discarded at another, "
        f"so the two backend shims disagree about it: {offenders}"
    )


def test_the_setup_shims_route_every_check_through_one_owner() -> None:
    """Single owner: neither shim re-derives the status comparison inline.

    Two copies of the rule is how the shims drifted into one reading the
    verdict and the other discarding it, so the check has one home.
    """
    source = (_EXAMPLES_LIBERO / "run.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    inline: list[tuple[str, int]] = []
    for shim in [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in _SETUP_SHIMS]:
        for node in ast.walk(shim):
            if isinstance(node, ast.If) and _is_status_guard(node.test):
                inline.append((shim.name, node.lineno))
    assert not inline, f"a setup shim re-derives the status check instead of calling _require_ok: {inline}"


def test_the_shared_owner_reports_the_step_and_keeps_the_backend_reason() -> None:
    """The refusal names the step and carries the envelope the backend answered."""
    run = _load_example("run.py")
    refusal = {"status": "error", "content": [{"text": "Camera(s) not found: ['image']"}]}

    with pytest.raises(RuntimeError) as excinfo:
        run._require_ok(refusal, "start_cameras_recording")

    message = str(excinfo.value)
    assert message.startswith("start_cameras_recording failed:"), message
    assert "Camera(s) not found" in message, message


def test_the_shared_owner_returns_the_envelope_untouched_on_success() -> None:
    """Control: the Isaac recorder chains its ``on_frame`` handle off the check."""
    run = _load_example("run.py")
    envelope = {"status": "success", "content": [{"json": {"on_frame": "handle"}}]}

    returned = run._require_ok(envelope, "start_cameras_recording")

    assert returned is envelope


def test_the_refusal_wording_is_unchanged() -> None:
    """Control: the operator-visible template still reads ``<step> failed: <envelope>``.

    Written to match either spelling of it - the per-site raises the shims used
    before the shared owner existed, and the owner's own - so it pins the
    wording rather than which function produces it.
    """
    source = (_EXAMPLES_LIBERO / "run.py").read_text(encoding="utf-8")

    assert re.search(r"failed: \{(?:result|rec|what)\}", source), (
        "the refusal template moved away from '<step> failed: <envelope>'"
    )


def test_the_cleanup_and_retry_loop_calls_are_deliberately_unchecked() -> None:
    """Boundary: not every ``sim`` call in a shim answers a verdict worth reading.

    ``destroy`` runs while already unwinding, and ``step`` sits in the RTX
    warmup retry loop whose observable is the accumulated frame count - so both
    stay unchecked, in both shims. This fails if the fix is widened to every
    call rather than the setup steps.
    """
    calls = _setup_shim_sim_calls((_EXAMPLES_LIBERO / "run.py").read_text(encoding="utf-8"))

    for method in ("destroy", "step"):
        assert method in calls, f"premise: the shims still call sim.{method}"
        assert not calls[method]["checked"], (
            f"sim.{method} is not a setup step; checking it would refuse a rollout for a "
            f"cleanup/retry call: {calls[method]['checked']}"
        )


def test_a_setup_that_succeeds_still_returns_a_plan_and_reports(tmp_path, monkeypatch, capsys):
    """Control: the happy path is untouched - it still builds a plan and prints."""
    run = _load_example("run.py")
    sim, raised = _drive_mujoco_setup(run, monkeypatch, tmp_path, None)

    assert raised is None, f"a fully successful setup must not raise: {raised}"
    printed = capsys.readouterr().out
    assert "success_rate=0.40" in printed and "backend=mujoco" in printed, printed[:300]
    assert "evaluate_benchmark" in sim.calls, f"premise: the rollout ran (calls: {sim.calls})"


# ---------------------------------------------------------------------------
# A matrix row with no measurement must not report the same status as one that
# measured. `_run_one` reads the driver's exit code for the "did it run" half
# and `_RE_RESULT` for the numbers; a driver that exits 0 whose result line it
# cannot read has answered the first and failed the second, so "ok" asserts a
# measurement the row does not carry.
# ---------------------------------------------------------------------------


def _stub_driver(tmp_path, body: str):
    """Write a throwaway driver script and return its path.

    ``_run_one`` spawns the path with ``sys.executable``, so the stub stands in
    for a real per-backend driver without importing any backend.
    """
    path = tmp_path / "stub_driver.py"
    path.write_text(body + "\n", encoding="utf-8")
    return path


def _matrix_row(tmp_path, body: str):
    """Drive the real ``_run_one`` against a stub driver with the given body."""
    matrix = _load_example("libero_backend_matrix.py")
    return matrix._run_one(
        "mujoco",
        _stub_driver(tmp_path, body),
        [],
        [],
        policy="mock",
        task="libero-spatial-pick_up_the_red_cube",
        n_episodes=1,
        seed=42,
        port=8000,
        timeout=60.0,
    )


def _real_result_line(backend: str = "mujoco", success_rate: float = 0.4, wall_time: float = 12.5) -> str:
    """The driver's own result line, from the single formatter both subcommands print through."""
    run = _load_example("run.py")
    _, result_line = run._format_result_lines(
        policy="mock",
        requested_task="libero-spatial-pick_up_the_red_cube",
        resolved_task="libero-spatial-pick_up_the_red_cube",
        success_rate=success_rate,
        wall_time=wall_time,
        video_path="rollouts/2026_01_01/x.mp4",
        backend=backend,
    )
    return result_line


class TestARowWithNoMeasurementIsNotReportedAsOk:
    """A driver that exits 0 without a readable result line is not an ``ok`` row."""

    def test_a_driver_that_prints_no_result_line_is_not_ok(self, tmp_path) -> None:
        """Exit 0, no result line: the row has no numbers, so it cannot be ``ok``."""
        row = _matrix_row(tmp_path, "print('[setup] world ready')")

        assert row.success_rate is None and row.wall_time is None, (
            f"premise: nothing parseable was printed, so both cells stay empty: {row}"
        )
        assert row.status != "ok", (
            "a row whose success_rate and wall_time could not be read reports 'ok', so the "
            "status column claims a measurement that does not exist"
        )

    def test_a_driver_whose_result_line_does_not_parse_is_not_ok(self, tmp_path) -> None:
        """Exit 0 with a result line the parser cannot read is the same non-measurement."""
        row = _matrix_row(tmp_path, "print('policy=mock task=t success_rate=NaN wall_time=abcs')")

        assert row.success_rate is None and row.wall_time is None, f"premise: the line does not parse: {row}"
        assert row.status != "ok", (
            "a driver whose line drifted out of _RE_RESULT's shape reports 'ok', so a contract "
            "break between the driver and this parser reads as a clean cell"
        )

    def test_the_two_outcomes_no_longer_share_one_status(self, tmp_path) -> None:
        """The measured row and the unmeasured row must not report the same status.

        This is the whole defect in one assertion: before the fix both spellings
        answered ``ok`` and differed only in two ``--`` table cells, which a
        reader of the status column never sees.
        """
        measured = _matrix_row(tmp_path, f"print({_real_result_line()!r})")
        unmeasured = _matrix_row(tmp_path, "print('[setup] world ready')")

        assert measured.success_rate is not None, f"premise: the control row measured: {measured}"
        assert unmeasured.success_rate is None, f"premise: the other row did not: {unmeasured}"
        assert measured.status != unmeasured.status, (
            f"both rows report {measured.status!r}: the status column cannot distinguish a row "
            "that measured 0.40 from one that measured nothing"
        )

    def test_the_status_names_the_cause_and_is_not_a_skip(self, tmp_path) -> None:
        """The new status must be actionable and must not read as a host-capability skip.

        ``skip (...)`` means the driver refused to run; this driver ran and
        exited 0, so classing it as a skip would send the reader to check for a
        missing backend instead of a parse failure.
        """
        row = _matrix_row(tmp_path, "print('[setup] world ready')")

        assert not row.status.startswith("skip ("), (
            f"a driver that ran to completion must not be reported as a skip: {row.status!r}"
        )
        assert not row.status.startswith("unavailable ("), (
            f"the driver file exists and ran, so it is not unavailable: {row.status!r}"
        )
        assert "result line" in row.status, (
            f"the status must name what could not be read so the reader knows where to look: {row.status!r}"
        )


class TestTheOtherRowVerdictsAreUnchanged:
    """Controls: only the unreadable-result row moves."""

    @pytest.mark.parametrize("backend", ["mujoco", "isaac"])
    def test_a_real_result_line_still_reports_ok_with_its_numbers(self, tmp_path, backend: str) -> None:
        """The happy path keeps the ``ok`` spelling a downstream table reader greps for.

        The line is built by the driver's own formatter, so this stays tied to
        the single source of the format rather than restating it.
        """
        row = _matrix_row(tmp_path, f"print({_real_result_line(backend=backend)!r})")

        assert row.status == "ok", f"a parseable result line must still be 'ok': {row}"
        assert row.success_rate == pytest.approx(0.4)
        assert row.wall_time == pytest.approx(12.5)

    def test_a_nonzero_exit_is_still_a_skip_naming_the_reason(self, tmp_path) -> None:
        """A driver that refuses to run keeps its exit-code-derived skip reason."""
        row = _matrix_row(
            tmp_path,
            "import sys; print('Isaac Sim is not available on this host', file=sys.stderr); raise SystemExit(1)",
        )

        assert row.status.startswith("skip ("), f"a non-zero exit is a skip: {row.status!r}"
        assert "not available" in row.status, f"the skip must keep the driver's own reason: {row.status!r}"
        assert row.success_rate is None and row.wall_time is None

    def test_the_matrix_still_exits_zero_so_the_table_stays_the_verdict(self) -> None:
        """Boundary: the row's status is the report, not the process exit code.

        ``main`` returns 0 after printing the table however the rows turned out -
        an ``unavailable`` Isaac row on a CPU-only host is the documented normal
        case. Turning an ``incomplete`` row into a non-zero exit is a separate
        contract for the matrix's callers and is deliberately not taken here;
        this test fails if that is changed without deciding it.
        """
        source = (_EXAMPLES_LIBERO / "libero_backend_matrix.py").read_text(encoding="utf-8")
        main = next(
            node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        returns = [ast.unparse(node.value) for node in ast.walk(main) if isinstance(node, ast.Return) and node.value]

        assert returns == ["0"], f"main's exit code is not the row verdict; it returns only 0: {returns}"

    def test_the_parser_reports_both_numbers_or_neither(self) -> None:
        """The guard reads ``sr is None or wt is None``; this is why that is safe.

        ``_parse_driver_output`` yields both floats on a match and ``(None, None)``
        otherwise, so a half-read row cannot occur and ``or`` cannot differ from
        ``and`` today. Pinning the invariant is what keeps the defensive spelling
        correct: if the parser ever grows a path that reports one number without
        the other, this goes red and the guard's ``or`` is what still catches it.
        """
        matrix = _load_example("libero_backend_matrix.py")
        stdouts = (
            "",
            "[setup] world ready",
            "policy=mock task=t success_rate=NaN wall_time=abcs",
            "policy=mock  task=t  success_rate=0.40  wall_time=1.5s  videos=x  backend=mujoco",
            "policy=mock  task=t  success_rate=0.40  videos=x  backend=mujoco",
            "policy=mock  task=t  wall_time=1.5s  videos=x  backend=mujoco",
        )
        for stdout in stdouts:
            sr, wt = matrix._parse_driver_output(stdout)
            assert (sr is None) == (wt is None), (
                f"_parse_driver_output reported one number without the other for {stdout!r}: {(sr, wt)}"
            )


class TestTheDriverHasNoZeroExitPathThatSkipsItsResultLine:
    """Why the new status means a parse failure and not an early exit.

    ``_parse_driver_output``'s docstring rests on this: a driver that gives up
    before the eval starts exits non-zero and is recorded by exit code, so a
    zero exit with no readable line means the driver's format and this parser
    disagree. If ``run.py`` ever grows a path that returns normally without
    printing, that reasoning stops holding and this goes red.
    """

    @staticmethod
    def _run_py_function(name: str) -> ast.FunctionDef:
        source = (_EXAMPLES_LIBERO / "run.py").read_text(encoding="utf-8")
        return next(
            node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.FunctionDef) and node.name == name
        )

    def test_main_has_no_early_return(self) -> None:
        """``main`` runs the eval unconditionally - it has no return at all."""
        main = self._run_py_function("main")
        returns = [ast.unparse(node) for node in ast.walk(main) if isinstance(node, ast.Return)]

        assert returns == [], (
            f"run.py main() can return before printing its result line, so a zero exit no longer "
            f"implies a completed eval: {returns}"
        )

    def test_the_reporter_raises_rather_than_returning_on_a_failed_eval(self) -> None:
        """``_evaluate_and_report`` refuses loudly; it never returns a silent failure."""
        report = self._run_py_function("_evaluate_and_report")
        raises = [ast.unparse(node.exc) for node in ast.walk(report) if isinstance(node, ast.Raise) and node.exc]
        returns = [node for node in ast.walk(report) if isinstance(node, ast.Return) and node.value]

        assert raises, "premise: _evaluate_and_report is the function that refuses a failed eval"
        assert any("evaluate_benchmark failed" in text for text in raises), (
            f"a failed evaluate_benchmark must raise, not return: {raises}"
        )
        assert returns == [], f"_evaluate_and_report reports by printing and raising, not by a value: {returns}"

    def test_the_scan_reaches_both_functions(self) -> None:
        """Non-vacuity: the two functions above are found by name, not silently missed."""
        for name in ("main", "_evaluate_and_report"):
            assert self._run_py_function(name).body, f"{name} was not found with a body in run.py"
