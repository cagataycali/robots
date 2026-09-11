"""``get_robot_state`` answers with a configuration the robot was actually in.

The readback assembles one answer out of several mjData arrays: each joint's
``qpos``/``qvel``, a floating base's pose and twist, and the world position of
the frame ``move_to`` drives (``site_xpos``/``xpos``). A concurrent ``mj_step``
- from a ``PolicyRunner`` worker, from the ``step()`` loop, or from the camera
recorder daemon, none of which the blanket dispatch lock covers - lands between
two of those reads, and the returned "state" is then a splice of two physics
steps: joint angles that never coexisted, or joints from one step beside an
end-effector position from another. The second one is the damaging shape,
because the documented use of that field is to offset a ``move_to`` target from
it, so the target is computed in a configuration the arm was not in.

With a writer alternating an SO-101 between two coherent configurations under
the lock (so every state the robot is EVER in is entirely one or the other),
4,323 of 45,414 unserialised samples returned a mixed joint vector and 2,958
reported joints from one configuration beside the end-effector position of the
other; serialised, 0 of 40,791 did.

:meth:`~strands_robots.simulation.mujoco.rendering.RenderingMixin.render`,
``render_depth``, ``get_frame``, ``get_observation``, ``get_body_state`` and the
joint writers already serialise their mjData access - of the public methods that
touch it, this one and ``apply_force``'s default-point read were the two that
did not.

Two things are pinned here:

* **Behaviour** - ``get_robot_state`` cannot complete while another thread holds
  ``self._lock`` (an ``Event`` that times out, not a wall-clock measurement: a
  reader blocked on the lock cannot signal completion while the writer is inside
  its critical section, whatever the machine's load), and no sample it returns
  mixes two configurations.
* **The root cause** - no public method of the MuJoCo backend touches mjData
  outside a ``with self._lock`` block, derived from the package's own AST rather
  than a hand-written list, so a method added later is graded without an edit
  here. This subsumes the renderer-scoped relation in
  ``test_frame_readers_serialize_the_mjdata_read``, which additionally pins the
  rendering consequence against a stand-in renderer. ``apply_force`` is graded
  by this relation alone: its critical section already existed and the defect was
  only which side of it the CoM read fell on, so there is no ordering an
  ``Event`` could distinguish.
"""

from __future__ import annotations

import ast
import inspect
import threading
import time
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco import simulation as simulation_mod  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

# Long enough that an unserialised reader (sub-millisecond here) finishes inside
# it by a wide margin, and irrelevant to the serialised verdict: a reader waiting
# on the lock can never signal completion while the writer holds it.
_READER_BUDGET_S = 1.0

# The mjData arrays the backend's readers touch. Every one of them is rewritten
# by mj_step, so reading any of them off-lock races a stepping thread.
_MJDATA_ARRAYS = (
    "qpos",
    "qvel",
    "qacc",
    "ctrl",
    "act",
    "xpos",
    "xquat",
    "xmat",
    "xipos",
    "site_xpos",
    "site_xmat",
    "contact",
    "ncon",
    "sensordata",
    "qfrc_applied",
    "xfrc_applied",
)


@pytest.fixture
def sim():
    s = Simulation(tool_name="state_reader_lock", mesh=False)
    s.create_world()
    assert s.add_robot("so101")["status"] == "success"
    yield s
    s.cleanup(policy_stop_timeout=0.5)


def _state(sim: Simulation) -> dict[str, Any]:
    result = sim.get_robot_state()
    assert result["status"] == "success", result
    return next(block["json"] for block in result["content"] if "json" in block)


class TestGetRobotStateAnswersWithAStateTheRobotWasIn:
    """One answer, one configuration - never a splice of two physics steps."""

    def test_it_cannot_complete_while_a_writer_holds_the_lock(self, sim):
        done = threading.Event()
        box: dict[str, Any] = {}

        def run() -> None:
            try:
                box["state"] = _state(sim)
            finally:
                done.set()

        with sim._lock:
            reader = threading.Thread(target=run, daemon=True, name="state-reader")
            reader.start()
            completed_under_the_lock = done.wait(_READER_BUDGET_S)
        reader.join(_READER_BUDGET_S * 5)

        assert "state" in box, "premise: the reader ran to completion once the lock was free"
        assert not completed_under_the_lock, (
            "get_robot_state finished while another thread held sim._lock, so it read "
            "qpos/qvel/site_xpos a concurrent mj_step was free to rewrite between reads"
        )

    def test_it_still_answers_once_the_writer_releases(self, sim):
        """Serialising must not deadlock or drop the readback."""
        state = _state(sim)
        assert state["state"], "the per-joint readback still arrives"
        assert "position" in state["end_effector"], "the end-effector frame still arrives"

    def test_no_answer_mixes_two_configurations(self, sim):
        """A writer flips between two coherent poses; no sample may straddle them."""
        joints = list(_state(sim)["state"])
        poses = ({name: 0.0 for name in joints}, {name: 0.3 for name in joints})
        reference = []
        for pose in poses:
            assert sim.set_joint_positions(pose)["status"] == "success"
            snapshot = _state(sim)
            reference.append(
                (
                    {name: entry["position"] for name, entry in snapshot["state"].items()},
                    snapshot["end_effector"]["position"],
                )
            )

        stop = threading.Event()

        def writer() -> None:
            flip = False
            while not stop.is_set():
                sim.set_joint_positions(poses[flip])
                flip = not flip

        thread = threading.Thread(target=writer, daemon=True, name="state-writer")
        thread.start()
        try:
            seen: set[int] = set()
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                sample = _state(sim)
                tags = {
                    next(
                        (i for i, (q, _) in enumerate(reference) if abs(entry["position"] - q[name]) < 0.05),
                        -1,
                    )
                    for name, entry in sample["state"].items()
                }
                assert tags != {-1}, f"premise: every joint reads as one of the two poses, got {sample['state']}"
                assert len(tags) == 1, (
                    f"get_robot_state spliced two physics steps into one joint vector: {sample['state']} "
                    f"is neither {reference[0][0]} nor {reference[1][0]}"
                )
                config = tags.pop()
                ee = sample["end_effector"]["position"]
                near = [
                    i
                    for i, (_, ref_ee) in enumerate(reference)
                    if max(abs(a - b) for a, b in zip(ee, ref_ee, strict=True)) < 0.01
                ]
                assert near == [config], (
                    f"get_robot_state reported pose {config}'s joints beside end-effector position {ee}, "
                    f"which belongs to pose {near} - a move_to target offset from it would be computed "
                    "in a configuration the arm was not in"
                )
                seen.add(config)
        finally:
            stop.set()
            thread.join(timeout=5.0)

        assert seen == {0, 1}, f"premise: the writer really moved the arm between both poses, saw {seen}"


def _under_a_lock(fn: ast.AST) -> set[int]:
    """``id()`` of every node lexically inside a ``with self._lock`` block."""
    under: set[int] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.With) and any("_lock" in ast.unparse(item.context_expr) for item in node.items):
            under.update(id(inner) for inner in ast.walk(node))
    return under


def _mjdata_touches() -> dict[str, list[tuple[int, str, bool]]]:
    """Map ``<module>.<public method>`` -> its mjData touches and whether each is locked.

    Derived from the MuJoCo backend package's own AST. Private helpers are
    excluded: each is reached through a public facade that holds the lock (and
    several document that requirement), which the public entries pin.
    """
    package = Path(str(inspect.getsourcefile(simulation_mod))).parent
    out: dict[str, list[tuple[int, str, bool]]] = {}
    for path in sorted(package.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
            for fn in (m for m in cls.body if isinstance(m, ast.FunctionDef | ast.AsyncFunctionDef)):
                if fn.name.startswith("_"):
                    continue
                under = _under_a_lock(fn)
                touches = [
                    (node.lineno, ast.unparse(node), id(node) in under)
                    for node in ast.walk(fn)
                    if isinstance(node, ast.Attribute)
                    and node.attr in _MJDATA_ARRAYS
                    and ast.unparse(node.value).endswith(("data", "_data"))
                ]
                if touches:
                    out[f"{path.name}:{fn.name}"] = touches
    return out


class TestEveryPublicMjDataTouchIsSerialised:
    """The root cause, derived from the tree so a later method is graded too."""

    def test_no_public_method_touches_mjdata_outside_the_lock(self):
        offenders = {
            name: [(line, src) for line, src, locked in touches if not locked]
            for name, touches in _mjdata_touches().items()
        }
        offenders = {name: rows for name, rows in offenders.items() if rows}
        assert not offenders, (
            "these public methods read or write mjData outside 'with self._lock', so a "
            f"concurrent mj_step can tear the value they answer with: {offenders}"
        )

    def test_the_derived_population_covers_the_known_readers(self):
        """Non-vacuity: an empty or shrunken population would pass the rule above."""
        population = _mjdata_touches()
        assert {"simulation.py:get_robot_state", "physics.py:apply_force", "physics.py:get_body_state"} <= set(
            population
        ), sorted(population)
        assert len(population) >= 10, sorted(population)
