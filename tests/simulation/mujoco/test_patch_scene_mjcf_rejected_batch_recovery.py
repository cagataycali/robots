"""Contract tests for what a *rejected* ``patch_scene_mjcf`` batch leaves behind.

``patch_scene_mjcf`` applies a batch of structured ops to the live ``MjSpec``
atomically: the batch mutates the live spec and a snapshot taken beforehand is
put back if any op raises, so the caller is left with a usable world rather than
a half-patched one.

``tests/simulation/mujoco/test_patch_scene_mjcf.py`` pins that the *compiled
model* is unchanged after a rejected batch. That assertion is nearly free -- a
rejected batch never reaches the recompile -- so the property that actually
makes the world usable again is unpinned by it: the **live spec** must be the
clean snapshot. Nothing else undoes the ops that already ran, and the spec is
what the next mutation recompiles from, so a batch that only restored the
compiled model would hand the next ``add_object`` an orphan body.

This module pins the three things a caller can observe after a rejection:

1. The message names **which** op failed, so a batch is debuggable.
2. The live spec carries none of the ops that already ran, and a subsequent
   mutation therefore recompiles cleanly.
3. The world is still steppable.
"""

from __future__ import annotations

import ast
import pathlib
from typing import Any

import pytest

pytest.importorskip("mujoco")


from strands_robots.simulation.mujoco import scene_ops  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

#: A batch whose first op is valid and whose second is not, so a rejection has
#: something half-applied to roll back.
REJECTED_BATCH: list[dict[str, Any]] = [
    {"op": "add_body", "name": "doomed", "pos": [0, 0, 1]},
    {"op": "totally_made_up", "name": "whatever"},
]


def _functions_calling(symbol: str) -> list[str]:
    """Names of the ``scene_ops`` functions whose body mentions ``symbol``."""
    source = pathlib.Path(scene_ops.__file__).read_text(encoding="utf-8")
    return [
        node.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and symbol in (ast.get_source_segment(source, node) or "")
    ]


@pytest.fixture
def sim():
    s = Simulation(tool_name="devx_patch_rejected_batch", mesh=False)
    try:
        yield s
    finally:
        s.cleanup(policy_stop_timeout=0.5)


class TestRejectedBatchIsDebuggable:
    def test_the_message_names_which_op_failed(self, sim: Simulation) -> None:
        """A batch is only debuggable if the rejection says which op broke it."""
        sim.create_world()

        result = sim.patch_scene_mjcf(list(REJECTED_BATCH))

        assert result["status"] == "error"
        message = result["content"][0]["text"].lower()
        assert "patch op #2" in message, f"the failing op is not identified: {message}"


class TestRejectedBatchRestoresTheLiveSpec:
    """The spec -- not just the compiled model -- is what must come back clean.

    A rejected batch never recompiles, so ``_model`` is trivially unchanged. The
    spec is the mutable object the ops wrote to and the one the next mutation
    recompiles from, so it is the only place a half-applied body can survive.
    """

    def test_the_live_spec_carries_no_half_applied_body(self, sim: Simulation) -> None:
        sim.create_world()
        assert sim._world is not None
        spec_before = sorted(b.name for b in sim._world._backend_state["spec"].bodies)

        result = sim.patch_scene_mjcf(list(REJECTED_BATCH))
        assert result["status"] == "error"

        assert sim._world is not None
        spec_after = sorted(b.name for b in sim._world._backend_state["spec"].bodies)
        assert spec_after == spec_before, "op #1's body survived the rejection on the live spec"

    def test_a_later_mutation_recompiles_without_the_orphan(self, sim: Simulation) -> None:
        """The consequence the restore exists for: the next mutation stays clean."""
        sim.create_world()
        assert sim._world is not None
        mj = sim._mj
        nbody_before = sim._world._model.nbody

        assert sim.patch_scene_mjcf(list(REJECTED_BATCH))["status"] == "error"

        # This recompiles from the live spec, so a surviving op #1 would be
        # baked into the model here rather than at the rejected batch.
        added = sim.add_object(name="crate", shape="box", size=[0.1, 0.1, 0.1], position=[0.4, 0, 0.05])
        assert added["status"] == "success", added

        assert sim._world is not None
        model = sim._world._model
        assert mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "crate") >= 0, "the later mutation did not land"
        assert mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "doomed") == -1, (
            "the rejected batch's body was recompiled into the model by a later mutation"
        )
        assert model.nbody == nbody_before + 1


class TestRejectedBatchLeavesTheWorldSteppable:
    def test_the_world_still_steps(self, sim: Simulation) -> None:
        sim.create_world()
        assert sim._world is not None
        nbody_before = sim._world._model.nbody

        assert sim.patch_scene_mjcf(list(REJECTED_BATCH))["status"] == "error"

        assert sim._world is not None
        assert sim._world._model.nbody == nbody_before
        sim.step(1)


class TestTheRestoreIsAnObjectSwap:
    """Pins the mechanism, because this module used to test a different one.

    The snapshot moved from a ``spec.to_xml()`` round trip to ``MjSpec.copy``
    (see :func:`strands_robots.simulation.mujoco.scene_ops._snapshot_spec`,
    whose docstring records why the round trip was abandoned). The rollback is
    now a plain reassignment of the cached spec, so ``SpecBuilder.from_mjcf_string``
    is unreachable from the patch path -- a test that stubs it to force a
    "restore itself failed" branch intercepts nothing and passes on the ordinary
    rollback instead.
    """

    def test_the_patch_path_does_not_rebuild_the_spec_from_mjcf(self) -> None:
        callers = _functions_calling("from_mjcf_string")
        assert callers, "from_mjcf_string vanished from scene_ops; this guard needs re-reading"
        assert "patch_scene_mjcf" not in callers, (
            f"patch_scene_mjcf now rebuilds the spec from MJCF (callers: {callers}); the rollback "
            "is no longer a plain object swap and this module's contract needs re-reading"
        )
