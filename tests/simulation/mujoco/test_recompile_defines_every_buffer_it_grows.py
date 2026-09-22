"""A scene rebuild trusts the compiler with nothing a reset would write.

``scene_ops._recompile_preserving_state`` hands ``spec.recompile(model, data)``
the live state and gets back a fresh ``MjData`` with the old values transferred
positionally. Which entries of that data the compiler leaves undefined is a
property of the MuJoCo build: through 3.13 it was the new tail of ``qpos``,
``qvel``, ``ctrl`` and ``act``, with everything else zeroed; 3.14.0 rewrote the
transfer to carry ``qacc_warmstart``, ``eq_active``, both applied-force buffers
and mocap poses by element, and when the spec was grown by attaching a sub-spec
that carries a ``<keyframe>`` - every robot description in the registry - the
slices of the attached elements come back as heap garbage in every one of those
buffers. Measured on 3.14.0 over six rebuilds: ``nan``, ``6.98e-316``,
``1.12e+219``; the pre-existing slices intact every time.

A non-finite entry there is not a nonsense number in one joint. ``mj_checkCtrl``
disables actuation for the whole model on one non-finite ``ctrl``, and the
solver starts from ``qacc_warmstart``, so one non-finite entry on a NEW dof
poisons the accelerations of every dof on the first step after the rebuild -
``Nan, Inf or huge value in QACC at DOF 0``, reported against an arm that was
parked and healthy. Whether that happens depends on what the freed memory held,
so the parked-arm cell in ``test_add_robot_preserves_scene_state.py`` failed on
some runs and passed on others against the same tree.

These cells remove the luck. A proxy around the live spec runs the real
``recompile`` and then writes into the returned data exactly the garbage the
3.14.0 transfer can leave - NaN in every new slice, an inactive byte for a new
equality declared active - so the rebuild is measured against the worst case on
every build, and a buffer the rebuild forgets to define fails here whether or
not the compiler happened to zero it. The behavioural half stays where it was:
the parked arm holding its pose after a second robot is added is what the
definitions buy, and that cell is the one that reported the defect.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

mj = pytest.importorskip("mujoco")

from strands_robots.simulation import Simulation  # noqa: E402

# A two-hinge arm with position servos, a filter actuator (so ``act`` is
# non-empty), an equality declared active, a mocap body and a keyframe - the
# keyframe being the ingredient that makes the 3.14.0 transfer hand back garbage.
_ARM_XML = """<mujoco model="probe_arm">
  <compiler angle="radian"/>
  <worldbody>
    <body name="link" pos="0 0 0.2">
      <joint name="pan" type="hinge" axis="0 0 1" range="-2 2" limited="true" damping="4"/>
      <geom type="capsule" fromto="0 0 0 0.25 0 0" size="0.03"/>
      <body name="fore" pos="0.25 0 0">
        <joint name="lift" type="hinge" axis="0 1 0" range="-2 2" limited="true" damping="4"/>
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.025"/>
      </body>
    </body>
    <body name="marker" mocap="true" pos="1 1 1">
      <geom type="sphere" size="0.02" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <actuator>
    <position name="pan_act" joint="pan" kp="60"/>
    <general name="lift_act" joint="lift" dyntype="filter" dynprm="0.1" gainprm="60" biasprm="0 -60"/>
  </actuator>
  <equality>
    <joint name="couple" joint1="pan" joint2="lift" active="true" polycoef="0 0.01 0 0 0"/>
  </equality>
  <keyframe>
    <key name="stow" qpos="1.2 -1.1"/>
  </keyframe>
</mujoco>"""


class _PoisoningSpec:
    """The live spec, with a ``recompile`` that returns what 3.14.0 can return.

    Every other attribute is the real spec's, so the attach, the snapshot copy
    and the XML sync all run against the real object; only the data handed back
    from the compile is rewritten, and only in the slices of the elements the
    rebuild added.
    """

    def __init__(self, spec, old_sizes: dict[str, int]):
        self._spec = spec
        self._old = old_sizes

    def __getattr__(self, name):
        return getattr(self._spec, name)

    def recompile(self, model, data):
        new_model, new_data = self._spec.recompile(model, data)
        old = self._old
        new_data.qpos[old["nq"] :] = math.nan
        new_data.qvel[old["nv"] :] = math.nan
        new_data.qacc_warmstart[old["nv"] :] = math.nan
        new_data.qfrc_applied[old["nv"] :] = math.nan
        new_data.ctrl[old["nu"] :] = math.nan
        new_data.act[old["na"] :] = math.nan
        new_data.xfrc_applied[old["nbody"] :] = math.nan
        new_data.eq_active[old["neq"] :] = 0
        new_data.mocap_pos[old["nmocap"] :] = math.nan
        new_data.mocap_quat[old["nmocap"] :] = math.nan
        return new_model, new_data


def _sizes(model) -> dict[str, int]:
    return {k: int(getattr(model, k)) for k in ("nq", "nv", "nu", "na", "nbody", "neq", "nmocap")}


@pytest.fixture
def arm_path(tmp_path):
    path = tmp_path / "arm.xml"
    path.write_text(_ARM_XML, encoding="utf-8")
    return str(path)


@pytest.fixture
def sim(arm_path):
    engine = Simulation(backend="mujoco", tool_name="recompile_defines_sim", mesh=False)
    assert engine.create_world()["status"] == "success"
    assert engine.add_robot(name="a", urdf_path=arm_path)["status"] == "success"
    try:
        yield engine
    finally:
        engine.cleanup()


def _grow_with_poisoned_transfer(sim, arm_path) -> tuple[dict[str, int], dict[str, np.ndarray]]:
    """Add a second robot through a transfer that garbles every new slice.

    Returns the old sizes and the latched forces the rebuild had to carry.
    """
    world = sim._world
    model, data = world._model, world._data
    old = _sizes(model)
    # Latch one wrench and one joint force on the first robot, so the carry
    # of the pre-existing rows is measured beside the definition of the new.
    link = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "a/link")
    lift = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "a/lift")
    data.xfrc_applied[link] = [1.0, 2.0, 3.0, 0.0, 0.0, 0.5]
    data.qfrc_applied[int(model.jnt_dofadr[lift])] = 0.25
    latched = {"xfrc": data.xfrc_applied.copy(), "qfrc": data.qfrc_applied.copy()}

    world._backend_state["spec"] = _PoisoningSpec(world._backend_state["spec"], old)
    try:
        assert sim.add_robot(name="b", urdf_path=arm_path)["status"] == "success"
    finally:
        world._backend_state["spec"] = world._backend_state["spec"]._spec
    return old, latched


class TestEveryGrownBufferIsDefined:
    def test_no_buffer_the_rebuild_grows_holds_a_non_finite_entry(self, sim, arm_path):
        _grow_with_poisoned_transfer(sim, arm_path)
        data = sim._world._data
        for name in (
            "qpos",
            "qvel",
            "qacc_warmstart",
            "qfrc_applied",
            "ctrl",
            "act",
            "xfrc_applied",
            "mocap_pos",
            "mocap_quat",
        ):
            values = np.asarray(getattr(data, name), dtype=float)
            assert np.all(np.isfinite(values)), (name, values)

    def test_the_new_slices_hold_what_a_reset_writes(self, sim, arm_path):
        old, _ = _grow_with_poisoned_transfer(sim, arm_path)
        model, data = sim._world._model, sim._world._data
        assert model.nq > old["nq"] and model.neq > old["neq"] and model.nmocap > old["nmocap"]

        assert np.array_equal(data.qpos[old["nq"] :], model.qpos0[old["nq"] :])
        assert not np.any(data.qvel[old["nv"] :])
        assert not np.any(data.qacc_warmstart[old["nv"] :])
        assert not np.any(data.ctrl[old["nu"] :])
        assert not np.any(data.act[old["na"] :])
        assert np.array_equal(data.eq_active[old["neq"] :], model.eq_active0[old["neq"] :])
        assert int(model.eq_active0[old["neq"]]) == 1  # the declared activation is the non-trivial one
        for bid in range(model.nbody):
            mid = int(model.body_mocapid[bid])
            if mid >= old["nmocap"]:
                assert np.array_equal(data.mocap_pos[mid], model.body_pos[bid])
                assert np.array_equal(data.mocap_quat[mid], model.body_quat[bid])

    def test_the_applied_forces_are_exactly_what_was_latched(self, sim, arm_path):
        old, latched = _grow_with_poisoned_transfer(sim, arm_path)
        model, data = sim._world._model, sim._world._data
        # Pre-existing rows carried by name; every other row zero - including the
        # rows of the elements the rebuild added, which the transfer garbled.
        link = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "a/link")
        lift = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "a/lift")
        expected_xfrc = np.zeros_like(data.xfrc_applied)
        expected_xfrc[link] = latched["xfrc"][mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "a/link")]
        expected_qfrc = np.zeros_like(data.qfrc_applied)
        expected_qfrc[int(model.jnt_dofadr[lift])] = 0.25
        assert np.array_equal(data.xfrc_applied, expected_xfrc)
        assert np.array_equal(data.qfrc_applied, expected_qfrc)

    def test_the_first_step_after_the_rebuild_is_finite(self, sim, arm_path):
        _grow_with_poisoned_transfer(sim, arm_path)
        sim.step(1)
        data = sim._world._data
        assert np.all(np.isfinite(data.qacc)), data.qacc
        assert np.all(np.isfinite(data.qpos)), data.qpos


class TestThePoisonIsNotVacuous:
    def test_the_proxy_garbles_every_slice_the_rebuild_must_define(self, sim, arm_path):
        """The proxy reaches what it claims to, so a green cell above measured something."""
        world = sim._world
        old = _sizes(world._model)
        real = world._backend_state["spec"]
        proxy = _PoisoningSpec(real, old)
        # Reproduce the attach the rebuild performs, then read the raw transfer.
        sub = mj.MjSpec.from_file(arm_path)
        frame = real.worldbody.add_frame()
        real.attach(sub, prefix="probe/", frame=frame)
        new_model, new_data = proxy.recompile(world._model, world._data)
        assert new_model.nq > old["nq"] and new_model.neq > old["neq"] and new_model.nmocap > old["nmocap"]
        for name in (
            "qpos",
            "qvel",
            "qacc_warmstart",
            "qfrc_applied",
            "ctrl",
            "act",
            "xfrc_applied",
            "mocap_pos",
            "mocap_quat",
        ):
            assert not np.all(np.isfinite(np.asarray(getattr(new_data, name), dtype=float))), name
        assert not np.any(new_data.eq_active[old["neq"] :])
