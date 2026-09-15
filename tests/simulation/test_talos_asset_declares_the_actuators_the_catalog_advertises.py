"""The ``talos`` entry names files its pack ships, carrying the DOF it advertises.

Menagerie's ``pal_talos`` pack splits the robot across four documents.
``talos.xml`` is a shared include holding the bodies alone - it declares no
``<actuator>`` at all. The drivable models are ``talos_position.xml`` (32
``<position>`` actuators) and ``talos_motor.xml`` (32 ``<motor>``), each with a
matching ``scene_position.xml`` / ``scene_motor.xml``. The pack ships no
``scene.xml``.

An entry naming the bare include still compiles, which is why this hid: the
robot loads, renders, seats on the terrain and reports its 45 joints, so
``add_robot`` returns success and goes on to advertise ``run_policy`` on it. It
merely has ``nu == 0``, so :meth:`robot_action_keys` is empty and every command
is dropped for want of a driving actuator - a robot the catalog describes as a
32-DOF humanoid that nothing can move. The entry's own ``description`` states
the count that settles which document it meant, so
:meth:`test_the_actuator_count_is_the_dof_the_description_advertises` re-derives
it rather than restating a number here.

The asset is not downloaded. Fetching it would clone a third-party repository
during a test run and turn a host with no network into a failure rather than a
skip, so the pack is located through the search paths and every cell skips when
it is absent - which is the case on a clean checkout. The gate is the pack
*directory*, deliberately not the declared file: gating on the file would turn
an entry naming a document the pack does not ship into a silent skip, which is
half of what went wrong here.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ROBOTS_JSON = REPO_ROOT / "strands_robots" / "registry" / "robots.json"

#: The document in the pack that holds the bodies and no actuator. An entry
#: naming this compiles to an unactuated robot, so the entry must not name it.
SHARED_INCLUDE = "talos.xml"


def _entry() -> dict:
    return json.loads(ROBOTS_JSON.read_text(encoding="utf-8"))["robots"]["talos"]


def _pack() -> Path:
    """Return the downloaded ``pal_talos`` directory, or skip the cell."""
    from strands_robots.utils import get_search_paths

    asset_dir = str(_entry()["asset"]["dir"])
    present = next((candidate for root in get_search_paths() if (candidate := Path(root) / asset_dir).is_dir()), None)
    if present is None:
        pytest.skip(f"{asset_dir} is not downloaded, so the compiled shape cannot be read")
    return present


def _model(file_name: str):
    mujoco = pytest.importorskip("mujoco")
    path = _pack() / file_name
    assert path.exists(), (
        f"the pack does not ship {file_name}; it holds {sorted(p.name for p in _pack().glob('*.xml'))}"
    )
    return mujoco, mujoco.MjModel.from_xml_path(str(path))


def _joint_names(mujoco, model) -> list[str]:
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]


class TestTheEntryNamesDocumentsThePackShips:
    """A declared file the pack does not hold resolves to something else."""

    @pytest.mark.parametrize("key", ["model_xml", "scene_xml"])
    def test_the_declared_document_is_present_in_the_pack(self, key: str) -> None:
        pack = _pack()
        declared = str(_entry()["asset"][key])
        assert (pack / declared).exists(), (
            f"talos declares {key}={declared!r}, which {pack} does not ship. "
            f"Resolution then falls through to another document of the pack, so the robot that loads "
            f"is not the one the entry names. The pack holds {sorted(p.name for p in pack.glob('*.xml'))}."
        )

    def test_the_entry_does_not_name_the_actuatorless_shared_include(self) -> None:
        """The include is a real document, so naming it loads an inert robot."""
        _, shared = _model(SHARED_INCLUDE)
        assert shared.nu == 0, f"premise: {SHARED_INCLUDE} is the bodies-only include, so it declares no actuator"
        assert shared.njnt > 0, f"premise: {SHARED_INCLUDE} still compiles to a jointed robot, which is why it loads"
        assert str(_entry()["asset"]["model_xml"]) != SHARED_INCLUDE


class TestTheDeclaredDocumentsCarryTheAdvertisedActuators:
    """Both entry points must reach a robot a caller can command."""

    def test_the_actuator_count_is_the_dof_the_description_advertises(self) -> None:
        """``(32-DOF)`` in the description is the model's ``nu``."""
        described = re.search(r"\((\d+)-DOF\)", str(_entry()["description"]))
        assert described is not None, "premise: the entry states its DOF count, which fixes the document it meant"
        _, model = _model(str(_entry()["asset"]["model_xml"]))
        assert model.nu == int(described.group(1)), (
            "the declared model has no actuator for the DOF the catalog advertises, so robot_action_keys is "
            "empty and every command is dropped for want of a driving actuator"
        )

    def test_the_declared_scene_carries_the_same_actuators_as_the_model(self) -> None:
        """So ``prefer_scene`` resolution reaches an equally drivable robot."""
        _, model = _model(str(_entry()["asset"]["model_xml"]))
        _, scene = _model(str(_entry()["asset"]["scene_xml"]))
        assert (scene.nu, scene.njnt) == (model.nu, model.njnt)

    def test_the_declared_actuators_take_a_joint_position_target(self) -> None:
        """The pack's other 32-actuator variant would read the same value as torque.

        ``talos_motor.xml`` also declares 32 actuators, so the count alone does
        not settle which document the entry should name. It declares ``<motor>``,
        which MODULE strands_robots.simulation.mujoco.scene_ops classifies as
        ``torque_only_actuation`` (``biastype == mjBIAS_NONE``): a caller sending
        a joint angle there commands that many newton-metres instead. The
        position variant carries the affine bias that makes the commanded value
        a target the servo holds.
        """
        mujoco, model = _model(str(_entry()["asset"]["model_xml"]))
        biastypes = {int(kind) for kind in model.actuator_biastype}
        assert biastypes == {int(mujoco.mjtBias.mjBIAS_AFFINE)}, (
            "the declared model's actuators carry no position bias, so a commanded joint angle is applied "
            f"as a raw force rather than held as a target (biastypes: {sorted(biastypes)})"
        )

    def test_the_declared_joint_count_is_the_models_joint_total(self) -> None:
        """``joints: 45`` counts every joint, driven or not, base included."""
        _, model = _model(str(_entry()["asset"]["model_xml"]))
        assert model.njnt == _entry()["joints"]

    def test_only_the_base_and_the_passive_gripper_linkage_go_undriven(self) -> None:
        """Every other joint has an actuator, so the 32 cover the whole robot.

        Named rather than counted: the count alone would still pass if an
        upstream retune moved an actuator from an arm to a fingertip.
        """
        mujoco, model = _model(str(_entry()["asset"]["model_xml"]))
        names = _joint_names(mujoco, model)
        driven = {names[int(model.actuator_trnid[i, 0])] for i in range(model.nu)}
        free = {
            name
            for name, kind in zip(names, model.jnt_type, strict=True)
            if int(kind) == int(mujoco.mjtJoint.mjJNT_FREE)
        }
        undriven = [name for name in names if name not in driven]
        assert free == {"reference"}, f"premise: the pack spells talos' floating base 'reference', got {sorted(free)}"
        unexplained = [name for name in undriven if name not in free and "gripper" not in name]
        assert not unexplained, (
            f"joints with no driving actuator that are neither the base nor gripper linkage: {unexplained}"
        )
