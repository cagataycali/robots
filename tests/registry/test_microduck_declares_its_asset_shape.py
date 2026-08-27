"""The ``microduck`` entry describes the robot it points at.

Pollen's Microduck is a 14-DOF biped: fourteen hinges driven by fourteen
position actuators, plus a floating base. Three registry fields carry that shape
to a reader who cannot compile the asset - ``joints``, the ``description`` and
the ``asset`` block - and two discovery surfaces report them verbatim
(:func:`~strands_robots.registry.get_robot` returns the entry, and
:func:`~strands_robots.registry.list_robots` prints ``joints`` in the ``Joints``
column an agent reads to size an action vector).

``joints`` is 15, not 14, because the registry counts MuJoCo's ``njnt`` - the
floating base included. ``docs/robots/arms.md`` states that convention ("Joint
counts include any free joints"); ``asimov_v0`` declares 15 for the same
one-free-plus-fourteen-hinge shape; and of the sixteen humanoids whose asset
compiles, twelve declare their ``njnt`` and none declares its actuator count.
The *hardware* figure - fourteen XL330 servos - is what the description carries,
matching ``op3`` (21 against "20-DOF") and ``unitree_h1`` (20 against "19-DOF").

What ``joints`` means registry-wide is deliberately unsettled - see
``tests/registry/test_asset_family_joint_counts.py``, which grades only in-family
agreement because 22 of the 50 compilable entries declare a figure that is
neither their ``njnt`` nor their movable-joint count. Microduck has no
same-model sibling, so that guard says nothing about it; this file states the
convention the entry was written against so the figure is not a bare number.

No home pose is copied into ``robots.json``. No entry declares one, ``add_robot``
reaches a pose by name from the source model (``keyframe="STAND"``), and upstream
has already retuned this one once - the superseded ``STAND`` is still commented
out beside it. A duplicate would drift silently, so the pose is asserted against
the shipped keyframe in the simulation-side companion to this file instead.

Everything here is graded from ``robots.json`` alone, so it holds on any install:
no MuJoCo, no downloaded assets, no network. The shape claims that need the
compiled model live in ``tests/simulation/`` because ``tests/registry`` repoints
``STRANDS_ASSETS_DIR`` at a per-test temp dir, where no asset is ever present.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ROBOTS_JSON = REPO_ROOT / "strands_robots" / "registry" / "robots.json"

#: The actuated joints, in the order the asset declares them.
DOCUMENTED_ORDER: tuple[str, ...] = (
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
)

#: One floating base joint: counted by ``joints``, not by the description.
FLOATING_BASE_JOINTS = 1


def _entry() -> dict:
    return json.loads(ROBOTS_JSON.read_text(encoding="utf-8"))["robots"]["microduck"]


class TestTheEntryIsWellFormed:
    """Graded from ``robots.json`` alone, so it holds with no MuJoCo installed."""

    def test_the_registry_declares_the_robot(self) -> None:
        entry = _entry()
        assert entry["category"] == "humanoid"
        assert "Microduck" in entry["description"]

    def test_the_declared_count_is_the_hinges_plus_the_floating_base(self) -> None:
        """15, following the ``njnt`` convention the catalog documents."""
        assert _entry()["joints"] == len(DOCUMENTED_ORDER) + FLOATING_BASE_JOINTS

    def test_the_description_carries_the_hardware_dof(self) -> None:
        """Fourteen servos - the figure a reader sizes an action vector from."""
        assert f"{len(DOCUMENTED_ORDER)}-DOF" in _entry()["description"]

    def test_the_asset_declares_an_auto_download_source(self) -> None:
        """A github source, so the asset is fetchable without a naming guess."""
        source = _entry()["asset"]["source"]
        assert source["type"] == "github"
        assert source["repo"] == "pollen-robotics/microduck_rl"
        assert source["subdir"] == "src/mjlab_microduck/robot/microduck"

    def test_no_alias_repeats_the_canonical_name(self) -> None:
        """A self-alias would make every registry read raise, not just this one.

        ``loader._validate_robots`` refuses an alias that collides with a
        canonical robot name, and it runs on every load - so a self-alias here
        does not degrade this entry, it makes importing the registry raise for
        all 73. The policy validator exempts a provider naming itself; the robot
        validator has no such guard, which is easy to miss when copying an
        alias list that includes the robot's own name.
        """
        assert "microduck" not in _entry()["aliases"]

    def test_the_declared_aliases_resolve_to_the_robot(self) -> None:
        from strands_robots.registry import resolve_name

        assert _entry()["aliases"], "the entry advertises alternative spellings"
        for alias in _entry()["aliases"]:
            assert resolve_name(alias) == "microduck"
