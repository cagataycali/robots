"""Every Microduck skill the page advertises names the scene it needs.

``docs/policies/microduck.md`` opens by listing the nine shipped Pollen weights
:class:`~strands_robots.policies.microduck.MicroduckPolicy` wraps and says they
drive the biped "through the standard ``Robot(...).run_policy`` seam - in MuJoCo
or on hardware". Five of those nine run on the scene the registry entry declares.
Four do not: ``roller`` and ``roller_crouch`` need the four passive ankle wheels
that only ``scene_rollers.xml`` carries, and ``ball_kick_left`` /
``ball_kick_right`` need the ball prop that only ``scene_ball.xml`` places.

Running one of the four on the default scene is not an error. The policy writes
the same fourteen control targets, the rollout reports success, and the physics
simply has nothing to roll on or nothing to kick - so a reader who follows the
page gets a duck standing still and no indication why. ``render_video.py`` builds
a bare ``Robot("microduck")``, which is why the page used to say any shipped
weight "drops straight in": true of the five, false of the four.

The scenes are not missing. All three ship in the one asset directory the entry
already downloads, and the entry keeps naming the fourteen-hinge model on purpose
- ``tests/simulation/test_microduck_asset_matches_the_declared_shape.py`` pins
that, and its ``TestTheEntryPointsAtTheDocumentedLayout`` records the reason ("a
caller can load it by path"). So the gap was never reachability; it was that no
page said which scene a skill needs, and nothing graded the claim.

This file grades it in two layers. The first reads the page: every skill named in
the opening paragraph must appear in the Skill scenes table, so a tenth weight
cannot be advertised without naming its scene. The second reads the compiled
models, so the table's claims are true rather than asserted - the scene a row
names really does carry the wheels or the ball, and the default scene really does
not. The second layer skips when the asset is absent, which is the case on a
clean checkout; the first holds on any install, with no MuJoCo and no network.

A registry ``variant=`` spelling would be a nicer front door and is deliberately
not invented here: no entry among the seventy-three declares one, so the schema
is a public-API decision rather than a docs fix.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PAGE = REPO_ROOT / "docs" / "policies" / "microduck.md"

#: The directory the ``microduck`` entry downloads into, and the scene it names.
ASSET_DIR = "microduck"
DEFAULT_SCENE = "scene.xml"

#: A skill list shorter than this means the opening paragraph stopped listing
#: weights, so the cross-reference below would pass by having nothing to check.
MINIMUM_SKILLS = 9

#: Fewer rows than this means the table stopped covering the scenes, so the
#: per-scene cells would pass by never reaching a non-default scene.
MINIMUM_TABLE_ROWS = 3


def _page() -> str:
    return PAGE.read_text(encoding="utf-8")


def _section(text: str, heading: str) -> str:
    """Return one ``##`` section's body, so a rule reads only its own prose."""
    start = text.index(heading)
    rest = text[start + len(heading) :]
    end = rest.find("\n## ")
    return rest if end == -1 else rest[:end]


def _advertised_skills(text: str) -> set[str]:
    """The weights named in the opening sentence's parenthetical.

    The list is read from the parenthetical that follows "policies", rather than
    from the whole opening: the paragraph below it names metadata fields
    (``joint_names``, ``action_scale``) in the same backticked style, and those
    are not skills. Anchoring on the structure keeps the rule derived - a tenth
    weight added to the list is graded - where a list of names to ignore would
    have to be edited every time the prose grew.

    A pair is spelled ``ball_kick_left``/``ball_kick_right`` inside one span
    pair, so each backticked run is split on ``/`` rather than taken whole.
    """
    opening = text[: text.index("## Walking in MuJoCo")]
    start = opening.index("policies (") + len("policies (")
    listed = opening[start : opening.index(")", start)]
    return {
        token.strip()
        for span in re.findall(r"`([^`]+)`", listed)
        for token in span.split("/")
        if re.fullmatch(r"[a-z][a-z0-9_]*", token.strip())
    }


def _scene_table(text: str) -> dict[str, str]:
    """Map every skill named in the Skill scenes table to the scene it needs."""
    rows: dict[str, str] = {}
    for line in _section(text, "## Skill scenes").splitlines():
        if not line.startswith("|") or "---" in line:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 2 or cells[0] == "skill":
            continue
        scene = re.search(r"`([A-Za-z0-9_]+\.xml)`", cells[1])
        if scene is None:
            continue
        for skill in re.findall(r"`([a-z][a-z0-9_]*)`", cells[0]):
            rows[skill] = scene.group(1)
    return rows


def _scene_model(scene: str):
    """Compile a scene the asset directory carries, or skip.

    Reads the search paths rather than resolving through the asset manager,
    which downloads a missing asset: a test that clones a third-party
    repository fails on a host with no network instead of skipping.
    """
    mujoco = pytest.importorskip("mujoco")
    from strands_robots.utils import get_search_paths

    present = next(
        (candidate for root in get_search_paths() if (candidate := Path(root) / ASSET_DIR / scene).exists()),
        None,
    )
    if present is None:
        pytest.skip(f"microduck {scene} is not downloaded, so its contents cannot be read")
    return mujoco, mujoco.MjModel.from_xml_path(str(present))


def _joint_names(mujoco, model) -> list[str]:
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]


def _wheel_joints(mujoco, model) -> list[str]:
    return [name for name in _joint_names(mujoco, model) if name and "wheel" in name]


def _ball_bodies(mujoco, model) -> list[str]:
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]
    return [name for name in names if name and "ball" in name]


class TestThePageNamesASceneForEverySkill:
    """Read from the page alone, so it holds with no asset and no MuJoCo."""

    def test_the_opening_paragraph_still_lists_the_shipped_weights(self) -> None:
        """Non-vacuity: the cross-reference needs a skill list to check."""
        skills = _advertised_skills(_page())
        assert len(skills) >= MINIMUM_SKILLS, f"expected at least {MINIMUM_SKILLS} skills, found {sorted(skills)}"

    def test_the_table_still_covers_more_than_the_default_scene(self) -> None:
        """Non-vacuity: a one-row table would name no variant to verify."""
        scenes = set(_scene_table(_page()).values())
        assert len(scenes) >= MINIMUM_TABLE_ROWS, f"expected at least {MINIMUM_TABLE_ROWS} scenes, found {scenes}"

    def test_every_advertised_skill_appears_in_the_scene_table(self) -> None:
        """A weight advertised without a scene is the gap this file closes."""
        text = _page()
        unnamed = sorted(_advertised_skills(text) - set(_scene_table(text)))
        assert not unnamed, f"advertised with no scene named: {unnamed}"

    def test_the_table_names_no_skill_the_page_does_not_advertise(self) -> None:
        """Over-reach guard: the table describes the page, not a wider set."""
        text = _page()
        unadvertised = sorted(set(_scene_table(text)) - _advertised_skills(text))
        assert not unadvertised, f"in the table but never advertised: {unadvertised}"


class TestTheTablesClaimsAreTrueOfTheAssets:
    """Re-derive each row from the model it names, so the table cannot drift."""

    def test_every_scene_the_table_names_is_a_scene_that_compiles(self) -> None:
        """A row pointing at a file that is not there is a dead instruction."""
        for scene in sorted(set(_scene_table(_page()).values())):
            mujoco, model = _scene_model(scene)
            assert model.nu > 0, f"{scene} compiled with no actuators"

    def test_the_default_scene_carries_neither_a_wheel_nor_a_ball(self) -> None:
        """This is why four of the nine skills need a different scene."""
        mujoco, model = _scene_model(DEFAULT_SCENE)
        assert _wheel_joints(mujoco, model) == []
        assert _ball_bodies(mujoco, model) == []

    def test_a_roller_skill_is_pointed_at_a_scene_that_has_wheels(self) -> None:
        """The rollers scene adds the four passive ankle wheels."""
        scene = _scene_table(_page())["roller"]
        assert scene != DEFAULT_SCENE, "roller must not be pointed at the wheel-less default"
        mujoco, model = _scene_model(scene)
        assert len(_wheel_joints(mujoco, model)) == 4

    def test_a_ball_kick_skill_is_pointed_at_a_scene_that_has_a_ball(self) -> None:
        """The ball scene places the prop the kick policies were trained on."""
        scene = _scene_table(_page())["ball_kick_left"]
        assert scene != DEFAULT_SCENE, "ball_kick must not be pointed at the ball-less default"
        mujoco, model = _scene_model(scene)
        assert _ball_bodies(mujoco, model) == ["ball"]

    def test_the_two_roller_skills_share_one_scene(self) -> None:
        """Both roller weights want the same wheels; one row covers them."""
        table = _scene_table(_page())
        assert table["roller"] == table["roller_crouch"]

    def test_the_two_ball_kick_skills_share_one_scene(self) -> None:
        """The ball sits in front of the duck; the side is the policy's."""
        table = _scene_table(_page())
        assert table["ball_kick_left"] == table["ball_kick_right"]


class TestTheJointLayoutClaimsHold:
    """The page tells a raw ``qpos`` reader which scene renumbers the slice."""

    def test_the_actuator_order_is_the_same_on_every_documented_scene(self) -> None:
        """Why a policy writing ``ctrl`` is unaffected by the scene choice."""
        mujoco, default = _scene_model(DEFAULT_SCENE)

        def actuators(model) -> list[str]:
            return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]

        expected = actuators(default)
        assert len(expected) == 14
        for scene in sorted(set(_scene_table(_page()).values())):
            _, model = _scene_model(scene)
            assert actuators(model) == expected, f"{scene} permutes the actuator order"

    def test_the_ball_scene_leaves_the_robots_position_slice_alone(self) -> None:
        """The ball's free joint is appended, so ``qpos[7:21]`` still holds."""
        mujoco, default = _scene_model(DEFAULT_SCENE)
        _, ball = _scene_model(_scene_table(_page())["ball_kick_left"])

        def layout(model) -> dict[str, int]:
            return {
                name: int(model.jnt_qposadr[i])
                for i, name in enumerate(_joint_names(mujoco, model))
                if name is not None
            }

        base, with_ball = layout(default), layout(ball)
        assert {name: adr for name, adr in with_ball.items() if name in base} == base

    def test_the_rollers_scene_moves_nine_of_the_fourteen_joints(self) -> None:
        """The count the page and the asset-shape guard both state."""
        mujoco, default = _scene_model(DEFAULT_SCENE)
        _, rollers = _scene_model(_scene_table(_page())["roller"])

        def layout(model) -> dict[str, int]:
            return {
                name: int(model.jnt_qposadr[i])
                for i, name in enumerate(_joint_names(mujoco, model))
                if name is not None
            }

        base, moved_layout = layout(default), layout(rollers)
        moved = [name for name, adr in base.items() if name in moved_layout and moved_layout[name] != adr]
        assert len(moved) == 9, f"expected nine renumbered joints, got {sorted(moved)}"

    def test_the_wheels_land_where_the_page_says_the_head_joints_sit(self) -> None:
        """The concrete mis-read the page warns a slice reader about."""
        mujoco, default = _scene_model(DEFAULT_SCENE)
        _, rollers = _scene_model(_scene_table(_page())["roller"])

        def at(model, addresses: set[int]) -> list[str]:
            return [
                name
                for i, name in enumerate(_joint_names(mujoco, model))
                if name is not None and int(model.jnt_qposadr[i]) in addresses
            ]

        assert at(default, {12, 13}) == ["neck_pitch", "head_pitch"]
        assert at(rollers, {12, 13}) == ["passive_LF_wheel", "passive_LR_wheel"]
