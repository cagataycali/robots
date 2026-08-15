"""End-effector hint matching is component-aware, not bare-substring.

:func:`strands_robots.simulation.ik.discover_ee_frame` resolves an IK target
frame by looking for hint words in body/site names. The hints name *components*
of a name, so they must match on word boundaries: the short hints (``"ee"``,
``"eef"``, ``"tcp"``) otherwise fire inside unrelated words - ``"ee"`` occurs in
``"knee"``, ``"wheel"`` and ``"unitree"`` - and resolve a leg link or a drive
wheel as a robot's end-effector, which then silently becomes the frame every
Cartesian target and every eef-delta policy chunk is applied to.

This is the same guarantee as the namespace-stripping one (a namespace that
contains a hint substring must not trigger a hint match), applied within the
name itself.

Models are built from inline MJCF so the real ``mj_id2name`` traversal runs;
skips cleanly when the ``sim-mujoco`` extra is absent.
"""

import pytest

pytest.importorskip("mujoco")

import mujoco  # noqa: E402

from strands_robots.simulation.ik import discover_ee_frame  # noqa: E402


def _model(xml: str) -> "mujoco.MjModel":
    return mujoco.MjModel.from_xml_string(xml)


def _chain(namespace: str, links: str) -> str:
    """A single serial chain of hinge-jointed bodies named by ``links``."""
    body_names = links.split()
    xml = f'<mujoco><worldbody><body name="{namespace}base">'
    xml += f'<joint name="{namespace}j0" type="hinge"/><geom type="box" size=".1 .1 .1"/>'
    for i, name in enumerate(body_names):
        xml += f'<body name="{namespace}{name}">'
        xml += f'<joint name="{namespace}j{i + 1}" type="hinge"/><geom type="box" size=".05 .05 .05"/>'
    xml += "</body>" * len(body_names)
    xml += "</body></worldbody></mujoco>"
    return xml


# --------------------------------------------------------------------------
# A hint must not match inside an unrelated word.
# --------------------------------------------------------------------------


def test_knee_is_not_an_end_effector_when_a_wrist_exists() -> None:
    """A humanoid resolves its wrist, not the ``kn-ee`` the ``ee`` hint used to
    match: ``ee`` precedes ``wrist`` in the body-hint order, so a substring
    match on a leg link outranked the arm entirely."""
    xml = _chain("g1/", "left_knee_link left_elbow_link left_wrist_roll_link")
    assert discover_ee_frame(_model(xml), "g1/") == ("g1/left_wrist_roll_link", "body")


def test_drive_wheel_is_not_an_end_effector_on_a_mobile_manipulator() -> None:
    """A wheeled base carrying an arm resolves the arm's wrist, not a
    ``wh-ee-l`` link."""
    xml = _chain("lekiwi/", "wheel_hub_back_link wheel_back_link Wrist_Pitch_Roll")
    assert discover_ee_frame(_model(xml), "lekiwi/") == ("lekiwi/Wrist_Pitch_Roll", "body")


def test_hyphen_separated_knee_is_not_an_end_effector() -> None:
    """Hyphens delimit name components too, so ``left-knee`` does not match
    ``ee`` and discovery falls through to the leaf body."""
    xml = _chain("cassie/", "left-knee left-shin left-foot")
    assert discover_ee_frame(_model(xml), "cassie/") == ("cassie/left-foot", "body")


def test_site_hint_does_not_match_inside_a_word() -> None:
    """The site rung is subject to the same rule: a ``knee``-named site does not
    win rung 1, so a genuine hand body on rung 2 is resolved instead."""
    xml = """
    <mujoco><worldbody>
      <body name="r/upper">
        <joint name="r/j0" type="hinge"/><geom type="box" size=".1 .1 .1"/>
        <site name="r/knee_marker" pos="0 0 .1"/>
        <body name="r/hand">
          <joint name="r/j1" type="hinge"/><geom type="box" size=".05 .05 .05"/>
        </body>
      </body>
    </worldbody></mujoco>
    """
    assert discover_ee_frame(_model(xml), "r/") == ("r/hand", "body")


def test_robot_name_in_an_unnamespaced_world_does_not_match() -> None:
    """With no namespace the full name is searched, so the ``unitr-ee`` in a
    robot's own name must not resolve that robot's pelvis site as its TCP."""
    xml = """
    <mujoco><worldbody>
      <body name="unitree_g1/pelvis">
        <joint name="unitree_g1/j0" type="hinge"/><geom type="box" size=".1 .1 .1"/>
        <site name="unitree_g1/imu_in_pelvis" pos="0 0 .1"/>
        <body name="unitree_g1/left_wrist_roll_link">
          <joint name="unitree_g1/j1" type="hinge"/><geom type="box" size=".05 .05 .05"/>
        </body>
      </body>
    </worldbody></mujoco>
    """
    assert discover_ee_frame(_model(xml), None) == ("unitree_g1/left_wrist_roll_link", "body")


# --------------------------------------------------------------------------
# Every way a hint is legitimately spelled still matches.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("site_name", "hint_described"),
    [
        ("attachment_site", "the menagerie multi-token convention"),
        ("attachment", "a single-token prefix hint"),
        ("grasp_point", "a hint as the leading token"),
        ("left_pinch", "a hint as the trailing token"),
        ("tcp", "a whole-name hint"),
        ("ee_site", "a multi-token short hint"),
        ("ee", "a bare short hint as the whole name"),
        ("gripper_ee", "a short hint as the trailing token"),
        ("tool_flange", "a hint as the trailing token"),
    ],
)
def test_a_hinted_site_is_still_resolved(site_name: str, hint_described: str) -> None:
    """Component matching keeps every legitimate hint spelling resolving,
    including the short hints when they really are a name component."""
    xml = f"""
    <mujoco><worldbody>
      <body name="a/link0">
        <joint name="a/j0" type="hinge"/><geom type="box" size=".1 .1 .1"/>
        <site name="a/{site_name}" pos="0 0 .2"/>
      </body>
    </worldbody></mujoco>
    """
    assert discover_ee_frame(_model(xml), "a/") == (f"a/{site_name}", "site"), hint_described


@pytest.mark.parametrize(
    "body_name",
    ["hand", "gripper", "tool0", "wristYawLeft", "eef_link", "end_effector_frame", "robotiq_2f85_flange"],
)
def test_a_hinted_body_is_still_resolved(body_name: str) -> None:
    """Body hints match across the separator conventions MuJoCo names use:
    snake_case, camelCase, and a trailing digit (``tool0``)."""
    xml = _chain("b/", f"link1 {body_name}")
    assert discover_ee_frame(_model(xml), "b/") == (f"b/{body_name}", "body")


def test_hint_priority_is_unchanged() -> None:
    """Earlier hints still outrank later ones within a rung."""
    xml = """
    <mujoco><worldbody>
      <body name="a/l0">
        <joint name="a/j0" type="hinge"/><geom type="box" size=".1 .1 .1"/>
        <site name="a/tcp_site" pos="0 0 .1"/>
        <site name="a/grasp_point" pos="0 0 .2"/>
      </body>
    </worldbody></mujoco>
    """
    assert discover_ee_frame(_model(xml), "a/") == ("a/grasp_point", "site")


def test_leaf_fallback_when_no_component_matches() -> None:
    """A chain whose names carry no hint component still resolves its leaf
    body, so a robot with no tool-like name keeps a usable frame."""
    xml = _chain("plain/", "link1 link2 link3")
    assert discover_ee_frame(_model(xml), "plain/") == ("plain/link3", "body")
