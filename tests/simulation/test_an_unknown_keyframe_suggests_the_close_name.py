"""An unknown keyframe name is refused with the close name it was a typo of.

``Robot("panda", keyframe="hom")`` answered ``Keyframe 'hom' not found ...
Available: 'home'.`` while every other unknown-name refusal in the package
(robots, shapes, policy providers, keywords) points at the near miss. The
three backend sites that resolve ``add_robot(keyframe=...)`` against an MJCF
now share the package's ``did_you_mean`` clause; a distant name still gets the
plain list, so the hint never guesses.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

mujoco = pytest.importorskip("mujoco")

from strands_robots.simulation.isaac.joint_names import mjcf_keyframe_joint_positions  # noqa: E402
from strands_robots.simulation.mjlab import simulation as mjlab_sim  # noqa: E402
from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine  # noqa: E402

_MJCF = """
<mujoco>
  <worldbody>
    <body name="arm"><joint name="j0" type="hinge"/><geom size="0.05"/></body>
  </worldbody>
  <keyframe>
    <key name="home" qpos="0.5"/>
    <key name="rest" qpos="0"/>
  </keyframe>
</mujoco>
"""


def _mujoco(path: Path, keyframe: str) -> str:
    _, _, error = MuJoCoSimEngine()._keyframe_home_state(str(path), keyframe)
    assert error is not None and error["status"] == "error"
    return str(error["content"][0]["text"])


def _mjlab(path: Path, keyframe: str) -> str:
    with pytest.raises(KeyError) as exc:
        mjlab_sim._resolve_key(mujoco.MjModel.from_xml_path(str(path)), keyframe)
    return str(exc.value)


def _isaac(path: Path, keyframe: str) -> str:
    positions, reason = mjcf_keyframe_joint_positions(str(path), keyframe)
    assert positions is None and reason is not None
    return reason


@pytest.mark.parametrize("refusal", [_mujoco, _mjlab, _isaac], ids=["mujoco", "mjlab", "isaac"])
@pytest.mark.parametrize(
    ("typed", "hint"),
    [
        ("hom", "'hom' -> 'home'"),
        ("Home", "'Home' -> 'home'"),
        ("hoem", "'hoem' -> 'home'"),
        ("start", None),
    ],
)
def test_the_refusal_names_the_close_keyframe(
    tmp_path: Path, refusal: Callable[[Path, str], str], typed: str, hint: str | None
) -> None:
    path = tmp_path / "robot.xml"
    path.write_text(_MJCF)
    text = refusal(path, typed)
    assert "home" in text and "rest" in text, text  # the full list still lands
    if hint is None:
        assert "Did you mean" not in text, text
    else:
        assert f"Did you mean: {hint}?" in text, text


@pytest.mark.parametrize("refusal", [_mujoco, _mjlab, _isaac], ids=["mujoco", "mjlab", "isaac"])
def test_an_unnamed_keyframe_does_not_break_the_refusal(tmp_path: Path, refusal: Callable[[Path, str], str]) -> None:
    # An unnamed <key> is legal MJCF (used by index); it is listed by its index.
    path = tmp_path / "robot.xml"
    path.write_text(_MJCF.replace('<key name="rest" qpos="0"/>', '<key qpos="0"/>'))
    text = refusal(path, "hom")
    assert "Did you mean: 'hom' -> 'home'?" in text and "None" not in text, text
