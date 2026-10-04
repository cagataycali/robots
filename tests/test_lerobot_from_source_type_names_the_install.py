"""A robot whose lerobot type is only on lerobot main names the from-source install.

``hardware.requires_lerobot_from_source`` marks a registry entry whose lerobot
type the PyPI release does not ship. On such an install lerobot's registry has
no choice for it, and the generic ``Known lerobot robot types`` listing cannot
name the robot the caller asked for. The refusal names the install instead.
"""

from __future__ import annotations

import pytest

from strands_robots import Robot
from strands_robots.registry.robots import lerobot_from_source_entry

pytest.importorskip("lerobot")


@pytest.mark.parametrize(
    ("lerobot_type", "entry"),
    [
        ("rebot_b601_follower", "rebot_b601"),
        ("bi_rebot_b601_follower", "bi_rebot_b601"),
        ("so101_follower", None),  # flag absent: the listing stays the answer
        ("no_such_type", None),
    ],
)
def test_the_registry_names_the_entry_that_needs_lerobot_main(lerobot_type: str, entry: str | None) -> None:
    assert lerobot_from_source_entry(lerobot_type) == entry


@pytest.fixture
def pypi_lerobot(monkeypatch: pytest.MonkeyPatch) -> None:
    """lerobot as the PyPI release ships it: no reBot B601 robot types."""
    from strands_robots.utils import ensure_lerobot_family_registered

    ensure_lerobot_family_registered("robots")
    from lerobot.robots.config import RobotConfig

    real = RobotConfig.get_choice_class.__func__  # type: ignore[attr-defined]
    missing = {"rebot_b601_follower", "bi_rebot_b601_follower"}

    def get_choice_class(cls: type, name: str) -> type:
        if name in missing:
            raise KeyError(name)
        return real(cls, name)

    monkeypatch.setattr(RobotConfig, "get_choice_class", classmethod(get_choice_class))


@pytest.mark.parametrize(
    ("name", "entry"),
    [("rebot_b601", "rebot_b601"), ("b601_dm", "rebot_b601"), ("bi_rebot_b601", "bi_rebot_b601")],
)
def test_robot_names_the_from_source_install(pypi_lerobot: None, name: str, entry: str) -> None:
    with pytest.raises(ValueError) as excinfo:
        Robot(name, mode="real", driver="lerobot", port="/dev/ttyACM0")
    message = str(excinfo.value)
    assert "pip install 'git+https://github.com/huggingface/lerobot'" in message
    assert f"docs/robots/{entry}.md" in message
    assert "Known lerobot robot types" not in message
