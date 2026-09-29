"""``Robot(name)`` refuses a non-string name with the ``ValueError`` the docstring promises.

``Robot(None)``, ``Robot(123)`` and ``Robot(["so101"])`` escaped as
``AttributeError: 'NoneType' object has no attribute 'lower'`` from
``normalize_robot_name`` and ``Robot(b"so101")`` as a ``TypeError`` from a regex
(#4150), while ``Robot("")`` was already the clean ``ValueError`` naming
``list_robots()`` and ``urdf_path=``. The four now get that same refusal, and it
names the type they passed.
"""

from __future__ import annotations

import pytest

from strands_robots import Robot


@pytest.mark.parametrize("bad", [None, 123, b"so101", ["so101"], 1.5, {"name": "so101"}], ids=type)
def test_a_non_string_name_is_a_value_error_naming_the_remedy(bad: object) -> None:
    with pytest.raises(ValueError) as exc:
        Robot(bad)  # type: ignore[call-overload]
    text = str(exc.value)
    assert text.startswith("Invalid robot name "), text
    assert type(bad).__name__ in text
    assert "list_robots()" in text and "urdf_path=" in text


def test_the_empty_string_keeps_its_refusal() -> None:
    with pytest.raises(ValueError, match=r"^Invalid robot name ''\. Pass a registered name"):
        Robot("")


def test_a_non_string_name_is_refused_even_with_a_urdf_path(tmp_path) -> None:
    # ``urdf_path=`` lets an unregistered NAME through; it does not make a non-name a name.
    with pytest.raises(ValueError, match="a robot name is a string"):
        Robot(None, urdf_path=str(tmp_path / "arm.xml"))  # type: ignore[call-overload]
