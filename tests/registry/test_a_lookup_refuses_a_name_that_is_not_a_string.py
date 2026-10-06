"""Every registry read refuses a non-string name with a ``ValueError`` naming the remedy.

The read API folds its argument through ``normalize_robot_name`` first, so
``get_robot(None)`` used to escape as ``AttributeError: 'NoneType' object has no
attribute 'lower'`` and ``get_robot(b"so100")`` as a ``TypeError`` from
``str.replace`` - neither says a robot name was wanted. ``Robot(name)`` already
refused these at its own door; the registry functions now give the same answer.
"""

from __future__ import annotations

import pytest

from strands_robots.registry import (
    get_driver,
    get_hardware_type,
    get_robot,
    has_hardware,
    has_sim,
    joint_labels,
    normalize_robot_name,
    resolve_name,
)
from strands_robots.registry.discovery import is_discoverable, is_urdf_only

LOOKUPS = [
    normalize_robot_name,
    resolve_name,
    get_robot,
    has_sim,
    has_hardware,
    joint_labels,
    get_driver,
    get_hardware_type,
    is_discoverable,
    is_urdf_only,
]
NOT_NAMES = [None, 42, 3.14, True, b"so100", ["so100"], {"name": "so100"}]


@pytest.mark.parametrize("bad", NOT_NAMES, ids=lambda v: type(v).__name__)
@pytest.mark.parametrize("lookup", LOOKUPS, ids=lambda f: f.__name__)
def test_a_non_string_name_is_a_value_error_naming_the_remedy(lookup, bad: object) -> None:
    with pytest.raises(ValueError, match=r"^Invalid robot name .* a robot name is a string\. ") as exc:
        lookup(bad)
    assert f"({type(bad).__name__})" in str(exc.value)
    assert "list_robots()" in str(exc.value)
