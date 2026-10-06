"""``normalize_robot_name`` refuses a non-string with the same wording ``Robot(name)`` does.

The nine siblings on the public read surface exported by
:mod:`strands_robots.registry` (``get_robot``, ``resolve_name``, ``has_sim``,
``has_hardware``, ``joint_labels``, ``get_driver``, ``get_hardware_type``,
``is_discoverable``, ``is_urdf_only``) all funnel their ``name`` through
:func:`strands_robots.registry.normalize_robot_name`, whose body is
``name.lower().strip().replace("-", "_")``.

Before this pin they leaked raw ``AttributeError`` from ``.lower()`` on a
non-string, or ``TypeError`` from ``.replace("-", "_")`` on ``bytes`` -
neither names what to pass instead. ``Robot(name)`` was already guarded at
the door with a structured ``ValueError`` (see
``tests/test_a_robot_name_that_is_not_a_string_is_refused_by_name.py``) and
the in-code comment at ``robot.py:800-803`` explicitly admits the sibling
leak. This file pins the registry layer to the same refusal, by putting the
guard at the fold itself so every caller gets it at once.

Three siblings on the SAME surface were already structured: ``get_policy_provider``
and ``resolve_policy`` return ``None`` for a non-string via dict lookup,
and ``lerobot_from_source_entry`` returns ``None`` via a loop. They are not
retested here because they never used ``normalize_robot_name`` - this test
pins the sibling SURFACE, not the fold at every reader.
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

#: The nine ``strands_robots.registry``-exported read functions whose first
#: non-trivial step is :func:`normalize_robot_name`. A sibling on the same
#: public surface that does NOT touch ``normalize_robot_name``
#: (``get_policy_provider``, ``resolve_policy``, ``lerobot_from_source_entry``)
#: handles non-strings with its own control flow and is not pinned here.
_LEAKY_SIBLINGS = pytest.mark.parametrize(
    "fn",
    [
        get_robot,
        resolve_name,
        has_sim,
        has_hardware,
        joint_labels,
        get_driver,
        get_hardware_type,
        is_discoverable,
        is_urdf_only,
    ],
    ids=lambda f: f.__name__,
)

#: The six non-string types reported by the repro - ``None``, ``int``,
#: ``float``, ``bool``, ``bytes``, and the two container types a tool call
#: might land here if a schema is wrong. ``str`` subclasses are deliberately
#: NOT in this list: the ``isinstance(..., str)`` guard accepts them.
_NON_STRING = [None, 42, 3.14, True, b"so101", ["so101"], {"name": "so101"}]


def test_normalize_robot_name_refuses_non_string_with_value_error() -> None:
    """The fold itself raises the structured refusal, so every reader inherits it."""
    for bad in _NON_STRING:
        with pytest.raises(ValueError) as exc:
            normalize_robot_name(bad)  # type: ignore[arg-type]
        text = str(exc.value)
        assert text.startswith("Invalid robot name "), text
        assert type(bad).__name__ in text, (bad, text)
        # The wording matches ``Robot(name)``'s refusal - a caller asking the
        # same question of two public surfaces reads the same remedy.
        assert "list_robots()" in text and "urdf_path=" in text, text


@_LEAKY_SIBLINGS
@pytest.mark.parametrize("bad", _NON_STRING, ids=type)
def test_registry_read_api_refuses_non_string(fn, bad: object) -> None:
    """Every sibling that goes through ``normalize_robot_name`` reports the same refusal.

    Pre-fix: raw ``AttributeError``/``TypeError`` from CPython. Post-fix:
    structured ``ValueError`` whose text names the type and the remedy.
    """
    with pytest.raises(ValueError) as exc:
        fn(bad)  # type: ignore[arg-type]
    text = str(exc.value)
    assert "Invalid robot name" in text and type(bad).__name__ in text, (fn.__name__, bad, text)


def test_known_name_still_resolves() -> None:
    """The guard does not displace positive-path lookups.

    One good spelling per sibling - the pin would be vacuous if the guard
    swallowed every caller, so these are the smoke test that the behavior
    the sibling is exported for is still reachable.
    """
    assert resolve_name("franka") == "panda"
    assert get_robot("so100") is not None
    assert has_sim("so101") is True
    assert has_hardware("so101") is True
    assert joint_labels("so100") != {}
    assert get_hardware_type("so100") == "so100_follower"


def test_empty_string_is_not_refused_by_the_type_guard() -> None:
    """The empty string is a string.

    ``normalize_robot_name("")`` returns ``""`` just as it did before: the
    pin is only on the type, not the content. ``Robot("")`` already has its
    own ``_validate_known_robot`` refusal and the sibling test file
    ``tests/test_a_robot_name_that_is_not_a_string_is_refused_by_name.py``
    pins that separately.
    """
    assert normalize_robot_name("") == ""
    assert normalize_robot_name(" ") == ""
    assert resolve_name("") == ""  # fallthrough to canonical-set miss, no crash
