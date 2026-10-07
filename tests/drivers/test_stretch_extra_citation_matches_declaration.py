"""StretchDriver's missing-SDK refusal cites a declared extra.

Before this fire, the refusal said ``pip install hello-robot-stretch-body`` --
the bare PyPI wheel name -- while every sibling with a PyPI-shipped SDK
(``crazyflie``, ``earthrover``, ``rby1``, ``spot``, ``ur``, ``xarm``) cited
``pip install 'strands-robots[<name>]'``. A user who followed the pattern and
typed ``pip install 'strands-robots[stretch]'`` saw pip exit 0 (pip 24+ does
not warn on unknown extras), then hit a refusal naming a different install
line entirely.

This guard pins the two halves of the fix together:

1. ``stretch`` IS a declared extra in ``pyproject.toml``.
2. The refusal in :func:`strands_robots.drivers.stretch._resolve_sdk` cites
   that extra by name, matching the sibling family's shape.
"""

from __future__ import annotations

import importlib.metadata as im

from strands_robots.drivers.stretch import _resolve_sdk


def test_stretch_is_a_declared_extra() -> None:
    """``pip install 'strands-robots[stretch]'`` resolves to a declared extra.

    The installed distribution's ``Provides-Extra`` metadata is the ground
    truth pip and importlib.metadata both read; it lists every extra
    pyproject.toml declares. Before this fire, ``stretch`` was absent and
    ``pip install 'strands-robots[stretch]'`` succeeded as a bare install
    with no feedback that the user had asked for something the project did
    not declare.
    """
    extras = set(im.metadata("strands-robots").get_all("Provides-Extra") or [])
    assert "stretch" in extras, (
        "`stretch` is not in Provides-Extra. `pip install 'strands-robots[stretch]'` "
        "will succeed as a bare install (pip is silent on unknown extras), and the "
        "driver's refusal will not match what the user typed. "
        f"Declared extras: {sorted(extras)}"
    )


def test_stretch_refusal_cites_the_declared_extra() -> None:
    """The ImportError-handler refusal is extras-shaped, like its siblings.

    This test runs on an environment without ``stretch_body`` installed (every
    CI host; ``stretch_body`` ships on the robot). The resolver returns the
    refusal string -- the exact text the agent surfaces back to the caller --
    and this guard pins the install line to the ``strands-robots[stretch]``
    shape so the Spot/UR/xArm sibling pattern now includes Stretch.
    """
    refusal = _resolve_sdk()
    assert isinstance(refusal, str), (
        f"stretch_body is importable in this environment; the refusal path is unreachable. "
        f"_resolve_sdk returned: {refusal!r}"
    )
    assert "pip install 'strands-robots[stretch]'" in refusal, (
        f"The refusal does not cite the declared [stretch] extra. "
        f"Siblings (spot/rby1/xarm/ur/crazyflie) all do. Got: {refusal!r}"
    )
    # Negative: the bare PyPI wheel name is no longer the user-facing instruction.
    # (The underlying package is still 'hello-robot-stretch-body'; the extra just
    # routes to it.)
    assert "pip install hello-robot-stretch-body" not in refusal, (
        f"The refusal still points at the bare PyPI name; the user who followed "
        f"the extras-shaped pattern will see a mismatched remedy. Got: {refusal!r}"
    )
