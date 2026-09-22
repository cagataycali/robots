# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The SO-101 cuRobo demo only offers a URDF cuRobo can build a model from.

``examples/so101_curobo/planner.py`` plans to a ``gripper_frame_link`` tool
frame and measures its grasp offsets in that frame. The SO-ARM100 revision the
strands-robots asset cache pins declares no such link (its links are ``base``,
``shoulder``, ``upper_arm``, ``lower_arm``, ``wrist``, ``gripper``, ``jaw``), so
offering that URDF makes cuRobo's builder raise ``Link gripper_frame_link not
found in parent map`` - swallowed by the demo's scripted fallback, which records
a dataset from a different planner and reports ``planner: 'curobo'`` unusable
only in a warning quoting cuRobo's internals.

The same resolver handed cuRobo the cache's ``assets/`` subdir as the mesh
search path while the URDF spells its meshes ``assets/<f>.stl`` relative to its
own directory, so every ref resolved as ``assets/assets/<f>.stl`` and none
loaded.

Both are graded here on synthetic URDFs: one flavour per row, no cuRobo, no GPU
and no asset download.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

_WITH_FRAME = """<robot name="so101">
  <link name="base_link"><visual><geometry>
    <mesh filename="assets/base.stl"/></geometry></visual></link>
  <link name="gripper_link"/>
  <link name="gripper_frame_link"/>
</robot>
"""

_WITHOUT_FRAME = """<robot name="so101">
  <link name="base"><visual><geometry>
    <mesh filename="assets/base.stl"/></geometry></visual></link>
  <link name="gripper"/>
  <link name="jaw"/>
</robot>
"""


@dataclass(frozen=True)
class _Flavour:
    """One SO-101 URDF naming generation and what the resolver owes it."""

    label: str
    urdf: str
    offered: bool


_FLAVOURS = [
    _Flavour("declares the tool frame", _WITH_FRAME, True),
    _Flavour("the pinned cache revision, no tool frame", _WITHOUT_FRAME, False),
]


def _cache(tmp_path: Path, urdf_text: str, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write a cache-shaped SO-101 model dir and point the asset resolver at it."""
    import strands_robots.assets as assets

    model_dir = tmp_path / "robotstudio_so101"
    (model_dir / "assets").mkdir(parents=True)
    (model_dir / "assets" / "base.stl").write_bytes(b"solid\n")
    urdf = model_dir / "so101_new_calib.urdf"
    urdf.write_text(urdf_text)
    monkeypatch.setattr(assets, "resolve_model_dir", lambda *_a, **_k: str(model_dir))
    return urdf


@pytest.mark.parametrize("flavour", _FLAVOURS, ids=lambda f: f.label)
def test_only_a_urdf_with_the_tool_frame_is_offered(
    flavour: _Flavour, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A URDF without the planner's tool frame is declined, not handed to cuRobo."""
    from examples.so101_curobo import planner as P

    monkeypatch.delenv("SO101_URDF", raising=False)
    urdf = _cache(tmp_path, flavour.urdf, monkeypatch)

    resolved = P.resolve_so101_urdf(None)
    assert resolved == (str(urdf) if flavour.offered else None)


def test_the_mesh_dir_is_the_one_the_urdf_refs_resolve_under(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``assets/base.stl`` resolves under the URDF's dir, so that is the search path."""
    from examples.so101_curobo import planner as P

    monkeypatch.delenv("SO101_URDF", raising=False)
    monkeypatch.delenv("SO101_ASSET", raising=False)
    urdf = _cache(tmp_path, _WITH_FRAME, monkeypatch)

    assert P.resolve_so101_asset("") == str(urdf.parent)


def test_a_urdf_without_the_tool_frame_is_refused_with_the_remedy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit URDF missing the frame is refused before cuRobo is imported."""
    from examples.so101_curobo import planner as P

    urdf = tmp_path / "old_calib.urdf"
    urdf.write_text(_WITHOUT_FRAME)
    monkeypatch.setattr(P, "curobo_available", lambda: True)

    planner = P.CuroboMotionPlanner(urdf_path=str(urdf))
    with pytest.raises(RuntimeError) as excinfo:
        planner._ensure()

    message = str(excinfo.value)
    assert "gripper_frame_link" in message
    assert "--curobo-urdf" in message
