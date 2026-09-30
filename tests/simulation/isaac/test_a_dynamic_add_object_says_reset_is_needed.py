"""A dynamic ``add_object`` tells the caller that ``reset()`` comes next.

Adding a dynamic body invalidates PhysX's tensor view, and ``step`` /
``send_action`` refuse until ``reset()`` rebuilds it. The refusal is correct, but
it arrived one call late and only to callers that read it: ``add_object``
answered a plain ``success``, and a loop over ``step()`` that discards its
envelope (a live GPU probe did, as does ``examples/isaac_gs``) watched a
dropped cube hang at z=0.5 for 480 steps while MuJoCo's fell to the floor.
MuJoCo needs no reset here; an implicit one is not an option, because reset()
returns every robot to its default pose. So the add says it.
"""

from __future__ import annotations

from .test_a_failed_add_object_marks_the_scene_stale import _engine, _succeeder


def _add(is_static: bool) -> dict:
    return _engine(construct=_succeeder()).add_object(
        name="cube", shape="cuboid", position=[0.0, 0.0, 0.5], size=[0.05] * 3, is_static=is_static
    )


def test_a_dynamic_add_names_the_reset() -> None:
    result = _add(is_static=False)

    assert result["status"] == "success"
    block = result["content"][0]
    assert "call reset() before step()/send_action()" in block["text"]
    assert block["json"]["requires_reset"] is True


def test_a_static_add_does_not() -> None:
    result = _add(is_static=True)

    block = result["content"][0]
    assert "reset()" not in block["text"]
    assert block["json"]["requires_reset"] is False
