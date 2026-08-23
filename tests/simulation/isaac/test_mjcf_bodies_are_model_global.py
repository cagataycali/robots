# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""An MJCF's bodies are read from the whole model, not just the top file.

MuJoCo treats ``<include file=...>`` as a textual splice and MERGES every
``<worldbody>`` the spliced model carries, exactly as it treats ``<compiler>``
and ``<asset>`` as model-global. ``test_mjcf_mesh_assets_are_model_global``
states that rule for the mesh registry and the search directory; this module
states it for the bodies, which is the one question both loaders in
:mod:`strands_robots.simulation.isaac.loaders` still answered from
``root.find("worldbody")`` alone.

Reading only the top file's direct children broke in two ways, and they failed
differently. A model whose ``<worldbody>`` lives entirely in an included
fragment read as having none, so ``load_mjcf`` refused a model MuJoCo compiles -
``aloha/scene.xml`` is ``<include file="aloha.xml"/>`` plus a table, and it was
rejected as a "phantom robot" while ``aloha/aloha.xml`` loaded all 21 bodies and
16 joints. A model that keeps some bodies in the top file and includes the rest
read only the ones it could see and silently dropped the others, their subtrees
and their joints: ``franka_emika_panda/mjx_single_cube.xml`` returned 2 bodies
and ZERO joints for a model with 10, the whole Panda arm absent under a
successful load - and body/joint topology is the function's product.

Every fixture here is a model real MuJoCo compiles, and
``TestMuJoCoAgreesOnTheModelsBodies`` compares the loader's top-level body names
against MuJoCo's own (the bodies whose parent is the world), so the expectation
is derived from the compiler rather than restated. That is what makes a loader
disagreement a defect rather than invalid scaffolding.

Scope: an ``<include>`` nested INSIDE a ``<worldbody>`` (legal MJCF, splicing
bare ``<body>`` elements out of a ``<mujocoinclude>`` fragment) is deliberately
left out. No ``<worldbody>`` in the shipped registry carries one - 0 of 227
measured - and the rule here is about which ``<worldbody>`` elements the model
has, not about splicing inside one. ``TestAnIncludeInsideAWorldbodyIsOutOfScope``
pins that boundary so widening it later is a decision rather than an accident.
"""

from __future__ import annotations

import pathlib

import pytest

from strands_robots.simulation.isaac.loaders import load_mjcf, load_mjcf_scene_objects

#: A body MuJoCo accepts on its own: one hinge joint and one box geom.
_LINK = (
    '<body name="{name}" pos="{pos}">'
    '<joint name="{name}_j" type="hinge" axis="0 0 1"/>'
    '<geom type="box" size="0.05 0.05 0.05"/>'
    "</body>"
)


def _link(name: str, pos: str = "0 0 0.3") -> str:
    return _LINK.format(name=name, pos=pos)


def _model(*children: str) -> str:
    return '<mujoco model="m">' + "".join(children) + "</mujoco>"


def _worldbody(*children: str) -> str:
    return "<worldbody>" + "".join(children) + "</worldbody>"


def _write(root: pathlib.Path, top: str, fragments: dict[str, str] | None = None) -> str:
    """Write a model tree under ``root``; return the top file's path.

    A fragment name spells a subdirectory with ``__`` so nested includes can be
    built without the caller creating directories. Taken as a mapping rather
    than keywords because a fragment name carries a ``.xml`` suffix.
    """
    for name, text in (fragments or {}).items():
        path = root / name.replace("__", "/")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    top_path = root / "model.xml"
    top_path.write_text(top, encoding="utf-8")
    return str(top_path)


#: ``(id, top file, fragments, expected top-level body names in document order)``.
#:
#: Each model is one MuJoCo compiles, and the expected names are the bodies whose
#: parent is the world - the same set ``TestMuJoCoAgreesOnTheModelsBodies`` reads
#: back out of the compiler.
_MODEL_GLOBAL_FIXTURES = [
    (
        "worldbody-lives-in-a-fragment",
        _model('<include file="frag/arm.xml"/>'),
        {"frag__arm.xml": _model(_worldbody(_link("frag_link")))},
        ["frag_link"],
    ),
    (
        "a-second-worldbody-arrives-by-include",
        _model(_worldbody(_link("direct", "0 0 0.1")), '<include file="frag/arm.xml"/>'),
        {"frag__arm.xml": _model(_worldbody(_link("frag_link")))},
        ["direct", "frag_link"],
    ),
    (
        "the-include-precedes-the-direct-worldbody",
        _model('<include file="frag/arm.xml"/>', _worldbody(_link("direct", "0 0 0.1"))),
        {"frag__arm.xml": _model(_worldbody(_link("frag_link")))},
        ["frag_link", "direct"],
    ),
    (
        "a-nested-include-chain",
        _model('<include file="frag/outer.xml"/>'),
        {
            "frag__outer.xml": _model('<include file="inner/deep.xml"/>'),
            "frag__inner__deep.xml": _model(_worldbody(_link("deep_link"))),
        },
        ["deep_link"],
    ),
    (
        "two-bodies-in-one-included-worldbody",
        _model('<include file="frag/arm.xml"/>'),
        {"frag__arm.xml": _model(_worldbody(_link("a"), _link("b", "0.3 0 0.3")))},
        ["a", "b"],
    ),
]

_FIXTURE_IDS = [f[0] for f in _MODEL_GLOBAL_FIXTURES]
_FIXTURES = [f[1:] for f in _MODEL_GLOBAL_FIXTURES]


def _loaded_top_bodies(scene_path: str) -> list[str]:
    """``load_mjcf``'s top-level body names, minus the synthetic ``world`` root."""
    robot = load_mjcf(scene_path)
    return [b.name for b in robot.bodies if b.name != "world"]


class TestTheModelsBodiesAreReadFromTheWholeModel:
    """A body the model declares is a body the loader reads, whatever file it is in."""

    @pytest.mark.parametrize(("top", "fragments", "expected"), _FIXTURES, ids=_FIXTURE_IDS)
    def test_load_mjcf_reads_every_declared_body(self, tmp_path, top, fragments, expected):
        assert _loaded_top_bodies(_write(tmp_path, top, fragments)) == expected

    @pytest.mark.parametrize(("top", "fragments", "expected"), _FIXTURES, ids=_FIXTURE_IDS)
    def test_the_scene_loader_reads_every_declared_body(self, tmp_path, top, fragments, expected):
        objects = load_mjcf_scene_objects(_write(tmp_path, top, fragments))
        assert [o.name for o in objects] == expected

    def test_an_included_bodys_joints_reach_the_robot(self, tmp_path):
        """The joints, not just the bodies: topology is what ``load_mjcf`` produces."""
        model = _write(
            tmp_path,
            _model(_worldbody(_link("direct", "0 0 0.1")), '<include file="frag/arm.xml"/>'),
            {"frag__arm.xml": _model(_worldbody(_link("frag_link")))},
        )
        robot = load_mjcf(model)
        assert [j.name for j in robot.joints] == ["direct_j", "frag_link_j"]

    def test_an_included_bodys_subtree_reaches_the_robot(self, tmp_path):
        """A nested child of an included body is read too, not just its root."""
        nested = (
            '<body name="upper" pos="0 0 0.3">'
            '<joint name="upper_j" type="hinge" axis="0 0 1"/>'
            '<geom type="box" size="0.05 0.05 0.05"/>'
            '<body name="lower" pos="0 0 0.2">'
            '<joint name="lower_j" type="slide" axis="1 0 0"/>'
            '<geom type="box" size="0.02 0.02 0.02"/>'
            "</body></body>"
        )
        model = _write(
            tmp_path,
            _model('<include file="frag/arm.xml"/>'),
            {"frag__arm.xml": _model(_worldbody(nested))},
        )
        robot = load_mjcf(model)
        assert [b.name for b in robot.bodies] == ["world", "upper", "lower"]
        assert [j.name for j in robot.joints] == ["upper_j", "lower_j"]

    def test_the_phantom_robot_guard_reads_an_included_worldbody(self, tmp_path):
        """A worldbody read through an include is still subject to the guard.

        The guard's own verdict changes here: before the fix this model reported
        "has no <worldbody>", so the two refusals were told apart by which FILE a
        body lived in rather than by whether the model declares one at all.
        """
        model = _write(
            tmp_path, _model('<include file="frag/empty.xml"/>'), {"frag__empty.xml": _model("<worldbody/>")}
        )
        with pytest.raises(ValueError, match="phantom robot guard"):
            load_mjcf(model)

    def test_the_skip_list_applies_to_an_included_fragments_bodies(self, tmp_path):
        """An included body goes through the same floor/robot skip list as a direct one."""
        model = _write(
            tmp_path,
            _model('<include file="frag/arm.xml"/>'),
            {
                "frag__arm.xml": _model(
                    _worldbody(_link("floor"), _link("robot0_base", "0.3 0 0.3"), _link("mug", "0.6 0 0.3"))
                )
            },
        )
        assert [o.name for o in load_mjcf_scene_objects(model)] == ["mug"]


class TestTheRefusalsAreUnchanged:
    """A top-file-only model, and both guards, are untouched.

    Every test here passes before and after the fix: the change widens which
    files are read, never which models are accepted once read. They are what
    fails if a wider reader starts accepting a model a guard should refuse, or
    stops reading a model that never used an ``<include>`` at all.
    """

    def test_a_model_with_no_worldbody_anywhere_is_still_refused(self, tmp_path):
        model = _write(tmp_path, _model('<include file="frag/empty.xml"/>'), {"frag__empty.xml": _model("<asset/>")})
        with pytest.raises(ValueError, match="has no <worldbody>"):
            load_mjcf(model)

    def test_a_top_file_only_empty_worldbody_is_still_a_phantom_robot(self, tmp_path):
        with pytest.raises(ValueError, match="phantom robot guard"):
            load_mjcf(_write(tmp_path, _model("<worldbody/>")))

    def test_a_top_file_only_model_is_unchanged(self, tmp_path):
        model = _write(tmp_path, _model(_worldbody(_link("only", "0 0 0.1"))))
        robot = load_mjcf(model)
        assert [b.name for b in robot.bodies] == ["world", "only"]
        assert [j.name for j in robot.joints] == ["only_j"]
        assert [o.name for o in load_mjcf_scene_objects(model)] == ["only"]

    def test_a_top_file_only_scene_still_skips_the_floor_and_the_robot(self, tmp_path):
        model = _write(
            tmp_path,
            _model(_worldbody(_link("floor"), _link("robot0_base", "0.3 0 0.3"), _link("mug", "0.6 0 0.3"))),
        )
        assert [o.name for o in load_mjcf_scene_objects(model)] == ["mug"]


class TestAnIncludeInsideAWorldbodyIsOutOfScope:
    """A ``<worldbody>``-nested include splices bodies in MuJoCo and not here.

    Legal MJCF, and no shipped registry ``<worldbody>`` uses it (0 of 227). The
    rule this module states is about which ``<worldbody>`` elements the model
    has; splicing inside one is a separate question, and this pins the boundary
    so widening it is a decision rather than a surprise.
    """

    def test_a_worldbody_nested_include_is_not_spliced(self, tmp_path):
        model = _write(
            tmp_path,
            _model(_worldbody(_link("direct", "0 0 0.1"), '<include file="frag/bare.xml"/>')),
            {"frag__bare.xml": "<mujocoinclude>" + _link("nested", "0.4 0 0.3") + "</mujocoinclude>"},
        )
        assert _loaded_top_bodies(model) == ["direct"]

    def test_mujoco_does_splice_it(self, tmp_path):
        """The premise: MuJoCo reads the body this loader does not."""
        mujoco = pytest.importorskip("mujoco")
        model = _write(
            tmp_path,
            _model(_worldbody(_link("direct", "0 0 0.1"), '<include file="frag/bare.xml"/>')),
            {"frag__bare.xml": "<mujocoinclude>" + _link("nested", "0.4 0 0.3") + "</mujocoinclude>"},
        )
        compiled = mujoco.MjModel.from_xml_path(model)
        assert _mujoco_top_bodies(mujoco, compiled) == ["direct", "nested"]


class TestAnUnusableIncludeDoesNotFailTheModel:
    """MuJoCo names the offending file on the load that follows; this reader does not guess.

    Mirrors ``TestAnUnusableIncludeDoesNotFailTheScene`` for the mesh registry:
    the rest of the model still resolves, so a stale or broken fragment cannot
    turn a readable model into a refusal this reader invented.
    """

    @pytest.mark.parametrize(
        ("label", "fragments"),
        [
            ("missing", {}),
            ("malformed", {"frag__arm.xml": "<mujoco><worldbody>"}),
            ("not-a-file", {"frag__arm.xml__x": "ignored"}),
        ],
        ids=["missing", "malformed", "a-directory"],
    )
    def test_the_rest_of_the_model_still_loads(self, tmp_path, label, fragments):
        model = _write(
            tmp_path,
            _model(_worldbody(_link("direct", "0 0 0.1")), '<include file="frag/arm.xml"/>'),
            fragments,
        )
        assert _loaded_top_bodies(model) == ["direct"]

    def test_an_include_cycle_terminates(self, tmp_path):
        model = _write(
            tmp_path,
            _model(_worldbody(_link("direct", "0 0 0.1")), '<include file="frag/a.xml"/>'),
            {
                "frag__a.xml": _model('<include file="b.xml"/>'),
                "frag__b.xml": _model('<include file="a.xml"/>'),
            },
        )
        assert _loaded_top_bodies(model) == ["direct"]

    def test_an_include_without_a_file_attribute_is_ignored(self, tmp_path):
        model = _write(tmp_path, _model(_worldbody(_link("direct", "0 0 0.1")), "<include/>"))
        assert _loaded_top_bodies(model) == ["direct"]


def _mujoco_top_bodies(mujoco, model) -> list[str]:
    """The compiled model's top-level body names, in MuJoCo's own index order."""
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        for i in range(1, model.nbody)
        if model.body_parentid[i] == 0
    ]


class TestMuJoCoAgreesOnTheModelsBodies:
    """Premise + oracle: every fixture compiles, and the compiler names the same bodies.

    The expected names above are not a restatement - this reads them back out of
    MuJoCo, so a fixture that stopped being a valid model, or a loader that
    stopped agreeing with the compiler, fails here.
    """

    @pytest.mark.parametrize(("top", "fragments", "expected"), _FIXTURES, ids=_FIXTURE_IDS)
    def test_the_compiler_reads_the_same_top_level_bodies(self, tmp_path, top, fragments, expected):
        mujoco = pytest.importorskip("mujoco")
        model = _write(tmp_path, top, fragments)
        compiled = mujoco.MjModel.from_xml_path(model)
        assert _mujoco_top_bodies(mujoco, compiled) == expected
        assert _loaded_top_bodies(model) == _mujoco_top_bodies(mujoco, compiled)

    def test_the_compiler_refuses_no_fixture(self, tmp_path):
        """Non-vacuity: the parametrized set is non-empty and every model compiles."""
        assert len(_FIXTURES) >= 5
