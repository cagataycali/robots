"""A scene-creation call refuses a name that cannot address the entity it creates.

``add_object`` / ``add_camera`` / ``add_robot`` each CLAIM a name: they write it
into the world registry as a key and into the MJCF spec as an element name. Three
kinds of value break the link between those two layers, and each did so silently:

* A non-``str`` is hashable often enough to be registered - ``7`` and ``True``
  both are - and only then reaches the spec build, where pybind11 raises
  ``TypeError: add_body(): incompatible function arguments``. That escapes the
  result dict these methods document, and it lands AFTER the registry write, so
  the world was left holding a key for an entity with no body in the model.
* An empty name is MuJoCo's own sentinel for an unnamed element, so the entity
  compiled anonymously. ``get_body_state(body_name="")`` then reports the body as
  missing while it simulates, and ``render(camera_name="")`` routes to the FREE
  camera by an explicit token check - so a camera created under that name is
  never the camera rendered from, and the caller is handed an image anyway.
* A NUL leaves the layers disagreeing: MuJoCo compares names only up to it, so
  ``"a\\x00b"`` compiles as ``"a"`` while the registry keeps the full string.

The domain stops there because nothing else is broken: ``"a/b"``, ``"a b"``,
``"a-b"`` and a name carrying a newline or a quote each compile under the name
given and are addressable by it, so they are pinned as accepted here to keep the
guard from widening into a character allowlist that would refuse working names.

``add_robot`` documents a falsy ``name`` as "derive one from the model", so its
guard is gated on the same truthiness test that branch already uses: every value
that derives today still derives, and only a supplied name is held to the domain.
"""

import mujoco as mj
import pytest

from strands_robots.simulation.mujoco.simulation import Simulation
from strands_robots.utils import entity_name_error

# Values that cannot address the entity they would name, with the layer each
# breaks. Parametrized as (label, value) so a failure names the case.
UNUSABLE: list[tuple[str, object]] = [
    ("empty", ""),
    ("nul", "a\x00b"),
    ("int", 7),
    ("bool", True),
    ("unhashable", ["x"]),
]

# Names that round-trip: the compiled element answers to exactly the string
# given. These are the guard's over-reach controls.
USABLE = ["plain", "a/b", "a b", "a-b", 'a"b']

_ARM_XML = """<mujoco>
  <compiler angle="radian"/>
  <worldbody>
    <body name="link" pos="0 0 0.1">
      <joint name="pan" type="hinge" axis="0 0 1" damping="1" range="-2 2" limited="true"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02"/>
    </body>
  </worldbody>
  <actuator><position name="pan_act" joint="pan" kp="20" ctrlrange="-1 1"/></actuator>
</mujoco>
"""


# ``sim`` is deliberately un-annotated: ``Simulation`` is a lazy re-export, and
# annotating it makes mypy read ``sim._world`` as ``SimWorld | None`` at every
# model probe below.
@pytest.fixture
def sim():
    engine = Simulation(backend="mujoco", tool_name="name_domain", mesh=False)
    engine.create_world()
    yield engine
    engine.cleanup()


@pytest.fixture
def arm_xml(tmp_path):
    path = tmp_path / "arm.xml"
    path.write_text(_ARM_XML, encoding="utf-8")
    return str(path)


def _body_names(sim) -> list[str]:
    model = sim._world._model
    return [mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]


class TestMuJoCoTreatsTheseNamesAsUnusable:
    """The premises the refusals rest on, measured against MuJoCo itself.

    Without these, the guard's domain would be a matter of taste. With them it is
    a property of the engine the names are compiled into.
    """

    def test_an_empty_name_is_mujocos_unnamed_sentinel(self, sim):
        model = sim._world._model
        assert mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "") == -1

    def test_mujoco_compares_a_name_only_up_to_a_nul(self, sim):
        # Compiled under the full string; MuJoCo answers to the prefix, and to
        # the prefix ALONE - one entity reachable under a name nobody asked for.
        sim._world.objects.clear()
        assert sim.add_object(name="a", shape="box", size=[0.05] * 3, position=[0, 0, 0.3])["status"] == "success"
        model = sim._world._model
        assert mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "a\x00b") == mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "a")

    def test_an_empty_camera_name_is_the_free_camera_token(self, sim):
        # ``render`` documents "", None, "default" and "free" as free-camera
        # tokens, so a camera stored under "" could never be the one rendered
        # from - the caller is handed the free view and told it succeeded.
        assert sim.add_camera(name="probe", position=[0.6, 0.6, 0.5], target=[0, 0, 0])["status"] == "success"
        free = sim.render(camera_name="", width=64, height=48)
        assert free["status"] == "success"
        assert any("image" in chunk for chunk in free["content"])


class TestACreatorRefusesAnUnaddressableName:
    @pytest.mark.parametrize(("label", "value"), UNUSABLE, ids=[label for label, _ in UNUSABLE])
    def test_add_object_refuses_it_and_registers_nothing(self, sim, label, value):
        result = sim.add_object(name=value, shape="box", size=[0.05] * 3, position=[0, 0, 0.3])
        assert result["status"] == "error"
        assert "add_object: name" in result["content"][0]["text"]
        # The refusal precedes the registry write, so there is no orphan entry
        # to roll back - which is what the escaping TypeError used to leave.
        assert sim._world.objects == {}
        assert _body_names(sim) == ["world"]

    @pytest.mark.parametrize(("label", "value"), UNUSABLE, ids=[label for label, _ in UNUSABLE])
    def test_add_camera_refuses_it_and_registers_nothing(self, sim, label, value):
        result = sim.add_camera(name=value, position=[0.6, 0.6, 0.5], target=[0, 0, 0])
        assert result["status"] == "error"
        assert "add_camera: name" in result["content"][0]["text"]
        assert list(sim._world.cameras) == ["default"]

    @pytest.mark.parametrize(("label", "value"), UNUSABLE, ids=[label for label, _ in UNUSABLE])
    def test_add_robot_refuses_a_supplied_one(self, sim, arm_xml, label, value):
        if not value:
            pytest.skip("a falsy name is documented as 'derive one from the model'")
        result = sim.add_robot(name=value, urdf_path=arm_xml)
        assert result["status"] == "error"
        assert "add_robot: name" in result["content"][0]["text"]
        assert sim._world.robots == {}


class TestAWorkingNameIsStillAccepted:
    """The guard must not widen into a character allowlist."""

    @pytest.mark.parametrize("name", USABLE)
    def test_add_object_accepts_it_and_the_body_answers_to_it(self, sim, name):
        assert sim.add_object(name=name, shape="box", size=[0.05] * 3, position=[0, 0, 0.3])["status"] == "success"
        assert name in _body_names(sim)
        assert sim.get_body_state(body_name=name)["status"] == "success"

    @pytest.mark.parametrize("name", USABLE)
    def test_add_camera_accepts_it_and_renders_from_it(self, sim, name):
        assert sim.add_camera(name=name, position=[0.6, 0.6, 0.5], target=[0, 0, 0])["status"] == "success"
        rendered = sim.render(camera_name=name, width=64, height=48)
        assert rendered["status"] == "success"
        assert any("image" in chunk for chunk in rendered["content"])

    @pytest.mark.parametrize("omitted", [None, ""])
    def test_add_robot_still_derives_a_label_when_none_is_supplied(self, sim, arm_xml, omitted):
        result = sim.add_robot(name=omitted, urdf_path=arm_xml)
        assert result["status"] == "success"
        assert list(sim._world.robots) == ["arm"]


class TestTheThreeCreatorsShareOneDomain:
    """A name one creator refuses must be refused by the others.

    Parametrized over factories rather than values so that each creator is
    handed its own instance of the value.
    """

    @pytest.mark.parametrize(
        ("label", "value"),
        [(label, value) for label, value in UNUSABLE if value],
        ids=[label for label, value in UNUSABLE if value],
    )
    def test_every_creator_refuses_the_same_unusable_name(self, sim, arm_xml, label, value):
        verdicts = {
            "add_object": sim.add_object(name=value, shape="box", size=[0.05] * 3)["status"],
            "add_camera": sim.add_camera(name=value, position=[0.6, 0.6, 0.5])["status"],
            "add_robot": sim.add_robot(name=value, urdf_path=arm_xml)["status"],
        }
        assert set(verdicts.values()) == {"error"}, verdicts

    @pytest.mark.parametrize("name", USABLE)
    def test_every_creator_accepts_the_same_usable_name(self, sim, arm_xml, name):
        verdicts = {
            "add_object": sim.add_object(name=name, shape="box", size=[0.05] * 3)["status"],
            "add_camera": sim.add_camera(name=f"{name}_cam", position=[0.6, 0.6, 0.5])["status"],
            "add_robot": sim.add_robot(name=f"{name}_arm", urdf_path=arm_xml)["status"],
        }
        assert set(verdicts.values()) == {"success"}, verdicts


class TestEntityNameErrorMessages:
    def test_a_usable_name_returns_none(self):
        assert entity_name_error("crate", "name", "add_object") is None

    def test_a_non_string_names_the_type_and_offers_the_string_form(self):
        message = entity_name_error(7, "name", "add_object")
        assert message is not None
        assert "must be a string" in message
        assert "(int)" in message
        # The remedy has to be pasteable, so the quoted form is the one to pass.
        assert "'7'" in message

    def test_an_empty_name_says_why_nothing_could_address_it(self):
        message = entity_name_error("", "name", "add_camera")
        assert message is not None
        assert message.startswith("add_camera: name must be a non-empty string")
        assert "unnamed element" in message

    def test_a_nul_name_reports_the_name_the_entity_would_answer_to(self):
        message = entity_name_error("crate\x00x", "name", "add_object")
        assert message is not None
        assert "NUL" in message
        assert "'crate'" in message
