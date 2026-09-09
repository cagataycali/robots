"""A registry-stored asset path is joined the same way wherever it is joined.

``register_robot`` stores ``model_xml``/``scene_xml`` relative to the robot's
asset directory, and every *reader* joins them back through
:func:`strands_robots.utils.safe_join` - both branches of
:func:`~strands_robots.assets.manager.resolve_model_path` and of
:func:`~strands_robots.assets.manager.is_robot_asset_present`. Three writers and
deciders joined the same value raw, so each of them answered a question about a
file the readers will never open:

* ``register_robot`` validated existence with ``resolved_dir / model_xml``. An
  absolute value discards ``resolved_dir`` entirely, so any existing host file
  satisfied the check the registration exists to make - and the deferred
  ``add_robot()`` failure it was added to prevent came back, now with the entry
  persisted and ``_user_asset_path`` naming a directory the stored path is not in.
* :func:`~strands_robots.assets.download._needs_download` read
  ``search_dir / asset_dir / xml_file``, so an out-of-tree model made a robot
  read as present and needing no fetch while the resolver found nothing - the
  two readings of one entry that :func:`_mjcf_missing_meshes` exists to keep
  together.
* :func:`~strands_robots.assets.download._download_via_robot_descriptions`
  validated a freshly linked directory with ``dst / model_xml``, so an absolute
  value reported ``downloaded`` for a link holding no model at all.

These pin the containment, that the refusal names which of the two values to
correct, and that the documented relative shapes are untouched.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from strands_robots.assets import download as dl
from strands_robots.assets.manager import is_robot_asset_present, resolve_model_path
from strands_robots.registry.user_registry import get_user_robots, register_robot

# A minimal model with no mesh references: present on disk means present, so a
# containment verdict is never confused with a missing-mesh verdict.
_MJCF = '<mujoco model="probe"><worldbody><geom type="box" size="1 1 1"/></worldbody></mujoco>'

# Kept as a list of pairs, not a dict: the reason is what a failure should read
# out, and two values may share one.
_ESCAPING = [
    ("../outsider.xml", "traverses one level out of the asset directory"),
    ("../../../../../../etc/hostname", "traverses out to a host file that exists"),
    ("/etc/hostname", "absolute: a raw join discards the asset directory"),
    ("nested/../../outsider.xml", "escapes after a legal-looking first component"),
]

_CONTAINED = [
    ("bot.xml", "the documented shape: a file in the asset directory"),
    ("xml/bot.xml", "a nested component, still inside"),
]


@pytest.fixture
def asset_dir(tmp_path: Path) -> Path:
    """A robot asset directory, with an XML one level outside it that exists."""
    root = tmp_path / "assets" / "probe_arm"
    (root / "xml").mkdir(parents=True)
    (root / "bot.xml").write_text(_MJCF)
    (root / "xml" / "bot.xml").write_text(_MJCF)
    (root.parent / "outsider.xml").write_text(_MJCF)
    return root


@pytest.mark.parametrize(("value", "why"), _ESCAPING, ids=[v for v, _ in _ESCAPING])
def test_register_robot_refuses_a_model_xml_outside_the_asset_dir(value: str, why: str, asset_dir: Path) -> None:
    """The existence check cannot be satisfied by a file the readers refuse."""
    with pytest.raises(ValueError, match="model_xml") as excinfo:
        register_robot(name="probe_arm", model_xml=value, asset_dir=str(asset_dir), joints=6)
    assert value in str(excinfo.value), f"the refusal does not quote the value ({why})"
    assert "probe_arm" not in get_user_robots(), "an escaping registration was persisted anyway"


def test_register_robot_refuses_a_scene_xml_outside_the_asset_dir(asset_dir: Path) -> None:
    """A scene is read back the same way, so it is contained the same way.

    It is not existence-checked (a scene may be authored after registration),
    which is why the refusal has to name it: the model beside it is valid.
    """
    with pytest.raises(ValueError, match="scene_xml") as excinfo:
        register_robot(
            name="probe_arm",
            model_xml="bot.xml",
            scene_xml="../outsider.xml",
            asset_dir=str(asset_dir),
            joints=6,
        )
    assert "model_xml" not in str(excinfo.value), "the refusal names the wrong value"
    assert "probe_arm" not in get_user_robots()


@pytest.mark.parametrize(("value", "why"), _CONTAINED, ids=[v for v, _ in _CONTAINED])
def test_a_contained_model_path_registers_and_resolves(value: str, why: str, asset_dir: Path) -> None:
    """Containment refuses only what the readers already refused."""
    entry = register_robot(name="probe_arm", model_xml=value, asset_dir=str(asset_dir), joints=6)
    assert entry["asset"]["model_xml"] == value, why
    resolved = resolve_model_path("probe_arm", allow_download=False)
    assert resolved == asset_dir / value, f"registered but not resolvable ({why})"
    assert is_robot_asset_present("probe_arm") is True


def test_a_model_outside_the_search_path_is_not_read_as_already_present(tmp_path: Path) -> None:
    """The fetch decision and the resolver agree about one entry.

    A hand-edited ``user_robots.json`` reaches
    :func:`~strands_robots.assets.download._needs_download` without passing
    ``register_robot``, and an absolute ``model_xml`` pointed it at a readable
    model outside every search path: no meshes missing, so nothing to fetch,
    for a robot :func:`~strands_robots.assets.manager.resolve_model_path`
    cannot resolve.
    """
    outside = tmp_path / "outside.xml"
    outside.write_text(_MJCF)
    info = {"asset": {"dir": "probe_arm", "model_xml": str(outside), "scene_xml": str(outside)}}

    assert dl._needs_download("probe_arm", info) is True, (
        "a model outside every asset search path read as present and needing no fetch"
    )


def test_a_robot_descriptions_link_is_not_validated_by_a_file_outside_it(tmp_path: Path, monkeypatch) -> None:
    """``downloaded`` means the linked directory holds the declared model."""
    package = tmp_path / "rd_package"
    package.mkdir()  # the linked robot_descriptions package holds no model at all
    outside = tmp_path / "outside.xml"
    outside.write_text(_MJCF)
    monkeypatch.setitem(
        sys.modules,
        "robot_descriptions.fake_mj_description",
        SimpleNamespace(PACKAGE_PATH=str(package)),
    )
    dest = tmp_path / "assets"  # STRANDS_ASSETS_DIR, from the shared isolation fixture
    info = {
        "asset": {
            "dir": "faker",
            "model_xml": str(outside),
            "scene_xml": str(outside),
            "robot_descriptions_module": "fake_mj_description",
        }
    }

    result = dl._download_via_robot_descriptions({"faker": info}, dest)

    assert not result["faker"].startswith("downloaded"), f"a link holding no model reported {result['faker']!r}"


def test_no_stored_asset_path_is_joined_raw() -> None:
    """No ``/`` join of a stored asset path is left in the package.

    A scan rather than a roster of modules, so a fourth writer of the same
    field cannot be added without either using ``safe_join`` or failing here.
    """
    package = Path(dl.__file__).parent.parent
    raw: list[str] = []
    for path in sorted(package.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                joined = ast.unparse(node)
                if any(field in joined for field in ("model_xml", "scene_xml", "xml_file")):
                    raw.append(f"{path.relative_to(package)}:{node.lineno}: {joined}")

    assert not raw, "join these through strands_robots.utils.safe_join:\n" + "\n".join(raw)
