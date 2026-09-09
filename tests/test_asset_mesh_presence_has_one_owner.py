"""Mesh presence is answered by one owner, so every reader agrees about a model.

"Are this model's meshes on disk?" has a single owner,
:func:`strands_robots.assets.download._mjcf_missing_meshes`, which resolves each
``file=`` reference the way MuJoCo does: against the model's ``<compiler
meshdir>``, read once across every ``<include>``d fragment. Two readers ask it -
:func:`~strands_robots.assets.download._needs_download` (should the assets be
fetched?) and
:meth:`~strands_robots.simulation.mujoco.simulation.MuJoCoSimEngine._ensure_meshes`
(may ``add_robot`` proceed?).

:func:`~strands_robots.assets.manager.resolve_model_path` is the third reader,
and it documents the same question as its own second download trigger: "XML is
found but mesh files are missing, downloads the asset". Answering that by walking
the model directory for files with a mesh extension is a different reading, and
the two disagree in BOTH directions:

* ``<compiler meshdir="../meshes/"/>`` puts the meshes outside the model's own
  directory, so a downward walk reports a complete asset as mesh-less and the
  resolver reaches for a fetch that cannot change its answer - on every call,
  for a model that already loads.
* a model missing one of the meshes it declares still has its other meshes on
  disk, so a walk reports it as fine, the documented fetch never fires, and the
  caller is handed a path MuJoCo refuses to load.

The cells below pin both directions through the public resolver, plus the
agreement with ``_needs_download`` that makes "one owner" observable.
"""

import ast
import inspect
from pathlib import Path

import pytest

import strands_robots.assets.manager as manager
from strands_robots.assets.download import _mjcf_missing_meshes, _needs_download
from strands_robots.registry import get_robot, list_robots
from strands_robots.registry.user_registry import _invalidate_cache, register_robot


def _mjcf(*, meshdir: str | None = None, declares: tuple[str, ...] = ()) -> str:
    """A minimal MJCF that declares *declares* resolved against *meshdir*."""
    compiler = f'<compiler meshdir="{meshdir}"/>' if meshdir else ""
    refs = "".join(f'<mesh name="m{i}" file="{m}"/>' for i, m in enumerate(declares))
    asset = f"<asset>{refs}</asset>" if refs else ""
    return f'<mujoco>{compiler}{asset}<worldbody><body><geom size="0.1"/></body></worldbody></mujoco>'


@pytest.fixture(autouse=True)
def _isolate_assets(tmp_path, monkeypatch):
    """Point the asset search at a temp tree and clear the registry cache."""
    assets = tmp_path / "assets"
    assets.mkdir()
    monkeypatch.setenv("STRANDS_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("STRANDS_ASSETS_DIR", str(assets))
    _invalidate_cache()
    yield
    _invalidate_cache()


def _register(
    assets: Path,
    *,
    model_xml: str = "unitbot.xml",
    meshdir: str | None = None,
    declares: tuple[str, ...] = (),
    present: tuple[str, ...] = (),
) -> Path:
    """Write and register a model, then place *present* under its ``meshdir``.

    *present* is relative to the resolved mesh directory, so a reference in
    *declares* that is absent from *present* is a mesh the model asks for and
    does not have.
    """
    model = assets / "unitbot" / model_xml
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_text(_mjcf(meshdir=meshdir, declares=declares))
    mesh_root = (model.parent / meshdir) if meshdir else model.parent
    for ref in present:
        target = (mesh_root / ref).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"meshbytes")
    register_robot(
        name="unitbot",
        model_xml=model_xml,
        description="unit test robot",
        category="arm",
        joints=6,
        overwrite=True,
    )
    _invalidate_cache()
    return model


def _record_attempts(monkeypatch) -> list[str]:
    """Install a downloader that records the names it is asked for and declines.

    Declining is what keeps this off the network: a failed fetch is a documented
    outcome of both triggers, so the resolved path is the same one a real
    download failure yields.
    """
    seen: list[str] = []

    def _decline(name: str, _info: dict) -> bool:
        seen.append(name)
        return False

    monkeypatch.setattr(manager, "_auto_download_robot", _decline)
    return seen


def _attempts(monkeypatch, **kwargs) -> tuple[list[str], Path | None]:
    """Resolve ``unitbot``, recording every download attempt without running one."""
    seen = _record_attempts(monkeypatch)
    return seen, manager.resolve_model_path("unitbot", **kwargs)


class TestAModelWhoseMeshesAreOutsideItsOwnDirectory:
    """The shipped ``meshdir="../meshes/"`` layout is a complete asset.

    ``aliengo``, ``unitree_a1``, ``jvrc`` and ``asimov_v0`` all ship as
    ``<robot>/xml/<model>.xml`` declaring ``<compiler meshdir="../meshes/"/>``,
    with the mesh files in ``<robot>/meshes/``. MuJoCo loads every one of them.
    """

    @staticmethod
    def _shipped_layout(assets: Path) -> Path:
        return _register(
            assets,
            model_xml="xml/unitbot.xml",
            meshdir="../meshes/",
            declares=("trunk.stl", "calf.stl"),
            present=("trunk.stl", "calf.stl"),
        )

    def test_the_owner_reports_the_asset_complete(self, tmp_path):
        """Premise: the references really do resolve, one level up."""
        model = self._shipped_layout(tmp_path / "assets")
        assert _mjcf_missing_meshes(model) == []
        assert (model.parent.parent / "meshes" / "trunk.stl").exists()
        assert not list(model.parent.glob("*.stl")), "the meshes are NOT beside the model"

    def test_a_complete_asset_is_resolved_without_reaching_for_a_download(self, tmp_path, monkeypatch):
        model = self._shipped_layout(tmp_path / "assets")
        seen, resolved = _attempts(monkeypatch)
        assert resolved == model
        assert seen == [], f"fetched a model whose declared meshes are all on disk: {seen}"

    def test_the_fetch_it_would_attempt_is_not_satisfiable(self, tmp_path, monkeypatch):
        """A download cannot change a reading taken in the wrong directory.

        This is what separates a slow first call from a permanent one: the
        condition is false for a reason no fetch addresses, so it is false again
        on the next call, and every call after that.
        """
        self._shipped_layout(tmp_path / "assets")
        seen, _ = _attempts(monkeypatch)
        first = list(seen)
        seen.clear()
        manager.resolve_model_path("unitbot")
        assert (first, seen) == ([], []), "the fetch repeats on every call"


class TestAModelMissingOneOfTheMeshesItDeclares:
    """The documented second trigger: XML present, a declared mesh absent."""

    @staticmethod
    def _partial(assets: Path) -> Path:
        return _register(
            assets,
            meshdir="assets",
            declares=("base.stl", "gripper.stl"),
            present=("base.stl",),  # gripper.stl is never written
        )

    def test_the_owner_names_the_absent_reference(self, tmp_path):
        """Premise: one of two declared references is missing."""
        model = self._partial(tmp_path / "assets")
        assert _mjcf_missing_meshes(model) == ["gripper.stl"]

    def test_the_download_the_resolver_documents_is_attempted(self, tmp_path, monkeypatch):
        self._partial(tmp_path / "assets")
        seen, resolved = _attempts(monkeypatch)
        assert seen == ["unitbot"], "a declared mesh is absent and no fetch was attempted"
        assert resolved is not None, "a failed fetch still yields the XML"

    def test_the_meshes_that_are_present_do_not_hide_it(self, tmp_path, monkeypatch):
        """The sibling mesh on disk is exactly what a directory walk finds first."""
        model = self._partial(tmp_path / "assets")
        assert list((model.parent / "assets").glob("*.stl")), "a mesh IS present beside the missing one"
        seen, _ = _attempts(monkeypatch)
        assert seen == ["unitbot"]


#: ``(id, meshdir, declared, present, the asset is complete)`` - the layouts a
#: model can ship in, including the two the readings disagree about.
_LAYOUTS = [
    ("meshes-beside-the-model", None, ("a.stl",), ("a.stl",), True),
    ("meshes-one-level-down", "assets", ("a.stl",), ("a.stl",), True),
    ("meshes-one-level-up", "../meshes", ("a.stl",), ("a.stl",), True),
    ("declares-nothing", None, (), (), True),
    ("one-of-two-absent", "assets", ("a.stl", "b.stl"), ("a.stl",), False),
    ("the-only-one-absent", "assets", ("a.stl",), (), False),
    ("absent-under-an-upward-meshdir", "../meshes", ("a.stl", "b.stl"), ("a.stl",), False),
]


class TestTheResolverAgreesWithTheDownloadDecision:
    """One owner, observable: the two readers reach the same verdict.

    ``_needs_download`` is the download module's own answer to "should these
    assets be fetched?". The resolver documents the same question as its second
    trigger, so a layout where one fetches and the other does not is a model the
    package judges present and absent at once.
    """

    @pytest.mark.parametrize(
        ("meshdir", "declared", "present", "complete"),
        [pytest.param(*row[1:], id=row[0]) for row in _LAYOUTS],
    )
    def test_the_fetch_decision_matches(self, tmp_path, monkeypatch, meshdir, declared, present, complete):
        model = _register(
            tmp_path / "assets",
            model_xml="xml/unitbot.xml" if meshdir and meshdir.startswith("..") else "unitbot.xml",
            meshdir=meshdir,
            declares=declared,
            present=present,
        )
        assert (_mjcf_missing_meshes(model) == []) is complete, "fixture does not pose what it claims"

        wants = _needs_download("unitbot", get_robot("unitbot"))
        seen, resolved = _attempts(monkeypatch)

        assert bool(seen) == wants, f"resolver fetched={bool(seen)} while _needs_download said {wants}"
        assert wants is not complete, "a complete asset has nothing to fetch"
        assert resolved == model


class TestTheResolverKeepsItsOtherBehaviour:
    """Controls: the trigger this change does not touch, and the declining knob."""

    def test_an_absent_xml_still_reports_a_miss_after_attempting(self, tmp_path, monkeypatch):
        """First trigger, unchanged: no XML anywhere."""
        model = _register(tmp_path / "assets", declares=("a.stl",), present=("a.stl",))
        model.unlink()
        _invalidate_cache()
        seen, resolved = _attempts(monkeypatch)
        assert (seen, resolved) == (["unitbot"], None)

    def test_declining_never_attempts_on_either_trigger(self, tmp_path, monkeypatch):
        _register(tmp_path / "assets", meshdir="assets", declares=("a.stl",), present=())
        seen, resolved = _attempts(monkeypatch, allow_download=False)
        assert seen == []
        assert resolved is not None, "declining keeps the XML it found"


class TestOnlyOneReaderAnswersTheQuestion:
    """Structural: the resolver holds no second reading of mesh presence.

    A copy of the rule is what let the readings drift apart, so the absence of
    one is the property worth pinning rather than the wording of the one that
    remains.
    """

    @staticmethod
    def _tree() -> ast.Module:
        return ast.parse(Path(inspect.getfile(manager)).read_text())

    def test_the_resolver_asks_the_owner(self):
        calls = {
            n.func.id
            for n in ast.walk(ast.parse(inspect.getsource(manager._model_meshes_resolve)))
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        assert "_mjcf_missing_meshes" in calls, f"mesh presence is answered locally: {sorted(calls)}"

    def test_no_mesh_extension_set_is_declared_here(self):
        """A module-level set of mesh suffixes is the shape of a second reader."""
        literals = [
            ast.unparse(n)
            for n in ast.walk(self._tree())
            if isinstance(n, ast.Constant)
            and isinstance(n.value, str)
            and n.value.lower() in {".stl", ".obj", ".msh", ".ply"}
        ]
        assert literals == [], f"a mesh-extension reading lives beside the owner: {literals}"

    def test_every_mesh_presence_read_routes_through_the_one_helper(self):
        """Non-vacuity: the helper exists and the resolver is the only caller."""
        names = [
            n.func.id
            for n in ast.walk(self._tree())
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "_model_meshes_resolve"
        ]
        assert len(names) == 2, f"expected the two ranking passes to ask, got {len(names)}"


class TestTheShippedCorpusAgrees:
    """The same agreement over whatever assets are actually installed.

    Skipped where no asset is on disk. Where they are, this is the reading that
    reports a shipped ``meshdir="../meshes/"`` model as needing a fetch it
    cannot use.
    """

    def test_no_installed_robot_is_both_complete_and_fetched(self, monkeypatch):
        monkeypatch.delenv("STRANDS_ASSETS_DIR", raising=False)
        monkeypatch.delenv("STRANDS_BASE_DIR", raising=False)
        _invalidate_cache()
        seen = _record_attempts(monkeypatch)

        disagree = []
        checked = 0
        for entry in list_robots(mode="sim"):
            name = entry["name"]
            if not manager.is_robot_asset_present(name):
                continue
            checked += 1
            seen.clear()
            manager.resolve_model_path(name)
            wants = _needs_download(name, get_robot(name))
            if bool(seen) != wants:
                disagree.append((name, bool(seen), wants))

        if checked == 0:
            pytest.skip("no robot assets installed on this machine")
        assert disagree == [], f"resolver and _needs_download disagree about {disagree}"
