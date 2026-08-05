"""The declared huggingface_hub floor must actually ship the bucket CLI.

``sync_dataset_to_bucket`` runs two subcommands of the ``hf`` CLI:
``hf buckets create`` and ``hf sync``. Both first ship in **huggingface_hub
1.5.0**, as ``huggingface_hub/cli/buckets.py`` registered by ``cli/hf.py``.
Every earlier release installs an ``hf`` entry point without that module and
answers either invocation with ``Error: No such command 'buckets'`` / ``'sync'``.

The library's version gate, its upgrade instructions and the ``[wbc]`` extra's
pin all previously named ``>=1.0``, which is two-and-a-half minor releases below
the capability. That made three claims wrong at once:

* a caller on 1.0-1.4.x passed the gate, so the gate that exists to replace CLI
  usage noise with an upgrade instruction stayed silent for exactly the releases
  it describes;
* the upgrade instruction the library printed - ``pip install -U
  'huggingface_hub>=1.0'`` - could be followed to the letter and still resolve a
  CLI that cannot run a bucket sync; and
* the ``[wbc]`` pin let a fresh resolve land on such a CLI while satisfying the
  documented minimum.

Every assertion below reads the floor from
``dataset_recorder._HF_BUCKET_CLI_MIN_VERSION`` rather than restating it, so the
gate, the messages, the docs and the packaging pin cannot drift apart again.
"""

from __future__ import annotations

import re
import subprocess
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

from strands_robots import dataset_recorder

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PYPROJECT = _REPO_ROOT / "pyproject.toml"
_README = _REPO_ROOT / "README.md"
_RECORDER_SRC = Path(dataset_recorder.__file__)

_FLOOR = Version(".".join(str(part) for part in dataset_recorder._HF_BUCKET_CLI_MIN_VERSION))

# Releases whose `hf` entry point exists but carries no `buckets`/`sync`
# subcommand. Verified against the published wheels: `huggingface_hub/cli/
# buckets.py` is absent from every one of them, and 1.4.1's CLI answers
# `hf buckets create demo` with "Error: No such command 'buckets'" (rc=2).
_WITHOUT_THE_SUBCOMMANDS = ["0.36.2", "1.0.0", "1.0.1", "1.1.0", "1.2.4", "1.3.7", "1.4.0", "1.4.1"]

# Releases that ship them. 1.5.0 is the first; the CLI there answers the same
# invocation with rc=0.
_WITH_THE_SUBCOMMANDS = ["1.5.0", "1.6.0", "1.7.0", "1.26.0"]


def _dataset_root(tmp_path: Path) -> Path:
    """A finalized-looking dataset directory (``meta/`` present)."""
    root = tmp_path / "cube_pick"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text('{"fps": 30}')
    return root


def _install_hint_floors(text: str) -> list[Version]:
    """Every ``huggingface_hub>=X`` floor a piece of guidance names.

    The version is matched without a trailing separator so a pin that ends a
    sentence (``... needs huggingface_hub>=1.5.``) parses as ``1.5`` rather than
    raising ``InvalidVersion`` on ``"1.5."``.
    """
    return [Version(m) for m in re.findall(r"huggingface_hub>=([0-9]+(?:\.[0-9]+)*)", text)]


class TestTheGateRefusesEveryReleaseWithoutTheSubcommands:
    """The accepted domain is exactly the releases that can honor a sync."""

    @pytest.mark.parametrize("version", _WITHOUT_THE_SUBCOMMANDS)
    def test_a_release_without_the_subcommands_is_refused(self, version, monkeypatch):
        import huggingface_hub

        monkeypatch.setattr(huggingface_hub, "__version__", version)
        problem = dataset_recorder._huggingface_hub_version_error()
        assert problem is not None, f"huggingface_hub {version} has no `hf buckets`/`hf sync` but the gate accepted it"
        assert version in problem, "the refusal must quote the installed version"

    @pytest.mark.parametrize("version", _WITH_THE_SUBCOMMANDS)
    def test_a_release_that_ships_them_is_accepted(self, version, monkeypatch):
        import huggingface_hub

        monkeypatch.setattr(huggingface_hub, "__version__", version)
        assert dataset_recorder._huggingface_hub_version_error() is None, (
            f"huggingface_hub {version} ships the subcommands and must not be refused"
        )

    def test_the_floor_itself_is_the_boundary(self, monkeypatch):
        """The refused/accepted split falls exactly at the declared floor."""
        import huggingface_hub

        major, minor = dataset_recorder._HF_BUCKET_CLI_MIN_VERSION
        monkeypatch.setattr(huggingface_hub, "__version__", f"{major}.{minor}.0")
        assert dataset_recorder._huggingface_hub_version_error() is None
        monkeypatch.setattr(huggingface_hub, "__version__", f"{major}.{minor - 1}.99")
        assert dataset_recorder._huggingface_hub_version_error() is not None


class TestTheUpgradeInstructionNamesAReleaseThatShipsThem:
    """A remedy that resolves a CLI without the subcommands is not a remedy."""

    def test_the_version_gate_remedy_is_at_least_the_capability_floor(self, monkeypatch):
        import huggingface_hub

        monkeypatch.setattr(huggingface_hub, "__version__", "1.4.1")
        problem = dataset_recorder._huggingface_hub_version_error()
        assert problem is not None
        floors = _install_hint_floors(problem)
        assert floors, f"the upgrade instruction names no huggingface_hub floor: {problem!r}"
        assert all(f >= _FLOOR for f in floors), (
            f"the upgrade instruction advises a release without the subcommands: {problem!r}"
        )
        assert "pip install -U" in problem, "the remedy must be a runnable install command"

    def test_the_cli_not_found_message_names_the_same_floor(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dataset_recorder, "_hf_executable", lambda: None)
        result = dataset_recorder.sync_dataset_to_bucket(root=_dataset_root(tmp_path), bucket="my-org/robot-fave")
        assert result["status"] == "error"
        floors = _install_hint_floors(result["message"])
        assert floors and all(f >= _FLOOR for f in floors), (
            f"the `hf` CLI-not-found remedy advises a release without the subcommands: {result['message']!r}"
        )


class TestTheRefusalPrecedesTheSubprocess:
    """A too-old hub is reported without spawning the CLI that cannot serve it."""

    def test_no_subprocess_runs_for_a_release_without_the_subcommands(self, tmp_path, monkeypatch):
        import huggingface_hub

        monkeypatch.setattr(dataset_recorder, "_hf_executable", lambda: "hf")
        monkeypatch.setattr(huggingface_hub, "__version__", "1.4.1")

        def _boom(*_args, **_kwargs):
            raise AssertionError("the CLI must not be spawned when it cannot serve the request")

        monkeypatch.setattr(subprocess, "run", _boom)
        result = dataset_recorder.sync_dataset_to_bucket(root=_dataset_root(tmp_path), bucket="my-org/robot-fave")
        assert result["status"] == "error"
        assert "No such command" not in result["message"], (
            "the caller got raw CLI usage noise instead of an upgrade instruction"
        )


class TestEveryDeclaredFloorAgrees:
    """Guidance, packaging and the gate name one version."""

    @pytest.mark.parametrize(
        "path", [_README, _RECORDER_SRC, _PYPROJECT], ids=["README", "dataset_recorder", "pyproject"]
    )
    def test_no_documented_floor_is_below_the_capability(self, path):
        floors = _install_hint_floors(path.read_text())
        assert floors, f"{path.name} names no huggingface_hub floor to check"
        assert all(f >= _FLOOR for f in floors), (
            f"{path.name} names a huggingface_hub floor below {_FLOOR}, which has no "
            f"`hf buckets`/`hf sync`: {[str(f) for f in floors]}"
        )

    def test_the_packaging_floor_is_at_least_the_capability_floor(self):
        extras = tomllib.loads(_PYPROJECT.read_text())["project"]["optional-dependencies"]
        specs = [s for s in extras["wbc"] if Requirement(s).name.replace("-", "_").lower() == "huggingface_hub"]
        assert len(specs) == 1, f"expected exactly one huggingface_hub pin in [wbc], got {specs!r}"
        lower = min(s.version for s in Requirement(specs[0]).specifier if s.operator == ">=")
        assert Version(lower) >= _FLOOR, (
            f"[wbc] resolves an `hf` CLI that may lack the buckets/sync subcommands: {specs[0]!r}"
        )


class TestTheFloorsClaimIsExecutable:
    """The installed hub really carries what the floor says it does.

    A prose claim about a dependency goes stale silently. Reading the CLI
    surface out of the installed package means a rename upstream fails here
    instead of turning the advice quietly wrong.
    """

    def test_the_installed_hub_ships_the_registered_subcommands(self):
        huggingface_hub = pytest.importorskip("huggingface_hub")
        installed = Version(huggingface_hub.__version__)
        if installed < _FLOOR:
            pytest.skip(f"installed huggingface_hub {installed} predates the bucket CLI floor {_FLOOR}")
        buckets = pytest.importorskip("huggingface_hub.cli.buckets")
        assert hasattr(buckets, "buckets_cli"), "the `hf buckets` command group is gone"
        assert hasattr(buckets, "sync"), "the `hf sync` command is gone"
        entry = Path(huggingface_hub.__file__).parent / "cli" / "hf.py"
        registration = entry.read_text()
        assert 'name="buckets"' in registration, "`hf` no longer registers the buckets group"
        assert "(sync)" in registration, "`hf` no longer registers the sync command"


class TestTheGateStillFailsOpen:
    """Only a version it can read and compare is refused."""

    def test_an_unparseable_version_is_not_refused(self, monkeypatch):
        import huggingface_hub

        monkeypatch.setattr(huggingface_hub, "__version__", "not-a-version")
        assert dataset_recorder._huggingface_hub_version_error() is None

    def test_an_unimportable_hub_is_not_refused(self, monkeypatch):
        monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", None)
        assert dataset_recorder._huggingface_hub_version_error() is None
