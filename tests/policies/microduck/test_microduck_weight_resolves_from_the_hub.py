"""A bare Microduck weight name resolves to Pollen's Hub repository.

``docs/policies/microduck.md`` said the shipped weights "ship in Pollen's
``microduck`` repository under ``policies/*.onnx``", and every example under
``examples/microduck/`` defaulted ``--onnx`` to ``../microduck/policies/...``.
Pollen removed that directory upstream ("policies/ leaves the repository"); the
ten weights now live on the Hub at ``pollen-robotics/microduck-policies``. A
reader following the page got ``no such ONNX`` from the very first command, on
the headline feature of the release.

The fix is one resolver, :func:`resolve_microduck_weight`, that the policy's
session builder and the examples share. A path that exists is used as is. A bare
name that is not in the working directory is fetched from the Hub repository the
module names, so the page's ``MicroduckPolicy(onnx_path="alpha_walking.onnx")``
works on a fresh install with no clone beside it. A path *with* directories that
does not exist is refused, not downloaded: the caller made a claim about their
filesystem, and answering a typo with a different file would hide it.

Every cell is hermetic. The Hub client is a stub recorded through the module
cache ``require_optional`` reads, ``onnxruntime`` is a stub in the same cache,
and no network is reached.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from strands_robots import utils
from strands_robots.policies import microduck as microduck_pkg
from strands_robots.policies.microduck import (
    MICRODUCK_POLICIES_HF_REPO,
    MicroduckPolicy,
    resolve_microduck_weight,
)
from strands_robots.policies.microduck import policy as policy_module

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PAGE = _REPO_ROOT / "docs" / "policies" / "microduck.md"
_EXAMPLES = sorted((_REPO_ROOT / "examples" / "microduck").glob("*.py"))
_PYPROJECT = _REPO_ROOT / "pyproject.toml"

# The layout the weights left, as a file (``microduck/policies/alpha_walking.onnx``)
# and as the bare directory an example passed as ``--policy-dir``. Nothing shipped
# may send a reader to either.
_GONE = re.compile(r"microduck/policies|policies/\*\.onnx")


class _Hub:
    """A ``huggingface_hub`` stand-in that records the one download it serves."""

    def __init__(self, target: Path, *, fail: Exception | None = None) -> None:
        self.calls: list[tuple[str, str, str | None]] = []
        self._target = target
        self._fail = fail

    def hf_hub_download(self, repo_id: str, filename: str, revision: str | None = None) -> str:
        self.calls.append((repo_id, filename, revision))
        if self._fail is not None:
            raise self._fail
        return str(self._target)


@pytest.fixture
def hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Hub:
    cached = tmp_path / "cache" / "alpha_walking.onnx"
    cached.parent.mkdir()
    cached.write_bytes(b"onnx")
    stub = _Hub(cached)
    monkeypatch.setitem(utils._lazy_modules, "huggingface_hub", stub)
    return stub


class TestTheResolver:
    def test_a_file_that_exists_is_returned_as_is_and_the_hub_is_not_consulted(self, tmp_path: Path, hub: _Hub) -> None:
        local = tmp_path / "mine.onnx"
        local.write_bytes(b"onnx")

        assert resolve_microduck_weight(local) == local
        assert resolve_microduck_weight(str(local)) == local
        assert hub.calls == []

    def test_a_bare_name_not_in_the_working_directory_is_fetched_from_the_repo(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hub: _Hub
    ) -> None:
        monkeypatch.chdir(tmp_path)

        resolved = resolve_microduck_weight("alpha_walking.onnx")

        assert resolved.is_file()
        assert hub.calls == [(MICRODUCK_POLICIES_HF_REPO, "alpha_walking.onnx", None)]

    def test_a_bare_name_present_in_the_working_directory_wins_over_the_hub(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hub: _Hub
    ) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "alpha_walking.onnx").write_bytes(b"local")

        assert resolve_microduck_weight("alpha_walking.onnx") == Path("alpha_walking.onnx")
        assert hub.calls == []

    def test_a_revision_is_passed_through(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hub: _Hub) -> None:
        monkeypatch.chdir(tmp_path)

        resolve_microduck_weight("roller.onnx", revision="v1")

        assert hub.calls == [(MICRODUCK_POLICIES_HF_REPO, "roller.onnx", "v1")]

    def test_a_missing_path_with_directories_is_refused_not_downloaded(self, tmp_path: Path, hub: _Hub) -> None:
        missing = tmp_path / "weights" / "alpha_walking.onnx"

        with pytest.raises(FileNotFoundError) as excinfo:
            resolve_microduck_weight(missing)

        message = str(excinfo.value)
        assert str(missing) in message
        assert MICRODUCK_POLICIES_HF_REPO in message
        assert "'alpha_walking.onnx'" in message, "the remedy names the bare name that would fetch"
        assert hub.calls == []

    def test_a_name_the_repo_does_not_carry_is_refused_naming_both(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        stub = _Hub(tmp_path, fail=RuntimeError("404 Client Error: Entry Not Found"))
        monkeypatch.setitem(utils._lazy_modules, "huggingface_hub", stub)

        with pytest.raises(FileNotFoundError) as excinfo:
            resolve_microduck_weight("no_such_skill.onnx")

        message = str(excinfo.value)
        assert "'no_such_skill.onnx'" in message
        assert MICRODUCK_POLICIES_HF_REPO in message
        assert "Entry Not Found" in message
        assert isinstance(excinfo.value.__cause__, RuntimeError)

    def test_a_missing_hub_client_names_the_extra_that_installs_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.delitem(utils._lazy_modules, "huggingface_hub", raising=False)
        real_import = utils.importlib.import_module

        def _no_hub(name: str, package: str | None = None) -> ModuleType:
            if name == "huggingface_hub":
                raise ImportError(name)
            return real_import(name, package)

        monkeypatch.setattr(utils.importlib, "import_module", _no_hub)

        with pytest.raises(ImportError) as excinfo:
            resolve_microduck_weight("alpha_walking.onnx")

        assert "strands-robots[microduck]" in str(excinfo.value)
        assert "alpha_walking.onnx" in str(excinfo.value)


class TestThePolicyTakesTheSameDoor:
    def test_the_session_builder_resolves_a_bare_name_through_the_hub(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hub: _Hub
    ) -> None:
        monkeypatch.chdir(tmp_path)
        opened: list[str] = []

        class _Session:
            def __init__(self, path: str, providers: list[str]) -> None:
                opened.append(path)
                self._providers = providers

            def get_providers(self) -> list[str]:
                return self._providers

        monkeypatch.setitem(utils._lazy_modules, "onnxruntime", SimpleNamespace(InferenceSession=_Session))

        MicroduckPolicy._build_onnx_session(Path("alpha_walking.onnx"), ["CPUExecutionProvider"])

        assert hub.calls == [(MICRODUCK_POLICIES_HF_REPO, "alpha_walking.onnx", None)]
        assert opened == [str(hub._target)]

    def test_the_session_builder_refuses_a_missing_path_before_touching_onnxruntime(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hub: _Hub
    ) -> None:
        def _never(*args: object, **kwargs: object) -> None:
            raise AssertionError("onnxruntime was asked to open a file that does not exist")

        monkeypatch.setitem(utils._lazy_modules, "onnxruntime", SimpleNamespace(InferenceSession=_never))

        with pytest.raises(FileNotFoundError, match=re.escape(MICRODUCK_POLICIES_HF_REPO)):
            MicroduckPolicy._build_onnx_session(tmp_path / "gone" / "x.onnx", ["CPUExecutionProvider"])

    def test_the_resolver_and_the_repo_name_are_exported_from_the_package(self) -> None:
        assert microduck_pkg.resolve_microduck_weight is policy_module.resolve_microduck_weight
        assert "MICRODUCK_POLICIES_HF_REPO" in microduck_pkg.__all__
        assert "resolve_microduck_weight" in microduck_pkg.__all__


class TestNothingShippedSendsAReaderToTheGoneLayout:
    def test_the_docs_page_names_the_hub_repository_and_not_the_old_directory(self) -> None:
        text = _PAGE.read_text(encoding="utf-8")
        assert MICRODUCK_POLICIES_HF_REPO in text
        assert not _GONE.search(text), "docs/policies/microduck.md still points at Pollen's removed policies/ dir"

    @pytest.mark.parametrize("example", _EXAMPLES, ids=lambda p: p.name)
    def test_no_example_defaults_to_the_old_directory(self, example: Path) -> None:
        assert not _GONE.search(example.read_text(encoding="utf-8")), (
            f"{example.name} points at a directory Pollen removed"
        )

    def test_the_microduck_extra_installs_the_hub_client(self) -> None:
        with _PYPROJECT.open("rb") as fh:
            extras = tomllib.load(fh)["project"]["optional-dependencies"]
        names = [re.split(r"[<>=!~\[ ]", req, maxsplit=1)[0] for req in extras["microduck"]]
        assert "huggingface_hub" in names, "a bare weight name needs huggingface_hub to fetch it"
