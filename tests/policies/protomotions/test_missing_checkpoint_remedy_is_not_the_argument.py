"""The missing-checkpoint refusal must name a step, not the caller's argument.

``ProtoMotionsPolicy`` takes ``onnx_path`` as a *local* file and resolves no
HuggingFace model id - unlike ``WBCPolicy``, whose ``checkpoint`` is downloaded
through ``huggingface_hub``. The ``[protomotions]`` extra declares that hub
client all the same, and the install docs used to attribute the fetch to it, so
the documented next step for a reader was to pass the model id.

Doing that produced::

    ONNX artifact not found: cagataydev/protomotions-gtp-unitree-g1. Download
    from cagataydev/protomotions-gtp-unitree-g1 on HuggingFace.

- a remedy that names the argument back. The caller is told to download the
string they just supplied, with no step that turns it into the local path the
parameter wants. These cells pin the refusal on the step instead: the canonical
repo *and* file name, as a command whose output is the path to pass.

The value-domain cells for this constructor live in
``test_history_length_sizes_the_window_it_names.py``; this file grades only the
not-found remedy and the install sentence that sends a reader into it.
"""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path
from typing import Any

import pytest

from strands_robots import utils
from strands_robots.policies.protomotions import policy as policy_mod
from strands_robots.policies.protomotions.policy import ProtoMotionsPolicy

_ROOT = Path(__file__).resolve().parents[3]
_PACKAGE = _ROOT / "strands_robots" / "policies" / "protomotions"
_DOC = _ROOT / "docs" / "policies" / "protomotions.md"

#: Stated locally so these cells are an independent oracle rather than a
#: restatement of the module constants they grade.
REPO_ID = "cagataydev/protomotions-gtp-unitree-g1"
ONNX_FILENAME = "unified_pipeline.onnx"

_HUB = "huggingface_hub"


class _Sentinel(Exception):
    """Raised by the stub session to prove the not-found guard was passed."""


class _StubSession:
    def __init__(self, *_a: Any, **_k: Any) -> None:
        raise _Sentinel("session build reached")


@pytest.fixture
def _ort(monkeypatch: pytest.MonkeyPatch) -> None:
    """Seed ``require_optional``'s cache so the guard under test is reachable.

    ``_build_onnx_session`` resolves ``onnxruntime`` *before* it checks the
    path, so the not-found branch is unreachable on a host without it. Seeding
    the cache (rather than ``sys.modules``) is what ``require_optional`` reads
    first, so this is deterministic whether or not the real package is present.
    """
    stub: Any = type("onnxruntime", (), {"InferenceSession": _StubSession})
    monkeypatch.setitem(utils._lazy_modules, "onnxruntime", stub)


def _refusal(value: str, _ort: None) -> str:
    with pytest.raises(FileNotFoundError) as excinfo:
        ProtoMotionsPolicy._build_onnx_session(Path(value), ["CPUExecutionProvider"])
    return str(excinfo.value)


def _module_scope_imports(path: Path) -> set[str]:
    """Every module-scope import name in ``path`` (typing-only blocks skipped)."""
    names: set[str] = set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


class TestWhyTheWordingMatters:
    """Premises: the extra advertises a fetch this family never performs."""

    def test_the_extra_declares_a_hub_client(self) -> None:
        """``[protomotions]`` pins ``huggingface_hub`` - the docs' premise."""
        declared = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        extra = declared["project"]["optional-dependencies"]["protomotions"]
        assert any(_HUB in requirement for requirement in extra), extra

    def test_the_family_resolves_no_model_id(self) -> None:
        """No module in the family imports the hub client, unlike ``wbc``."""
        importers = sorted(p.name for p in _PACKAGE.glob("*.py") if _HUB in _module_scope_imports(p))
        assert importers == [], importers
        wbc = _ROOT / "strands_robots" / "policies" / "wbc" / "policy.py"
        assert _HUB in wbc.read_text(encoding="utf-8"), "sibling contrast is the point"


class TestTheRemedyNamesAStep:
    """Regression: the refusal must not hand the argument back."""

    def test_a_model_id_is_not_told_to_download_itself(self, _ort: None) -> None:
        """The remedy for a model id may not be that same model id."""
        message = _refusal(REPO_ID, _ort)
        assert f"Download from {REPO_ID}" not in message, message
        remedy = message.split("Fetch the checkpoint first:", 1)[-1]
        assert "hf_hub_download" in remedy, message

    def test_the_remedy_names_the_repo_and_the_file(self, _ort: None) -> None:
        """A reader needs both halves; the repo alone is not a fetchable step."""
        message = _refusal(REPO_ID, _ort)
        assert REPO_ID in message, message
        assert ONNX_FILENAME in message, message

    def test_the_remedy_is_a_runnable_one_liner(self, _ort: None) -> None:
        """The quoted command must be valid Python, not prose about one."""
        message = _refusal(REPO_ID, _ort)
        match = re.search(r'python -c "(.+?)"\n', message, re.S)
        assert match is not None, message
        compile(match.group(1), "<remedy>", "exec")

    def test_the_remedy_says_the_parameter_wants_a_local_file(self, _ort: None) -> None:
        """Naming the contract is what makes the extra step make sense."""
        assert "local file" in _refusal(REPO_ID, _ort)

    def test_the_remedy_is_single_sourced_from_the_module_constants(self) -> None:
        """The repo and file names are interpolated, not spelled twice.

        A second literal would let the message drift from the checkpoint the
        module docstring pins, which is how the circular wording survived.
        """
        assert policy_mod._GTP_G1_HF_REPO == REPO_ID
        assert policy_mod._GTP_G1_ONNX_FILENAME == ONNX_FILENAME
        source = ast.unparse(ast.parse(_PACKAGE.joinpath("policy.py").read_text(encoding="utf-8")))
        raised = [line for line in source.splitlines() if "ONNX artifact not found" in line]
        assert len(raised) == 1, raised
        assert "_GTP_G1_HF_REPO" in raised[0], raised[0]
        assert "_GTP_G1_ONNX_FILENAME" in raised[0], raised[0]


class TestTheInstallSectionDoesNotPromiseAFetch:
    """Regression: the docs sentence that sent readers down the dead end."""

    def test_the_hub_client_is_not_described_as_fetching_for_you(self) -> None:
        """The extra pulls the client; the caller calls it."""
        text = " ".join(_DOC.read_text(encoding="utf-8").split())
        assert "(fetches a checkpoint from a model id)" not in text
        assert _HUB in text, "the dependency is still worth naming"

    def test_the_install_section_says_the_paths_are_local(self) -> None:
        """A reader must not be left to infer that a model id would work."""
        text = " ".join(_DOC.read_text(encoding="utf-8").split())
        install = text.split("## Install", 1)[-1].split("##", 1)[0]
        assert "local file" in install, install


class TestWhatStillHolds:
    """Over-reach controls: nothing else about the guard may change."""

    def test_a_missing_local_path_still_reports_that_path(self, _ort: None) -> None:
        """The path a caller passed is still the first thing named."""
        message = _refusal("/no/such/dir/unified_pipeline.onnx", _ort)
        assert message.startswith("ONNX artifact not found: /no/such/dir/unified_pipeline.onnx")

    def test_an_existing_file_reaches_the_session_build(self, tmp_path: Path, _ort: None) -> None:
        """The guard refuses only a path that is absent."""
        artifact = tmp_path / ONNX_FILENAME
        artifact.write_bytes(b"not really onnx")
        with pytest.raises(_Sentinel):
            ProtoMotionsPolicy._build_onnx_session(artifact, ["CPUExecutionProvider"])
