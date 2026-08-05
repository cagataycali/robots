"""The advertised huggingface_hub floor for bucket sync must be the capability floor.

``sync_dataset_to_bucket`` shells out to ``hf buckets`` / ``hf sync``. Those
subcommands first ship in huggingface_hub 1.5.0: measured against the published
wheels, 1.4.1 carries no ``huggingface_hub/cli/buckets.py`` and answers
``hf sync`` with ``Error: No such command 'sync'``, while 1.5.0 registers both a
``buckets`` group and a top-level ``sync`` command.

Two things therefore have to hold, and neither did:

* the runtime gate must refuse every release below the capability floor. A gate
  floored lower admits releases whose CLI cannot route the subcommand, so the
  caller receives the unroutable-subcommand noise verbatim - precisely the
  outcome :func:`~strands_robots.dataset_recorder._huggingface_hub_version_error`
  exists to replace with an upgrade instruction.
* every floor the project *advertises* - the ``[wbc]`` pin a resolver reads, the
  README guidance, the docstrings, the remedy the gate prints - must be at least
  that capability floor. A remedy naming a release that still cannot sync sends
  the caller to install something that does not fix their problem.

The tests below pin both against one constant
(:data:`~strands_robots.dataset_recorder._BUCKET_CLI_MIN_HUB_VERSION`) so the
runtime gate, the packaging pin and the prose cannot drift apart from the real
capability. They are deliberately relation-based (``floor >= capability``) rather
than equality-based: raising a floor further is always safe and must not fail.
"""

from __future__ import annotations

import inspect
import pathlib
import re
from typing import Any

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

from strands_robots import dataset_recorder as dr

# Every literal `huggingface_hub>=X` floor the project declares or advertises.
# The gate's own messages are f-strings derived from the constant, so they carry
# no literal to drift - that is the point of the constant.
_HUB_FLOOR_RE = re.compile(r"huggingface_hub\s*>=\s*([0-9]+(?:\.[0-9]+)*)")

# Files that advertise a floor to a human or to a resolver. Rooted at the
# package the constant lives in so a move cannot silently empty the scan.
_REPO_ROOT = pathlib.Path(inspect.getfile(dr)).resolve().parents[1]
_ADVERTISING_FILES = (
    "pyproject.toml",
    "README.md",
    "strands_robots/dataset_recorder.py",
    "examples/06_agent_collect_and_stream.py",
)


def _capability() -> Version:
    """The version at which `hf buckets` / `hf sync` start existing."""
    return Version(dr._BUCKET_CLI_MIN_HUB_VERSION_STR)


def _advertised_floors(text: str) -> list[Version]:
    return [Version(m.group(1)) for m in _HUB_FLOOR_RE.finditer(text)]


def _wbc_hub_requirement() -> Requirement:
    """The huggingface_hub requirement the ``[wbc]`` extra declares."""
    import tomllib

    data = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())
    for spec in data["project"]["optional-dependencies"]["wbc"]:
        req = Requirement(spec)
        if req.name.replace("-", "_") == "huggingface_hub":
            return req
    raise AssertionError("no huggingface_hub requirement in the [wbc] extra")


class TestTheRuntimeGateRefusesEveryReleaseBelowTheCapabilityFloor:
    """A release whose `hf` cannot route `sync` must be refused, not admitted."""

    @pytest.mark.parametrize(
        "version",
        ["0.36.2", "1.0.0", "1.0.1", "1.1.7", "1.3.5", "1.4.0", "1.4.1"],
        ids=["0.36.2", "1.0.0", "1.0.1", "1.1.7", "1.3.5", "1.4.0", "1.4.1"],
    )
    def test_a_release_without_the_subcommands_is_refused(
        self, version: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import huggingface_hub

        monkeypatch.setattr(huggingface_hub, "__version__", version)
        message = dr._huggingface_hub_version_error()
        assert message is not None, (
            f"huggingface_hub {version} has no `hf buckets`/`hf sync`, so the gate "
            "must refuse it instead of letting the caller reach the unroutable CLI"
        )
        assert version in message
        assert dr._BUCKET_CLI_MIN_HUB_VERSION_STR in message

    @pytest.mark.parametrize(
        "version",
        ["1.5.0", "1.5.1", "1.6.0", "1.26.0", "2.0.0"],
        ids=["1.5.0", "1.5.1", "1.6.0", "1.26.0", "2.0.0"],
    )
    def test_a_release_with_the_subcommands_is_admitted(
        self, version: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import huggingface_hub

        monkeypatch.setattr(huggingface_hub, "__version__", version)
        assert dr._huggingface_hub_version_error() is None

    def test_the_boundary_is_the_capability_floor_itself(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The floor release is admitted and its immediate predecessor is not.

        A two-component version parse would read the floor's own ``1.5`` prefix
        as below ``(1, 5, 0)`` and refuse the very release that introduced the
        subcommands, so the boundary is pinned from both sides.
        """
        import huggingface_hub

        floor = _capability()
        monkeypatch.setattr(huggingface_hub, "__version__", str(floor))
        assert dr._huggingface_hub_version_error() is None
        monkeypatch.setattr(huggingface_hub, "__version__", f"{floor.major}.{floor.minor}")
        assert dr._huggingface_hub_version_error() is None, (
            "a two-component version string at the floor's own minor must not be refused"
        )
        monkeypatch.setattr(huggingface_hub, "__version__", f"{floor.major}.{floor.minor - 1}.99")
        assert dr._huggingface_hub_version_error() is not None

    def test_an_unparseable_version_still_fails_open(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import huggingface_hub

        monkeypatch.setattr(huggingface_hub, "__version__", "not-a-version")
        assert dr._huggingface_hub_version_error() is None


class TestTheRemedyTheGatePrintsCanActuallyBeFollowed:
    """Installing what the message advises must produce a CLI that can sync."""

    def test_the_advised_version_is_at_least_the_capability_floor(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import huggingface_hub

        monkeypatch.setattr(huggingface_hub, "__version__", "1.4.1")
        message = dr._huggingface_hub_version_error()
        assert message is not None
        advised = _advertised_floors(message)
        assert advised, f"the gate's remedy names no huggingface_hub version: {message!r}"
        for version in advised:
            assert version >= _capability(), (
                f"the gate advises huggingface_hub>={version}, which still has no "
                f"`hf buckets`/`hf sync` (they start at {_capability()})"
            )

    def test_following_the_advice_satisfies_the_gate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The advised version, once installed, passes the gate it was printed by."""
        import huggingface_hub

        monkeypatch.setattr(huggingface_hub, "__version__", "1.4.1")
        message = dr._huggingface_hub_version_error()
        assert message is not None
        advised = max(_advertised_floors(message))
        monkeypatch.setattr(huggingface_hub, "__version__", str(advised))
        assert dr._huggingface_hub_version_error() is None

    def test_the_missing_cli_remedy_also_names_a_usable_floor(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The `hf`-not-found branch advises an install too, and it must be usable."""
        (tmp_path / "meta").mkdir()
        (tmp_path / "meta" / "info.json").write_text("{}")
        monkeypatch.setattr(dr, "_hf_executable", lambda: None)

        result: dict[str, Any] = dr.sync_dataset_to_bucket(tmp_path, "my-org/robot-fave")

        assert result["status"] == "error"
        advised = _advertised_floors(str(result["message"]))
        assert advised, f"the missing-CLI remedy names no version: {result['message']!r}"
        for version in advised:
            assert version >= _capability()


class TestEveryAdvertisedFloorIsAtLeastTheCapabilityFloor:
    """A resolver pin or a doc line below the capability floor is unusable advice."""

    def test_the_packaging_pin_cannot_resolve_below_the_capability_floor(self) -> None:
        req = _wbc_hub_requirement()
        lowers = [Version(s.version) for s in req.specifier if s.operator == ">="]
        assert lowers, f"[wbc] huggingface_hub pin declares no lower bound: {req}"
        assert min(lowers) >= _capability(), (
            f"[wbc] pins huggingface_hub{req.specifier}, which permits a release with no "
            f"`hf buckets`/`hf sync` (they start at {_capability()})"
        )
        # keep the MAJOR cap per repo convention (>=1.0 deps cap the major)
        assert any(s.operator == "<" and Version(s.version) >= Version("2") for s in req.specifier), (
            f"[wbc] huggingface_hub pin lost its major cap: {req.specifier}"
        )

    @pytest.mark.parametrize("relative", _ADVERTISING_FILES, ids=_ADVERTISING_FILES)
    def test_no_advertised_floor_is_below_the_capability_floor(self, relative: str) -> None:
        path = _REPO_ROOT / relative
        assert path.exists(), f"{relative} moved; this guard would silently scan nothing"
        floors = _advertised_floors(path.read_text())
        assert floors, (
            f"{relative} advertises no huggingface_hub floor; it used to, so either the "
            "guidance moved (point this guard at it) or the floor was dropped"
        )
        for version in floors:
            assert version >= _capability(), (
                f"{relative} advertises huggingface_hub>={version}, below the "
                f"`hf buckets`/`hf sync` floor {_capability()}"
            )

    def test_no_file_recommends_an_unversioned_upgrade(self) -> None:
        """`pip install -U huggingface_hub` resolves anywhere; it must never be advised."""
        for relative in _ADVERTISING_FILES:
            text = (_REPO_ROOT / relative).read_text()
            assert "pip install -U huggingface_hub" not in text, (
                f"{relative} recommends an unversioned huggingface_hub install"
            )

    def test_the_scanner_reports_a_floor_below_the_capability_floor(self) -> None:
        """Planted-defect meta-test: an empty result must mean clean files, not a dead regex."""
        planted = 'pip install -U "huggingface_hub>=1.0" and run `hf auth login`.'
        found = _advertised_floors(planted)
        assert found == [Version("1.0")], found
        assert not all(v >= _capability() for v in found), (
            "the scanner must recognise a below-floor claim as below the floor"
        )
