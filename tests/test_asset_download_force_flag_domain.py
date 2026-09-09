"""A re-fetch must refuse a force posture it can only misread.

``force`` is the confirmation gate in front of the one write in
:mod:`strands_robots.assets.download` that removes something the caller already
has: a robot whose assets are present is re-fetched by deleting its cached
directory (``shutil.rmtree`` / ``Path.unlink``) and fetching it again. The flag
was read by truthiness - ``_needs_download`` returned it verbatim as its own
``bool`` verdict - so every non-empty string selected the re-fetch.

Measured on ``b882372`` with one present robot, a ``NOTES.md`` kept beside its
assets, and the ``robot_descriptions`` route stubbed to a local package:

=========================  ==================  ===========================
``force=``                 result              file kept beside the assets
=========================  ==================  ===========================
``True``                   ``downloaded: 1``   gone - as asked
``False``                  ``skipped: 1``      kept - as asked
``"false"`` / ``"no"``     ``downloaded: 1``   **gone**
``"0"`` / ``1`` / ``nan``  ``downloaded: 1``   **gone**
``None`` / ``0`` / ``[]``  ``skipped: 1``      kept - undeclared
=========================  ==================  ===========================

That directory is where a user's own files sit, and this is the only path in the
module that touches them: :func:`~strands_robots.assets.download._copy_external_tree`
filters on read rather than deleting afterwards precisely so a README or notes
kept beside the assets survive a download. A re-fetch also re-clones from the
network, so an opt-out spelling turned a no-op into 56 clones on the shipped
registry.

Both surfaces that read the flag now check it against the shared
:func:`~strands_robots.utils.boolean_flag_error` domain, which is the shape the
two other flags in front of an ``rmtree`` were given - ``start_recording``'s
``overwrite`` (``tests/test_dataset_recorder_posture_flag_domain.py``) and
``restore_calibrations``' (``tests/tools/test_calibration_restore_overwrite_flag_domain.py``).
The documented API refuses, because it is where the flag is read; the tool
refuses ahead of it, because the library's ``ValueError`` reaching the tool's
handler renders as ``Error: download_robots: ...`` - a refusal naming a function
the caller of a tool never addressed. The guard precedes the cache read, so no
refusal can arrive after the directory it was refusing to replace is gone.

Refs #3356.
"""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

import strands_robots.tools.download_assets as tool_mod
from strands_robots.assets import download as dl

download_assets = tool_mod.download_assets

_TOOL_MOD = "strands_robots.tools.download_assets"

# The spellings a caller reaches for when opting out (every one truthy), a truthy
# number and a truthy float, and the falsy values that are not a declared
# spelling of the negative posture either.
NOT_A_BOOLEAN = [
    pytest.param("false", id="str-false"),
    pytest.param("no", id="str-no"),
    pytest.param("off", id="str-off"),
    pytest.param("0", id="str-zero"),
    pytest.param(1, id="int-one"),
    pytest.param(float("nan"), id="nan"),
    pytest.param(None, id="none"),
    pytest.param(0, id="int-zero"),
    pytest.param([], id="empty-list"),
]

# Both python spellings plus the numpy booleans the shared domain also accepts.
A_BOOLEAN = [
    pytest.param(True, id="true"),
    pytest.param(False, id="false"),
    pytest.param(np.True_, id="np-true"),
    pytest.param(np.False_, id="np-false"),
]

_MODEL = '<mujoco model="probe"><worldbody><body name="link"/></worldbody></mujoco>\n'
_ENTRY: dict[str, Any] = {
    "name": "probe_arm",
    "category": "arm",
    "description": "a one-robot sim registry",
    "asset": {
        "dir": "probe_arm",
        "model_xml": "probe.xml",
        "robot_descriptions_module": "probe_arm_mj_description",
    },
}


class AssetCache:
    """A cache already holding one robot's assets, plus a file the user keeps there.

    The ``robot_descriptions`` route is pointed at a local package directory, so
    the re-fetch this flag decides runs for real - including the removal of the
    cached directory - without a clone or a network call.
    """

    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.cache = root / "cache"
        package = root / "site-packages" / "probe_arm_mj_description"
        package.mkdir(parents=True)
        (package / "probe.xml").write_text(_MODEL, encoding="utf-8")

        self.present = self.cache / "probe_arm"
        self.present.mkdir(parents=True)
        (self.present / "probe.xml").write_text(_MODEL, encoding="utf-8")
        self.notes = self.present / "NOTES.md"
        self.notes.write_text("hand-measured reach limits for this arm\n", encoding="utf-8")

        monkeypatch.setenv("STRANDS_ASSETS_DIR", str(self.cache))
        monkeypatch.setattr(dl, "registry_list_robots", lambda mode: [{"name": "probe_arm"}])
        monkeypatch.setattr(dl, "get_robot", lambda name: _ENTRY if name == "probe_arm" else None)
        monkeypatch.setattr(dl, "resolve_robot_name", lambda name: name)

        root_module = types.ModuleType("robot_descriptions")
        root_module.__path__ = []  # type: ignore[attr-defined]
        described = types.ModuleType("robot_descriptions.probe_arm_mj_description")
        described.PACKAGE_PATH = str(package)  # type: ignore[attr-defined]
        # setitem, not a bare assignment: a removal orphans every reference already
        # bound to the real package for the rest of the session.
        monkeypatch.setitem(sys.modules, "robot_descriptions", root_module)
        monkeypatch.setitem(sys.modules, "robot_descriptions.probe_arm_mj_description", described)

    def download(self, force: Any) -> dict[str, Any]:
        """Call the documented API through one funnel.

        The values under test are deliberately outside the declared ``bool``, which
        is the point; routing the call through here states that once rather than
        suppressing it at every call site.
        """
        return dl.download_robots(names=["probe_arm"], force=force)

    @property
    def user_file_kept(self) -> bool:
        """Whether the file the user keeps beside the assets survived the call."""
        return self.notes.exists()


@pytest.fixture
def assets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AssetCache:
    return AssetCache(tmp_path, monkeypatch)


def _run_tool(**kwargs: Any) -> dict[str, Any]:
    """Call the agent tool through one funnel, for the same reason as ``download``."""
    return dict(download_assets(**kwargs))


def _text(envelope: dict[str, Any]) -> str:
    return " ".join(item.get("text", "") for item in envelope.get("content", []))


class TestARefetchRefusesAPostureItCanOnlyMisread:
    """The documented API is held to the shared domain, ahead of any removal."""

    @pytest.mark.parametrize("value", NOT_A_BOOLEAN)
    def test_a_non_boolean_force_is_refused_by_name(self, assets: AssetCache, value: Any) -> None:
        with pytest.raises(ValueError, match=r"\bforce must be a boolean"):
            assets.download(value)

    @pytest.mark.parametrize("value", NOT_A_BOOLEAN)
    def test_the_refused_call_leaves_the_cached_assets_alone(self, assets: AssetCache, value: Any) -> None:
        with pytest.raises(ValueError):
            assets.download(value)
        assert assets.user_file_kept
        assert not assets.present.is_symlink(), "the cached directory was replaced by a refused call"

    @pytest.mark.parametrize("value", A_BOOLEAN)
    def test_a_usable_boolean_is_not_refused(self, assets: AssetCache, value: Any) -> None:
        assert assets.download(value)["failed"] == 0

    def test_the_refusal_precedes_the_partition_that_reads_the_flag(self, assets: AssetCache) -> None:
        """Nothing is asked to download before the flag deciding it is checked.

        ``_needs_download`` returned this flag verbatim as its own ``bool`` verdict,
        so an unchecked value became the partition's answer rather than an argument
        error.
        """
        with patch.object(dl, "_needs_download") as partition:
            with pytest.raises(ValueError, match=r"\bforce must be a boolean"):
                assets.download("false")
        partition.assert_not_called()


class TestTheTwoPosturesStillSelectWhatTheySpell:
    """The behaviour the flag exists for is unchanged."""

    def test_forcing_replaces_the_cached_directory(self, assets: AssetCache) -> None:
        result = assets.download(True)
        assert (result["downloaded"], result["failed"]) == (1, 0)
        assert not assets.user_file_kept

    def test_not_forcing_keeps_a_present_robot_untouched(self, assets: AssetCache) -> None:
        result = assets.download(False)
        assert (result["downloaded"], result["skipped"]) == (0, 1)
        assert assets.user_file_kept


class TestTheToolRefusesTheFlagItForwards:
    """The facade checks the flag rather than forwarding a posture it cannot read."""

    @pytest.mark.parametrize("value", NOT_A_BOOLEAN)
    def test_a_non_boolean_force_is_refused_before_the_download_layer(self, value: Any) -> None:
        with patch(f"{_TOOL_MOD}.download_robots") as download:
            envelope = _run_tool(action="download", robots="probe_arm", force=value)
        assert envelope["status"] == "error"
        assert "force must be a boolean" in _text(envelope)
        download.assert_not_called()

    def test_the_refusal_reads_as_a_bad_argument_rather_than_a_tool_crash(self) -> None:
        """The facade checks the flag instead of letting the library's raise reach its handler.

        With the facade guard removed the same call answers ``Error: download_robots:
        force must be a boolean ...`` - a refusal that reads as a crash and names a
        function the caller of a tool never addressed.
        """
        envelope = _run_tool(action="download", robots="probe_arm", force="false")
        assert envelope["status"] == "error"
        assert "download_robots" not in _text(envelope)
        assert not _text(envelope).startswith("Error:")

    @pytest.mark.parametrize("action", ["list", "status"])
    def test_an_action_that_reads_no_flag_is_not_refused_for_one(self, action: str) -> None:
        envelope = _run_tool(action=action, force="false")
        assert "must be a boolean" not in _text(envelope)

    @pytest.mark.parametrize("value", [True, False])
    def test_a_usable_boolean_is_forwarded_unchanged(self, value: bool) -> None:
        fake = {"downloaded": 0, "skipped": 0, "failed": 0, "method": "git clone"}
        with patch(f"{_TOOL_MOD}.download_robots", return_value=fake) as download:
            envelope = _run_tool(action="download", robots="probe_arm", force=value)
        assert envelope["status"] == "success"
        assert download.call_args.kwargs["force"] is value


class TestEveryBooleanParameterHasADomain:
    """A posture flag added to either surface later cannot skip the domain."""

    @staticmethod
    def _definition(module: Any, name: str) -> ast.FunctionDef:
        source = Path(module.__file__).read_text(encoding="utf-8")
        return next(
            node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.FunctionDef) and node.name == name
        )

    @pytest.mark.parametrize(
        ("module", "surface"),
        [(dl, "download_robots"), (tool_mod, "download_assets")],
        ids=["download_robots", "download_assets"],
    )
    def test_every_declared_bool_parameter_routes_through_the_shared_domain(self, module: Any, surface: str) -> None:
        definition = self._definition(module, surface)
        arguments = definition.args
        declared = {
            argument.arg
            for argument in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs)
            if isinstance(argument.annotation, ast.Name) and argument.annotation.id == "bool"
        }
        assert declared, f"{surface} declares no bool parameter; this rule has stopped reading it"
        checked = {
            node.args[0].id
            for node in ast.walk(definition)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "boolean_flag_error"
            and node.args
            and isinstance(node.args[0], ast.Name)
        }
        assert declared == checked
