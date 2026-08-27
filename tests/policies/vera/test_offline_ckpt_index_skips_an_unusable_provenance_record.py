"""One unusable ``provenance.json`` never costs the container its healthy checkpoints.

:mod:`strands_robots.policies.vera.docker.wandb_offline_resolve` exists for one
reason: VERA loads its Jacobian IDM by wandb run id, and the real
``download_checkpoint`` calls the wandb API before it looks at disk, so an
offline container never starts. The resolver indexes the mounted checkpoints by
the ``wandb_run`` recorded in each ``provenance.json`` and answers from disk
instead.

That index is built while the container is still starting - the launcher imports
the module (which self-installs) and then calls ``install()`` - and it is built
by scanning *every* ``provenance.json`` under the mounted root. So the scan is
handed whatever the operator mounted, and its disposition towards a file it
cannot use decides whether the server comes up at all. The module already skips
a file that does not parse and one that omits ``wandb_run``; the cases below are
the remaining ones, where the file parses into something that is simply not a
run record. Those used to be handed straight to the operations that assume a
string - hashing the value as a dict key and splitting it on ``/`` - so the scan
raised, the import raised, and the checkpoints that *were* healthy went with it.

The rule is the one :func:`strands_robots.transforms.provenance.load_provenance`
applies to a provenance payload: check the type before using the value. The two
differ only in disposition, and for the documented reason - a caller asking
which episodes are synthetic has no fallback and is refused, while this resolver
has one (the network) and the checkpoint it could not read is simply not in the
index.

The case set is not a hand-picked list of things that once crashed: each entry
is graded against the property that makes it interesting - it parses as JSON but
is not a usable record - so a case that stops being either is caught here rather
than quietly becoming a duplicate of the parse-failure path.
"""

from __future__ import annotations

import importlib
import json
import logging
import sys
import types
from pathlib import Path
from typing import Any

import pytest

# The container helpers are excluded from type checking (``tool.mypy.exclude``):
# they are written against the dependency set of the image, where ``vera`` and
# ``omegaconf`` are installed and unstubbed here. mypy's ``exclude`` does not
# apply to a module reached through a followed import, so a static import here
# would quietly pull the module back into the type-checked graph. Importing it by
# name keeps the declared exclusion effective.
resolver: Any = importlib.import_module("strands_robots.policies.vera.docker.wandb_offline_resolve")

# Two released-style records, laid out the way the hosted artifacts are: the run
# the entrypoint defaults to for mimicgen, and the pusht IDM beside it.
MIMICGEN_RUN = "your-wandb-entity/jacobian-learning/37oa162u"
PUSHT_RUN = "your-wandb-entity/jacobian-learning/pusht01"

# Payloads that parse as JSON but cannot be read as a run record. Either the
# payload is not an object, or its ``wandb_run`` is not a string.
UNUSABLE_PAYLOADS: dict[str, str] = {
    "run-id-is-a-number": json.dumps({"wandb_run": 12345}),
    "run-id-is-a-list": json.dumps({"wandb_run": ["your-wandb-entity", "jacobian-learning", "37oa162u"]}),
    "run-id-is-an-object": json.dumps({"wandb_run": {"entity": "your-wandb-entity", "id": "37oa162u"}}),
    "payload-is-an-array": json.dumps([{"wandb_run": MIMICGEN_RUN}]),
    "payload-is-a-string": json.dumps(MIMICGEN_RUN),
    "payload-is-a-number": json.dumps(7),
    "payload-is-null": json.dumps(None),
}

# Payloads the scan already skipped before, kept here so widening the skip does
# not widen the logging with it: none of these is a mistake worth a warning. A
# checkpoint directory that carries no wandb run is the ordinary case for every
# artifact that is not loaded by run id.
QUIETLY_SKIPPED_PAYLOADS: dict[str, str] = {
    "not-json-at-all": "{not json",
    "an-empty-file": "",
    "no-wandb_run-key": json.dumps({"artifact": "pusht-dfot"}),
    "wandb_run-is-null": json.dumps({"wandb_run": None}),
    "wandb_run-is-empty": json.dumps({"wandb_run": ""}),
}


def _module(name: str, **attrs: Any) -> Any:
    """A stand-in module carrying the attributes an importer will read off it."""
    module: Any = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _write_ckpt(root: Path, name: str, payload: str | None, *, model_ckpt: bool = True) -> Path:
    """Create one checkpoint directory the way the released artifacts lay them out."""
    ckpt_dir = root / name
    ckpt_dir.mkdir(parents=True)
    if payload is not None:
        (ckpt_dir / "provenance.json").write_text(payload)
    if model_ckpt:
        (ckpt_dir / "model.ckpt").write_text("weights")
    return ckpt_dir


class _RecordingWandbDownload:
    """The wandb-backed ``download_checkpoint`` the resolver falls back to.

    The real one reaches the network. This one records the arguments it was
    handed, so a test can say whether the fallback was taken and with what.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        run_path: str,
        download_dir: str,
        option: str = "latest",
        return_config: bool = False,
        force_redownload: bool = False,
    ) -> Path:
        self.calls.append(
            {
                "run_path": run_path,
                "download_dir": download_dir,
                "option": option,
                "return_config": return_config,
                "force_redownload": force_redownload,
            }
        )
        return Path("/downloaded/from/wandb/model.ckpt")


def _install_fake_vera(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any, _RecordingWandbDownload]:
    """Make ``vera`` importable, as it is inside the container.

    Returns the two modules the resolver patches and the recording fallback both
    of them start out holding.
    """
    wandb_download = _RecordingWandbDownload()
    ckpt_utils = _module("vera.utils.ckpt_utils", download_checkpoint=wandb_download)
    loading = _module("vera.policy.motion_policy_loading", download_checkpoint=wandb_download)
    utils = _module("vera.utils", __path__=[], ckpt_utils=ckpt_utils)
    policy = _module("vera.policy", __path__=[], motion_policy_loading=loading)
    vera = _module("vera", __path__=[], utils=utils, policy=policy)
    for name, module in (
        ("vera", vera),
        ("vera.utils", utils),
        ("vera.utils.ckpt_utils", ckpt_utils),
        ("vera.policy", policy),
        ("vera.policy.motion_policy_loading", loading),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return ckpt_utils, loading, wandb_download


class TestTheCaseSetIsWhatItClaims:
    """The unusable payloads are exactly the ones the parse check cannot catch."""

    @pytest.mark.parametrize("label", sorted(UNUSABLE_PAYLOADS))
    def test_each_case_parses_but_is_not_a_usable_record(self, label: str) -> None:
        value = json.loads(UNUSABLE_PAYLOADS[label])
        usable = isinstance(value, dict) and isinstance(value.get("wandb_run"), str)
        assert not usable, f"{label!r} is a usable record and grades nothing"

    def test_both_ways_of_being_unusable_are_represented(self) -> None:
        payloads = [json.loads(p) for p in UNUSABLE_PAYLOADS.values()]
        assert any(not isinstance(p, dict) for p in payloads), "no case exercises a non-object payload"
        assert any(isinstance(p, dict) for p in payloads), "no case exercises a non-string wandb_run"

    @pytest.mark.parametrize("label", sorted(QUIETLY_SKIPPED_PAYLOADS))
    def test_the_quiet_cases_carry_no_run_to_index(self, label: str) -> None:
        try:
            value = json.loads(QUIETLY_SKIPPED_PAYLOADS[label])
        except ValueError:
            return  # does not parse: skipped by the parse check, which is the point
        assert not (isinstance(value, dict) and value.get("wandb_run"))


class TestOneUnusableRecordDoesNotCostTheHealthyOnes:
    """The scan skips what it cannot read and keeps indexing the rest."""

    @pytest.mark.parametrize("label", sorted(UNUSABLE_PAYLOADS))
    def test_the_checkpoints_on_either_side_of_it_still_index(self, tmp_path: Path, label: str) -> None:
        _write_ckpt(tmp_path, "pusht-idm", json.dumps({"wandb_run": PUSHT_RUN}))
        _write_ckpt(tmp_path, "idm-omni-x21o0cwe", UNUSABLE_PAYLOADS[label])
        _write_ckpt(tmp_path, "idm-mimicgen-37oa162u", json.dumps({"wandb_run": MIMICGEN_RUN}))

        index = resolver._index_local_ckpts(str(tmp_path))

        assert set(index) == {MIMICGEN_RUN, "37oa162u", PUSHT_RUN, "pusht01"}
        assert index["37oa162u"] == tmp_path / "idm-mimicgen-37oa162u"
        assert index["pusht01"] == tmp_path / "pusht-idm"

    def test_the_run_the_entrypoint_defaults_to_resolves_with_no_wandb(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The launcher's ``install()`` is the caller that used to die here."""
        ckpt_utils, _, wandb_download = _install_fake_vera(monkeypatch)
        _write_ckpt(tmp_path, "idm-omni-x21o0cwe", UNUSABLE_PAYLOADS["run-id-is-an-object"])
        mimicgen = _write_ckpt(tmp_path, "idm-mimicgen-37oa162u", json.dumps({"wandb_run": MIMICGEN_RUN}))

        resolver.install(str(tmp_path))

        assert ckpt_utils.download_checkpoint is not wandb_download
        assert ckpt_utils.download_checkpoint("37oa162u", "/downloads") == mimicgen / "model.ckpt"
        assert wandb_download.calls == []

    @pytest.mark.parametrize("label", sorted(UNUSABLE_PAYLOADS))
    def test_the_ignored_file_is_named_with_the_type_that_made_it_unusable(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture, label: str
    ) -> None:
        payload = UNUSABLE_PAYLOADS[label]
        _write_ckpt(tmp_path, "idm-omni-x21o0cwe", payload)

        with caplog.at_level(logging.WARNING, logger="vera.offline_resolve"):
            assert resolver._index_local_ckpts(str(tmp_path)) == {}

        assert len(caplog.records) == 1
        message = caplog.records[0].getMessage()
        assert str(tmp_path / "idm-omni-x21o0cwe" / "provenance.json") in message
        value = json.loads(payload)
        offender = value.get("wandb_run") if isinstance(value, dict) else value
        assert type(offender).__name__ in message


class TestARecordTheScanCanUse:
    """What the scan indexed before is indexed on exactly the same terms."""

    def test_a_healthy_record_is_keyed_by_both_the_full_path_and_the_run_id(self, tmp_path: Path) -> None:
        ckpt_dir = _write_ckpt(tmp_path, "idm-mimicgen-37oa162u", json.dumps({"wandb_run": MIMICGEN_RUN}))

        index = resolver._index_local_ckpts(str(tmp_path))

        assert index == {MIMICGEN_RUN: ckpt_dir, "37oa162u": ckpt_dir}

    @pytest.mark.parametrize("label", sorted(QUIETLY_SKIPPED_PAYLOADS))
    def test_a_directory_with_no_run_to_index_is_skipped_without_a_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture, label: str
    ) -> None:
        _write_ckpt(tmp_path, "pusht-dfot", QUIETLY_SKIPPED_PAYLOADS[label])

        with caplog.at_level(logging.WARNING, logger="vera.offline_resolve"):
            assert resolver._index_local_ckpts(str(tmp_path)) == {}

        assert caplog.records == []

    def test_a_record_whose_checkpoint_is_missing_is_skipped(self, tmp_path: Path) -> None:
        _write_ckpt(tmp_path, "idm-mimicgen-37oa162u", json.dumps({"wandb_run": MIMICGEN_RUN}), model_ckpt=False)

        assert resolver._index_local_ckpts(str(tmp_path)) == {}


class TestThePatchedResolverPrefersTheLocalCheckpoint:
    """What ``install()`` puts in front of the server, once the index is built."""

    @pytest.fixture
    def container(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        ckpt_utils, loading, wandb_download = _install_fake_vera(monkeypatch)
        ckpt_dir = _write_ckpt(tmp_path, "idm-mimicgen-37oa162u", json.dumps({"wandb_run": MIMICGEN_RUN}))
        resolver.install(str(tmp_path))
        return {
            "ckpt_utils": ckpt_utils,
            "loading": loading,
            "wandb": wandb_download,
            "ckpt_dir": ckpt_dir,
            "root": tmp_path,
        }

    def test_the_full_run_path_resolves_to_the_mounted_checkpoint(self, container: dict[str, Any]) -> None:
        resolved = container["ckpt_utils"].download_checkpoint(MIMICGEN_RUN, "/downloads")

        assert resolved == container["ckpt_dir"] / "model.ckpt"
        assert container["wandb"].calls == []

    def test_a_different_entity_and_project_resolve_on_the_trailing_run_id(self, container: dict[str, Any]) -> None:
        """The code defaults and the released artifacts disagree about the prefix."""
        resolved = container["ckpt_utils"].download_checkpoint("someone-else/other-project/37oa162u", "/downloads")

        assert resolved == container["ckpt_dir"] / "model.ckpt"
        assert container["wandb"].calls == []

    def test_an_unknown_run_id_falls_back_with_every_argument_intact(self, container: dict[str, Any]) -> None:
        resolved = container["ckpt_utils"].download_checkpoint(
            "your-wandb-entity/jacobian-learning/unknown", "/downloads", option="best"
        )

        assert resolved == Path("/downloaded/from/wandb/model.ckpt")
        assert container["wandb"].calls == [
            {
                "run_path": "your-wandb-entity/jacobian-learning/unknown",
                "download_dir": "/downloads",
                "option": "best",
                "return_config": False,
                "force_redownload": False,
            }
        ]

    def test_a_forced_redownload_goes_to_wandb_even_though_the_run_is_local(self, container: dict[str, Any]) -> None:
        resolved = container["ckpt_utils"].download_checkpoint(MIMICGEN_RUN, "/downloads", force_redownload=True)

        assert resolved == Path("/downloaded/from/wandb/model.ckpt")
        assert [call["force_redownload"] for call in container["wandb"].calls] == [True]

    def test_a_run_config_is_read_from_the_sidecar_beside_the_checkpoint(
        self, container: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (container["ckpt_dir"] / "config.yaml").write_text("algo: jacobian\n")
        sidecar_config = {"algo": "jacobian"}
        omega = _module(
            "omegaconf",
            OmegaConf=_module(
                "OmegaConf",
                load=lambda path: {"path": str(path)},
                to_container=lambda cfg, resolve=True: sidecar_config,
            ),
        )
        monkeypatch.setitem(sys.modules, "omegaconf", omega)

        resolved, config = container["ckpt_utils"].download_checkpoint(MIMICGEN_RUN, "/downloads", return_config=True)

        assert resolved == container["ckpt_dir"] / "model.ckpt"
        assert config == sidecar_config

    def test_a_run_config_is_empty_when_the_checkpoint_carries_no_sidecar(self, container: dict[str, Any]) -> None:
        resolved, config = container["ckpt_utils"].download_checkpoint(MIMICGEN_RUN, "/downloads", return_config=True)

        assert resolved == container["ckpt_dir"] / "model.ckpt"
        assert config == {}

    def test_an_unreadable_sidecar_costs_the_config_not_the_checkpoint(
        self, container: dict[str, Any], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        (container["ckpt_dir"] / "config.yaml").write_text("algo: [unterminated\n")

        def _explode(path: Any) -> Any:
            raise ValueError("could not parse the sidecar")

        omega = _module("omegaconf", OmegaConf=_module("OmegaConf", load=_explode))
        monkeypatch.setitem(sys.modules, "omegaconf", omega)

        with caplog.at_level(logging.WARNING, logger="vera.offline_resolve"):
            resolved, config = container["ckpt_utils"].download_checkpoint(
                MIMICGEN_RUN, "/downloads", return_config=True
            )

        assert resolved == container["ckpt_dir"] / "model.ckpt"
        assert config == {}
        assert any("config.yaml" in record.getMessage() for record in caplog.records)

    def test_the_symbol_already_imported_by_name_is_patched_too(self, container: dict[str, Any]) -> None:
        """``motion_policy_loading`` bound the function at import, not by lookup."""
        assert container["loading"].download_checkpoint is container["ckpt_utils"].download_checkpoint
        assert container["loading"].download_checkpoint(MIMICGEN_RUN, "/downloads") == (
            container["ckpt_dir"] / "model.ckpt"
        )

    def test_installing_twice_still_resolves_locally_and_falls_back_once(self, container: dict[str, Any]) -> None:
        """The launcher imports the module and then calls ``install()`` again."""
        resolver.install(str(container["root"]))

        assert container["ckpt_utils"].download_checkpoint(MIMICGEN_RUN, "/downloads") == (
            container["ckpt_dir"] / "model.ckpt"
        )
        container["ckpt_utils"].download_checkpoint("your-wandb-entity/jacobian-learning/unknown", "/downloads")
        assert len(container["wandb"].calls) == 1


class TestWhatInstallDoesWhenItCannotHelp:
    """The resolver never becomes the reason the server fails to start."""

    def test_an_empty_checkpoint_root_leaves_wandb_in_place_and_says_so(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        ckpt_utils, _, wandb_download = _install_fake_vera(monkeypatch)

        with caplog.at_level(logging.WARNING, logger="vera.offline_resolve"):
            resolver.install(str(tmp_path))

        assert any("no local provenance.json" in record.getMessage() for record in caplog.records)
        assert ckpt_utils.download_checkpoint("any/run/id", "/downloads") == Path("/downloaded/from/wandb/model.ckpt")
        assert len(wandb_download.calls) == 1

    def test_nothing_is_patched_when_vera_is_not_importable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        _write_ckpt(tmp_path, "idm-mimicgen-37oa162u", json.dumps({"wandb_run": MIMICGEN_RUN}))
        monkeypatch.setitem(sys.modules, "vera", None)

        with caplog.at_level(logging.WARNING, logger="vera.offline_resolve"):
            resolver.install(str(tmp_path))

        assert any("could not import" in record.getMessage() for record in caplog.records)

    def test_the_checkpoint_utils_patch_survives_an_unimportable_loading_module(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The by-name re-import is a bonus; the primary patch must still land."""
        ckpt_utils, _, wandb_download = _install_fake_vera(monkeypatch)
        ckpt_dir = _write_ckpt(tmp_path, "idm-mimicgen-37oa162u", json.dumps({"wandb_run": MIMICGEN_RUN}))
        monkeypatch.setitem(sys.modules, "vera.policy.motion_policy_loading", None)

        resolver.install(str(tmp_path))

        assert ckpt_utils.download_checkpoint(MIMICGEN_RUN, "/downloads") == ckpt_dir / "model.ckpt"
        assert wandb_download.calls == []
