"""A checkpoint's ``norm_stats_filename`` names a file inside the checkpoint.

A stats-only checkpoint (the MolmoAct2 SO-100/101 family) gets its
normalization from :mod:`strands_robots.policies.lerobot_local.norm_stats`, and
the checkpoint's own ``config.json`` may name the file those statistics are read
from. Those numbers are not advisory: the postprocessor unnormalizes every
predicted action through them, so they are the scale of every motor command.

That name used to be followed wherever it pointed. It was coerced with ``str()``
and joined onto the checkpoint directory, so a ``..`` segment stepped out of it
and an absolute path discarded the directory altogether (``Path("/ckpt") /
"/tmp/x.json"`` is ``/tmp/x.json``). A checkpoint could therefore have its
actions scaled by a JSON file it does not contain -- measured on this fixture,
the same predicted action left the bridge as 500.9 degrees instead of 125.2 --
while the checkpoint's own ``norm_stats.json`` sat unread beside it, under a
``load_norm_stats`` call that reported success.

The joined component is now held to the rule the rest of the tree applies to an
externally sourced path component (:func:`strands_robots.utils.safe_join`): it
must be a relative name that stays inside the checkpoint. The rule is checked on
the value rather than on the join, because the local branch joins it onto a
directory and the Hub branch hands it to ``hf_hub_download``, and the rule is the
same one either way.

A malformed declaration is not an absence, so it does not share the ``None``
verdict used for "this checkpoint ships no stats" -- it raises
``NormStatsFilenameError``, and the reason reaches the load report in place of
the generic missing-postprocessor message, exactly as an undeclared ``norm_tag``
already does (``test_unknown_norm_tag_is_reported``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from strands_robots.policies.lerobot_local import norm_stats as ns
from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy
from strands_robots.policies.lerobot_local.processor import ProcessorBridge

_FIXTURE = Path(__file__).parent / "fixtures" / "molmoact2_norm_stats.json"


def _lerobot_pipeline_importable() -> bool:
    try:
        import lerobot.processor.pipeline  # noqa: F401
    except (ImportError, AttributeError):
        return False
    return True


_requires_lerobot_pipeline = pytest.mark.skipif(
    not _lerobot_pipeline_importable(),
    reason="requires LeRobot's processor pipeline (real ProcessorStep framework)",
)

# Spellings that leave the checkpoint. A dict/list/int is not a filename at all;
# ``str()`` used to turn each of them into one. The pairs are a list rather than
# a dict because ``True == 1`` and ``0 == False`` collapse in a mapping, and the
# bool/int distinction is exactly what the coercion erased.
_ESCAPING: list[tuple[str, Any]] = [
    ("parent segment", "../foreign.json"),
    ("segment then parents", "sub/../../foreign.json"),
    ("bare parent dir", ".."),
    ("absolute path", "/tmp/foreign.json"),
    ("absolute system file", "/etc/hostname"),
    ("int", 42),
    ("float", 1.5),
    ("bool", True),
    ("list", ["norm_stats.json"]),
    ("dict", {"norm_stats_filename": "norm_stats.json"}),
    ("empty string", ""),
    ("null byte", "norm\x00_stats.json"),
]

# Spellings that name a file inside the checkpoint, and must keep working.
_CONTAINED: list[tuple[str, str]] = [
    ("plain name", "norm_stats.json"),
    ("other plain name", "custom_stats.json"),
    ("nested name", "stats/custom.json"),
    ("dot-prefixed name", "./custom_stats.json"),
    ("name with a dotted stem", "norm_stats.v1.json"),
]


def _payload() -> dict[str, Any]:
    return json.loads(_FIXTURE.read_text())


def _tag(payload: dict[str, Any]) -> str:
    return next(iter(payload["metadata_by_tag"]))


def _foreign(root: Path) -> Path:
    """Where the outsider lives: one directory up from the checkpoint.

    Deliberately reachable by a single ``..`` segment, so an escaping name is a
    read that succeeds rather than one that happens to miss.
    """
    return root / "checkpoints" / "foreign.json"


def _checkpoint(root: Path, *, declared: Any = ..., stats_at: str | None = "norm_stats.json") -> Path:
    """A checkpoint dir with its own stats, plus a foreign stats file outside it.

    Args:
        root: Temporary directory holding both the checkpoint and the outsider.
        declared: Value for ``norm_stats_filename`` in the checkpoint's
            ``config.json``; the sentinel omits the key, ``None`` writes a JSON
            null, and no ``config.json`` is written when the key is omitted.
        stats_at: Checkpoint-relative path to write the real stats to, or
            ``None`` to ship no stats file at all.

    Returns:
        The checkpoint directory.
    """
    ckpt = root / "checkpoints" / "molmoact2"
    ckpt.mkdir(parents=True, exist_ok=True)
    if stats_at is not None:
        target = ckpt / stats_at
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(_FIXTURE.read_text())
    # A recognized stats file OUTSIDE the checkpoint, on a wider action scale, so
    # a read that escapes is visible in the numbers and not just in the path.
    foreign = _payload()
    action = foreign["metadata_by_tag"][_tag(foreign)]["action_stats"]
    action["q01"] = [v * 4 for v in action["q01"]]
    action["q99"] = [v * 4 for v in action["q99"]]
    _foreign(root).write_text(json.dumps(foreign))
    if declared is not ...:
        (ckpt / "config.json").write_text(json.dumps({"norm_stats_filename": declared}))
    return ckpt


class TestADeclaredNameThatLeavesTheCheckpointIsRefused:
    """The escaping spellings are refused by name, not followed."""

    @pytest.mark.parametrize(("label", "declared"), _ESCAPING, ids=[c[0] for c in _ESCAPING])
    def test_it_raises_naming_the_option_and_the_file_that_declared_it(self, tmp_path, label, declared):
        ckpt = _checkpoint(tmp_path, declared=declared)
        with pytest.raises(ns.NormStatsFilenameError) as excinfo:
            ns.load_norm_stats(str(ckpt))
        message = str(excinfo.value)
        assert "norm_stats_filename" in message, message
        # The declaring file, so the remedy is one path away.
        assert str(ckpt / "config.json") in message, message

    def test_the_refusal_is_a_valueerror_so_existing_callers_still_narrow(self, tmp_path):
        ckpt = _checkpoint(tmp_path, declared="../foreign.json")
        with pytest.raises(ValueError):
            ns.load_norm_stats(str(ckpt))

    def test_the_outside_file_is_never_read(self, tmp_path):
        """Not merely unreturned: the escaping path never reaches ``open``."""
        ckpt = _checkpoint(tmp_path, declared=str(_foreign(tmp_path)))
        opened: list[str] = []
        real_open = open

        def recording_open(file, *args, **kwargs):
            opened.append(str(file))
            return real_open(file, *args, **kwargs)

        with patch("builtins.open", recording_open), pytest.raises(ns.NormStatsFilenameError):
            ns.load_norm_stats(str(ckpt))
        assert not any("foreign.json" in path for path in opened), opened


class TestANameInsideTheCheckpointStillLoads:
    """Non-narrowing control: every contained spelling keeps working."""

    @pytest.mark.parametrize(("label", "declared"), _CONTAINED, ids=[c[0] for c in _CONTAINED])
    def test_a_contained_override_is_honored(self, tmp_path, label, declared):
        ckpt = _checkpoint(tmp_path, declared=declared, stats_at=declared.removeprefix("./"))
        assert ns.is_norm_stats_payload(ns.load_norm_stats(str(ckpt)))

    def test_no_config_json_uses_the_default_filename(self, tmp_path):
        assert ns.is_norm_stats_payload(ns.load_norm_stats(str(_checkpoint(tmp_path))))

    def test_a_json_null_declares_nothing_and_falls_back_to_the_default(self, tmp_path):
        """``null`` is how a config spells "unset"; it is not a malformed name."""
        ckpt = _checkpoint(tmp_path, declared=None)
        assert ns.is_norm_stats_payload(ns.load_norm_stats(str(ckpt)))

    def test_a_contained_name_that_is_absent_is_still_a_plain_absence(self, tmp_path):
        ckpt = _checkpoint(tmp_path, declared="not_written.json")
        assert ns.load_norm_stats(str(ckpt)) is None


class TestTheApiArgumentIsHeldToTheSameRule:
    """``filename=`` is joined onto the same checkpoint, so it is graded too."""

    def test_an_escaping_filename_argument_is_refused(self, tmp_path):
        ckpt = _checkpoint(tmp_path)
        with pytest.raises(ns.NormStatsFilenameError, match="filename"):
            ns.load_norm_stats(str(ckpt), filename="../foreign.json")

    def test_a_contained_filename_argument_is_honored(self, tmp_path):
        ckpt = _checkpoint(tmp_path, stats_at="custom.json")
        assert ns.is_norm_stats_payload(ns.load_norm_stats(str(ckpt), filename="custom.json"))


class TestTheHubBranchIsHeldToTheSameRule:
    """A Hub repo's ``config.json`` is as external as a local one."""

    @staticmethod
    def _hub(monkeypatch, tmp_path, declared: Any) -> list[str]:
        """Patch ``hf_hub_download`` to serve local files; record what was asked for."""
        (tmp_path / "config.json").write_text(json.dumps({"norm_stats_filename": declared}))
        (tmp_path / "norm_stats.json").write_text(_FIXTURE.read_text())
        requested: list[str] = []

        def fake_download(repo_id, filename, *args, **kwargs):
            requested.append(filename)
            return str(tmp_path / filename)

        monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
        return requested

    def test_an_escaping_declared_name_is_refused_before_it_is_fetched(self, monkeypatch, tmp_path):
        requested = self._hub(monkeypatch, tmp_path, "../../../etc/hostname")
        with pytest.raises(ns.NormStatsFilenameError, match="norm_stats_filename"):
            ns.load_norm_stats("acme/molmoact2-so101")
        assert requested == ["config.json"], requested

    def test_a_contained_declared_name_is_still_fetched(self, monkeypatch, tmp_path):
        requested = self._hub(monkeypatch, tmp_path, "norm_stats.json")
        assert ns.is_norm_stats_payload(ns.load_norm_stats("acme/molmoact2-so101"))
        assert requested == ["config.json", "norm_stats.json"], requested

    def test_a_missing_config_json_is_still_a_benign_absence(self, monkeypatch, tmp_path):
        """The two fetches are guarded separately, so this path is unchanged."""
        (tmp_path / "norm_stats.json").write_text(_FIXTURE.read_text())

        def fake_download(repo_id, filename, *args, **kwargs):
            if filename == "config.json":
                raise FileNotFoundError("no config on hub")
            return str(tmp_path / filename)

        monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
        assert ns.is_norm_stats_payload(ns.load_norm_stats("acme/molmoact2-so101"))


@_requires_lerobot_pipeline
class TestNoForeignStatsReachTheMotorCommand:
    """The measured harm: which numbers unnormalize the predicted action."""

    def test_a_redirected_checkpoint_does_not_unnormalize_through_the_outside_file(self, tmp_path):
        ckpt = _checkpoint(tmp_path, declared="../foreign.json")
        action = np.array([0.1, -0.5, 0.3, 0.9, -1.0, 0.0], dtype=np.float32)

        foreign_stats = json.loads(_foreign(tmp_path).read_text())
        _, foreign_post = ns.build_norm_stats_processors(foreign_stats)
        would_have_been = np.asarray(foreign_post.process_action(action.copy()), dtype=np.float64)

        # Pre-fix the outsider's file was found and its wider action range
        # applied, so this is the number the motors would have been sent.
        assert np.max(np.abs(would_have_been)) > 100.0

        bridge = ProcessorBridge.from_pretrained(str(ckpt), device="cpu")
        applied = np.asarray(bridge.postprocess(action.copy()), dtype=np.float64)
        assert not np.allclose(applied, would_have_been), (applied, would_have_been)

    def test_the_bridge_reports_the_refusal_instead_of_applying_anything(self, tmp_path):
        ckpt = _checkpoint(tmp_path, declared="../foreign.json")
        bridge = ProcessorBridge.from_pretrained(str(ckpt), device="cpu")
        assert bridge.has_postprocessor is False
        assert bridge.inert_reason is not None
        assert "norm_stats_filename" in bridge.inert_reason
        assert bridge.get_info()["inert_reason"] == bridge.inert_reason

    def test_a_checkpoint_with_no_override_still_applies_its_own_stats(self, tmp_path):
        bridge = ProcessorBridge.from_pretrained(str(_checkpoint(tmp_path)), device="cpu")
        assert bridge.has_postprocessor is True
        assert bridge.inert_reason is None


class TestTheLoadReportNamesTheAccurateCause:
    """The warning blames the declaration, not a postprocessor never shipped."""

    @staticmethod
    def _policy() -> LerobotLocalPolicy:
        with patch.object(LerobotLocalPolicy, "_load_model"):
            return LerobotLocalPolicy(pretrained_name_or_path="fake/ckpt")

    def test_the_refusal_replaces_the_missing_postprocessor_message(self, monkeypatch, caplog):
        reason = "norm_stats_filename='../foreign.json' in fake/ckpt/config.json must not contain a '..' segment"
        bridge = MagicMock(name="ProcessorBridge")
        bridge.is_active = False
        bridge.has_postprocessor = False
        bridge.inert_reason = reason
        bridge.inert_normalization_features.return_value = []
        monkeypatch.setattr(
            "strands_robots.policies.lerobot_local.policy.ProcessorBridge.from_pretrained",
            classmethod(lambda cls, *a, **k: bridge),
        )
        pol = self._policy()

        with caplog.at_level(logging.WARNING):
            pol._load_processor_bridge()

        msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert any(reason in m for m in msgs), msgs
        # Supplying a postprocessor does not fix a name pointing out of the
        # checkpoint, so that remedy must not be the one reported.
        assert not any("policy_postprocessor.json" in m for m in msgs), msgs
