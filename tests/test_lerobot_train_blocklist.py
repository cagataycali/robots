"""Regression tests for lerobot_train extra_flags security blocklist + HIL gate."""

from __future__ import annotations

import argparse
import contextlib
import io
from unittest.mock import MagicMock

import pytest

from strands_robots.tools.lerobot_train import (
    _BLOCKED_EXTRA_FLAGS,
    _approve_response,
    _gate_extra_flags,
    _normalize_hydra_key,
    _validate_extra_flags,
)


class TestValidateExtraFlags:
    """Pin the blocklist contract: dangerous flags detected, benign flags pass."""

    @pytest.mark.parametrize(
        "key",
        [
            "output_dir",
            "--output_dir",
            "+output_dir",
            "~output_dir",
            "++output_dir",
        ],
    )
    def test_output_dir_all_hydra_forms_blocked(self, key):
        blocked = _validate_extra_flags({key: "/tmp/evil"})
        assert len(blocked) == 1
        assert blocked[0][1] == "output_dir"

    @pytest.mark.parametrize(
        "key",
        [
            "config_path",
            "--config_path",
            "+config_path",
        ],
    )
    def test_config_path_blocked(self, key):
        blocked = _validate_extra_flags({key: "/tmp/malicious.yaml"})
        assert len(blocked) == 1

    @pytest.mark.parametrize(
        "key",
        [
            "wandb.enable",
            "--wandb.enable",
            "+wandb.enable",
            "wandb.project",
            "wandb.entity",
            "wandb.api_key",
        ],
    )
    def test_wandb_flags_blocked(self, key):
        blocked = _validate_extra_flags({key: "true"})
        assert len(blocked) == 1

    @pytest.mark.parametrize(
        "key",
        [
            "dataset.root",
            "--dataset.root",
            "policy.pretrained_path",
            "--policy.pretrained_path",
        ],
    )
    def test_data_and_model_paths_blocked(self, key):
        blocked = _validate_extra_flags({key: "/etc/shadow"})
        assert len(blocked) == 1

    @pytest.mark.parametrize(
        "key",
        [
            "push_to_hub",
            "policy.push_to_hub",
            "hub_repo_id",
        ],
    )
    def test_hub_push_flags_blocked(self, key):
        blocked = _validate_extra_flags({key: "attacker/repo"})
        assert len(blocked) == 1

    def test_benign_flags_pass(self):
        assert _validate_extra_flags({"lr": "1e-4"}) == []
        assert _validate_extra_flags({"--batch_size": "32"}) == []
        assert _validate_extra_flags({"training.num_workers": "4"}) == []

    def test_multiple_flags_all_blocked_returned(self):
        blocked = _validate_extra_flags({"lr": "1e-4", "output_dir": "/tmp", "wandb.enable": "true"})
        assert len(blocked) == 2
        norms = {b[1] for b in blocked}
        assert norms == {"output_dir", "wandb.enable"}

    def test_empty_dict_passes(self):
        assert _validate_extra_flags({}) == []


class TestNormalizeHydraKey:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("output_dir", "output_dir"),
            ("--output_dir", "output_dir"),
            ("+output_dir", "output_dir"),
            ("~output_dir", "output_dir"),
            ("++output_dir", "output_dir"),
        ],
    )
    def test_strips_prefixes(self, raw, expected):
        assert _normalize_hydra_key(raw) == expected


class TestGateExtraFlags:
    """Pin the HIL gate contract: allowlist, bypass, interrupt, decline."""

    @pytest.fixture(autouse=True)
    def _hermetic_gate_env(self, monkeypatch):
        """Neutralize ambient env that short-circuits the gate.

        Both BYPASS_TOOL_CONSENT and STRANDS_TRAIN_EXTRA_FLAGS_ALLOW cause the
        gate to allow blocked flags without prompting. A developer or CI shell
        that exports BYPASS_TOOL_CONSENT=true (common in agent/automation
        contexts) would otherwise make the no-context, allowlist, and interrupt
        cases pass silently and fail their assertions. Clearing both per-test
        makes each case deterministic regardless of the ambient environment;
        tests that exercise those paths opt in explicitly via monkeypatch.setenv.
        """
        monkeypatch.delenv("BYPASS_TOOL_CONSENT", raising=False)
        monkeypatch.delenv("STRANDS_TRAIN_EXTRA_FLAGS_ALLOW", raising=False)

    def test_benign_flags_pass(self):
        assert _gate_extra_flags({"lr": "1e-4"}, None) is None

    def test_blocked_flag_no_context_returns_error(self):
        result = _gate_extra_flags({"output_dir": "/tmp"}, None)
        assert result is not None
        assert result["status"] == "error"
        assert "approval" in result["content"][0]["text"].lower()

    def test_allowlist_skips_gate(self, monkeypatch):
        monkeypatch.setenv("STRANDS_TRAIN_EXTRA_FLAGS_ALLOW", "output_dir")
        assert _gate_extra_flags({"output_dir": "/tmp"}, None) is None

    def test_allowlist_partial(self, monkeypatch):
        """Allowlist covers one flag but not the other."""
        monkeypatch.setenv("STRANDS_TRAIN_EXTRA_FLAGS_ALLOW", "output_dir")
        result = _gate_extra_flags({"output_dir": "/tmp", "wandb.enable": "true"}, None)
        assert result is not None
        assert result["status"] == "error"

    def test_bypass_consent_allows(self, monkeypatch):
        monkeypatch.setenv("BYPASS_TOOL_CONSENT", "true")
        assert _gate_extra_flags({"output_dir": "/tmp"}, None) is None

    def test_interrupt_approved(self):
        ctx = MagicMock()
        ctx.interrupt.return_value = "y"
        assert _gate_extra_flags({"output_dir": "/tmp"}, ctx) is None
        ctx.interrupt.assert_called_once()
        reason = ctx.interrupt.call_args[1]["reason"]
        assert reason["action"] == "train"
        assert "output_dir" in str(reason["blocked_flags"])

    def test_interrupt_declined(self):
        ctx = MagicMock()
        ctx.interrupt.return_value = "no"
        result = _gate_extra_flags({"output_dir": "/tmp"}, ctx)
        assert result is not None
        assert result["status"] == "error"
        assert "declined" in result["content"][0]["text"]

    def test_interrupt_runtime_error_fails_closed(self):
        ctx = MagicMock()
        ctx.interrupt.side_effect = RuntimeError("no agent loop")
        result = _gate_extra_flags({"output_dir": "/tmp"}, ctx)
        assert result is not None
        assert result["status"] == "error"

    @pytest.mark.parametrize("response", ["y", "Y", "yes", "YES", "approve", "Approved"])
    def test_approve_response_affirmative(self, response):
        assert _approve_response(response) is True

    @pytest.mark.parametrize("response", ["n", "no", "nope", "", 42, None])
    def test_approve_response_negative(self, response):
        assert _approve_response(response) is False


class TestPretrainedPathGate:
    """Pin: the pretrained_path named parameter is gated identically to extra_flags."""

    @pytest.fixture(autouse=True)
    def _hermetic_env(self, monkeypatch):
        monkeypatch.delenv("BYPASS_TOOL_CONSENT", raising=False)
        monkeypatch.delenv("STRANDS_TRAIN_EXTRA_FLAGS_ALLOW", raising=False)

    def test_pretrained_path_blocked_without_approval(self):
        """The named parameter hits the same gate as extra_flags={'policy.pretrained_path': ...}."""
        result = _gate_extra_flags({"policy.pretrained_path": "evil/model"}, None)
        assert result is not None
        assert result["status"] == "error"

    def test_pretrained_path_allowed_via_allowlist(self, monkeypatch):
        monkeypatch.setenv("STRANDS_TRAIN_EXTRA_FLAGS_ALLOW", "policy.pretrained_path")
        result = _gate_extra_flags({"policy.pretrained_path": "trusted/model"}, None)
        assert result is None

    def test_pretrained_path_approved_via_interrupt(self):
        ctx = MagicMock()
        ctx.interrupt.return_value = "y"
        result = _gate_extra_flags({"policy.pretrained_path": "org/model"}, ctx)
        assert result is None
        ctx.interrupt.assert_called_once()


class TestAbbreviatedFlagsReachTheSameGate:
    """Pin: a key that abbreviates a gated flag is gated as that flag.

    ``extra_flags`` keys are emitted verbatim into the argv of
    ``lerobot.scripts.lerobot_train``, whose parser (draccus over stdlib
    :mod:`argparse`) honors any unambiguous prefix of a registered option. So
    ``{"ou": "/anywhere"}`` reaches ``--output_dir``, and a gate that compares
    whole keys sees a name that is on no list.

    The cells above vary the *Hydra prefix* of a key exhaustively and never vary
    the name, which is the half the parser resolves.
    """

    @staticmethod
    def _rule():
        """The blocked-flag resolver under test, imported inside the class.

        A module-level import would make every cell here a collection error on a
        tree without the resolver, and a collection error grades nothing.
        """
        from strands_robots.tools.lerobot_train import _blocked_flags_named

        return _blocked_flags_named

    #: Option names a lerobot train parser registers, in the shape draccus
    #: builds them: every gated flag, the nested configs that carry the dotted
    #: ones, and enough siblings for a short prefix to be ambiguous.
    _OPTIONS = tuple(
        sorted(
            set(_BLOCKED_EXTRA_FLAGS)
            | {
                "batch_size",
                "dataset",
                "dataset.repo_id",
                "observation",
                "optimizer",
                "optimizer.lr",
                "policy",
                "policy.type",
                "steps",
                "wandb",
                "wandb.notes",
            }
        )
    )

    @staticmethod
    def _argparse_verdict(options, candidate):
        """What argparse does with ``--candidate=X`` against ``options``.

        Returns the option name it resolved to, ``"ambiguous"`` when the prefix
        matches several, or ``"unrecognized"`` when it matches none.
        """
        parser = argparse.ArgumentParser(add_help=False)
        for name in options:
            parser.add_argument(f"--{name}", dest=name)
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                namespace, extras = parser.parse_known_args([f"--{candidate}=X"])
            except SystemExit:
                return "ambiguous"
        if extras:
            return "unrecognized"
        return next(name for name, value in vars(namespace).items() if value == "X")

    @pytest.mark.parametrize(
        "key,flag",
        [
            # Measured against lerobot 0.5.1's TrainPipelineConfig: each of these
            # spellings sets the named field through draccus.
            ("ou", "output_dir"),
            ("output", "output_dir"),
            ("outp", "output_dir"),
            ("co", "config_path"),
            ("config", "config_path"),
            ("wandb.p", "wandb.project"),
            ("wandb.ent", "wandb.entity"),
            ("wandb.ena", "wandb.enable"),
            ("dataset.ro", "dataset.root"),
            # The Hydra prefixes the cells above cover, on an abbreviation.
            ("--ou", "output_dir"),
            ("+ou", "output_dir"),
            ("~ou", "output_dir"),
        ],
    )
    def test_an_abbreviation_names_the_flag_it_reaches(self, key, flag):
        assert self._rule()(key) == (flag,)
        assert _validate_extra_flags({key: "/tmp/evil"}) == [(key, flag)]

    def test_a_nested_config_name_is_not_an_abbreviation_of_its_gated_child(self):
        """``--wandb`` is its own option, so argparse never reads it as a child.

        draccus registers an option for each nested config beside one per field,
        and argparse prefers an exact match over any abbreviation. Gating these
        would refuse three whole-config overrides that reach no gated flag.
        """
        for parent in ("wandb", "dataset", "policy"):
            assert self._rule()(parent) == ()
            assert _validate_extra_flags({parent: "{}"}) == []
            assert self._argparse_verdict(self._OPTIONS, parent) == parent

    def test_an_ambiguous_prefix_names_every_gated_flag_it_could_reach(self):
        """``wandb.e`` could be ``enable`` or ``entity``; it is held to both.

        argparse refuses an ambiguous prefix outright, so this is the
        conservative direction: it costs a prompt for a spelling the trainer
        would reject, and it means the allowlist has to clear both.
        """
        assert self._rule()("wandb.e") == ("wandb.enable", "wandb.entity")
        assert self._argparse_verdict(self._OPTIONS, "wandb.e") == "ambiguous"

    def test_every_partial_segment_prefix_of_every_gated_flag_is_gated(self):
        """Derived from the blocklist, so a flag added later is covered on arrival."""
        rule = self._rule()
        checked = 0
        for flag in _BLOCKED_EXTRA_FLAGS:
            for cut in range(1, len(flag)):
                prefix = flag[:cut]
                if flag[cut] == ".":
                    continue  # a whole leading segment: an option of its own
                assert flag in rule(prefix), f"{prefix!r} reaches {flag!r} ungated"
                checked += 1
        assert checked > 60, f"the blocklist stopped covering prefixes: {checked}"

    def test_the_rule_agrees_with_argparse_over_every_prefix_of_every_gated_flag(self):
        """The oracle: argparse itself decides which spellings reach a gated flag.

        For each candidate the verdict is what a lerobot-shaped parser does with
        it, and the rule must refuse exactly the candidates that land on a gated
        flag or that are ambiguous with one.
        """
        rule = self._rule()
        candidates = {flag[:cut] for flag in _BLOCKED_EXTRA_FLAGS for cut in range(1, len(flag) + 1)}
        candidates |= {"batch_size", "steps", "op", "observation", "dataset.repo_id", "lr"}
        for candidate in sorted(candidates):
            verdict = self._argparse_verdict(self._OPTIONS, candidate)
            reaches_gated = verdict in _BLOCKED_EXTRA_FLAGS or (
                verdict == "ambiguous" and any(flag.startswith(candidate) for flag in _BLOCKED_EXTRA_FLAGS)
            )
            assert bool(rule(candidate)) == reaches_gated, f"{candidate!r}: argparse says {verdict!r}"

    def test_pre_approving_a_flag_clears_its_abbreviations(self, monkeypatch):
        """One allowlist entry covers every spelling of the flag it names.

        The operator approves a flag, not an argv spelling, so
        ``STRANDS_TRAIN_EXTRA_FLAGS_ALLOW=output_dir`` has to clear ``ou`` too -
        otherwise the gate would prompt for a flag already approved.
        """
        monkeypatch.delenv("BYPASS_TOOL_CONSENT", raising=False)
        monkeypatch.delenv("STRANDS_TRAIN_EXTRA_FLAGS_ALLOW", raising=False)
        assert _gate_extra_flags({"ou": "/tmp/evil"}, None) is not None
        monkeypatch.setenv("STRANDS_TRAIN_EXTRA_FLAGS_ALLOW", "output_dir")
        assert _gate_extra_flags({"ou": "/tmp/evil"}, None) is None

    def test_the_operator_prompt_quotes_the_spelling_the_caller_wrote(self, monkeypatch):
        """The prompt has to show the argv, not only the flag it resolves to."""
        monkeypatch.delenv("BYPASS_TOOL_CONSENT", raising=False)
        monkeypatch.delenv("STRANDS_TRAIN_EXTRA_FLAGS_ALLOW", raising=False)
        ctx = MagicMock()
        ctx.interrupt.return_value = "y"
        assert _gate_extra_flags({"ou": "/tmp/evil"}, ctx) is None
        reason = ctx.interrupt.call_args[1]["reason"]
        assert reason["blocked_flags"] == {"ou": "/tmp/evil"}
        assert "ou" in reason["warning"]

    def test_one_key_naming_two_gated_flags_is_reported_once(self, monkeypatch):
        """Two pairs share a key, and the caller is told about the key once."""
        monkeypatch.delenv("BYPASS_TOOL_CONSENT", raising=False)
        monkeypatch.delenv("STRANDS_TRAIN_EXTRA_FLAGS_ALLOW", raising=False)
        result = _gate_extra_flags({"wandb.e": "true"}, None)
        assert result is not None
        assert result["content"][0]["text"].count("wandb.e") == 1
