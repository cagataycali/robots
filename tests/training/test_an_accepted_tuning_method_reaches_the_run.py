# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A tuning strategy a backend accepts must reach what that backend launches.

``TrainSpec.method`` is a request to train *differently*: ``lora`` trains a
rank-``r`` adapter, ``expert_only`` and ``frozen_backbone`` freeze part of the
model. ``full`` is the baseline - the absence of such a request - so any other
value only means something if the backend forwards it.

The Cosmos3 backend accepted ``method="lora"`` and forwarded nothing. Its run is
configured by the recipe TOML plus the Hydra override list ``build_overrides``
writes, and that list has no adapter entry, so ``validate`` returned no problems
and the override list for a LoRA request was byte-identical to the one for a
full fine-tune: the caller asked for an adapter, got a full fine-tune of the
whole model, and was told the run succeeded. ``lora_r`` went the same way - the
adapter-hyperparameter domain is owed only by a backend that reads the field, so
a rank of ``0`` was accepted here too.

The two peers show both halves of the correct posture. LeRobot honors the
request, emitting ``--peft.method_type=LORA`` with the rank and scaling. GR00T
refuses ``lora`` by name, because it has no config field to carry it. Cosmos3
now refuses it the same way, naming the recipe TOML as the one surface that can
express a different strategy, so the request can no longer be silently
downgraded to the run the caller did not ask for.

The sweep at the bottom derives its scope: a backend that declares its own set
of accepted methods *and* builds a launch description from the spec must read
``method`` somewhere other than the check that accepts it.
"""

from __future__ import annotations

import ast
import inspect
import json
import pathlib

import pytest

from strands_robots.training.base import Trainer, TrainSpec
from strands_robots.training.cosmos3 import Cosmos3Trainer
from strands_robots.training.groot import Gr00tTrainer
from tests.training._spec_field_reads import reads_spec_field

# Strategies that ask for something other than a full fine-tune.
ADAPTER_OR_FREEZE = ("lora", "expert_only", "frozen_backbone")

BASELINE = "full"


@pytest.fixture
def cosmos_spec(tmp_path: pathlib.Path) -> TrainSpec:
    """A launchable Cosmos3 spec whose ``method`` is the only thing under test.

    Every input Cosmos3's ``validate`` reads is present and usable, so an empty
    problem list means "launchable" and a non-empty one is about ``method``.
    """
    meta = tmp_path / "ds" / "meta"
    meta.mkdir(parents=True)
    meta.joinpath("info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "total_episodes": 10,
                "total_tasks": 1,
                "total_frames": 1200,
                "fps": 30,
                "features": {},
            }
        )
    )
    root = tmp_path / "cosmos"
    (root / "cosmos_framework").mkdir(parents=True)
    toml = tmp_path / "sft.toml"
    toml.write_text("[experiment]\nname = 'x'\n")
    return TrainSpec(
        dataset_root=str(tmp_path / "ds"),
        output_dir=str(tmp_path / "out"),
        base_model="nvidia/cosmos-base",
        extra={"sft_toml": str(toml), "cosmos_root": str(root)},
    )


def _method_problems(trainer: Trainer, spec: TrainSpec) -> list[str]:
    """The ``validate`` problems that are about ``method``."""
    return [p for p in trainer.validate(spec) if "method" in p]


class TestAStrategyTheBackendCannotForwardIsRefused:
    """Cosmos3 reports the strategy it cannot carry instead of running another."""

    @pytest.mark.parametrize("method", ADAPTER_OR_FREEZE)
    def test_it_is_reported_as_a_problem(self, cosmos_spec: TrainSpec, method: str) -> None:
        cosmos_spec.method = method
        assert _method_problems(Cosmos3Trainer(), cosmos_spec), f"method={method!r} was accepted"

    def test_a_lora_request_is_refused_even_with_usable_hyperparameters(self, cosmos_spec: TrainSpec) -> None:
        """A rank and a scaling do not make the request forwardable."""
        cosmos_spec.method = "lora"
        cosmos_spec.lora_r = 64
        cosmos_spec.lora_alpha = 128
        assert _method_problems(Cosmos3Trainer(), cosmos_spec)

    def test_the_refusal_names_the_surface_that_can_express_it(self, cosmos_spec: TrainSpec) -> None:
        """A dead end would leave the caller with nothing to do next."""
        cosmos_spec.method = "lora"
        problems = _method_problems(Cosmos3Trainer(), cosmos_spec)
        assert problems and "sft_toml" in problems[0], problems

    def test_the_refusal_names_the_backend_and_the_value(self, cosmos_spec: TrainSpec) -> None:
        cosmos_spec.method = "lora"
        problems = _method_problems(Cosmos3Trainer(), cosmos_spec)
        assert problems and "Cosmos3" in problems[0] and "'lora'" in problems[0]

    @pytest.mark.parametrize("method", ADAPTER_OR_FREEZE)
    def test_validate_reports_rather_than_raises(self, cosmos_spec: TrainSpec, method: str) -> None:
        """``validate`` is documented to *return* problems."""
        cosmos_spec.method = method
        assert isinstance(Cosmos3Trainer().validate(cosmos_spec), list)


class TestTheFullFineTuneIsUnchanged:
    """The baseline strategy - and everything it launches - is untouched.

    The fix narrows what ``validate`` accepts and writes no new override, so a
    full fine-tune must build exactly the run it built before.
    """

    def test_a_full_fine_tune_is_launchable(self, cosmos_spec: TrainSpec) -> None:
        assert Cosmos3Trainer().validate(cosmos_spec) == []

    def test_the_override_list_is_the_same_four_entries(self, cosmos_spec: TrainSpec) -> None:
        trainer = Cosmos3Trainer()
        assert trainer.build_overrides(cosmos_spec) == [
            f"trainer.max_iter={cosmos_spec.steps}",
            f"checkpoint.save_iter={cosmos_spec.save_freq}",
            f"checkpoint.load_path={cosmos_spec.output_dir}/_dcp_base",
            f"dataloader_train.max_samples_per_batch={cosmos_spec.global_batch_size}",
        ]

    def test_no_override_names_an_adapter(self, cosmos_spec: TrainSpec) -> None:
        """The refusal is not a stand-in for a wire format nobody verified."""
        overrides = Cosmos3Trainer().build_overrides(cosmos_spec)
        assert [o for o in overrides if "lora" in o.lower() or "peft" in o.lower()] == []


class TestThePeersShowBothHalvesOfThePosture:
    """The request is honored where a config field carries it, refused where none does.

    Non-vacuity for the refusal above: an adapter request is not universally
    unsupported, so refusing it is a statement about this backend.
    """

    def test_lerobot_forwards_the_adapter_the_caller_asked_for(self, tmp_path: pathlib.Path) -> None:
        pytest.importorskip("psutil")
        from strands_robots.training.lerobot import LerobotTrainer

        def command(method: str) -> list[str]:
            spec = TrainSpec(
                dataset_root=str(tmp_path / "ds"),
                output_dir=str(tmp_path / "out"),
                base_model="lerobot/act",
                method=method,
                lora_r=64,
                lora_alpha=128,
                extra={"policy_type": "act"},
            )
            return LerobotTrainer().build_command(spec)

        assert "--peft.method_type=LORA" in command("lora")
        assert command("lora") != command(BASELINE)

    def test_groot_refuses_the_strategy_it_has_no_field_for(self, tmp_path: pathlib.Path) -> None:
        spec = TrainSpec(
            dataset_root=str(tmp_path / "ds"),
            output_dir=str(tmp_path / "out"),
            base_model="nvidia/gr00t",
            embodiment="so101",
            method="lora",
        )
        assert _method_problems(Gr00tTrainer(), spec)


def _backends_that_declare_their_methods() -> dict[str, ast.Module]:
    """Trainer modules that declare an accepted-method set and build a run.

    Both halves are required and neither is listed. A module with no accepted
    set of its own does not decide the question (a transport backend forwards
    the value to a runner that does), and one that builds no launch description
    has nothing for the strategy to reach - the dependency-free reference
    trainer simulates a run rather than launching one.
    """
    root = pathlib.Path(inspect.getfile(Trainer)).parent
    found: dict[str, ast.Module] = {}
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text())
        declares = any(
            isinstance(node, ast.Assign)
            and any(getattr(target, "id", "") == "_SUPPORTED_METHODS" for target in node.targets)
            for node in tree.body
        )
        builds = any(
            isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name.startswith("build_")
            for node in ast.walk(tree)
        )
        if declares and builds:
            found[path.name] = tree
    return found


def _accepted_methods(tree: ast.Module) -> set[str]:
    """The literal ``_SUPPORTED_METHODS`` set of one backend module."""
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "_SUPPORTED_METHODS" for t in node.targets):
            return set(ast.literal_eval(node.value))
    raise AssertionError("module does not declare _SUPPORTED_METHODS")


def _reads_method_outside_validate(tree: ast.Module) -> bool:
    """Does the module read ``spec.method`` anywhere but in ``validate``?

    ``validate`` is where the value is *accepted*; reading it only there is the
    defect. The read itself is recognized by the rule the field-scoped domain
    guards share, so a backend that forwards ``method`` through a table counts
    as a reader too.
    """

    class _DropValidate(ast.NodeTransformer):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef | None:
            return None if node.name == "validate" else node

    return reads_spec_field(ast.unparse(_DropValidate().visit(tree)), ("method",))


class TestEveryAcceptedStrategyIsReadWhereTheRunIsBuilt:
    """A backend that accepts a non-baseline strategy must consult it again.

    Scope is derived from the tree, so a backend that starts accepting an
    adapter strategy fails this until it forwards the value.
    """

    def test_the_scan_finds_the_backends_that_declare_a_method_set(self) -> None:
        """Non-vacuity: a mis-rooted scan cannot report a clean sweep of nothing."""
        assert set(_backends_that_declare_their_methods()) == {"cosmos3.py", "groot.py", "lerobot.py"}

    def test_a_backend_accepting_more_than_a_full_fine_tune_reads_the_field(self) -> None:
        adrift = sorted(
            name
            for name, tree in _backends_that_declare_their_methods().items()
            if _accepted_methods(tree) - {BASELINE} and not _reads_method_outside_validate(tree)
        )
        assert adrift == [], f"backends accepting a strategy they never forward: {adrift}"

    def test_the_rule_grades_a_non_empty_population(self) -> None:
        """Non-vacuity: the rule above is not satisfied by an empty population."""
        in_scope = sorted(
            name
            for name, tree in _backends_that_declare_their_methods().items()
            if _accepted_methods(tree) - {BASELINE}
        )
        assert in_scope == ["groot.py", "lerobot.py"]

    def test_the_scanner_detects_a_planted_defect(self) -> None:
        """A scanner that matched nothing would look like a clean tree."""
        planted = ast.parse(
            '_SUPPORTED_METHODS = {"full", "lora"}\n'
            "class T:\n"
            "    def validate(self, spec):\n"
            "        return [] if spec.method in _SUPPORTED_METHODS else ['bad']\n"
            "    def build_command(self, spec):\n"
            "        return ['train']\n"
        )
        assert _accepted_methods(planted) - {BASELINE}
        assert not _reads_method_outside_validate(planted)

    def test_the_scanner_sees_a_forwarded_read(self) -> None:
        """A backend that forwards ``method`` by name is a reader, not an offender."""
        planted = ast.parse(
            '_SUPPORTED_METHODS = {"full", "lora"}\n'
            'FORWARDED = ("method",)\n'
            "class T:\n"
            "    def build_command(self, spec):\n"
            "        return [getattr(spec, f) for f in FORWARDED]\n"
        )
        assert _reads_method_outside_validate(planted)
