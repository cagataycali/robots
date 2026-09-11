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

LeRobot honors the request, emitting ``--peft.method_type=LORA`` with the rank
and scaling. Cosmos3 refuses it, naming the recipe TOML as the one surface that
can express a different strategy, so the request can no longer be silently
downgraded to the run the caller did not ask for.

GR00T went the same way one value further in. It refuses ``lora`` by name, and
it reads ``method`` again for ``frozen_backbone`` - so the field is forwarded
and the module looked compliant - while ``expert_only``, also accepted, was
named nowhere but the accepted set. GR00T freezes ``tune_llm`` / ``tune_visual``
/ ``tune_projector`` / ``tune_diffusion_model`` individually and has no single
expert-only switch, so an accepted ``expert_only`` produced the four flags of a
plain ``full`` fine-tune: the projector the caller asked to freeze trained, and
``validate`` reported no problems. lerobot's own gate already refuses
``method="expert_only"`` for its native ``groot`` policy for the same reason -
``GrootConfig`` carries the four switches, not a ``train_expert_only`` field -
and its docstring names this exact failure ("the run silently full-finetunes the
backbone while reporting success"). The component set is expressible here, just
not as a method: ``tune={"projector": False}`` is honored.

The sweep at the bottom derives its scope and grades per VALUE, not per field:
a backend that declares its own set of accepted methods *and* builds a launch
description from the spec must name every accepted non-baseline strategy
somewhere other than the declaration and the gate that accepts it. Grading the
field alone is what let ``expert_only`` through.
"""

from __future__ import annotations

import ast
import copy
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


@pytest.fixture
def groot_spec(tmp_path: pathlib.Path) -> TrainSpec:
    """A launchable GR00T spec whose ``method`` is the only thing under test.

    Every input GR00T's ``validate`` reads is present and usable - a v3 dataset
    root, a base model, an output dir, the required embodiment tag and a
    checkout holding ``launch_finetune.py`` - so an empty problem list means
    "launchable" and a non-empty one is about ``method``.
    """
    meta = tmp_path / "ds" / "meta"
    meta.mkdir(parents=True)
    meta.joinpath("info.json").write_text(json.dumps({"codebase_version": "v3.0"}))
    launch = tmp_path / "groot" / "gr00t" / "experiment"
    launch.mkdir(parents=True)
    launch.joinpath("launch_finetune.py").write_text("")
    return TrainSpec(
        dataset_root=str(tmp_path / "ds"),
        output_dir=str(tmp_path / "out"),
        base_model="nvidia/GR00T-N1.5-3B",
        embodiment="new_embodiment",
        extra={"groot_root": str(tmp_path / "groot")},
    )


def _tune_flags(trainer: Gr00tTrainer, spec: TrainSpec) -> dict[str, str]:
    """The four ``--tune_*`` flags of the command GR00T would launch."""
    return dict(flag.lstrip("-").split("=", 1) for flag in trainer.build_command(spec) if flag.startswith("--tune_"))


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

    def test_groot_refuses_the_strategy_it_has_no_field_for(self, groot_spec: TrainSpec) -> None:
        groot_spec.method = "lora"
        assert _method_problems(Gr00tTrainer(), groot_spec)


class TestGr00tReportsTheExpertOnlyRequestItCannotName:
    """An accepted ``expert_only`` trained the projector and reported success.

    GR00T's flags name components, not strategies, so the request only means
    something once the caller says which components. Refusing it is a statement
    about this backend: the peer above honors the same value.
    """

    def test_it_is_reported_as_a_problem(self, groot_spec: TrainSpec) -> None:
        groot_spec.method = "expert_only"
        assert _method_problems(Gr00tTrainer(), groot_spec), "method='expert_only' was accepted"

    def test_the_refusal_names_the_component_set_that_expresses_it(self, groot_spec: TrainSpec) -> None:
        """A dead end would leave the caller with nothing to do next."""
        groot_spec.method = "expert_only"
        problems = _method_problems(Gr00tTrainer(), groot_spec)
        assert problems and "tune={'projector': False}" in problems[0], problems

    def test_the_refusal_names_the_backend_and_the_value(self, groot_spec: TrainSpec) -> None:
        groot_spec.method = "expert_only"
        problems = _method_problems(Gr00tTrainer(), groot_spec)
        assert problems and "GR00T" in problems[0] and "'expert_only'" in problems[0]

    def test_validate_reports_rather_than_raises(self, groot_spec: TrainSpec) -> None:
        """``validate`` is documented to *return* problems."""
        groot_spec.method = "expert_only"
        assert isinstance(Gr00tTrainer().validate(groot_spec), list)

    def test_one_problem_is_reported_for_one_bad_value(self, groot_spec: TrainSpec) -> None:
        """The named refusal replaces the generic one rather than joining it."""
        groot_spec.method = "expert_only"
        assert len(_method_problems(Gr00tTrainer(), groot_spec)) == 1


class TestTheGr00tStrategiesThatSurviveAreUnchanged:
    """The fix narrows what ``validate`` accepts and writes no new flag.

    ``frozen_backbone`` is graded here as well as ``full``: its flags equal the
    default because ``_DEFAULT_TUNE`` already freezes the backbone, so it is
    honored rather than dropped, and it must keep launching what it launched.
    """

    @pytest.mark.parametrize("method", [BASELINE, "frozen_backbone"])
    def test_it_is_launchable(self, groot_spec: TrainSpec, method: str) -> None:
        groot_spec.method = method
        assert Gr00tTrainer().validate(groot_spec) == []

    @pytest.mark.parametrize("method", [BASELINE, "frozen_backbone"])
    def test_it_builds_the_backbone_frozen_action_head_run(self, groot_spec: TrainSpec, method: str) -> None:
        groot_spec.method = method
        assert _tune_flags(Gr00tTrainer(), groot_spec) == {
            "tune_llm": "false",
            "tune_visual": "false",
            "tune_projector": "true",
            "tune_diffusion_model": "true",
        }

    def test_the_component_set_the_refusal_names_is_honored(self, groot_spec: TrainSpec) -> None:
        """The refusal points at a route that works, not at a second dead end."""
        groot_spec.tune = {"projector": False}
        trainer = Gr00tTrainer()
        assert trainer.validate(groot_spec) == []
        assert _tune_flags(trainer, groot_spec) == {
            "tune_llm": "false",
            "tune_visual": "false",
            "tune_projector": "false",
            "tune_diffusion_model": "true",
        }


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


class _DropAcceptance(ast.NodeTransformer):
    """Strip the accepted-method declaration and every validation function.

    What remains is the part of a backend that *builds* something: naming a
    strategy in the set that admits it, or in the gate that reports it, says
    nothing about whether the run is configured differently.
    """

    def visit_Assign(self, node: ast.Assign) -> ast.Assign | None:
        declares = any(getattr(target, "id", "") == "_SUPPORTED_METHODS" for target in node.targets)
        return None if declares else node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef | None:
        return None if node.name.lstrip("_").startswith("validate") else node


def _unforwarded_methods(tree: ast.Module) -> set[str]:
    """Accepted non-baseline strategies this backend never names again.

    Graded per VALUE. A module that reads ``spec.method`` for one accepted
    strategy forwards the *field*, which is why the field-scoped question
    reported GR00T compliant while ``expert_only`` reached nothing: the read at
    ``_resolve_tune`` is about ``frozen_backbone``. A value named anywhere
    outside the declaration and the gate is credited, so a backend that
    forwards through a lookup table counts as forwarding too.
    """
    accepted = _accepted_methods(tree)
    remainder = _DropAcceptance().visit(copy.deepcopy(tree))
    named = {
        node.value
        for node in ast.walk(remainder)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in accepted
    }
    return (accepted - {BASELINE}) - named


class TestEveryAcceptedStrategyIsNamedWhereTheRunIsBuilt:
    """A backend that accepts a non-baseline strategy must consult it again.

    Scope is derived from the tree, so a backend that starts accepting an
    adapter or freeze strategy fails this until it forwards that value.
    """

    def test_the_scan_finds_the_backends_that_declare_a_method_set(self) -> None:
        """Non-vacuity: a mis-rooted scan cannot report a clean sweep of nothing."""
        assert set(_backends_that_declare_their_methods()) == {"cosmos3.py", "groot.py", "lerobot.py"}

    def test_no_backend_accepts_a_strategy_it_never_names_again(self) -> None:
        adrift = {
            name: sorted(unforwarded)
            for name, tree in _backends_that_declare_their_methods().items()
            if (unforwarded := _unforwarded_methods(tree))
        }
        assert adrift == {}, f"strategies accepted but never forwarded: {adrift}"

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
        assert _unforwarded_methods(planted) == {"lora"}

    def test_a_value_forwarded_by_name_is_not_an_offender(self) -> None:
        planted = ast.parse(
            '_SUPPORTED_METHODS = {"full", "lora"}\n'
            "class T:\n"
            "    def build_command(self, spec):\n"
            "        return ['--peft'] if spec.method == 'lora' else []\n"
        )
        assert _unforwarded_methods(planted) == set()

    def test_a_value_forwarded_through_a_table_is_not_an_offender(self) -> None:
        """A backend that maps strategies to flags forwards them too."""
        planted = ast.parse(
            '_SUPPORTED_METHODS = {"full", "lora"}\n'
            '_FLAGS = {"lora": "--peft.method_type=LORA"}\n'
            "class T:\n"
            "    def build_command(self, spec):\n"
            "        return [_FLAGS[spec.method]]\n"
        )
        assert _unforwarded_methods(planted) == set()

    def test_naming_a_value_only_in_the_gate_is_not_forwarding(self) -> None:
        """The distinction the field-scoped question could not draw.

        This module accepts two strategies, reads ``method`` outside
        ``validate`` for one of them, and reaches nothing for the other - the
        shape GR00T had.
        """
        planted = ast.parse(
            '_SUPPORTED_METHODS = {"full", "lora", "frozen_backbone"}\n'
            "class T:\n"
            "    def _validate_method(self, spec):\n"
            "        return [] if spec.method != 'lora' else ['no adapter']\n"
            "    def build_command(self, spec):\n"
            "        return ['--freeze'] if spec.method == 'frozen_backbone' else []\n"
        )
        assert reads_spec_field(ast.unparse(planted), ("method",))
        assert _unforwarded_methods(planted) == {"lora"}
