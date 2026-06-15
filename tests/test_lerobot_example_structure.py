"""Pin the LeRobot example to the agent-first shape: system prompt + tools.

The hub_to_hardware example exists to teach the canonical Strands pattern -
a system prompt, a set of tools (a ``Robot`` plus the mesh), and a handful of
natural-language invocations. It must NOT re-implement functionality that is
already baked into strands_robots (dataset state inspection, tool-call logging,
Bedrock model plumbing, an imperative record routine). Those scaffolding
helpers distract from the lesson and drift out of sync with the SDK.

This test loads the example module by path (it is not an installed package)
and asserts the agent-first shape stays intact.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

_EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "lerobot" / "hub_to_hardware.py"


def _load_example() -> ModuleType:
    spec = importlib.util.spec_from_file_location("hub_to_hardware_example", _EXAMPLE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def example() -> ModuleType:
    return _load_example()


def test_example_file_exists() -> None:
    assert _EXAMPLE.is_file()


def test_defines_system_prompt(example: ModuleType) -> None:
    """The agent-first pattern hinges on a real system prompt."""
    prompt = getattr(example, "SYSTEM_PROMPT", None)
    assert isinstance(prompt, str)
    assert prompt.strip()
    # The prompt must teach the abstraction, not raw lerobot.
    assert "strands_robots" in prompt
    assert "lerobot APIs" in prompt or "raw lerobot" in prompt


@pytest.mark.parametrize(
    "name",
    [
        "_lerobot_cache_root",
        "_read_dataset_state",
        "_log_dataset_summary",
        "_get",
        "_log_agent_tool_calls",
        "_log_prompt",
        "_build_bedrock_model",
        "record_demonstration",
    ],
)
def test_scaffolding_removed(example: ModuleType, name: str) -> None:
    """Functionality baked into strands_robots must not be re-implemented here."""
    assert not hasattr(example, name), (
        f"{name} re-implements SDK functionality in the example. "
        "The example should delegate to the Robot tool, not scaffold around it."
    )


def test_example_does_not_import_lerobot(example: ModuleType) -> None:
    """The example must demonstrate the SDK, never bypass it into lerobot."""
    source = _EXAMPLE.read_text(encoding="utf-8")
    assert "import lerobot" not in source
    assert "from lerobot" not in source


def test_build_agent_is_the_construction_surface(example: ModuleType) -> None:
    """A single build_agent wires the system prompt + tools together."""
    assert callable(example.build_agent)


def test_prompt_builders_are_pure_strings(example: ModuleType) -> None:
    """Each workflow phase composes one natural-language instruction string."""
    assert example.mesh_prompt().strip()
    mock = example.policy_prompt(policy="mock", checkpoint=None, instruction="pick up the red cube")
    assert "Mock policy" in mock and "pick up the red cube" in mock
    sim_rec = example.record_prompt(
        mode="sim",
        agent=None,
        repo_id="local/demo",
        num_steps=10,
        task="pick up the red cube",
        push_to_hub=False,
    )
    assert "start recording" in sim_rec.lower() and "local/demo" in sim_rec


def test_unknown_policy_rejected(example: ModuleType) -> None:
    with pytest.raises(SystemExit):
        example.policy_prompt(policy="nope", checkpoint=None, instruction="x")
