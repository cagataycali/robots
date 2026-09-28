"""A page that runs the mock policy promises what its own rollout does, not what its words say.

The old quickstart's front matter read "Five minutes from install to a robot
picking up a cube", and the only rollout on the page was ``policy_provider="mock"``.
:class:`~strands_robots.policies.mock.MockPolicy` declares
:attr:`~strands_robots.policies.base.Policy.reads_instruction` ``False``: it
drives every joint through a test motion whatever the task says, so the five
minutes the page sold ended with the cube exactly where it was put. The library
already refuses to let that read as a completed task: every envelope carries
:func:`~strands_robots.policies.base.instruction_not_read_notice`, whose own
docstring records that without it "MockPolicy | wave the arm ... completed" was
relayed as a wave that happened. The page was making the same claim one level
up, where no envelope reaches.

The new Start pages run no policy at all; the mock rollout that names a task
lives on ``docs/learn/policies/index.md``. So the rule is stated over the tree
rather than over one page: every hand-written page whose fence runs the mock
with an ``instruction=`` must show the notice the envelope carries and name the
attribute behind it, and no Start page may promise a manipulation its fences do
not perform. The provider is read out of each page's own fence, and the caveat
is required only while that provider's class declares it does not read the
instruction, so a future ``mock`` that acts on the words lifts the requirement
rather than outliving it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from strands_robots.policies.base import Policy, instruction_not_read_notice, provider_policy_class

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs"
START = DOCS / "start"

#: The words the old promise used; a Start page may not sell them unless a fence delivers them.
_MANIPULATION_PROMISES = ("picking up", "picks up", "pick up", "lifts the cube", "lift the cube", "grasp")


def _pages() -> list[Path]:
    """Every hand-written page; generated robot pages carry no rollout prose."""
    return [p for p in sorted(DOCS.rglob("*.md")) if "robots" not in p.relative_to(DOCS).parts]


def _fences(page: Path) -> list[str]:
    return re.findall(r"```python[^\n]*\n(.*?)```", page.read_text(encoding="utf-8"), re.DOTALL)


def _instructed_mock_rollouts() -> list[Path]:
    """Pages whose own fence runs ``policy_provider="mock"`` with an ``instruction=``.

    Read from the pages rather than named here: the cells below are about the
    policy a reader actually gets, so a page that switched providers should
    change what is required of it, not quietly keep passing.
    """
    out = []
    for page in _pages():
        for fence in _fences(page):
            if "run_policy(" in fence and 'policy_provider="mock"' in fence and "instruction=" in fence:
                out.append(page)
                break
    return out


def _first_paragraph(page: Path) -> str:
    text = page.read_text(encoding="utf-8")
    text = re.sub(r"^---\n.*?\n---\n", "", text, flags=re.DOTALL)
    body = text.split("\n# ", 1)[-1].split("\n", 1)[-1]
    for block in body.split("\n\n"):
        if block.strip() and not block.lstrip().startswith(("```", "#", "|", "<", "-")):
            return block
    return ""


def _description(page: Path) -> str:
    match = re.search(r"^---\n(.*?)\n---\n", page.read_text(encoding="utf-8"), re.DOTALL)
    if not match:
        return ""
    described = re.search(r"^description: (.+)$", match.group(1), re.MULTILINE)
    return described.group(1) if described else ""


def test_the_mock_does_not_read_the_instruction() -> None:
    """The premise of every cell below, established from the class itself.

    Without this the caveat cells would pass on a page that carries the words
    while the policy behind them had started acting on the task.
    """
    policy = provider_policy_class("mock")

    assert policy is not None
    assert issubclass(policy, Policy), policy
    assert policy.reads_instruction is False, policy


def test_some_page_runs_the_mock_against_an_instruction() -> None:
    """Non-vacuity: the site still shows the rollout the rule exists for."""
    assert _instructed_mock_rollouts(), (
        "no page runs policy_provider='mock' with an instruction; the guard grades nothing"
    )


@pytest.mark.parametrize("page", _instructed_mock_rollouts(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_the_page_names_the_attribute_behind_the_caveat(page: Path) -> None:
    """The attribute, by the name a reader can grep for in the package."""
    assert "reads_instruction = False" in page.read_text(encoding="utf-8"), (
        f"{page.relative_to(REPO_ROOT)} runs the mock against an instruction but never says "
        "reads_instruction = False, so a reader has no name for why the words did nothing"
    )


@pytest.mark.parametrize("page", _instructed_mock_rollouts(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_the_page_shows_the_notice_the_envelope_carries(page: Path) -> None:
    """The expected output on the page is the sentence the code emits, not a paraphrase."""
    policy = provider_policy_class("mock")
    notice = instruction_not_read_notice(policy)
    assert notice, "premise: the mock's envelope carries a notice"
    head = notice.split(".")[0]
    assert head in page.read_text(encoding="utf-8"), (
        f"{page.relative_to(REPO_ROOT)} does not show the notice the rollout prints: {head!r}"
    )


@pytest.mark.parametrize("page", sorted(START.glob("*.md")), ids=lambda p: p.name)
def test_no_start_page_sells_a_manipulation_its_fences_do_not_perform(page: Path) -> None:
    """The promise the mock rollout could not keep stays out of the Start pages.

    A Start page may promise a pick only if one of its fences runs a policy
    that reads the instruction, which none does today.
    """
    promise = (_description(page) + " " + _first_paragraph(page)).lower()
    sold = [p for p in _MANIPULATION_PROMISES if p in promise]
    if not sold:
        return
    providers = {m for fence in _fences(page) for m in re.findall(r'policy_provider="([a-z0-9_]+)"', fence)}
    delivering = [p for p in providers if (cls := provider_policy_class(p)) is not None and cls.reads_instruction]
    assert delivering, (
        f"{page.relative_to(REPO_ROOT)} promises {sold} but runs {sorted(providers) or 'no policy'}; "
        "none of those acts on the words"
    )


def test_the_old_promise_is_gone_not_softened() -> None:
    for page in _pages():
        assert "install to a robot picking up a cube" not in page.read_text(encoding="utf-8").lower(), page
