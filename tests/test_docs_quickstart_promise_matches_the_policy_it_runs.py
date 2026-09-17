"""The quickstart promises what its own rollout does, not what its words say.

The page's front matter read "Five minutes from install to a robot picking up a
cube", and the only rollout on the page is ``policy_provider="mock"``.
:class:`~strands_robots.policies.mock.MockPolicy` declares
:attr:`~strands_robots.policies.base.Policy.reads_instruction` ``False`` - it
drives every joint through a test motion whatever the task says - so the five
minutes the page sells end with the cube exactly where it was put. The library
already refuses to let that read as a completed task: every envelope carries
:func:`~strands_robots.policies.base.instruction_not_read_notice`, whose own
docstring records that without it "MockPolicy | wave the arm ... completed" was
relayed as a wave that happened. The page was making the same claim one level
up, where no envelope reaches.

These cells bind the page's prose to the attribute that makes it necessary: the
provider is read out of the page's own fence, and the caveat is required only
while that provider's class declares it does not read the instruction. So if a
future ``mock`` acts on the words, the requirement lapses with it rather than
outliving it. The pointer cells hold the page to an alternative that delivers
the promise - the reference pick, on the same install the page's first command
gives - and check that the file it names still lifts the cube.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from strands_robots.policies.base import Policy, provider_policy_class

REPO_ROOT = Path(__file__).resolve().parents[1]
QUICKSTART = REPO_ROOT / "docs" / "getting-started" / "quickstart.md"

#: The example the page sends a reader to for a cube that really leaves the table.
REFERENCE_PICK = "examples/18_so101_pick_and_lift.py"


def _quickstart() -> str:
    return QUICKSTART.read_text(encoding="utf-8")


def _description() -> str:
    """The front-matter ``description`` - the page's promise, and its meta tag."""
    match = re.search(r"^---\n(.*?)\n---\n", _quickstart(), re.DOTALL)
    assert match, "the quickstart lost its front matter"
    described = re.search(r"^description: (.+)$", match.group(1), re.MULTILINE)
    assert described, "the quickstart lost its front-matter description"
    return described.group(1)


def _rollout_provider() -> str:
    """The provider the page's own ``run_policy`` fence runs.

    Read from the page rather than named here: the cells below are about the
    policy a reader actually gets, so a page that switched providers should
    change what is required of it, not quietly keep passing.
    """
    fences = re.findall(r"```python\n(.*?)```", _quickstart(), re.DOTALL)
    rollouts = [f for f in fences if "sim.run_policy(" in f]
    assert rollouts, "the quickstart lost its run_policy fence"
    providers = re.findall(r'policy_provider="([a-z0-9_]+)"', rollouts[0])
    assert providers, "the quickstart's rollout no longer names a policy provider"
    return providers[0]


def test_the_rollout_the_page_gives_does_not_read_the_instruction() -> None:
    """The premise of every cell below, established from the class itself.

    Without this the caveat cells would pass on a page that carries the words
    while the policy behind them had started acting on the task.
    """
    policy = provider_policy_class(_rollout_provider())

    assert policy is not None, _rollout_provider()
    assert issubclass(policy, Policy), policy
    assert policy.reads_instruction is False, policy


@pytest.mark.parametrize(
    "claim",
    [
        # The attribute, by the name a reader can grep for in the package.
        "reads_instruction = False",
        # What that means for the scene the page just built.
        "the cube has not moved",
        # Where to go for the promise instead.
        REFERENCE_PICK,
    ],
)
def test_the_page_says_what_its_rollout_leaves_undone(claim: str) -> None:
    """One claim per case, so a page that drops just one of them reports which."""
    assert claim in _quickstart()


def test_the_description_no_longer_sells_the_mock_rollout_as_a_pick() -> None:
    """The promise that the rollout could not keep is gone, not merely softened."""
    assert "install to a robot picking up a cube" not in _description().lower()


def test_the_example_the_page_points_at_still_lifts_the_cube() -> None:
    """The alternative is a file that exists and promises the lift it is cited for.

    A pointer to a path that has moved, or to an example whose own contract no
    longer includes leaving the table, is the same defect as the description
    this change corrected.
    """
    example = REPO_ROOT / REFERENCE_PICK

    assert example.is_file(), example
    docstring = example.read_text(encoding="utf-8").split('"""')[1].lower()
    assert "pick ok - cube lifted" in docstring
