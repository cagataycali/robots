"""``extra["run_name"]`` is agent-supplied and names a directory under the operator's ``output_dir``.

Review on #4229: it flowed unvalidated into ``log_root / f"{stamp}_{run_name}"`` and then
``mkdir`` + ``write_text``, so ``run_name="x/../../../../tmp/anything"`` escaped the
root the operator approved. ``validate_train_inputs`` allowlists ``extra`` *keys* only. The
trainer now refuses any run name that is not a plain token, in ``validate()`` (the gate the
``Trainer`` contract promises runs before any config is built) and again at the write site.

No mjlab: the helper and the gate are pure.
"""

from __future__ import annotations

import pytest

from strands_robots.training import TrainSpec
from strands_robots.training.rsl_rl import RslRlTrainer, run_name_problem

TRAVERSALS = [
    "x/../../../../tmp/anything",
    "../up",
    "a/b",
    "a\\b",
    "-leading-dash",
    ".hidden",
    "..",
    "with space",
    "semi;colon",
    "",
]


@pytest.mark.parametrize("bad", TRAVERSALS)
def test_a_run_name_that_is_not_a_plain_token_is_refused_by_name(bad: str) -> None:
    problem = run_name_problem(bad)
    assert problem is not None
    assert "run_name" in problem and repr(bad) in problem


@pytest.mark.parametrize("ok", ["strands", "g1_velocity-v2", "Run01", "a"])
def test_a_plain_token_passes(ok: str) -> None:
    assert run_name_problem(ok) is None


def test_validate_reports_it_before_anything_is_built(tmp_path) -> None:
    spec = TrainSpec(
        dataset_root="",
        output_dir=str(tmp_path),
        embodiment="unitree_g1",
        extra={"run_name": "x/../../../../tmp/anything"},
    )
    problems = RslRlTrainer().validate(spec)
    assert any("run_name" in p for p in problems), problems
    assert not any(tmp_path.iterdir()), "validate() touched the output directory"
