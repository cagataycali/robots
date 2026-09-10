"""The documented GR00T example must pass the GR00T trainer's own validation.

``docs/training/vla_workflow.md`` shipped ``extra={"embodiment": ...}`` while
``Gr00tTrainer.validate`` reads ``spec.embodiment`` - so the copy-pasted example
returned ``status="error"`` ("embodiment is required for GR00T") on a machine
with everything installed. The keyword grader
(``test_docs_python_examples_are_callable``) cannot see this: ``extra=`` is a
real ``TrainSpec`` keyword. This test executes the documented block with the
trainer stubbed at the call boundary and validates the spec it built.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import strands_robots
from strands_robots.training.groot import Gr00tTrainer

_DOC = Path(strands_robots.__file__).resolve().parent.parent / "docs" / "training" / "vla_workflow.md"
_PYTHON_FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)

# Problems that depend on the reader's machine, not on the example's text.
_ENVIRONMENT_PROBLEMS = ("checkout not found", "dataset_root", "meta/info.json")


def _documented_groot_specs(dataset_root: Path) -> list:
    specs: list = []

    class _Captured:
        def train(self, spec):
            specs.append(spec)
            return type("R", (), {"checkpoint_dir": str(dataset_root)})()

        def export(self, spec, checkpoint_dir):
            return checkpoint_dir

    blocks = [b for b in _PYTHON_FENCE.findall(_DOC.read_text()) if 'create_trainer("groot")' in b]
    assert blocks, f"{_DOC.name} no longer documents create_trainer('groot')"
    for block in blocks:
        namespace: dict = {}
        exec("from strands_robots.training import TrainSpec\n", namespace)  # noqa: S102 - the doc's own import
        namespace["create_trainer"] = lambda *_a, **_k: _Captured()
        exec(re.sub(r"^from strands_robots\.training import .*$", "", block, flags=re.M), namespace)  # noqa: S102
    return specs


def test_documented_groot_spec_passes_the_trainers_spec_validation(tmp_path):
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "info.json").write_text(json.dumps({"codebase_version": "v3.0"}))
    specs = _documented_groot_specs(tmp_path)
    assert specs, "the documented block never called trainer.train(spec)"
    for spec in specs:
        problems = Gr00tTrainer().validate(spec)
        spec_problems = [p for p in problems if not any(tag in p for tag in _ENVIRONMENT_PROBLEMS)]
        assert spec_problems == [], f"documented GR00T example is rejected by Gr00tTrainer.validate: {spec_problems}"
        assert spec.embodiment, "embodiment must be a TrainSpec field; extra['embodiment'] is never read"
