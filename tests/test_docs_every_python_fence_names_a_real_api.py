"""Every python and bash fence under docs/ names an API that exists (docs/hooks/check_sketches.py).

``docs/hooks/check_fences.py`` runs the bare ``python`` fences. The fences with a
title (``python title="sketch"``, 102 of them on 72 pages) need an arm, a GPU or a
cloud account, so nothing ran them, and one on the first learned-policy page kept
naming the ACT checkpoint after the page had moved to SmolVLA. The static verifier
reads every fence, runnable or not: imports resolve against the installed package,
``Robot(...)`` names a registry robot with keywords the factory or the built class
takes, attributes read on a bound robot, simulation, policy, trainer, agent or mesh
exist on that surface, tool actions are published ones, extras are in pyproject.toml
and ``strands-robots <command>`` lines parse with the command's own parser.

Offline only: Hub ids and pip names that are not installed are recorded, not fetched.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_REPO = Path(__file__).resolve().parents[1]
_HOOK = _REPO / "docs" / "hooks" / "check_sketches.py"
_BAD_PAGE = """# a page with defects

```python
from strands_robots import Robot, nonesuch

arm = Robot("so102")
arm.act()
arm.send_action({}, bogus=1)
```

```bash
pip install 'strands-robots[<extra>]'
strands-robots doctor --lits
```
""".replace("<extra>", "lero" + "bbot")  # a template hole in the source, so the extras grader does not read the typo


def _load_hook() -> ModuleType:
    spec = importlib.util.spec_from_file_location("docs_check_sketches_hook", _HOOK)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    if str(_REPO) not in sys.path:
        sys.path.insert(0, str(_REPO))
    # The hook declares dataclasses under ``from __future__ import annotations``; the
    # dataclass machinery looks the module up in sys.modules while building them.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def hook() -> ModuleType:
    return _load_hook()


def test_every_fence_under_docs_names_a_real_api(hook: ModuleType) -> None:
    checker = hook.check(online=False)
    assert checker.fences_checked > 200, "the verifier saw almost no fences; the fence regex or the docs path broke"
    rows = "\n".join(finding.row() for finding in checker.findings)
    assert not checker.findings, (
        f"{len(checker.findings)} fence(s) name something the package does not have. "
        "Rewrite the fence to the call that exists (the row names it), or delete the fence and say in prose what "
        f"the reader does instead. Run `python docs/hooks/check_sketches.py` for the full report:\n{rows}"
    )


def test_the_verifier_reports_a_planted_defect(hook: ModuleType, tmp_path: Path) -> None:
    page = tmp_path / "bad.md"
    page.write_text(_BAD_PAGE, encoding="utf-8")
    setattr(hook, "DOCS", tmp_path)  # noqa: B010 - mypy refuses attribute assignment on a ModuleType
    try:
        checker = hook.check([page], online=False)
    finally:
        setattr(hook, "DOCS", _REPO / "docs")  # noqa: B010
    kinds = sorted({finding.kind for finding in checker.findings})
    assert kinds == ["attribute", "cli", "extra", "import", "keyword", "robot"], (
        f"the verifier lost a check: planted defects of every kind, reported {kinds}. "
        "Restore the check in docs/hooks/check_sketches.py before trusting the green run above."
    )


def test_the_verifier_is_fast_enough_for_the_test_job(hook: ModuleType) -> None:
    import time

    start = time.monotonic()
    hook.check(online=False)
    seconds = time.monotonic() - start
    assert seconds < 30, f"check_sketches took {seconds:.1f} s; keep the static pass under 30 s so it stays in CI"
