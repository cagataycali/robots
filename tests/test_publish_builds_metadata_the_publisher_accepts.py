"""The publish job builds a distribution its own publisher can read.

``hatchling`` 1.32 (2026-08-11) started writing ``Metadata-Version: 2.5``.  The
``twine`` bundled in the pinned ``pypa/gh-action-pypi-publish`` does not know
that version, so the ``v0.5.2`` publish (run 35206733145) passed 50 minutes of
tests, built, and then died at the last step with ``InvalidDistribution: '2.5'
is not a valid metadata version`` - after the release was already announced.
``v0.5.1`` had built a month earlier with hatchling 1.31 and Metadata 2.4.

Two things keep that from recurring, and this file pins both:

* The build job installs ``hatchling<1.32`` itself and builds with
  ``--no-isolation``.  A dispatch re-publishes a tag whose ``pyproject.toml``
  is frozen, so the tag's ``[build-system]`` cannot be the thing that decides
  the backend - the workflow on ``main`` has to.
* The build job checks the wheel's ``Metadata-Version`` and runs ``twine
  check`` before the artifact is handed to the deploy job, so the next
  backend move fails in the job that can be read, not the one with the
  trusted-publisher token.

``pyproject.toml`` carries the same pin so a local ``uv pip install -e .``
builds what CI builds.

Parsing is line-based rather than via ``yaml``, for the reason
``tests/test_workflow_jobs_are_bounded.py`` gives.
"""

from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_PUBLISH = _ROOT / ".github" / "workflows" / "pypi-publish-on-release.yml"
_PYPROJECT = _ROOT / "pyproject.toml"

#: The last hatchling that writes Metadata 2.4; 1.32.0 moved to 2.5.
_PIN = "hatchling<1.32"


def _publish() -> str:
    return _PUBLISH.read_text(encoding="utf-8")


def test_build_job_pins_the_backend_it_builds_with() -> None:
    text = _publish()
    assert re.search(r"pip install .*'" + re.escape(_PIN) + r"'", text), (
        f"the publish workflow's Install step must install {_PIN!r} explicitly; "
        "a dispatch builds a frozen tag, so only the workflow can pin the backend"
    )


def test_build_job_does_not_let_the_tag_choose_the_backend() -> None:
    text = _publish()
    assert "python -m build --no-isolation" in text, (
        "the Build step must run `python -m build --no-isolation` so the hatchling "
        "installed by the workflow builds the wheel, not whatever the tag's "
        "[build-system] resolves to today"
    )
    assert not re.search(r"^\s+hatch build\s*$", text, re.M), (
        "`hatch build` resolves [build-system] from the checked-out tag in an "
        "isolated environment, which is how 0.5.2 picked up hatchling 1.32"
    )


def test_build_job_rejects_metadata_the_publisher_cannot_read() -> None:
    text = _publish()
    build_step = text.split("- name: Build", 1)[1].split("- name:", 1)[0]
    assert "Metadata-Version: 2.4" in build_step and "Metadata-Version: 2.5" not in build_step, (
        "the Build step must assert the wheel's Metadata-Version is one the "
        "publisher's twine reads (2.4 at most), and must not allow 2.5"
    )
    assert "twine check --strict dist/*" in build_step, (
        "the Build step must `twine check --strict` the artifacts so a bad distribution fails before the deploy job"
    )


def test_pyproject_carries_the_same_pin() -> None:
    text = _PYPROJECT.read_text(encoding="utf-8")
    build_system = text.split("[build-system]", 1)[1].split("\n[", 1)[0]
    assert f'"{_PIN}"' in build_system, (
        f"[build-system] requires must pin {_PIN!r} so a local editable install "
        "builds the same metadata version CI publishes"
    )
