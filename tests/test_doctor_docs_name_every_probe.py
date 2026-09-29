"""Every probe :data:`strands_robots.doctor.CHECKS` runs is named in the docs.

Two docs pages present the doctor's checks: `docs/start/doctor.md` walks a
sample run and names every probe in its "The probes" table; `docs/reference/cli.md`
names them in one prose sentence in run order. A probe that runs at the CLI
but is absent from both is invisible to a user who reached the docs first -
they cannot pattern-match a `SKIP` line they never saw described, and the
next-step advice for its `FAIL` verdict lives only in the source.

On the pinned commit that added `IoT Direct` (between `Mesh` and `Sim Test`)
both pages listed 15 rows while the runtime ran 16. This test pins that they
stay in step from now on: every label the runtime emits (its first-column
label, which is what `--list` prints and what the report shows) appears in
both docs pages.

The rule is deliberately one-directional. Docs may still mention a probe by
one of its aliases (the row title in prose can differ from the label), so the
scan looks for the exact label the runtime chose - a match anywhere on the
page is enough. What it refuses is silence: a label the runtime prints and
neither docs page names.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from strands_robots.doctor import CHECKS

_ROOT = Path(__file__).resolve().parents[1]
_DOCTOR_MD = _ROOT / "docs" / "start" / "doctor.md"
_CLI_MD = _ROOT / "docs" / "reference" / "cli.md"


def _cli_doctor_section() -> str:
    """The ``## doctor`` section only.

    ``docs/reference/cli.md`` has a separate ``## iot`` subcommand section
    that legitimately mentions AWS IoT: scanning the whole file for "IoT"
    would let a probe row missing from the doctor prose slip through because
    the iot-provisioning section mentions IoT identities.
    """
    text = _CLI_MD.read_text(encoding="utf-8")
    return text.split("## verify-dataset", 1)[0]


@pytest.mark.parametrize("label", [label for label, _ in CHECKS])
def test_probe_label_named_in_start_doctor_md(label: str) -> None:
    text = _DOCTOR_MD.read_text(encoding="utf-8")
    assert label in text, (
        f"docs/start/doctor.md does not name the {label!r} probe. "
        f"The runtime prints one row with this label on every doctor run; a "
        f"reader landing on this page cannot recognise it in the report or "
        f"the 'not PASS when' table."
    )


@pytest.mark.parametrize("label", [label for label, _ in CHECKS])
def test_probe_label_named_in_reference_cli_md(label: str) -> None:
    section = _cli_doctor_section()
    assert label in section, (
        f"docs/reference/cli.md '## doctor' section does not name the "
        f"{label!r} probe in its run-order prose. That sentence is the "
        f"one-line contract for the CLI's probe list; drift here silently "
        f"desyncs the doc from `--list`."
    )
