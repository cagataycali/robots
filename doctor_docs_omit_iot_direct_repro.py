"""Repro: doctor's docs listed 15 probes; runtime runs 16.

Reproduces the drift filed in cagataycali/robots-harness#<TBD>.

- Before the docs fix (upstream d93d8a2..85e895e or any commit where neither
  ``docs/start/doctor.md`` nor ``docs/reference/cli.md`` names ``IoT Direct``),
  this script exits 0 with the drift printed.
- After the fix (this branch), it exits 1 because both docs pages now name
  the probe, and the assertion at the bottom refuses that outcome so a
  regression is visible.

The tests/test_doctor_docs_name_every_probe.py test that ships with the fix
is the durable version: it walks every ``CHECKS`` label and refuses one that
no docs page names, so the next probe to land is graded on arrival.
"""

from __future__ import annotations

from pathlib import Path

from strands_robots.doctor import CHECKS

_ROOT = Path(__file__).resolve().parent
_DOCTOR_MD = _ROOT / "docs" / "start" / "doctor.md"
_CLI_MD = _ROOT / "docs" / "reference" / "cli.md"


def main() -> int:
    labels = [label for label, _ in CHECKS]
    print(f"CHECKS has {len(labels)} rows:")
    for label in labels:
        print(f"  - {label}")
    print()

    missing_from: list[str] = []
    for md in (_DOCTOR_MD, _CLI_MD):
        text = md.read_text(encoding="utf-8")
        if md == _CLI_MD:
            # docs/reference/cli.md has a separate ``## iot`` section. Restrict
            # to the ``## doctor`` prose so its subcommand section is not
            # counted as documentation of the probe.
            text = text.split("## verify-dataset", 1)[0]
        has_iot_row = "IoT Direct" in text
        print(f"{md.relative_to(_ROOT)}: names 'IoT Direct'? {has_iot_row}")
        if not has_iot_row:
            missing_from.append(str(md.relative_to(_ROOT)))

    print()
    if missing_from:
        print("Result: source runs 'IoT Direct' between 'Mesh' and 'Sim Test';")
        print(f"        {len(missing_from)} docs page(s) omit it: {missing_from}")
        print("        Drift confirmed.")
        return 0
    else:
        print("Result: both docs pages name 'IoT Direct'; drift is closed.")
        # Deliberately non-zero: this script exists to reproduce the drift,
        # so a run that finds no drift is not what it was written to see.
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
