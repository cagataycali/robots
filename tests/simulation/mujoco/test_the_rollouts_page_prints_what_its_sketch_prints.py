# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``docs/learn/simulation/predicates-and-rollouts.md`` shows the output its own sketch prints.

The page's opening fence is deterministic (so101, the mock sinusoid, default
MuJoCo physics), so its "You should see" block can be compared line for line.
It once said the ``joint_above`` stop fired after 20 applied actions while the
sketch printed 19, which reads to a reader as a mistake on their side.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("mujoco")

_PAGE = Path(__file__).resolve().parents[3] / "docs" / "learn" / "simulation" / "predicates-and-rollouts.md"


def test_the_sketch_prints_the_you_should_see_block(tmp_path: Path) -> None:
    text = _PAGE.read_text(encoding="utf-8")
    sketch = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    expected = re.search(r"You should see:\n\n```text\n(.*?)```", text, re.DOTALL)
    assert sketch and expected, f"{_PAGE.name} lost its opening fence or its 'You should see' block"

    run = subprocess.run(
        [sys.executable, "-c", sketch.group(1)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )

    assert run.returncode == 0, run.stderr[-2000:]
    assert run.stdout.splitlines() == expected.group(1).splitlines()
