"""JSON-line result reporter.

Each rogue writes one ``RogueResult`` to ``$ROGUE_RESULT_FILE`` (set by
the orchestrator). The orchestrator aggregates them at the end. We use
file-per-rogue (not a shared file) to avoid append races between
concurrent rogues.

Fields:
* ``rogue_id``    -- e.g. ``"rogue_01_no_cert_outsider"``
* ``av_id``       -- canonical attack-vector id (e.g. ``"AV-01"``)
* ``title``       -- one-line summary
* ``posture``     -- victim deployment posture under test
* ``defence_held``-- True if the attack was rejected as expected
* ``observed``    -- what actually happened (free text)
* ``error``       -- traceback if the rogue itself crashed (vs. blocked)
* ``duration_s``  -- wall-clock for the attack
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import time


@dataclass
class RogueResult:
    rogue_id: str
    av_id: str
    title: str
    posture: str
    defence_held: bool
    observed: str = ""
    error: str = ""
    duration_s: float = 0.0
    started_at: float = field(default_factory=time)

    def to_dict(self) -> dict:
        return asdict(self)


def write_result(result: RogueResult) -> None:
    """Persist a result to ``$ROGUE_RESULT_FILE`` (or stdout if unset)."""
    line = json.dumps(result.to_dict()) + "\n"
    target = os.getenv("ROGUE_RESULT_FILE")
    if target:
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        with open(target, "a", encoding="utf-8") as f:
            f.write(line)
    else:
        sys.stdout.write(line)
        sys.stdout.flush()
