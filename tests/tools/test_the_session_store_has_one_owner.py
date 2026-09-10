"""The detached-session store has one reader, so it has one retention policy.

``lerobot_train`` and ``lerobot_teleoperate`` keep their detached sessions in the
same JSON document under
:data:`~strands_robots.tools._process_stop.SESSION_DIR` - deliberately, so every
robot session lives together and either tool's ``list`` shows all of them. Each
tool used to define its own ``SessionManager`` over it, and the two disagreed
about what a read may delete: the training store's docstring said "no record is
dropped here, and that is what keeps a detached session stoppable", while the
teleoperation store pruned every record whose pid did not answer as running *and
wrote the pruned map back to disk*.

A document with two readers cannot hold two policies. Whichever reader deletes a
record deletes it for the other, so the destructive one is the one that takes
effect and the retaining one's guarantee was never the store's - only its own.
Measured on the two-class tree, one ``lerobot_teleoperate(action="list")`` erased
two of three records the training tool had just written and documented itself as
keeping, and the same process logged "its record is kept" and "dropping the
record" about the same record.

The suite could not see it, either: every test redirected one tool module's own
``SESSION_DIR``, so two independently-redirectable names for one file kept each
half isolated from the other. These cells drive both tools against ONE store, the
way production does, and pin that the class has one definition so a third copy
cannot reintroduce the disagreement.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from typing import Any

import pytest

import strands_robots.tools.lerobot_teleoperate as tele_mod
import strands_robots.tools.lerobot_train as train_mod
from strands_robots.tools import _process_stop

#: A pid this process holds, so "the process exists" is settled and the only
#: thing under test is what a read does to the record.
LIVE_PID = os.getpid()


@pytest.fixture(autouse=True)
def _one_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the one store both tools read. There is only one name to set."""
    session_dir = tmp_path / ".sessions"
    session_dir.mkdir()
    monkeypatch.setattr(_process_stop, "SESSION_DIR", session_dir)
    return session_dir


def _stored(session_dir: Path) -> dict[str, Any]:
    """The records on disk, independent of what any load returned."""
    store = session_dir / "active_sessions.json"
    return json.loads(store.read_text()) if store.exists() else {}


#: Records a retaining store keeps and a pruning one drops, with what each is.
#: A finished run is kept for its log tail; a record whose ``pid`` field is not a
#: process id is kept because nothing can inspect it and converting it would name
#: a different process, so the record is the only handle on the run left.
RETAINED = [
    pytest.param({"pid": 999_999, "action": "train", "start_time": 0.0}, id="finished-run"),
    pytest.param({"pid": "not-a-pid", "action": "train", "start_time": 0.0}, id="pid-is-not-a-pid"),
]


@pytest.mark.parametrize("record", RETAINED)
def test_a_teleop_read_does_not_erase_a_training_record(_one_store: Path, record: dict[str, Any]) -> None:
    """The cross-tool erasure, driven against the shared store production uses.

    ``list`` is a query. Reading the store through either tool must leave the
    other tool's records on disk, because this file is the only place a detached
    run's pid is written down and the reads that follow it are load-modify-write.
    """
    train_mod.SessionManager().add_session("training", record)
    assert "training" in _stored(_one_store), "premise: the record reached disk"

    listed = tele_mod.lerobot_teleoperate(action="list")

    assert listed["status"] == "success"
    assert "training" in _stored(_one_store), (
        "a teleoperation read erased a training record; that store is the only place "
        "the detached run's pid was written down, so the run can no longer be stopped"
    )
    assert train_mod.SessionManager().get_session("training") is not None


def test_a_training_read_does_not_erase_a_teleop_record(_one_store: Path) -> None:
    """The same rule in the other direction, so neither tool is privileged."""
    tele_mod.SessionManager().add_session("arm", {"pid": 999_999, "action": "teleoperate", "start_time": 0.0})

    listed = train_mod.lerobot_train(dataset_root="/unused", action="list")

    assert listed["status"] == "success"
    assert "arm" in _stored(_one_store), "a training read erased a teleoperation record"


def test_both_tools_report_the_same_record_the_same_way(_one_store: Path) -> None:
    """One store, one verdict: a record cannot be live in one tool and gone in the other.

    Before one owner, a record whose ``pid`` field was a ``str`` was retained and
    reported by the training tool and dropped by the teleoperation tool, so the
    same file described two different fleets depending on which verb asked.
    """
    train_mod.SessionManager().add_session("shared", {"pid": "not-a-pid", "action": "train", "start_time": 0.0})

    from_train = train_mod.lerobot_train(dataset_root="/unused", action="list")
    from_teleop = tele_mod.lerobot_teleoperate(action="list")

    def names(result: dict[str, Any]) -> list[str]:
        return sorted(next(block["json"] for block in result["content"] if "json" in block)["sessions"])

    assert names(from_train) == names(from_teleop) == ["shared"]


def test_the_store_class_has_one_definition() -> None:
    """Both tools use the one class, and no module defines a second.

    Identity is what makes the policy shared rather than merely equal today: two
    copies that agree are two copies that can drift, and this store is the one
    where drift is measured in processes nothing can stop.
    """
    assert train_mod.SessionManager is tele_mod.SessionManager is _process_stop.SessionManager

    package = Path(_process_stop.__file__).resolve().parent.parent
    definitions = [
        str(path.relative_to(package))
        for path in sorted(package.rglob("*.py"))
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.ClassDef) and node.name == "SessionManager"
    ]
    assert len(list(package.rglob("*.py"))) > 10, "premise: the package must have been read"
    assert definitions == ["tools/_process_stop.py"], (
        f"the session store must have exactly one class over it, found {definitions}"
    )


def test_the_store_path_has_one_definition() -> None:
    """And one name for the file, or a redirect only reaches one of the readers.

    A per-tool ``SESSION_DIR`` is what let the suite grade each half in its own
    temp dir while production had both halves on one file - the reason a
    two-policy store went unnoticed. It is also an import-time ``mkdir`` under
    ``cwd`` per copy.
    """
    package = Path(_process_stop.__file__).resolve().parent.parent
    assignments = [
        str(path.relative_to(package))
        for path in sorted(package.rglob("*.py"))
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id == "SESSION_DIR"
    ]
    assert assignments == ["tools/_process_stop.py"], (
        f"the session store directory must be defined once, found {assignments}"
    )
