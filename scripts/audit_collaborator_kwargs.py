"""Does the dashboard ever call an injected collaborator with a kwarg the REAL class lacks? (Q56/Q57)

Q56: `training.export` called `push_to_hub(repo_id=...)` on a DatasetRecorder that has never had that
parameter. Every dashboard test injects a FAKE recorder, so no test could see it — the feature had
never published anything. This audit is the direction that would have caught it on the day it was
written: read the dashboard's call sites, check the kwargs against the class that really answers.

Run:  .venv/bin/python scripts/audit_collaborator_kwargs.py
Exit code is the number of mismatches, so it can gate anything. tests/test_dashboard_collaborator_kwargs.py
runs it in-process on every suite.

WHY THIS IS SCOPED TO THE DASHBOARD, measured 2026-08-20 and worth not repeating:
a whole-package AST variant (call sites vs every class in the tree defining that method name) produced
71 hits and ZERO real defects. AST cannot type the receiver, so `arr.mean(axis=0)`, `img.save(format=…)`,
`Path.open(encoding=…)`, `executor.shutdown(wait=True)`, lerobot's `robot.connect(calibrate=…)` and even a
module-level `config_api.snapshot(bridge=…)` all "matched" an unrelated same-named method on one of our
classes. The import-based check below is narrow BECAUSE it is true: the collaborator is named explicitly,
so a hit is a hit. Grow COLLABORATORS rather than widening the match.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import sys

DASH = pathlib.Path(__file__).resolve().parent.parent / "strands_robots" / "dashboard"

#: The classes the dashboard drives through an injected/held collaborator — i.e. the objects a test
#: is most likely to replace with a fake, which is exactly where a wrong kwarg hides — keyed by the
#: RECEIVER NAME the dashboard uses for them.
#:
#: Matching on the receiver, not just the method name, is what makes a hit trustworthy. A first cut
#: matched by method name alone and immediately produced four false alarms INSIDE the dashboard:
#: PIL's `img.save(format=…, quality=…)` collided with ProfileStore.save, `path.open(encoding=…)` with
#: RecordController.open, a module-level `config_api.snapshot(bridge=…)` with MeshBridge.snapshot, and
#: `worker.close(upload=…, repo_id=…)` — which is CORRECT for RecordWorker.close — with
#: RecordController.close. An audit that cries wolf gets muted, so it names its receivers.
RECEIVERS = {
    "recorder": ("strands_robots.dataset_recorder", "DatasetRecorder"),
    "_recorder": ("strands_robots.dataset_recorder", "DatasetRecorder"),
    "worker": ("strands_robots.dashboard.record_worker", "RecordWorker"),
    "_worker": ("strands_robots.dashboard.record_worker", "RecordWorker"),
    "devices": ("strands_robots.dashboard.device_manager", "DeviceManager"),
    "_devices": ("strands_robots.dashboard.device_manager", "DeviceManager"),
    "profiles": ("strands_robots.dashboard.device_manager", "ProfileStore"),
    "_profiles": ("strands_robots.dashboard.device_manager", "ProfileStore"),
    "bridge": ("strands_robots.dashboard.mesh_bridge", "MeshBridge"),
    "_bridge": ("strands_robots.dashboard.mesh_bridge", "MeshBridge"),
    "controller": ("strands_robots.dashboard.record_api", "RecordController"),
    "_controller": ("strands_robots.dashboard.record_api", "RecordController"),
    "mesh": ("strands_robots.mesh.core", "Mesh"),
    "_mesh": ("strands_robots.mesh.core", "Mesh"),
}

COLLABORATORS = tuple(dict.fromkeys(RECEIVERS.values()))


def real_methods() -> dict[str, list[tuple[str, inspect.Signature]]]:
    """method name -> [(class name, signature)] across every collaborator that imports."""
    out: dict[str, list[tuple[str, inspect.Signature]]] = {}
    for mod, cname in COLLABORATORS:
        try:
            cls = getattr(__import__(mod, fromlist=[cname]), cname)
        except Exception as exc:  # a torch-less install must not fail the audit, only narrow it
            print(f"note: {cname} unavailable ({type(exc).__name__}) - not checked", file=sys.stderr)
            continue
        for mname, fn in inspect.getmembers(cls, predicate=inspect.isfunction):
            if mname.startswith("__"):
                continue
            try:
                out.setdefault(mname, []).append((cname, inspect.signature(fn)))
            except (TypeError, ValueError):
                pass
    return out


def _receiver_name(node: ast.Attribute) -> str | None:
    """The name the call is made ON: `self._recorder.x` -> "_recorder", `worker.x` -> "worker"."""
    val = node.value
    if isinstance(val, ast.Name):
        return val.id
    if isinstance(val, ast.Attribute):
        return val.attr
    return None


def audit() -> list[str]:
    by_name = real_methods()
    problems: list[str] = []
    for path in sorted(DASH.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            kws = [k.arg for k in node.keywords if k.arg]
            if not kws:
                continue
            recv = _receiver_name(node.func)
            target = RECEIVERS.get(recv or "")
            if target is None:
                continue
            wanted = target[1]
            for cname, sig in by_name.get(node.func.attr, []):
                if cname != wanted:
                    continue
                params = sig.parameters
                if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
                    continue  # **kwargs accepts anything; nothing to prove
                bad = [k for k in kws if k not in params]
                if bad:
                    problems.append(
                        f"{path.name}:{node.lineno} calls .{node.func.attr}({', '.join(bad)}=…) "
                        f"but {cname}.{node.func.attr}{sig} has no such parameter"
                    )
    return problems


if __name__ == "__main__":
    found = audit()
    for line in found:
        print("MISMATCH", line)
    print(f"\n{len(found)} mismatch(es); {len(COLLABORATORS)} collaborator classes checked")
    sys.exit(len(found))
