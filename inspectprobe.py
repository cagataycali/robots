"""Separate mere path RESOLUTION from actual on-disk INSPECTION of the shared cache."""
from __future__ import annotations
import json, pathlib

RESOLVES: list[dict] = []
INSPECTS: list[dict] = []
_CUR = {"nodeid": "?"}
REAL_HOME = (pathlib.Path.home() / ".cache/huggingface/lerobot").resolve()


def _under(p) -> bool:
    try:
        r = pathlib.Path(p).resolve()
    except Exception:
        return False
    return r == REAL_HOME or REAL_HOME in r.parents


def pytest_runtest_setup(item):
    _CUR["nodeid"] = item.nodeid


def pytest_configure(config):
    from strands_robots import dataset_recorder as dr

    orig_resolve = dr.resolve_dataset_dir
    orig_prepare = dr._prepare_create_target

    def resolve(repo_id, root=None):
        out = orig_resolve(repo_id, root)
        if _under(out):
            RESOLVES.append({"nodeid": _CUR["nodeid"], "repo_id": str(repo_id), "path": str(out)})
        return out

    def prepare(dataset_dir, *, overwrite):
        if _under(dataset_dir):
            INSPECTS.append({"nodeid": _CUR["nodeid"], "path": str(dataset_dir), "overwrite": bool(overwrite)})
        return orig_prepare(dataset_dir, overwrite=overwrite)

    dr.resolve_dataset_dir = resolve
    dr._prepare_create_target = prepare


def pytest_sessionfinish(session, exitstatus):
    pathlib.Path("/tmp/split_hits.json").write_text(
        json.dumps({"resolves": RESOLVES, "inspects": INSPECTS}, indent=1), encoding="utf-8"
    )
    print(f"\n[inspectprobe] resolves={len(RESOLVES)} inspects={len(INSPECTS)}")
