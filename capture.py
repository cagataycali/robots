"""Measure what an agent following the push_to_hub description actually gets."""
import json, os, pathlib, sys
from unittest.mock import MagicMock

os.environ.pop("BYPASS_TOOL_CONSENT", None)  # or every consent gate is bypassed
import strands_robots.tools.lerobot_train as tr

TREE = str(pathlib.Path(tr.__file__).parents[2])
print("TREE:", TREE)

OUT = pathlib.Path(sys.argv[1])
_call = tr.lerobot_train.original if hasattr(tr.lerobot_train, "original") else tr.lerobot_train


class _Proc:
    pid = 4242
    returncode = None

    def poll(self):
        return None


def drive(allow: str | None, spelling: str, tmp: pathlib.Path):
    """Return (asked, published, status) for one (allowlist, spelling) cell."""
    for k in ("STRANDS_TRAIN_EXTRA_FLAGS_ALLOW",):
        os.environ.pop(k, None)
    if allow:
        os.environ["STRANDS_TRAIN_EXTRA_FLAGS_ALLOW"] = allow
    ds = tmp / "ds"
    (ds / "meta").mkdir(parents=True, exist_ok=True)
    (ds / "meta" / "info.json").write_text(json.dumps({"total_episodes": 4, "fps": 30}))
    argv: list[list[str]] = []
    ctx = MagicMock()
    ctx.interrupt.return_value = ""  # nobody answers a headless prompt
    real_popen = tr.subprocess.Popen
    tr.subprocess.Popen = lambda cmd, **kw: (argv.append(list(cmd)), _Proc())[1]
    real_dir = tr.SESSION_DIR
    tr.SESSION_DIR = tmp / "sessions"
    tr.SESSION_DIR.mkdir(parents=True, exist_ok=True)
    try:
        kw: dict = {
            "action": "start", "dataset_root": str(ds),
            "policy_type": "act", "steps": 8, "batch_size": 2, "tool_context": ctx,
        }
        if spelling == "named":
            kw["push_to_hub"] = True
        else:
            kw["extra_flags"] = {"push_to_hub": True}
        res = _call(**kw)
    finally:
        tr.subprocess.Popen = real_popen
        tr.SESSION_DIR = real_dir
        os.environ.pop("STRANDS_TRAIN_EXTRA_FLAGS_ALLOW", None)
    published = any("push_to_hub=true" in t or "push_to_hub=True" in t for c in argv for t in c)
    return {
        "asked": bool(ctx.interrupt.called),
        "published": published,
        "status": res.get("status"),
        "argv_flag": next((t for c in argv for t in c if "push_to_hub" in t), None),
    }


import tempfile

facts: dict = {"tree": TREE, "cells": {}, "description": {}}
with tempfile.TemporaryDirectory() as d:
    tmp = pathlib.Path(d)
    for allow, label in [(None, "no entry"), ("push_to_hub", "push_to_hub"),
                         ("policy.push_to_hub", "policy.push_to_hub")]:
        for spelling in ("named", "raw"):
            k = f"{label}|{spelling}"
            facts["cells"][k] = drive(allow, spelling, tmp)
            print(" ", k, "->", facts["cells"][k])

# What the description tells the reader, and what following it yields.
import inspect, re

doc = inspect.getdoc(_call) or ""
m = re.search(r"\n    push_to_hub:(.*?)(?=\n    [a-z_]+:)", doc, re.S)
entry = " ".join((m.group(1) if m else "").split())
# The key a reader would set: an explicitly named ALLOW=<key>, else the key the
# prose points at through its ``extra_flags={...}`` example.
named = re.findall(r"STRANDS_TRAIN_EXTRA_FLAGS_ALLOW=([A-Za-z_.]+)", entry)
example = re.findall(r"extra_flags=\{'([A-Za-z_.]+)'", entry)
facts["description"] = {
    "entry": entry,
    "names_allow_key": named[0] if named else None,
    "example_key": example[0] if example else None,
    "reader_would_set": named[0] if named else (example[0] if example else None),
    "claims_parity": "exactly as the" in entry,
}
print(" reader would set:", facts["description"]["reader_would_set"])

# Following the description: set whatever key it names (if any) and drive the parameter.
with tempfile.TemporaryDirectory() as d:
    key = facts["description"]["reader_would_set"]
    facts["following_the_description"] = drive(key, "named", pathlib.Path(d)) if key else {
        "asked": None, "published": None, "status": None, "argv_flag": None,
    }
print(" following the description ->", facts["following_the_description"])

OUT.write_text(json.dumps(facts, indent=2))
print("wrote", OUT)
