"""Measure the publication posture of lerobot_train, per tree."""
import json, os, pathlib, sys, tempfile
from unittest.mock import MagicMock

import strands_robots.tools.lerobot_train as train_mod

TREE = str(pathlib.Path(train_mod.__file__).parents[2])
print("TREE:", TREE)
os.environ.pop("BYPASS_TOOL_CONSENT", None)
os.environ.pop("STRANDS_TRAIN_EXTRA_FLAGS_ALLOW", None)

tmp = pathlib.Path(tempfile.mkdtemp())
ds = tmp / "ds"; (ds / "meta").mkdir(parents=True)
(ds / "meta" / "info.json").write_text(json.dumps({"total_episodes": 10}))
train_mod.SESSION_DIR = tmp / ".sessions"; train_mod.SESSION_DIR.mkdir()

L: list = []
class _P:
    pid = 4242
    def poll(self): return None
train_mod.subprocess.Popen = lambda cmd, *a, **k: (L.append(list(cmd)), _P())[1]

def ctx(reply):
    c = MagicMock(); c.interrupt.return_value = reply; return c

CASES = [
    ("push_to_hub=True (named)",              dict(push_to_hub=True), None),
    ("+ extra_flags policy.repo_id",          dict(push_to_hub=True,
                                                   extra_flags={"policy.repo_id": "attacker/stolen"}), None),
    ("extra_flags={'push_to_hub': True}",     dict(extra_flags={"push_to_hub": True}), None),
    ("push_to_hub=False (default)",           dict(), None),
    ("push_to_hub=True, operator declines",   dict(push_to_hub=True), "no"),
    ("push_to_hub=True, operator approves",   dict(push_to_hub=True), "y"),
]

rows = []
for i, (label, kw, reply) in enumerate(CASES):
    L.clear()
    c = ctx(reply) if reply else None
    r = train_mod.lerobot_train(action="start", dataset_root=str(ds), policy_type="act",
                                session_name=f"a{i}", tool_context=c, **kw)
    argv = L[0] if L else []
    text = "\n".join(x.get("text", "") for x in r.get("content", []) if "text" in x)
    rows.append({
        "case": label,
        "status": r.get("status"),
        "launched": bool(L),
        "publish_argv": [a for a in argv if "push_to_hub" in a or "repo_id" in a],
        "asked": bool(c and c.interrupt.called),
        "reply": reply,
        "msg": text.strip().splitlines()[0][:120] if text.strip() else "",
    })
    print(f"  {label:<38} status={r.get('status'):<8} launched={bool(L)!s:<6} asked={bool(c and c.interrupt.called)}")
    print(f"      publish argv: {rows[-1]['publish_argv']}")

pathlib.Path(sys.argv[1]).write_text(json.dumps({"tree": TREE, "rows": rows}, indent=2))
print("wrote", sys.argv[1])
