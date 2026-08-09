"""Record what posture actually reaches the remote store, per flag value."""
import json, pathlib, sys
import strands_robots.dataset_recorder as dr

TREE = str(pathlib.Path(dr.__file__).parents[1])
print("TREE:", TREE)

root = pathlib.Path("/tmp/art_ds"); (root / "meta").mkdir(parents=True, exist_ok=True)

calls = []
class CP:
    returncode = 0; stdout = ""; stderr = ""
import subprocess as sp
sp.run = lambda cmd, **kw: (calls.append(list(cmd)), CP())[1]
dr._hf_executable = lambda: "hf"
dr._huggingface_hub_version_error = lambda: None

TRUTHY = ["false", "no", "off", "0", 1, float("nan")]
FALSY = [0, "", None, []]
PROBE = TRUTHY + FALSY

def label(v):
    return {True: "True", False: "False"}.get(v, repr(v)) if isinstance(v, bool) else repr(v)

rows = {}
for flag in ("create", "private", "delete"):
    for v in PROBE:
        calls.clear()
        try:
            r = dr.sync_dataset_to_bucket(root, "acme/robotdata", run_id="run1", **{flag: v})
            st, msg = r.get("status"), r.get("message", "")
        except BaseException as e:
            st, msg = f"raised {type(e).__name__}", str(e)
        argv = [" ".join(str(x) for x in c) for c in calls]
        rows[f"sync:{flag}:{label(v)}"] = dict(
            surface="sync_dataset_to_bucket", flag=flag, value=label(v), status=st,
            msg=msg[:120], argv=argv,
            mirror_deleted=any("--delete" in c for c in calls),
            created=any(len(c) > 1 and c[1] == "buckets" for c in calls),
            private_flag=any("--private" in c for c in calls),
        )

# push_to_hub(private=)
class FakeDS:
    repo_id = "acme/robotdata"; root = "/tmp/art_ds"
    def __init__(self): self.pushed = []
    def push_to_hub(self, tags=None, private=None): self.pushed.append(private)

for v in PROBE:
    ds = FakeDS()
    rec = dr.DatasetRecorder.__new__(dr.DatasetRecorder)
    rec.dataset = ds; rec.frame_count = 10; rec.episode_count = 1
    try:
        r = rec.push_to_hub(private=v); st, msg = r.get("status"), r.get("message", "")
    except BaseException as e:
        st, msg = f"raised {type(e).__name__}", str(e)
    rows[f"push:private:{label(v)}"] = dict(
        surface="push_to_hub", flag="private", value=label(v), status=st, msg=msg[:120],
        argv=[], published=len(ds.pushed) > 0, published_as=repr(ds.pushed[0]) if ds.pushed else None)

# honoured controls: the accepted domain must be byte-identical across trees
controls = {}
for name, kw in [("defaults", {}), ("delete=True", dict(delete=True)),
                 ("private=False", dict(private=False)), ("create=False", dict(create=False))]:
    calls.clear()
    r = dr.sync_dataset_to_bucket(root, "acme/robotdata", run_id="run1", **kw)
    controls[name] = dict(status=r.get("status"), argv=[" ".join(str(x) for x in c) for c in calls])

out = dict(tree=TREE, rows=rows, controls=controls)
pathlib.Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
print("wrote", sys.argv[1], "rows:", len(rows), "controls:", len(controls))
