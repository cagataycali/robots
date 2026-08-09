"""Measure the allow_insecure resolver on whichever tree this script sits in."""
import asyncio, json, logging, os, pathlib, sys
from unittest.mock import patch

import strands_robots.device_connect as dc
TREE = str(pathlib.Path(dc.__file__).parents[2])
print("TREE:", TREE)

SPELLINGS = ["true", "1", "yes", "false", "no", "0", "off", "on", "enabled"]

def verdict(fn):
    try:
        return {"posture": "insecure" if fn() else "secure", "refused": False}
    except ValueError as e:
        return {"posture": "refused", "refused": True, "message": str(e)}
    except Exception as e:  # noqa: BLE001 - an escape past the declared return is an answer
        return {"posture": f"raised {type(e).__name__}", "refused": False, "message": str(e)}

rows = {}
for s in SPELLINGS:
    rows[s] = {
        "argument": verdict(lambda s=s: dc.resolve_allow_insecure(s)),
        "environment": verdict(lambda s=s: dc.resolve_allow_insecure(None, s)),
    }

# --- what init_device_connect does with the off spelling, end to end ---
class _Robot:
    tool_name_str = "so100"

def entrypoint(value):
    captured, warned = {}, []
    class _Rt:
        def __init__(self, **kw): captured.update(kw)
        def set_heartbeat_provider(self, *a, **k): pass
        async def run(self): return None
    class _H(logging.Handler):
        def emit(self, r): warned.append(r.getMessage())
    lg = logging.getLogger("strands_robots.device_connect"); h = _H(); h.setLevel(logging.WARNING)
    lg.addHandler(h)
    os.environ.pop("DEVICE_CONNECT_ALLOW_INSECURE", None)
    try:
        with patch.object(dc, "DeviceRuntime", _Rt):
            asyncio.run(dc.init_device_connect(_Robot(), peer_id="p1", allow_insecure=value))
        return {"outcome": "started", "runtime_setting": repr(captured.get("allow_insecure")),
                "real_bool": isinstance(captured.get("allow_insecure"), bool),
                "insecure_warning": any("INSECURE mode" in w for w in warned)}
    except ValueError as e:
        return {"outcome": "refused", "runtime_setting": "-", "real_bool": None,
                "insecure_warning": any("INSECURE mode" in w for w in warned), "message": str(e)}
    finally:
        lg.removeHandler(h)

entry = {repr(v): entrypoint(v) for v in ["false", True, False, None]}

# --- numpy boolean normalization (the declared -> bool return) ---
import numpy as np
npb = {}
for label, v in [("np.True_", np.True_), ("np.array(False)", np.array(False))]:
    try:
        out = dc.resolve_allow_insecure(v)
        npb[label] = {"returned": type(out).__name__, "real_bool": isinstance(out, bool)}
    except Exception as e:  # noqa: BLE001
        npb[label] = {"returned": f"raised {type(e).__name__}", "real_bool": False}

json.dump({"tree": TREE, "rows": rows, "entry": entry, "numpy": npb},
          open(sys.argv[1], "w"), indent=1)
print("wrote", sys.argv[1])
