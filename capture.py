"""Measure what a caller and an operator learn from each bring-up outcome."""
import asyncio, builtins, json, logging, pathlib, sys, time, types

import strands_robots.device_connect as dc
import strands_robots.robot as rmod

TREE = str(pathlib.Path(dc.__file__).parents[2])
print("TREE:", TREE)

async def _completes(*a, **k): return types.SimpleNamespace(name="runtime")
async def _fails(*a, **k):     raise RuntimeError("no broker at tcp://127.0.0.1:7447")
async def _hangs(*a, **k):     await asyncio.sleep(300)

CASES = (("bring-up completes", _completes), ("bring-up fails", _fails), ("budget expires", _hangs))
facts = {"tree": TREE, "budget_is_a_knob": hasattr(dc, "_INIT_TIMEOUT_S"), "rows": {}}

def _short_budget():
    """Shrink the budget where it is reachable; main's is an inline literal."""
    if hasattr(dc, "_INIT_TIMEOUT_S"):
        dc._INIT_TIMEOUT_S = 0.05
        return 0.05
    return 30.0

for label, fake in CASES:
    orig = dc.init_device_connect
    dc.init_device_connect = fake
    budget = _short_budget()
    # (a) what the direct caller receives
    t0 = time.time()
    try:
        rt = dc.init_device_connect_sync(types.SimpleNamespace(tool_name_str="arm"), peer_id="arm-1")
        caller = f"returned {type(rt).__name__}" if rt is not None else "returned None"
    except BaseException as exc:  # noqa: BLE001 - classifying the outcome
        caller = f"raised {type(exc).__name__}"
    elapsed = time.time() - t0

    # (b) what the operator is told, through the production foreground runner
    records, printed = [], []
    class _Cap(logging.Handler):
        def emit(self, r): records.append(f"{r.levelname} {r.getMessage()}")
    inst = types.SimpleNamespace(_peer_id="arm-1", _peer_type="robot", mesh=None,
                                 tool_name_str="arm", _device_connect_runtime="UNSET")
    o_sleep, o_exit, o_print = time.sleep, rmod.os._exit, builtins.print
    time.sleep = lambda _s: (_ for _ in ()).throw(KeyboardInterrupt())
    rmod.os._exit = lambda _c: None
    builtins.print = lambda *a, **k: printed.append(" ".join(str(x) for x in a))
    h = _Cap(); logging.getLogger("strands_robots.robot").addHandler(h)
    try:
        rmod._run_device_connect_foreground(inst)
    except BaseException:
        pass
    finally:
        builtins.print = o_print
        logging.getLogger("strands_robots.robot").removeHandler(h)
        time.sleep, rmod.os._exit = o_sleep, o_exit
        dc.init_device_connect = orig
    failure = [r for r in records if "Device Connect init failed" in r]
    online = [p for p in printed if "is online" in p]
    facts["rows"][label] = {
        "caller": caller,
        "elapsed_s": round(elapsed, 2),
        "budget_s": budget,
        "runtime_stored": ("None" if inst._device_connect_runtime is None
                           else "UNSET (assignment skipped)" if inst._device_connect_runtime == "UNSET"
                           else "the runtime"),
        "operator_log": failure[0] if failure else "(nothing logged about the bring-up)",
        "operator_stdout": online[0] if online else "(no online line)",
        "reported": bool(failure),
    }
    print(f"  {label:20s} caller={caller:26s} logged={bool(failure)}  ({elapsed:.2f}s)")

out = pathlib.Path(sys.argv[1])
out.write_text(json.dumps(facts, indent=2))
print("wrote", out)
