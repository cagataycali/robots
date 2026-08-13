"""Measure what the delta path commands the servo, and what each deferral defers to."""
import json, math, pathlib, serial
import strands_robots
TREE = str(pathlib.Path(strands_robots.__file__).parents[1])
print("TREE:", TREE)
import strands_robots.tools.pose_tool as pt

JOINT, RAW = "shoulder_pan", 3980
LOW, HIGH = pt._DEFAULT_MOTOR_CONFIGS[JOINT]["range"]
SPAN = HIGH - LOW

def packet(raw=RAW):
    return bytes([0xFF, 0xFF, 0x01, 0x04, 0x00, raw & 0xFF, (raw >> 8) & 0xFF, 0, 0, 0])

class Reading:
    def __init__(s, port, baudrate, timeout=1.0):
        s.writes = []; s.is_open = True
    def write(s, d): s.writes.append(bytes(d))
    def read(s, n=1): return packet()
    def close(s): s.is_open = False

INST = []
def ctor(port, baudrate, timeout=1.0):
    fs = Reading(port, baudrate, timeout); INST.append(fs); return fs
serial.Serial = ctor
PORT = "/dev/fake-artifact"

def goals(insts):
    out = []
    for fs in insts:
        for w in fs.writes:
            if len(w) >= 9 and w[4] == 0x03 and w[5] == 0x2A:
                out.append(w[6] | (w[7] << 8))
    return out

mc = pt.MotorController(PORT)
start = mc.position_to_degrees(JOINT, RAW)
END_STOP = mc.degrees_to_position(JOINT, HIGH)

facts = {"tree": TREE, "joint": JOINT, "range": [LOW, HIGH], "span": SPAN,
         "start_deg": start, "end_stop": END_STOP, "raw": RAW, "sweep": []}

for delta in range(-400, 401, 10):
    verdict = pt._joint_delta_error("incremental_move", JOINT, delta)
    INST.clear()
    r = pt.pose_tool(action="incremental_move", motor_name=JOINT, delta=delta, port=PORT)
    g = goals(INST)
    target = start + delta
    facts["sweep"].append({
        "delta": delta, "refused": verdict is not None, "status": r["status"],
        "target_deg": target, "goal": (g[0] if g else None),
        # what an unclamped linear scale would have produced
        "unclamped": int((target - LOW) / (HIGH - LOW) * pt._DEFAULT_MOTOR_CONFIGS[JOINT]["resolution"]),
        "endpoints_rule_refuses": not (LOW <= delta <= HIGH),
    })

# the two deferrals, driven through the public tool
ledger = []
for label, motor, delta in [
    ("unknown motor, unbounded delta", "no_such_joint", 5000),
    ("configured motor, same call", JOINT, -90),
    ("configured motor, delta inside travel", JOINT, 300),
]:
    INST.clear()
    r = pt.pose_tool(action="incremental_move", motor_name=motor, delta=delta, port=PORT)
    ledger.append({
        "label": label, "motor": motor, "delta": delta,
        "domain": pt._joint_delta_error("incremental_move", motor, delta),
        "status": r["status"], "text": r["content"][0]["text"], "goals": goals(INST),
    })
facts["ledger"] = ledger
pathlib.Path(f"/tmp/art-facts-{pathlib.Path(TREE).name}.json").write_text(json.dumps(facts, indent=2))
print("start_deg=%.2f end_stop=%d span=%d" % (start, END_STOP, SPAN))
for row in ledger:
    print(f"  {row['label']:<38} domain={'refuse' if row['domain'] else 'defer '} "
          f"status={row['status']:<7} goals={row['goals']}")
