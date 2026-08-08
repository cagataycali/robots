import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

A = json.load(open("/tmp/art_main.json")); B = json.load(open("/tmp/art_branch.json"))
A2 = json.load(open("/tmp/art_main2.json")); B2 = json.load(open("/tmp/art_branch2.json"))
assert A["tree"] != B["tree"] and A2["tree"] != B2["tree"], "before/after must be two trees"

# --- self-audit against the measured JSON --------------------------------
by = {r["label"]: r for r in A["rows"]}; byb = {r["label"]: r for r in B["rows"]}
assert by["0"]["verdict"] == "reported unreachable" and by["nan"]["verdict"] == "reported unreachable"
assert by["inf"]["verdict"] == "raised OverflowError" and by["inf"]["leaked"] is True
assert by["'15'"]["verdict"] == "raised TypeError" and by["'15'"]["leaked"] is True
assert by["15.0"]["verdict"] == "connected" and byb["15.0"]["verdict"] == "connected"
assert all(byb[k]["verdict"] == "refused at construction" for k in ("0", "-1", "nan", "inf", "True", "'15'", "None"))
s2a = {r["label"]: r for r in A2["rows"]}; s2b = {r["label"]: r for r in B2["rows"]}
assert s2a["None"]["verdict"] == "never returns" and s2a["None"]["lock_held"] is True
assert s2a["0.5"] == s2b["0.5"] or abs(s2a["0.5"]["waited_ms"] - s2b["0.5"]["waited_ms"]) < 60
assert A["ledger"] == B["ledger"] or (
    A["ledger"]["connecting_broker"] == B["ledger"]["connecting_broker"]
    and A["ledger"]["silent_broker"]["connect"] == B["ledger"]["silent_broker"]["connect"]
    and A["ledger"]["silent_broker"]["client_stopped"] == B["ledger"]["silent_broker"]["client_stopped"]
)
n_bad = sum(1 for r in A["rows"] if r["verdict"] not in ("connected",) or r["label"] in ("True", "None"))
GREEN, RED, AMBER, GREY = "#1b7f3b", "#b3261e", "#a8620a", "#54596d"
placed = []
def put(ax, x, y, s, **kw):
    kw.setdefault("va", "top"); kw.setdefault("fontsize", 9.4)
    placed.append((ax, y, "axes" if kw.get("transform") is not None else "data"))
    return ax.text(x, y, s, **kw)

fig = plt.figure(figsize=(16.2, 10.4), dpi=124)
gs = fig.add_gridspec(3, 1, height_ratios=[1.30, 0.92, 0.30], hspace=0.16,
                      left=0.035, right=0.972, top=0.925, bottom=0.035)
fig.suptitle("IotMqttTransport(connect_timeout=...) - the third surface carrying the knob, and the only one without a domain",
             fontsize=15.5, fontweight="bold", y=0.978)
fig.text(0.5, 0.947, "Measured on the real connect() body with a fake MQTT5 client. "
         "No policy, simulation, rendering, recording or asset behaviour changes, so the artifact is a measured verdict table rather than a rollout.",
         ha="center", fontsize=10.2, color="#333", style="italic")

# ---------------- Panel A: a broker that IS connecting -------------------
axA = fig.add_subplot(gs[0]); axA.axis("off"); axA.set_xlim(0, 1); axA.set_ylim(0, 1)
put(axA, 0.0, 1.045, "A.  Broker reports CONNACK 50 ms after start()  -  i.e. reachable, healthy, connecting normally",
    fontsize=12.6, fontweight="bold", transform=axA.transAxes)
cols = [(0.005, "connect_timeout"), (0.115, "main: connect() outcome"), (0.375, "what the caller is told"),
        (0.700, "MQTT5 client left"), (0.845, "this change")]
for x, h in cols:
    put(axA, x, 0.965, h, fontsize=10.0, fontweight="bold", color="#111")
axA.plot([0.0, 1.0], [0.935, 0.935], color="#999", lw=0.9)
ROWS_A = ["15.0", "0.5", "0", "-1", "nan", "inf", "True", "'15'", "None"]
TOP, FLOOR, PAD = 0.905, 0.045, 0.012
STEP = (TOP - FLOOR - PAD * len(ROWS_A)) / len(ROWS_A)
assert STEP > 0.030, STEP
y = TOP
for lab in ROWS_A:
    r, s = by[lab], byb[lab]
    honored = r["verdict"] == "connected" and lab not in ("True", "None")
    colour = GREEN if honored else RED
    if not honored:
        axA.add_patch(Rectangle((0.0, y - STEP - 0.004), 1.0, STEP + 0.012,
                                facecolor=RED, alpha=0.055, zorder=0))
    put(axA, 0.005, y, lab, fontsize=11.0, fontweight="bold", family="monospace")
    put(axA, 0.115, y, r["verdict"], fontsize=10.4, color=colour, fontweight="bold")
    put(axA, 0.375, y, r["detail"], fontsize=9.6, color="#222")
    leak = "STARTED, never stopped" if r["leaked"] else ("running (connected)" if honored else "stopped")
    put(axA, 0.700, y, leak, fontsize=9.6, color=RED if r["leaked"] else GREY,
        fontweight="bold" if r["leaked"] else "normal")
    put(axA, 0.845, y, s["verdict"], fontsize=10.2,
        color=GREEN if s["verdict"] == "connected" else "#0b57d0", fontweight="bold")
    y -= STEP + PAD
assert y > 0.030, y
put(axA, 0.115, 0.028, "A 'timeout' that never waited: 0 / -1 / nan each returned in under 0.5 ms and tore down the connecting client.",
    fontsize=9.4, color=RED, style="italic")

# ---------------- Panel B: a broker that never answers ------------------
axB = fig.add_subplot(gs[1]); axB.axis("off"); axB.set_xlim(0, 1); axB.set_ylim(0, 1)
put(axB, 0.0, 1.055, "B.  Broker never reports CONNACK  -  i.e. genuinely unreachable, the case the budget exists for",
    fontsize=12.6, fontweight="bold", transform=axB.transAxes)
for x, h in [(0.005, "connect_timeout"), (0.115, "main: waited"), (0.300, "main: outcome"),
             (0.520, "instance lock"), (0.700, "this change")]:
    put(axB, x, 0.955, h, fontsize=10.0, fontweight="bold", color="#111")
axB.plot([0.0, 1.0], [0.915, 0.915], color="#999", lw=0.9)
ROWS_B = [r["label"] for r in A2["rows"]]
TOPB, FLOORB, PADB = 0.875, 0.075, 0.014
STEPB = (TOPB - FLOORB - PADB * len(ROWS_B)) / len(ROWS_B)
assert STEPB > 0.030, STEPB
yb = TOPB
for lab in ROWS_B:
    r, s = s2a[lab], s2b[lab]
    usable = s["verdict"] != "refused at construction"
    waited = "never" if r["waited_ms"] is None else f"{r['waited_ms']:.1f} ms"
    put(axB, 0.005, yb, lab, fontsize=11.0, fontweight="bold", family="monospace")
    put(axB, 0.115, yb, waited, fontsize=10.4, family="monospace",
        color=GREEN if usable else (RED if r["waited_ms"] is None else AMBER))
    put(axB, 0.300, yb, r["verdict"], fontsize=10.2,
        color=GREEN if usable else (RED if r["waited_ms"] is None else AMBER))
    put(axB, 0.520, yb, "HELD FOREVER" if r["lock_held"] else "released",
        fontsize=10.0, color=RED if r["lock_held"] else GREY, fontweight="bold" if r["lock_held"] else "normal")
    put(axB, 0.700, yb, s["verdict"] + ("   (unchanged on both trees)" if usable else ""),
        fontsize=10.2, color=GREEN if usable else "#0b57d0", fontweight="bold")
    yb -= STEPB + PADB
assert yb > 0.030, yb
put(axB, 0.115, 0.045,
    "None blocks forever inside connect(), which holds self._lock for its whole body - so close() and every subscription call block behind it.",
    fontsize=9.4, color=RED, style="italic")

# ---------------- Panel C: no-regression ledger + gate ------------------
axC = fig.add_subplot(gs[2]); axC.axis("off"); axC.set_xlim(0, 1); axC.set_ylim(0, 1)
led_a, led_b = A["ledger"], B["ledger"]
put(axC, 0.0, 1.02, "C.  No regression on the usable path, measured on both trees",
    fontsize=12.0, fontweight="bold", transform=axC.transAxes)
lines = [
    f"connecting broker, connect_timeout=0.5   ->  main connect()={led_a['connecting_broker']['connect']}, client stopped={led_a['connecting_broker']['client_stopped']}"
    f"   |   this change connect()={led_b['connecting_broker']['connect']}, client stopped={led_b['connecting_broker']['client_stopped']}",
    f"silent broker,     connect_timeout=0.05  ->  main connect()={led_a['silent_broker']['connect']} after {led_a['silent_broker']['elapsed_ms']} ms, client stopped+cleared={led_a['silent_broker']['client_stopped']}/{led_a['silent_broker']['client_cleared']}"
    f"   |   this change connect()={led_b['silent_broker']['connect']} after {led_b['silent_broker']['elapsed_ms']} ms, {led_b['silent_broker']['client_stopped']}/{led_b['silent_broker']['client_cleared']}",
    "gate: 23571 passed / 257 skipped / 0 failed (MUJOCO_GL=egl, full suite)   -   ruff clean   -   mypy 0 errors outside examples/   -   pre-fix 32 failed / 22 passed",
]
TOPC, FLOORC = 0.78, 0.12
STEPC = (TOPC - FLOORC) / len(lines)
assert STEPC > 0.10, STEPC
yc = TOPC
for i, ln in enumerate(lines):
    put(axC, 0.005, yc, ln, fontsize=9.5, family="monospace",
        color=GREY if i < 2 else "#0b57d0", fontweight="bold" if i == 2 else "normal")
    yc -= STEPC
assert yc > 0.05, yc

for ax, yy, kind in placed:
    if kind == "axes":
        assert -0.03 <= yy <= 1.07, (yy, kind)
    else:
        lo, hi = ax.get_ylim()
        assert lo - 0.05 <= yy <= hi + 0.07, (yy, lo, hi)

out = "/tmp/iot_connect_timeout_domain.png"
fig.savefig(out, bbox_inches="tight", pad_inches=0.30, facecolor="white")
im = np.asarray(Image.open(out).convert("RGB"))
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band.astype(int) - 255).sum(axis=2) > 12).sum())
    assert n == 0, (name, n)
print(f"OK {out} {im.shape[1]}x{im.shape[0]}  rows_A={len(ROWS_A)} rows_B={len(ROWS_B)} border clean")
