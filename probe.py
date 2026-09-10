"""Issue model-shaped tool calls at one tree and record what reached the SDK."""
import asyncio, base64, json, sys, time, types

TREE, OUT = sys.argv[1], sys.argv[2]
sys.path.insert(0, TREE)
import strands_robots  # noqa: E402
assert strands_robots.__file__.startswith(TREE), strands_robots.__file__

FRAME = base64.b64encode(bytes.fromhex("89504e470d0a1a0a") + b"\x00" * 32).decode()


class Resp:
    def __init__(self, code=200, payload=None, text=""):
        self.status_code, self._p, self.text = code, payload, text

    def json(self):
        return self._p


class Session:
    """A recording stand-in for the earth-rovers-sdk. Timestamps every frame."""

    def __init__(self):
        self.control = []   # (t, command dict)
        self.other = []
        self.t0 = time.monotonic()

    def get(self, url, timeout=0.0, **k):
        if url.endswith("/data"):
            return Resp(200, {"battery": 87, "signal_level": 3, "orientation": 128,
                              "latitude": 41.015, "longitude": 28.979, "speed": 0,
                              "lamp": 0, "gps_signal": 1})
        return Resp(200, {"front_frame": FRAME})

    def post(self, url, json=None, timeout=0.0, **k):
        if url.endswith("/control"):
            self.control.append((round(time.monotonic() - self.t0, 4), dict(json["command"])))
        else:
            self.other.append((url.rsplit("/", 1)[-1], json))
        return Resp(200, {})

    def close(self):
        pass


session = Session()
fake = types.ModuleType("requests")
fake.Session = lambda: session
sys.modules["requests"] = fake

from strands_robots.drivers import get_native_driver_class  # noqa: E402
from strands_robots.drivers.base import declared_verbs  # noqa: E402

driver = get_native_driver_class("earthrover")(tool_name="earthrover", cameras=None, data_config=None)
assert driver.connect_eagerly() is None


def invoke(tool, payload):
    async def go():
        return [e async for e in tool.stream({"toolUseId": "v", "name": "earthrover", "input": payload}, {})]
    events = asyncio.run(go())
    result = events[-1].get("tool_result", events[-1])
    text = " ".join(str(b.get("text", "")) for b in result.get("content") or [] if isinstance(b, dict))
    kinds = [k for b in result.get("content") or [] if isinstance(b, dict) for k in b]
    return {"status": result.get("status"), "text": text[:200], "blocks": kinds}


# What a model may put in the input. Exactly what it would emit as JSON.
REQUESTS = [
    ("drive forward, hold 1s", {"action": "move", "linear": 0.35, "duration_s": 1.0}),
    ("headlamp on", {"action": "lamp", "on": True}),
    ("look at the front camera", {"action": "camera", "camera": "front"}),
    ("announce it", {"action": "speak", "text": "scanning ahead"}),
    ("read telemetry", {"action": "sensors"}),
]

report = {"tree": TREE, "declared": declared_verbs(driver.tool_spec),
          "properties": sorted(driver.tool_spec["inputSchema"]["json"]["properties"]), "calls": []}
for label, payload in REQUESTS:
    report["calls"].append({"label": label, "input": payload, "through": "the robot handle",
                            **invoke(driver, payload)})

# On the pre-fix tree the same capability had a second door: the rover_* tools.
# A model can only fill `driver` with JSON, so probe that too rather than
# comparing against a door nobody claimed existed.
try:
    from strands_robots.tools.earthrover import rover_move
    report["calls"].append({"label": "drive forward, hold 1s", "through": "rover_move tool",
                            "input": {"driver": "earthrover", "linear": 0.35, "duration_s": 1.0},
                            **invoke(rover_move, {"driver": "earthrover", "linear": 0.35, "duration_s": 1.0})})
except (ImportError, AttributeError) as exc:
    report["rover_move_tool"] = f"absent: {type(exc).__name__}"

report["control_frames"] = session.control
report["other_posts"] = session.other
open(OUT, "w").write(json.dumps(report, indent=1))
print(f"{TREE}: {len(session.control)} /control frames, verbs={report['declared']}")
