"""
Minimal repro: Robot("lekiwi", mode="real", robot_ip=...) silently builds the
wrong class and drops the kwarg.

Documented path (docs/learn/hardware/feetech-arms.md:73):
    `Robot("lekiwi", mode="real", robot_ip="192.168.1.50")` builds the client.

Actual behavior:
  1. The kwarg `robot_ip` is on the cross-robot passthrough allowlist
     (strands_robots/hardware_robot.py:159) but LeKiwi (host)'s dataclass
     does not declare it, so it is silently dropped (never applied).
  2. `Robot("lekiwi", mode="real", ...)` builds `LeKiwi` (the *host* class,
     which runs on the Pi), NOT `LeKiwiClient` (the laptop-side class the
     docs claim is being built).
  3. The correct address kwarg for the client is `remote_ip`, not `robot_ip`
     (see strands_robots/hardware_robot.py:220 `_ADDRESS_FIELDS` and
     the LeKiwiClient error at hardware_robot.py:1406).

End-user impact: a user copy-pastes the docs sketch, gets no error, and their
laptop-side code silently constructs a Pi-side host object that tries to open
serial ports and never dials the IP they provided.

Run: python lekiwi_robot_ip_silent_wrong_repro.py
"""
from strands_robots import Robot

# ---- What the docs sketch tells us to run ----
r = Robot("lekiwi", mode="real", robot_ip="192.168.1.50")

# ---- What actually got built ----
built_class = type(r.robot).__name__
built_module = type(r.robot).__module__
print(f"Built class : {built_module}.{built_class}")

# The kwarg vanished into thin air
config = r.robot.config
has_robot_ip = hasattr(config, "robot_ip")
has_remote_ip = hasattr(config, "remote_ip")
has_port      = hasattr(config, "port")
print(f"config has 'robot_ip'  : {has_robot_ip}")
print(f"config has 'remote_ip' : {has_remote_ip}")
print(f"config has 'port'      : {has_port}")
print(f"config.port            : {getattr(config, 'port', None)!r}")

# ---- Assertions that lock the defect ----
# 1. Docs claim the client is built. The host is built.
assert built_class == "LeKiwi", (
    f"Docs feetech-arms.md:73 claims this builds LeKiwiClient; got {built_class}"
)
assert built_module == "lerobot.robots.lekiwi.lekiwi", (
    f"Expected the LeKiwi host module (silent-wrong evidence). Got {built_module}"
)

# 2. `robot_ip` never made it to the config.
assert not has_robot_ip, (
    "If robot_ip had been forwarded, LeKiwiConfig would carry it; it does not."
)

# 3. Correct spelling for the client is remote_ip, and it needs lekiwi_client:
try:
    Robot("lekiwi_client", mode="real", robot_ip="192.168.1.50")
    raise AssertionError("lekiwi_client should reject robot_ip and demand remote_ip")
except ValueError as e:
    msg = str(e)
    assert "remote_ip" in msg, msg
    print("\n[good] lekiwi_client correctly demands remote_ip:")
    print("       " + msg.split(".")[0] + ".")

# 4. The correct invocation actually works:
client = Robot("lekiwi_client", mode="real", remote_ip="192.168.1.50")
assert type(client.robot).__name__ == "LeKiwiClient", type(client.robot).__name__
print(f"[good] Robot('lekiwi_client', mode='real', remote_ip=...) -> "
      f"{type(client.robot).__module__}.{type(client.robot).__name__}")

print("\nDEFECT CONFIRMED: docs sketch silently builds the wrong class + drops kwarg.")
