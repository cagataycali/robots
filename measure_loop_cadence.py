"""Measure the real loop bodies with the optional deps absent."""
import threading, time, sys
from pathlib import Path
import strands_robots.hardware_rtps_bridge as rtps_mod
import strands_robots.hardware_ros_bridge as ros_mod
print("TREE:", Path(rtps_mod.__file__).parents[1])

class FakeReader:
    def __init__(self): self.takes = 0
    def take(self, N=10):
        self.takes += 1
        return []

class FakeRclpy:
    def __init__(self): self.spins = []
    def spin_once(self, node, timeout_sec=None):
        self.spins.append(timeout_sec)

def measure_poll(period, window=0.30):
    b = rtps_mod.HardwareRtpsBridge.__new__(rtps_mod.HardwareRtpsBridge)
    b._stop = threading.Event()
    b._command_reader = FakeReader()
    try:
        b._poll_period = float(period)
    except Exception as e:
        return ("coercion", f"{type(e).__name__}: {e}", None)
    th = threading.Thread(target=b._poll_loop, daemon=True)
    t0 = time.perf_counter()
    th.start()
    time.sleep(window)
    b._stop.set()
    th.join(timeout=2.0)
    dt = time.perf_counter() - t0
    alive = th.is_alive()
    return ("ran", b._command_reader.takes, (dt, alive))

def measure_spin(period, window=0.30):
    b = ros_mod.HardwareRosBridge.__new__(ros_mod.HardwareRosBridge)
    b._stop = threading.Event()
    fake = FakeRclpy()
    b._rclpy = fake
    b._node = object()
    try:
        b._spin_period = float(period)
    except Exception as e:
        return ("coercion", f"{type(e).__name__}: {e}", None)
    th = threading.Thread(target=b._spin_loop, daemon=True)
    th.start()
    time.sleep(window)
    b._stop.set()
    th.join(timeout=2.0)
    return ("ran", len(fake.spins), (set(fake.spins), th.is_alive()))

CASES = [0.02, 0, -1, float("nan"), True, float("inf"), "0.02", None]
print("\n=== RTPS _poll_loop: reader.take() calls in a 0.30 s window ===")
for c in CASES:
    kind, n, extra = measure_poll(c)
    print(f"  poll_period={c!r:>10}  {kind:9} -> {n}" + (f"   {extra}" if extra else ""))
print("\n=== ROS _spin_loop: spin_once() calls in a 0.30 s window ===")
for c in CASES:
    kind, n, extra = measure_spin(c)
    print(f"  spin_period={c!r:>10}  {kind:9} -> {n}" + (f"   {extra}" if extra else ""))
