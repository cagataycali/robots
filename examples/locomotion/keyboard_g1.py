#!/usr/bin/env python3
"""Steer a Unitree G1 from the keyboard in real time (WASD + style keys).

Self-contained teleop: this script owns its own thread-safe current-goal state.
A background reader thread updates the goal from keypresses; the main loop drives
the WBC policy with short ``run_policy`` segments, each reading the current goal
out of the shared state and passing it through ``policy_kwargs``. Re-issuing the
short segment is the closed loop - there is no library locomotion abstraction.

Bindings: w/s forward/back, a/d strafe, q/e turn, space halt, 1-8 style, x quit.
Needs a TTY. Install the optional ``pynput`` reader, or pipe single chars on stdin.

Usage::

    pip install "strands-robots[wbc,sim-mujoco]" pynput
    MUJOCO_GL=egl python examples/locomotion/keyboard_g1.py --checkpoint /path/to/grootwbc-g1
"""

from __future__ import annotations

import argparse
import threading

from strands_robots import Robot

_STYLES = ["run", "happy", "stealth", "injured", "hand_crawling", "elbow_crawling", "boxing"]
_STEP = 0.3  # m/s or rad/s per keypress


class GoalState:
    """Thread-safe current locomotion goal, mutated by the key reader."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._vel = [0.0, 0.0, 0.0]  # [forward, lateral, yaw_rate]
        self._style = "run"
        self.running = True

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {"target_velocity": list(self._vel), "locomotion_style": self._style}

    def apply_key(self, key: str) -> None:
        with self._lock:
            if key == "w":
                self._vel[0] += _STEP
            elif key == "s":
                self._vel[0] -= _STEP
            elif key == "a":
                self._vel[1] += _STEP
            elif key == "d":
                self._vel[1] -= _STEP
            elif key == "q":
                self._vel[2] += _STEP
            elif key == "e":
                self._vel[2] -= _STEP
            elif key == " ":
                self._vel = [0.0, 0.0, 0.0]
            elif key in "12345678":
                self._style = _STYLES[min(int(key) - 1, len(_STYLES) - 1)]
            elif key == "x":
                self.running = False


def _read_keys(state: GoalState) -> None:
    """Feed keypresses into the shared goal until the user quits."""
    try:
        from pynput import keyboard  # optional dependency
    except ImportError:
        print("pynput not installed; reading single chars from stdin (one per line).")
        while state.running:
            line = input()
            if line:
                state.apply_key(line[0])
        return

    def on_press(k: object) -> bool:
        char = getattr(k, "char", None)
        if char:
            state.apply_key(char)
        return state.running

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="WBC checkpoint dir (policy.onnx + walk_policy.onnx)")
    parser.add_argument("--segment", type=float, default=0.5, help="Seconds per run_policy segment")
    args = parser.parse_args()

    state = GoalState()
    reader = threading.Thread(target=_read_keys, args=(state,), daemon=True)
    reader.start()

    robot = Robot("unitree_g1", mode="sim")
    policy_config = {"checkpoint": args.checkpoint, "walk": True}
    print("Steer the G1: WASD move, QE turn, space halt, 1-8 style, x quit.")
    while state.running:
        # Re-read the live goal each short segment: this re-issue IS the loop.
        result = robot.run_policy(
            robot_name="unitree_g1",
            policy_provider="wbc",
            policy_config=policy_config,
            policy_kwargs=state.snapshot(),
            duration=args.segment,
            control_frequency=50.0,
        )
        if result.get("status") != "success":
            print(result["content"][0]["text"])
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
