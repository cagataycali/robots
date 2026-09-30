#!/usr/bin/env python3
"""Capture the "Talk to it" transcript for real: a Strands agent building a scene step by step.

The agent holds two robot tools, the simulated SO-101 (``so101_sim``) and a ``mode="real"``
SO-101 on a mocked servo bus (``so101``, lerobot driver with ``mock=True``, so no arm is
needed and nothing physical can move). Four turns:

1. add a camera and a red cube to the simulation and look at the scene,
2. run the mock policy in the simulation (never gated),
3. run it on the real arm: the operator gate interrupts, the operator answers ``n``,
4. ask again: the gate interrupts, the operator answers ``y``, the rollout is dispatched.

After every tool call that changes the simulated scene a frame is rendered from the same
framing camera ``sim_frames.py`` uses, into ``$STRANDS_DOCS_FRAME_DIR/talk-to-it-<n>.png``.
The full message log (every tool_use, tool_result, interrupt and answer) is written next to
the frames as ``talk-to-it.json`` and a readable ``talk-to-it.md``; the page quotes from it.

Needs a model provider (Bedrock by default) and ``strands-robots[lerobot,sim-mujoco]``:

    STRANDS_DOCS_FRAME_DIR=docs/assets/sim python3 docs/hooks/transcripts/talk_to_it.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

FRAME_DIR = Path(os.environ.get("STRANDS_DOCS_FRAME_DIR", "docs/assets/sim"))
PREFIX = "talk-to-it"
W, H = 960, 540
_frames: list[dict[str, Any]] = []


def _frame_camera(engine: Any) -> tuple[list[float], list[float]]:
    data, model = engine.mj_data, engine.mj_model
    pts = np.asarray(data.xpos[1:]) if model.nbody > 1 else np.zeros((1, 3))
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    centre = (lo + hi) / 2
    radius = max(float(np.linalg.norm(hi - lo)) / 2, 0.15)
    direction = np.array([1.7, -1.4, 0.95])
    position = centre + direction / np.linalg.norm(direction) * radius * 3.2
    target = centre + np.array([0.0, 0.0, radius * 0.15])
    return [float(x) for x in position], [float(x) for x in target]


def snapshot(engine: Any, label: str) -> None:
    """Render the scene as it is now into the next numbered frame."""
    n = len(_frames) + 1
    if "docs_frame" not in engine.list_cameras():
        position, target = _frame_camera(engine)
        engine.add_camera(name="docs_frame", position=position, target=target, fov=45.0, width=W, height=H)
    frame = engine.render(camera_name="docs_frame")
    png = next(b["image"]["source"]["bytes"] for b in frame["content"] if "image" in b)
    path = FRAME_DIR / f"{PREFIX}-{n}.png"
    path.write_bytes(png)
    _frames.append({"n": n, "label": label, "file": path.name})
    print(f"[frame {n}] {label} -> {path}", file=sys.stderr)


def watch(engine: Any) -> None:
    """Snapshot after every scene-changing action the agent takes on the engine."""
    for verb in ("add_camera", "add_object", "run_policy", "set_joint_positions", "move_to"):
        original = getattr(engine, verb)

        def wrapped(*a: Any, _orig: Any = original, _verb: str = verb, **k: Any) -> Any:
            out = _orig(*a, **k)
            if not (k.get("name") == "docs_frame"):
                snapshot(engine, f"after {_verb}")
            return out

        setattr(engine, verb, wrapped)


def _text(block: dict[str, Any]) -> str:
    if "text" in block:
        return block["text"]
    if "toolUse" in block:
        use = block["toolUse"]
        return f"tool_use {use['name']} {json.dumps(use['input'], ensure_ascii=True)}"
    if "toolResult" in block:
        res = block["toolResult"]
        parts = []
        for c in res.get("content", []):
            if "text" in c:
                parts.append(c["text"])
            elif "json" in c:
                parts.append("json " + json.dumps(c["json"], ensure_ascii=True)[:300])
            elif "image" in c:
                parts.append("image (png)")
        return f"tool_result {res.get('status', '')} " + " | ".join(parts)
    return json.dumps(block, ensure_ascii=True)[:300]


def main() -> int:
    """Run the four turns and write frames, JSON and markdown."""
    from strands import Agent

    from strands_robots import Robot

    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    sim = Robot("so101")
    arm = Robot("so101", mode="real", port="/dev/null", mock=True)
    watch(sim)
    snapshot(sim, "the scene Robot('so101') built")
    agent = Agent(tools=[sim, arm], callback_handler=None)
    log: list[dict[str, Any]] = []

    def turn(prompt: Any, label: str) -> Any:
        before = len(agent.messages)
        result = agent(prompt)
        log.append(
            {
                "turn": label,
                "prompt": prompt if isinstance(prompt, str) else "interrupt responses",
                "stop_reason": result.stop_reason,
                "interrupts": [
                    {"id": i.id, "name": i.name, "reason": i.reason}
                    for i in (getattr(result, "interrupts", None) or [])
                ],
                "messages": agent.messages[before:],
            }
        )
        return result

    turn(
        "In the simulation, add a camera named 'front' that looks at the so101 from the front, then add a red cube "
        "5 cm wide named 'red_cube' on the table 20 cm in front of the arm base, and render the front camera.",
        "1 build the scene",
    )
    turn(
        "Run the mock policy in the simulation with the instruction 'pick up the red cube' for 3 seconds.",
        "2 run a policy in sim",
    )
    result = turn(
        "Now run the same mock policy on the real so101 for 2 seconds with the same instruction. Call the tool directly.",
        "3 ask the real arm",
    )
    if result.interrupts or []:
        answers = [{"interruptResponse": {"interruptId": i.id, "response": "n"}} for i in result.interrupts]
        turn(answers, "3b operator answers n")
    result = turn(
        "Try the real so101 once more, same policy, same instruction, 2 seconds. Call the tool directly.",
        "4 ask again",
    )
    if result.interrupts or []:
        answers = [{"interruptResponse": {"interruptId": i.id, "response": "y"}} for i in result.interrupts]
        turn(answers, "4b operator answers y")

    sim.cleanup()
    arm.cleanup()

    (FRAME_DIR / f"{PREFIX}.json").write_text(
        json.dumps({"frames": _frames, "turns": log}, indent=1, ensure_ascii=True, default=str), encoding="utf-8"
    )
    lines = ["# Talk to it: the captured transcript", ""]
    for entry in log:
        lines += [f"## {entry['turn']}", "", f"> {entry['prompt']}", ""]
        for message in entry["messages"]:
            for block in message.get("content", []):
                lines.append(f"[{message['role']}] {_text(block)}")
        for i in entry["interrupts"]:
            lines.append(f"[interrupt] {i['name']}: {i['reason'].get('warning', '')}")
        lines.append(f"stop_reason: {entry['stop_reason']}")
        lines.append("")
    lines += ["## frames", ""] + [f"- {f['file']}: {f['label']}" for f in _frames]
    (FRAME_DIR / f"{PREFIX}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(_frames)} frames, {len(log)} turns -> {FRAME_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
