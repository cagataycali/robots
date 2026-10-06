"""
Repro: sim tool_spec declares 12 params with no description.

The declared global param pool mixes well-documented params (which lead their
description with the action they belong to, e.g. 'n_steps' -> 'step: physics
steps to advance...') with 12 params whose entry is a bare type hint. The
worst offender is 'duration', which is the natural way to say "run for 5
seconds" (vs n_steps). An agent reading tool_spec reaches for it on step()
first because that's the obvious time-axis action, hits a Valid=['n_steps']
refusal, and only finds the real home for duration (run_policy family) by
parsing a parenthetical in control_frequency's description.

Run:
    MUJOCO_GL=egl python tool_spec_duration_undocumented_repro.py
"""
from strands_robots import Robot
import asyncio
from strands.types.tools import ToolUse


async def call(sim, action, inp=None):
    tu = ToolUse(toolUseId="t", name=sim.tool_name, input={"action": action, **(inp or {})})
    async for ev in sim.stream(tu, {}):
        return ev


async def main() -> None:
    r = Robot("so100")

    # (1) Inventory: how many top-level schema props have no description?
    props = r.tool_spec["inputSchema"]["json"]["properties"]
    missing = [k for k, v in props.items() if isinstance(v, dict) and not v.get("description")]
    print(f"Total props: {len(props)}")
    print(f"Missing description: {len(missing)}")
    for n in missing:
        print(f"  - {n}: {props[n]}")

    # (2) The worst papercut: duration is advertised and refused on step().
    print()
    r1 = await call(r, "step", {"duration": 2.0})
    tr = r1["tool_result"]
    print("step(duration=2.0):", tr["status"])
    print("  text:", tr["content"][0]["text"])

    # (3) duration IS valid on run_policy -- schema is correct to list it;
    #     the bug is the empty description + no mention outside a parenthetical.
    r2 = await call(r, "run_policy", {"robot_name": "so100", "duration": 0.1})
    tr2 = r2["tool_result"]
    print("run_policy(duration=0.1):", tr2["status"])


if __name__ == "__main__":
    asyncio.run(main())
