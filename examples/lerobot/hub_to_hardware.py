"""From Hugging Face Hub to robot hardware with Strands Agents and LeRobot.

A runnable end-to-end example that mirrors the blog post of the same name.
The whole thing is just three ingredients - the same recipe any Strands agent
uses:

    1. A system prompt that explains the robot and the job.
    2. A set of tools - here, a Robot('so100') plus the mesh tool.
    3. A few natural-language invocations.

``Robot('so100')`` returns a simulation (or a hardware connection with
``mode='real'``) that is itself a Strands tool. It already knows how to compose
a scene, record a ``LeRobotDataset``, run a policy, render frames, and push to
the Hub - so the example does not re-implement any of that. It tells the agent
what to do in English and lets the tool do the work.

Quick start (no hardware, no GPU, no Hub credentials needed):

    # Dev/lab mesh posture
    export STRANDS_MESH_LOCAL_DEV=1

    python hub_to_hardware.py

(For sim-only runs you can disable the mesh entirely with STRANDS_MESH=false.)

Push the recorded dataset to the Hub (requires HF_TOKEN with write scope):

    export HF_TOKEN=hf_...
    python hub_to_hardware.py --hf-user my_user

Override the LLM (verify the exact Bedrock ID in your AWS console):

    python hub_to_hardware.py --model-id global.anthropic.claude-sonnet-4-6

Run with the GR00T container as the policy (requires Docker + NVIDIA GPU):

    python hub_to_hardware.py --policy groot --checkpoint nvidia/GR00T-N1.7-LIBERO

Run on physical hardware (assumes SO-101 already calibrated via lerobot-calibrate):

    python hub_to_hardware.py --mode real --port /dev/ttyACM0 --leader-port /dev/ttyACM1

Repository: https://github.com/strands-labs/robots
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any

logger = logging.getLogger("hub_to_hardware")


# ---------------------------------------------------------------------------
# Model default
# ---------------------------------------------------------------------------
# Claude Opus 4.8 on Bedrock orchestrates the LeRobot tool surface cleanly in
# one shot - lower-tier models work but issue more defensive state-querying
# calls. Verify the exact ID in your AWS Bedrock console (Model catalog ->
# Anthropic); cross-region inference profile IDs are prefixed by ``us.``,
# ``eu.``, etc. Override via --model-id or STRANDS_BEDROCK_MODEL_ID.
DEFAULT_MODEL_ID = "global.anthropic.claude-opus-4-8"  # verify in AWS console


# ---------------------------------------------------------------------------
# System prompt - the agent's standing instructions
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a robotics operator driving an SO-100 arm through the strands_robots
SDK. You have one robot tool bound to this conversation; calling it dispatches
into the live simulation (or hardware in real mode). The robot is already
initialised - a world exists and the arm is loaded - so never create or destroy
the world yourself.

The robot tool already knows how to compose a scene, record a LeRobotDataset,
run a policy, and render frames. Use it directly; do not reach for raw lerobot
APIs. The recorder writes the standard LeRobotDataset layout on disk and can
push to the Hugging Face Hub when asked.

Work in the fewest tool calls that get the job done. When you record, the
lifecycle is: start_recording -> run_policy -> stop_recording (stop_recording
finalizes and saves the episode; there is no separate save step). Report what
you did in plain language, grounded in the tool results you actually got back -
do not claim a dataset or frame count you did not observe.
"""


# ---------------------------------------------------------------------------
# Agent construction
# ---------------------------------------------------------------------------
def build_agent(
    *,
    mode: str = "sim",
    policy: str = "mock",
    port: str | None = None,
    leader_port: str | None = None,
    model_id: str | None = None,
    aws_region: str | None = None,
) -> Any:
    """Build the Strands agent: a system prompt, a Robot tool, and the mesh.

    The Robot factory returns a simulation (mode='sim') or a hardware
    connection (mode='real'); either way it is a Strands tool the agent calls
    to act on the world. ``model`` resolution is best-effort - if a Bedrock
    model id is given but cannot be constructed, the agent falls back to the
    Strands default and the workflow still runs.
    """
    from strands import Agent

    from strands_robots import Robot, robot_mesh

    robot_kwargs: dict[str, Any] = {"data_config": "so100_dualcam"}
    if mode == "real":
        if not port:
            raise SystemExit("--mode real requires --port (e.g. /dev/ttyACM0). Hardware paths can't be guessed safely.")
        robot_kwargs.update(
            port=port,
            cameras={
                "front": {"type": "opencv", "index_or_path": "/dev/video0", "fps": 30},
                "wrist": {"type": "opencv", "index_or_path": "/dev/video2", "fps": 30},
            },
        )

    robot = Robot("so100", mode=mode, **robot_kwargs)
    tools: list[Any] = [robot, robot_mesh]

    if policy == "groot":
        from strands_robots import gr00t_inference

        tools.append(gr00t_inference)

    agent = Agent(model=_resolve_model(model_id, aws_region), system_prompt=SYSTEM_PROMPT, tools=tools)
    # Stash for cleanup() and the hardware recording prompt. Both Simulation
    # and HardwareRobot expose cleanup(); without it the Zenoh mesh listener
    # and the MuJoCo executor keep non-daemon threads alive and the process
    # never exits after the workflow finishes.
    agent._robot = robot  # type: ignore[attr-defined]
    agent._leader_port = leader_port  # type: ignore[attr-defined]
    return agent


def _resolve_model(model_id: str | None, aws_region: str | None) -> Any:
    """Return a BedrockModel for the resolved id, or None for the Strands default.

    None is returned on any failure (import error, model not enabled, wrong
    region) so the example stays runnable - Strands picks its default model and
    the workflow continues, just with more defensive tool orchestration.
    """
    resolved_id = model_id or os.environ.get("STRANDS_BEDROCK_MODEL_ID") or DEFAULT_MODEL_ID
    resolved_region = aws_region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    try:
        from strands.models import BedrockModel

        kwargs: dict[str, Any] = {"model_id": resolved_id}
        if resolved_region:
            kwargs["region_name"] = resolved_region
        model = BedrockModel(**kwargs)
        logger.info("Using Bedrock model %s (region %s)", resolved_id, resolved_region or "<from AWS env>")
        return model
    except Exception as exc:  # noqa: BLE001 - best-effort; fall back to default
        logger.warning(
            "Could not build BedrockModel(%s): %s. Falling back to the Strands default model.",
            resolved_id,
            exc,
        )
        return None


# ---------------------------------------------------------------------------
# Workflow phases - each is one natural-language instruction to the agent
# ---------------------------------------------------------------------------
def record_prompt(*, mode: str, agent: Any, repo_id: str, num_steps: int, task: str, push_to_hub: bool) -> str:
    """Compose the recording instruction for sim or hardware."""
    push_clause = (
        f"Push the result to {repo_id} when done."
        if push_to_hub
        else "Keep the dataset local - do not push to the Hub."
    )
    if mode == "sim":
        return (
            f"Compose the scene and record one demonstration. Add a small red cube "
            f"(about 3cm) near the robot and a front-facing camera named 'front' looking "
            f"at it. Start recording to repo_id={repo_id!r} at 30 fps with task={task!r}, "
            f"run the Mock policy for {num_steps} steps with instruction={task!r}, then "
            f"stop recording to finalize the episode. {push_clause}"
        )
    leader_port = getattr(agent, "_leader_port", None)
    if not leader_port:
        raise SystemExit("Hardware recording requires --leader-port (e.g. /dev/ttyACM1)")
    return (
        f"Record one demonstration of {task!r} on the SO-101, with the leader on "
        f"{leader_port}. Write the dataset to {repo_id} at 30 fps. {push_clause}"
    )


def policy_prompt(*, policy: str, checkpoint: str | None, instruction: str) -> str:
    """Compose the policy-rollout instruction for the chosen provider."""
    if policy == "mock":
        return (
            f"Run the Mock policy on the robot for 200 steps with the instruction "
            f"{instruction!r}. Render the final frame."
        )
    if policy == "groot":
        if not checkpoint:
            raise SystemExit("--policy groot requires --checkpoint <hf_repo>, e.g. nvidia/GR00T-N1.7-LIBERO")
        return (
            f"Use gr00t_inference lifecycle='full' to bring up the GR00T container on port "
            f"5555 with checkpoint {checkpoint}. Then run the policy on the robot with the "
            f"instruction {instruction!r} for 200 steps. Render the final frame."
        )
    if policy == "lerobot_local":
        if not checkpoint:
            raise SystemExit(
                "--policy lerobot_local requires --checkpoint <hf_repo>, e.g. lerobot/act_aloha_sim_transfer_cube_human"
            )
        if os.environ.get("STRANDS_TRUST_REMOTE_CODE") != "1":
            raise SystemExit(
                "lerobot_local requires STRANDS_TRUST_REMOTE_CODE=1 to opt in to "
                "trust_remote_code loading. Re-run with that env var."
            )
        # MolmoAct2 checkpoints (e.g. allenai/MolmoAct2-SO100_101) also run through
        # this path: LerobotLocalPolicy auto-detects model_type from config.json and
        # routes to the right loader. No separate provider needed.
        return (
            f"Run the LerobotLocal policy {checkpoint!r} on the robot with the instruction "
            f"{instruction!r} for 200 steps. Render the final frame."
        )
    raise SystemExit(f"Unknown policy: {policy!r}")


def mesh_prompt(instruction: str = "go to home pose") -> str:
    """Compose the mesh-broadcast instruction."""
    return (
        f"Use robot_mesh to list every peer and local robot, then broadcast the "
        f"instruction {instruction!r} to each one with a 5-second timeout."
    )


def cleanup(agent: Any) -> None:
    """Release the robot's sim world / hardware connection and mesh session.

    Both Simulation and HardwareRobot expose cleanup(); these own non-daemon
    threads (the mesh listener, the MuJoCo executor) that would otherwise keep
    the interpreter alive and make the script appear to hang after it finishes.
    """
    robot = getattr(agent, "_robot", None)
    if robot is not None and hasattr(robot, "cleanup"):
        try:
            robot.cleanup()
        except Exception as exc:  # noqa: BLE001 - best-effort teardown
            logger.warning("Robot cleanup reported: %s", exc)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="hub_to_hardware",
        description="From Hugging Face Hub to robot hardware - the runnable example.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mode", choices=("sim", "real"), default="sim", help="Robot execution mode (default: sim).")
    p.add_argument(
        "--policy",
        choices=("mock", "groot", "lerobot_local"),
        default="mock",
        help="Policy provider (default: mock, no GPU required).",
    )
    p.add_argument("--checkpoint", default=None, help="HF repo for the policy (groot / lerobot_local).")
    p.add_argument(
        "--model-id",
        default=None,
        help=f"Bedrock model ID to drive the agent. Default: {DEFAULT_MODEL_ID}. "
        f"Override here or via STRANDS_BEDROCK_MODEL_ID.",
    )
    p.add_argument("--aws-region", default=None, help="AWS region for Bedrock (else resolves from AWS env).")
    p.add_argument("--port", default=None, help="USB device of the SO-101 follower (--mode real only).")
    p.add_argument("--leader-port", default=None, help="USB device of the SO-101 leader arm (--mode real only).")
    p.add_argument("--hf-user", default=None, help="HF username for the dataset repo. If unset, stays local.")
    p.add_argument("--dataset-name", default="strands-cube-pick", help="Dataset name (default: strands-cube-pick).")
    p.add_argument("--num-steps", type=int, default=1000, help="Policy steps to record (default: 1000, ~33s @30fps).")
    p.add_argument("--instruction", default="pick up the red cube", help="Natural-language task instruction.")
    p.add_argument("--clean-cache", action="store_true", help="Tell the agent to wipe the local cache first.")
    p.add_argument("--skip-record", action="store_true", help="Skip the recording step.")
    p.add_argument("--skip-mesh", action="store_true", help="Skip the mesh broadcast step.")
    return p.parse_args(argv)


def banner(title: str) -> None:
    logger.info("=" * 60)
    logger.info(title)
    logger.info("=" * 60)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    push_to_hub = bool(args.hf_user)
    repo_id = f"{args.hf_user}/{args.dataset_name}" if push_to_hub else f"local/{args.dataset_name}"

    logger.info(
        "Starting workflow (mode=%s, policy=%s, push_to_hub=%s, repo=%s, num_steps=%d)",
        args.mode,
        args.policy,
        push_to_hub,
        repo_id,
        args.num_steps,
    )

    banner("Step 1: Build the agent")
    agent = build_agent(
        mode=args.mode,
        policy=args.policy,
        port=args.port,
        leader_port=args.leader_port,
        model_id=args.model_id,
        aws_region=args.aws_region,
    )

    try:
        if args.clean_cache:
            agent(f"Delete the local dataset cache for repo_id={repo_id!r} so we record from a clean slate.")

        if not args.skip_record:
            banner("Step 2: Record a demonstration")
            agent(
                record_prompt(
                    mode=args.mode,
                    agent=agent,
                    repo_id=repo_id,
                    num_steps=args.num_steps,
                    task=args.instruction,
                    push_to_hub=push_to_hub,
                )
            )
        else:
            logger.info("Skipping Step 2 (--skip-record)")

        banner("Step 3/4: Run a policy on the robot")
        agent(policy_prompt(policy=args.policy, checkpoint=args.checkpoint, instruction=args.instruction))

        if not args.skip_mesh:
            banner("Step 5: Mesh broadcast")
            agent(mesh_prompt())
        else:
            logger.info("Skipping Step 5 (--skip-mesh)")
    finally:
        cleanup(agent)

    logger.info("Workflow finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
