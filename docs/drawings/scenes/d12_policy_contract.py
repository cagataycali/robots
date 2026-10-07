"""D12: the Policy contract. What a policy declares, what the runtime supplies, what crosses between them.

Left card: the Policy class with what the provider declares (reads_instruction, requires_images,
required_bodies, requires_action_controller, provider_name) and what it implements (get_actions,
set_robot_state_keys, reset, preflight). Right card: the runtime inside run_policy, which supplies
control_frequency and the state keys, runs preflight against the observation keys, and installs
the action controller or refuses. Between them the two wires of every control tick: an observation
plus the instruction in, a chunk of action dicts out (the one accent element, the thing a policy is
for). Bottom: create_policy and the one string that swaps providers. Every name is on
learn/policies/index.md.
"""
from scene import Scene

L, LW = 60, 440
R, RW = 700, 440
MID = 600


def scene() -> Scene:
    s = Scene(
        "d12_policy_contract",
        "The Policy contract",
        "A provider declares what it needs and implements get_actions; the runtime supplies the rest and calls it every control tick.",
        "Left card, the Policy: declared attributes reads_instruction, instruction_free_actions, "
        "requires_images, required_bodies, requires_action_controller and provider_name; implemented "
        "methods get_actions, set_robot_state_keys, reset and preflight. Right card, the runtime inside "
        "run_policy: it sets control_frequency and rtc_observed_delay_steps, calls set_robot_state_keys with "
        "the robot's joint names, runs preflight against the observation keys before the first tick, and "
        "installs the action controller the policy names or refuses the rollout. Between the cards, every "
        "control tick: an observation dict and the instruction go in; a chunk, one dict per tick, joint name "
        "to a float, comes back, the one green wire. Bottom: create_policy(provider, **policy_config) "
        "builds any of them; swapping providers is one string. Footnote: planners read target_pose or "
        "target_joints and ignore the words; a non-reader says so in its report.",
        h=690,
    )

    # ---------------------------------------------------------------- the policy
    s.section(L, 122, "the provider writes")
    s.box(L, 134, LW, 344, "class Policy(ABC)", "twenty lines for your own; built-ins come from the registry",
          size=14, subsize=12, id="policy")
    s.text(L + 14, 200, "DECLARES", cls="mono muted", size=10.5, spacing="0.05em")
    s.chips(L + 14, 208, ["reads_instruction", "instruction_free_actions"])
    s.chips(L + 14, 238, ["requires_images", "required_bodies", "provider_name"])
    s.chips(L + 14, 268, ["requires_action_controller"])
    s.text(L + 14, 316, "IMPLEMENTS", cls="mono muted", size=10.5, spacing="0.05em")
    s.chips(L + 14, 326, ["get_actions(observation_dict, instruction)"])
    s.chips(L + 14, 356, ["set_robot_state_keys", "reset(seed)"])
    s.chips(L + 14, 386, ["preflight(observation_keys)"])
    s.para(L + 14, 428, "get_actions returns the chunk: one dict per control tick, joint name to a float.",
           LW - 28, size=12, cls="grot muted")

    # ---------------------------------------------------------------- the runtime
    s.section(R, 122, "the runtime supplies, inside run_policy")
    s.box(R, 134, RW, 344, "run_policy(policy_provider, policy_config)",
          "the engine or the real arm; the same loop on every backend", size=14, subsize=12, id="runtime")
    s.text(R + 14, 200, "SETS", cls="mono muted", size=10.5, spacing="0.05em")
    s.chips(R + 14, 208, ["control_frequency", "rtc_observed_delay_steps"])
    s.chips(R + 14, 238, ["set_robot_state_keys(joint names)"])
    s.text(R + 14, 316, "CHECKS BEFORE THE FIRST TICK", cls="mono muted", size=10.5, spacing="0.05em")
    s.chips(R + 14, 326, ["preflight against the observation keys"])
    s.chips(R + 14, 356, ["installs the action controller, or refuses"])
    s.para(R + 14, 428, "a policy declaring requires_action_controller runs only where the engine can install it.",
           RW - 28, size=12, cls="grot muted")

    # ---------------------------------------------------------------- the two wires of every tick
    s.arrow([(R, 260), (L + LW, 260)], id="obs")
    s.text(MID, 250, "observation + instruction", cls="mono muted", size=10.5, anchor="middle")
    s.arrow([(L + LW, 340), (R, 340)], accent=True, id="chunk")
    s.text(MID, 330, "action chunk", cls="mono accent", size=10.5, anchor="middle")
    s.text(MID, 364, "every control tick", cls="grot muted", size=11.5, anchor="middle")
    s.motion = [("obs", "flow"), ("chunk", "flow")]

    # ---------------------------------------------------------------- the factory
    s.down(MID, 478, 508, id="build")
    s.box(L, 508, 1080, 104, "create_policy(provider, **policy_config)",
          "mock, lerobot_local, remote, wbc, holosoma, curobo, cosmos3 and the rest of the registry; swapping "
          "providers is one string, and run_policy takes the same name with a policy_config",
          size=14, subsize=12)
    s.chips(L + 14, 576, ['create_policy("mock")', 'create_policy("ws://gpu:8765")', 'embodiment=...'])

    s.footnote(654, "planners read target_pose or target_joints and ignore the words; a provider that never reads "
                    "the instruction says so in its report.")
    return s
