"""D11: one SimEngine contract, four backends behind create_simulation.

Top, the door: create_simulation resolves an alias and imports the backend lazily; Robot("so101")
calls it. Middle, the contract every backend implements (SimEngine, the one accent element): world
lifecycle, entities, observation, actuation. Under it the dashed layer the base class implements
once and every backend inherits: run_policy, eval_policy, replay_episode, recording. Bottom, the
four backends with their aliases and what each needs. Every name here is on learn/simulation/index.md.
"""
from scene import Scene

X, W = 60, 1080
COLS = 4
GAP = 16
BW = (W - GAP * (COLS - 1)) / COLS


def scene() -> Scene:
    s = Scene(
        "d11_sim_backends",
        "One SimEngine, four backends",
        "create_simulation picks the engine; the calls your code and your agent make are the same on every one.",
        "Top: create_simulation(name) resolves an alias, imports the backend lazily and passes the "
        "remaining keywords to the constructor; Robot(\"so101\") calls it for you. Middle, the one green "
        "element: the SimEngine contract, world lifecycle (create_world, reset, step, destroy), entities "
        "(add_robot, add_object), observation (get_observation, render, get_contacts) and actuation "
        "(send_action). Under it a dashed layer, implemented once on the base class and inherited: "
        "run_policy, run_multi_policy, eval_policy, replay_episode and dataset recording. Bottom, four "
        "cards: mujoco (mj, mjc, mjx; a CPU, MUJOCO_GL for offscreen rendering), newton (nt; an NVIDIA "
        "GPU with Warp), isaac (isaac_sim, isaacsim, nvidia; Isaac Sim on an RTX GPU) and mjlab (mjl, "
        "mujoco_warp; an NVIDIA GPU, MuJoCo-Warp). Footnote: a third-party engine registers under the "
        "strands_robots.backends entry-point group, or with register_backend at runtime.",
        h=700,
    )

    # ---------------------------------------------------------------- the door
    s.section(X, 122, "the door")
    s.box(X, 134, W, 70, 'create_simulation("mujoco")',
          'resolves an alias, imports the backend lazily, passes the remaining keywords to the constructor; '
          'Robot("so101") calls it for you and adds the robot', size=14, subsize=12)
    s.down(600, 204, 234, label="one engine", label_dx=10, label_dy=4, id="pick")

    # ---------------------------------------------------------------- the contract (the one green element)
    s.section(X, 226, "the contract")
    s.box(X, 234, W, 112, "SimEngine",
          "the abstract class every backend implements; the same object is the agent's tool",
          accent=True, size=14, subsize=12, id="engine")
    s.chips(X + 14, 286, ["create_world", "reset", "step", "destroy", "add_robot", "add_object"])
    s.chips(X + 14, 314, ["get_observation", "render", "get_contacts", "send_action", "add_camera"])

    # ---------------------------------------------------------------- inherited once (dashed layer)
    s.box(X, 362, W, 92, None, None, dashed=True)
    s.text(X + 14, 384, "IMPLEMENTED ONCE ON THE BASE, INHERITED BY EVERY BACKEND", cls="mono muted",
           size=10.5, spacing="0.05em")
    s.chips(X + 14, 400, ["run_policy", "run_multi_policy", "eval_policy", "evaluate_benchmark",
                          "start_policy", "stop_policy", "replay_episode", "dataset recording"])

    # ---------------------------------------------------------------- the four backends
    s.down(600, 454, 486, label="the same calls", label_dx=10, label_dy=4, id="same")
    s.arrow([(X + BW / 2, 486), (X + W - BW / 2, 486)], head=False)
    cards = [
        ("mujoco", "a CPU; MUJOCO_GL for offscreen rendering; the default", ["mj", "mjc", "mjx"]),
        ("newton", "an NVIDIA GPU with Warp; the same MJCF assets", ["nt"]),
        ("isaac", "Isaac Sim on an RTX GPU; photoreal rendering, USD scenes", ["isaac_sim", "nvidia"]),
        ("mjlab", "an NVIDIA GPU, MuJoCo-Warp; thousands of worlds", ["mjl", "mujoco_warp"]),
    ]
    for i, (name, sub, aliases) in enumerate(cards):
        x = X + i * (BW + GAP)
        s.down(x + BW / 2, 486, 510, id=f"to_{name}")
        s.box(x, 510, BW, 120, name, sub, size=14, subsize=12, id=name)
        s.chips(x + 14, 596, aliases)
        s.motion.append((f"to_{name}", "flow"))
    s.motion[:0] = [("engine", "pulse"), ("pick", "flow"), ("same", "flow")]

    s.footnote(672, "a third-party engine registers under the strands_robots.backends entry-point group, "
                    "or with register_backend at runtime.")
    return s
