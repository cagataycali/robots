"""D3: the same checkpoint, sim or real.

One Policy object built by create_policy from lerobot/smolvla_base; the embodiment map speaks the sim
dialect (keys 1..6 in radians, converted both ways) and the driver dialect (<motor>.pos in degrees,
bound as they are); run_policy(policy_object=...) is the same call on Robot("so101") and
Robot("so101", mode="real"). Sources: policies/lerobot_local/embodiment.py, policies/factory.py,
simulation/base.py and hardware_robot.py run_policy, docs/start/first-policy.md.
"""
from scene import Scene

C, CW = 330, 540         # the centre column: checkpoint, policy
LX, RX, BW = 100, 620, 480   # the two dialect / robot columns


def scene() -> Scene:
    s = Scene(
        "d03_same_checkpoint",
        "Same checkpoint, sim or real",
        "One Policy object, one run_policy call; the embodiment map is the only thing that knows which body is underneath.",
        "Top, the checkpoint lerobot/smolvla_base: an instruction, observation.state and three camera images "
        "in the units it was trained on go in, an action chunk in the same units comes out. Under it the one "
        "green element, a single Policy object from create_policy with an embodiment. Under that a dashed "
        "embodiment map layer with two dialects: sim, keys 1 to 6 in radians converted both ways, cameras "
        "front and wrist; driver, shoulder_pan.pos to gripper.pos in degrees bound as they are, cameras "
        "front, wrist and top. At the bottom two robots, Robot(\"so101\") in MuJoCo and Robot(\"so101\", "
        "mode=\"real\") on the lerobot driver, each fed actions and returning observations through its "
        "dialect, each reached by the same run_policy(policy_object=policy) call drawn as a wire from the "
        "policy down each side. Footnote: the checkpoint never sees a robot.",
        h=814,
    )
    # ---------------------------------------------------------------- checkpoint
    s.section(C, 122, "the checkpoint")
    s.box(C, 134, CW, 108, "lerobot/smolvla_base",
          "a vision-language-action model from the Hub: the instruction, observation.state and three camera "
          "images go in, a chunk of actions in the same units comes out", size=14, subsize=12)
    s.chips(C + 14, 206, ["observation.state", "observation.images.camera1..3", "action"])
    s.down(600, 242, 276, label="loaded once", label_dx=10, label_dy=4)

    # ---------------------------------------------------------------- the policy (the one green element)
    s.box(C, 276, CW, 104, "one Policy object",
          'create_policy("lerobot_local", pretrained_name_or_path="lerobot/smolvla_base", embodiment=...)',
          accent=True, size=14, subsize=12)
    s.chips(C + 14, 346, ["get_actions(observation, instruction)"])
    s.down(600, 380, 404, label="observation in, action chunk out: the same tensors either way",
           label_dx=10, label_dy=4)

    # ---------------------------------------------------------------- embodiment map (dashed layer)
    s.box(80, 404, 1040, 206, None, None, dashed=True)
    s.text(94, 426, "EMBODIMENT MAP: WHAT THE POLICY SEES IS NOT WHAT THE BODY SPEAKS", cls="mono muted",
           size=10.5, spacing="0.05em")
    s.box(LX, 442, BW, 150, "sim dialect",
          "joints 1 to 6 read in radians under MuJoCo names; the map converts to the model's units and "
          "back, and renames the cameras", size=14, subsize=12)
    s.chips(LX + 14, 532, ['state_keys=["1", "2", "3", "4", "5", "6"]'])
    s.chips(LX + 14, 562, ["radians, converted", 'obs_rename: "front", "wrist"'])
    s.box(RX, 442, BW, 150, "driver dialect",
          "the lerobot driver reports shoulder_pan.pos to gripper.pos already in degrees; the map binds "
          "them as they are and renames the cameras", size=14, subsize=12)
    s.chips(RX + 14, 532, ['state_keys=["shoulder_pan.pos", ..., "gripper.pos"]'])
    s.chips(RX + 14, 562, ["degrees, bound", 'obs_rename: "front", "wrist", "top"'])

    # ---------------------------------------------------------------- the two robots
    s.section(LX, 650, "simulation")
    s.box(LX, 662, BW, 84, 'Robot("so101")',
          'MuJoCo, add_camera("front") and add_camera("wrist"); runs on a laptop', size=14, subsize=12)
    s.section(RX, 650, "hardware")
    s.box(RX, 662, BW, 84, 'Robot("so101", mode="real")',
          'the lerobot driver on a USB port, cameras={"front": ..., "wrist": ..., "top": ...}; execute waits for a yes',
          size=14, subsize=12)
    for x0 in (LX, RX):
        s.arrow([(x0 + 190, 592), (x0 + 190, 662)], label="actions", label_dx=-10, label_dy=4, label_anchor="end")
        s.arrow([(x0 + 290, 662), (x0 + 290, 592)], label="observation", label_dx=10, label_dy=4)
    # the same call, drawn twice, down the outside of the map
    s.arrow([(C, 320), (66, 320), (66, 704), (LX, 704)], label="run_policy(policy_object=policy)",
            label_dx=0, label_dy=-8, label_anchor="middle")
    s.arrow([(C + CW, 320), (1134, 320), (1134, 704), (RX + BW, 704)], label="the same line, the other body",
            label_dx=0, label_dy=-8, label_anchor="middle")

    s.footnote(786, 'the checkpoint never sees a robot; the map is the only thing that knows which one it is. "dim_policy": "pad" fills the state the model expects.')
    return s
