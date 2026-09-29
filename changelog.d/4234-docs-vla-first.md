### Docs: the site leads with running VLA checkpoints on real robots

A reader could go Home, Start, First real arm, `robots/so101` and never meet
the words VLA, checkpoint, SmolVLA, Pi0, ACT, GR00T or `run_policy`. The
landing hero now says what the package is for and its code pair is one Hub
checkpoint (`robotfuel/act_so101_t16b`) run in MuJoCo and, as a sketch, on the
physical SO-101 through `HardwareRobot.run_policy(create_policy(...))`;
Policies is the first card. A new Start page, First policy on the real arm,
runs `lerobot/smolvla_base` on the simulated SO-101 from three cameras with the
inline native embodiment, shows the identical call for the real arm, and the
tool's `execute` stopping at the operator gate with the checkpoint named in its
question. Every generated robot page ends with "Policies that ran on this
robot", read from `docs/hooks/data/checkpoints.json`, where each row carries
the script, fence, log, issue or pull request its numbers came from; a robot
without a verified checkpoint says so and links the recording page. Policies
moves above Agents under Learn, and its index opens with which checkpoints
ran where and the three open gaps (#4157, #4180, #4159); the `lerobot_local`
page's own example, which refused as written (#4157), is now the shape that
runs and is executed by `check_fences.py`, and its dead `lerobot/act_so101`
id (#4158) is gone. The README's first paragraph says the same sentence. The
site word ceiling is raised once, 46,418 to 49,800, for the new page and the
per-robot section.
