### Docs: learned policies on real robots, said where readers look

A reader could go Home, Start, First real arm, `robots/so101` and never meet
the words checkpoint, SmolVLA, Pi0, ACT, GR00T, Cosmos or `run_policy`. The
site's framing stays; the policies are added where a reader looks. The landing
lead gains one sentence, a seventh card, Policies, joins the six, and a second
section below the untouched send_action pair, "The same checkpoint, sim or
real", runs one Hub checkpoint (`robotfuel/act_so101_t16b`) in MuJoCo and, as a
sketch, on the physical SO-101 through `HardwareRobot.run_policy(create_policy(...))`.
A new Start page, First learned policy, runs `lerobot/smolvla_base` on the
simulated SO-101 from three cameras with the inline native embodiment, shows
the identical call for the real arm, and the tool's `execute` stopping at the
operator gate with the checkpoint named in its question; one line says the
same shape runs a Cosmos or WBC provider. Every generated robot page ends with
"Policies verified on this robot", read from `docs/hooks/data/checkpoints.json`,
where each row carries the script, fence, log, issue or pull request its numbers
came from; a robot without a verified checkpoint says so and links the recording
page. Policies moves above Agents under Learn; its index keeps its opening and
gains "Which policies run where" across every family (vision-language-action
models, world foundation models, whole-body controllers, reinforcement learning,
remote and planning) with the three open gaps (#4157, #4180, #4159); the
`lerobot_local` page's own example, which refused as written (#4157), is now
the shape that runs and is executed by `check_fences.py`, and its dead
`lerobot/act_so101` id (#4158) is gone. The README's opening paragraph gains
one sentence. The site word ceiling is raised once, 46,418 to 49,800, for the
new page and the per-robot section.
