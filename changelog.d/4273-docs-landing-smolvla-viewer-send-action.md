### Docs: the landing runs SmolVLA, and the viewer speaks `send_action`

The home page's "The same checkpoint, sim or real" pair now runs
`lerobot/smolvla_base`, a vision-language-action model that reads the three
cameras and the instruction, in place of the ACT checkpoint
`robotfuel/act_so101_t16b`, which had no language input; the ACT rows move
to the `lerobot_local` page, where they were already documented. The 3D
viewer's code card mirrored the sliders as `robot.act({...})`, a method
the `Robot` object does not have; it now writes `robot.send_action({...})`,
the call every fence on the site uses. First learned policy loses a few
words so the site stays under its word ceiling.
