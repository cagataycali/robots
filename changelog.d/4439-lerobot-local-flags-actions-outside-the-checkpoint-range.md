### Fixed: `lerobot_local` flags actions far outside the range its checkpoint's action stats record

A π0-FAST LIBERO checkpoint whose action stats span `[-0.94, 1.0]` emitted a pitch of
131 and a gripper of 2062 on an out-of-distribution scene, and strands forwarded them
unchecked - a joint-space policy would drive the robot there. After unnormalization
each action is now compared, column by column, with the `[low, high]` the checkpoint's
stats record (`min`/`max`, else `q01`/`q99`; only the columns both carry, so a model
padded to 32 dims is not misread): a value more than one recorded range beyond it is
logged once per episode with the columns and values (`out_of_range_actions="warn"`, the
default), clipped to the recorded range (`"clip"`), or ignored (`"off"`);
`out_of_range_action_count` counts them.
