### Fixed: `aloha` no longer declares a Feetech lerobot type for its Dynamixel servos

`aloha` is two ViperX 300s arms on Dynamixel Protocol 2.0, and its registry
entry declared `hardware.lerobot_type = "bi_so_follower"` - two SO-100/SO-101
followers on a Feetech STS3215 bus. `Robot("aloha", mode="real")` built that
`BiSOFollower`, so the arm a caller got spoke a wire protocol its servos do not
answer. lerobot ships no ALOHA robot, so there was nothing correct to point at:
the entry is now simulation-only, and `mode="real"` refuses by name instead,
reporting the native Dynamixel driver registered for it.

`bi_so_follower` was also an *alias* of `aloha`, so the owner of two SO-101 arms
who asked for that name got the 16-joint ViperX twin. It is now a robot of its
own - `bi_so_follower`, aliases `bi_so100`/`bi_so101` - carrying the
`bi_so_follower` lerobot type that describes it. Both SO spellings resolve to
one entry because lerobot registers `so100_follower` and `so101_follower` on a
single config class.

`test_every_strands_lerobot_type_is_real` could not see this: `bi_so_follower`
is a name lerobot really registers, which is all that check asks.
`test_a_declared_lerobot_type_drives_the_motor_family_the_description_names`
now derives each lerobot type's `MotorsBus` from lerobot's own source and
refuses an entry whose description names the other servo family.
