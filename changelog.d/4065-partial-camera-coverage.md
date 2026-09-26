### Fixed: a robot with fewer cameras than a checkpoint declares reaches the policy

`lerobot/smolvla_base` declares `observation.images.camera1..3`, so a one-camera
arm was refused with a remedy it could not follow ("Add the missing camera(s) to
the observation"). lerobot does not require them all: the flow-matching VLAs
build their view list from the declared features PRESENT in the batch, pad the
rest up to `config.empty_cameras`, and refuse only when none is present. Both
camera routers now share one rule -
`resolution.accepts_partial_images()` names the families that run on the views
they were given (`smolvla`, `pi0`, `pi05`, `pi0_fast`, `xvla`), which route what
there is and log a WARN naming the absent features. Every other family, an
unresolved policy type, and an observation with no camera at all keep the
refusal that names both sides, because those checkpoints index each declared
feature and the alternative is a `KeyError` raised inside lerobot.
