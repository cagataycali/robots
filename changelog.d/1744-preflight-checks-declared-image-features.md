### Fixed: the camera pre-flight check refuses an `image_keys` list that withholds a fed feature

`Policy.preflight` exists to catch a camera-routing mismatch before the
multi-minute weight download, but it only checked one direction: that every
image rename TARGET the embodiment declares has a source key present in the
runtime observation. It never checked that those targets are features the model
will actually declare.

An explicit `image_keys=` is priority 1 in `derive_image_keys`, so it replaces
the feature list that would otherwise be derived from the embodiment's
`obs_rename` targets. A list that does not cover them therefore builds a model
without the inputs the embodiment routes, and `EmbodimentMap.validate` refuses
exactly that -- but only after the download, and on the MolmoAct2 load path the
call is unguarded, so the load aborts. Measured with `embodiment='so_real'`
(which feeds `observation.images.image` and `observation.images.wrist_image`),
both `image_keys=['base', 'wrist']` and `image_keys=['observation.images.top']`
passed pre-flight and then failed post-download.

Both sides of that verdict are already in `policy_config` before anything is
fetched, so pre-flight now reports it, naming the features the embodiment feeds,
the list the caller declared, and three remedies. The check applies to the
MolmoAct2 load path only, where `image_keys` is honored -- it is documented as
inert for every other policy type -- so a list another checkpoint would ignore is
not refused.
