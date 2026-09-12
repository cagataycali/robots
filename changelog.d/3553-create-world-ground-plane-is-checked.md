### Fixed: `create_world` checks `ground_plane` instead of reading it by truthiness

The MuJoCo and Newton backends validated every knob beside `ground_plane`
(`timestep`, `gravity`, `terrain`, `difficulty`) and read this one by
truthiness, so `"false"`, `"no"`, `"off"` and `"0"` laid the floor the word
declines and `None`, `0`, `""` and `[]` omitted it without being a declared
spelling - every one reporting `status="success"`. The flag is now held to the
shared `boolean_flag_error` domain through the class's existing posture-flag
envelope, ahead of the world-exists report that reads it and ahead of the
build, so a misread posture is refused under its own name and the refused call
builds nothing. The Isaac backend reads the same flag and follows once the open
Isaac backend rewrite lands.
