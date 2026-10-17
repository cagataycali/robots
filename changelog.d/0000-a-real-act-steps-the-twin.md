### Changed: a real `act()` steps its twin's world; `RequestRefused` is a top-level name

`act()` on a hardware Robot used to write the bus and mirror the targets into `real.twin` without stepping the
twin's world, so the twin kept its old pose until the caller stepped `real.twin.sim` by hand. It now advances
that world one control step, as `act()` on a simulated body does; `manual_time()` still suppresses it and
`observe()` mirrors without stepping. `from strands_robots import RequestRefused` catches every refusal (the
operator gate, a validator, the latched e-stop, whose `EStopped` is a subclass); it is still a `PermissionError`.
