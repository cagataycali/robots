### Fixed: a positive-number refusal names the whole domain, so `inf` is no longer told it is not `> 0`

`positive_finite_number_error`, the shared guard behind `hz=`, `control_frequency=`,
`duration=`, timeouts, learning rates and the other positive-finite knobs,
answered every refusal with `must be > 0`, including `inf` (which is `> 0`),
`nan`, `None` and strings. Every refusal now reads
`<param> must be a positive finite number, got <value>.`, matching the wording of
`positive_whole_number_error` and `positive_count_error`. Values past the float64
range keep their own `must be within the range of a 64-bit float` text. Which
values are accepted is unchanged.
