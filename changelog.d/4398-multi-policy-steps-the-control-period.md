### Fixed: a `run_multi_policy` step advances one control period, as `run_policy` does

The synchronized multi-robot loop stepped physics once per control step: on
MuJoCo's default 2 ms dt, 50 steps at 50 Hz covered 0.1 s of sim time instead
of 1.0 s, each position servo moved a tenth of the way toward its target (an
elbow reached 0.040 of a 0.6 rad command in 0.5 s; now 0.544), and a recording
through it stamped frames 1/fps apart while the sim had moved 2 ms. Isaac had
the same shape with one 1/120 s tick per step. Both backends now step the
control period's physics steps, derived the way `run_policy` derives them.
