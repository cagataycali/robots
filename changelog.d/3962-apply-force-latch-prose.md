### Docs: `apply_force` latches a wrench, it does not push for one step

`examples/14_save_state_and_perturb.py` described `apply_force` as applying a
force "for the next step". The wrench is latched: MuJoCo re-applies it on every
step until the next `apply_force` on that body or a `reset()`. The two readings
predict opposite signs for the example's own numbers - its 2 N on a 50 g cube
held for 20 steps lifts the cube `+23.0 mm`, where a real one-step impulse of
the same force leaves it `-7.4 mm`, having fallen - so the printed result
contradicted the docstring above it, and a reader sizing a disturbance for a
robustness sweep holds a thruster on believing they tapped the object.

Every other caller-facing surface already says latched, so the example is
corrected and the rule is pinned in the file that owns the latch behaviour: each
docs page and example that teaches `apply_force` is graded for a one-step
lifetime claim, with a floor assert so a corpus the grader failed to collect
cannot pass silently.
