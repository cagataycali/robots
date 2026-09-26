### Tests: the RL engine cells share one stand-in, bound to the SimEngine seam

Eleven modules under `tests/training/` each carried a duck-typed double for the
engine `strands_robots.training.rl.SimEnv` drives -- `_FakeEngine` six times,
plus `_StandInEngine`, `_OneJointEngine`, `_CountdownEngine`, `_Recorder` and
`_TwoVocabEngine` -- and every one of them restated the five-method seam itself.

A double that declares its own signature is bound to nothing. `SimEngine`
publishes `list_robots`, `robot_action_keys`, `get_observation`, `reset` and
`send_action`, and `robot_action_keys` is the one that is not abstract: the base
defines it to mirror `robot_joint_names`, which is what a backend whose actuator
set matches its joint set inherits. None of the doubles inherited it, so four
restated it under the same five-line comment explaining that, two omitted it
entirely, and three call sites needed a `cast` or a `type: ignore[arg-type]` to
present the double as the seam. Renaming `send_action`'s `n_substeps` keyword on
the base, with no call site updated, left all 300 cells green.

`tests/training/_engine_stand_in.py` now holds one `EngineStandIn` that
subclasses `SimEngine`: the action-key default is the production one, mypy
checks every override against the published signature, its refusals are the
backend rules (a vector whose width is not the action-key count, a non-finite
command) rather than test-local inventions, and the nine methods the RL stack
does not reach raise instead of quietly answering.

`tests/training/test_the_rl_stack_reaches_the_engine_surface_it_declares.py`
reads the engine call sites out of `strands_robots/training/rl/` and binds each
against `SimEngine`, so that renamed keyword now names the call site it broke.
Coverage of `rl/env.py`, `rl/gym_env.py` and `rl/vec_env.py` is unchanged over
the same uncovered lines.
