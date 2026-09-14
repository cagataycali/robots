### Fixed: `eval_policy` / `evaluate_benchmark` refuse an evaluation that never commanded the robot

Both evaluation routes tolerate a policy call that returns an empty action
chunk - they advance one physics step so a degenerate policy cannot hang the
episode. That per-step tolerance is right, but nothing decided the aggregate:
when *every* call of an evaluation came back empty, `send_action` was never
reached and the run still reported `success_rate`, `avg_reward` and
`pass_hat_k` under `status="success"`. Measured on an SO-101 in MuJoCo, every
published field was identical between a policy that applied 0 actions and one
that applied 45 - so a results table could not tell an unexercised policy from
one that was exercised and scored zero. The sharpest case published
`success_rate: 1.0` for a policy that commanded nothing at all.

`run_policy` already refuses an empty chunk on the first occurrence and
`replay` refuses the aggregate of the same condition for a recorded episode
whose frames carry no action, so the two eval routes were the only consumers of
that condition with no verdict on it. The rule now has one owner
(`uncommanded_eval_error`) that both routes call. Both report `actions_applied`
beside `steps_advanced` in the result json and per episode - an advanced step is
not a commanded action, so one number cannot carry both facts - and return
`status="error"` when `actions_applied` is zero. A *partial* shortfall stays a
reported count rather than a refusal, since some empty calls are real policy
behaviour and refusing them would contradict the per-step tolerance.
