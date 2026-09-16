### Fixed: `stop_policy` says when the rollout it found nothing to stop had died

`start_policy` answers `Policy started` before its worker has built a policy or
taken a step, so a rollout that fails after that fails where no caller is
looking - the reason lands in `_rollouts_ended_in_error`, which only
`list_policies_running` read. That is not the verb an agent reaches for: every
rollout gate names `stop_policy` as the way out, and both of its "nothing is
running" answers were silent about the failure, which is the same reading as a
rollout that ran to completion. Measured on `MuJoCoSimEngine` with a policy that
raised on its first inference: `stop_policy` answered `stop_policy requires
'robot_name'. No policy is running now; robots: 'so101'.`, and following that
advice answered `status="success"` with `Was not running on 'so101'`. Both now
carry the reason, rendered from one seam shared with `list_policies_running` so
the two readers cannot drift. The verdict is unchanged - a stop with nothing in
flight is still an idempotent success with `was_running=False` - and a backend
that records no failure (the ABC default) appends nothing.
