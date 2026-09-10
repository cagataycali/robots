### Fixed: the last-push-approval check reads a workflow run's `actor`, so approving a held run no longer names the approver as the pusher

`scripts/check_last_push_approval.py` resolved the pusher from `triggering_actor`,
which GitHub rewrites when a maintainer approves a run held at `action_required`
or re-runs one. On a first-time contributor's fork that named the maintainer who
released the CI as the pusher, so their own approval was then discounted and the
pull request was reported as owing a second reviewer while GitHub's
`require_last_push_approval` rule -- which reads the real pusher -- was satisfied.
It now reads `actor`, which is the account whose event created the run.
