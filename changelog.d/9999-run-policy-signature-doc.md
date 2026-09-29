### Fixed: docs promised `run_policy` is portable across sim and hardware; the two signatures differ

`docs/reference/api/robot.md` opened with "Both expose the same agent-facing surface: `act`, `observe`, `run_policy`, `cleanup`," but `SimEngine.run_policy` and `HardwareRobot.run_policy` accept different first positional arguments — the sim binds a Policy instance to `robot_name` and refuses it as a missing robot, naming the Python `repr()` of the policy where a docs-side-by-side reader expected `policy_object=` to be recognised.

Rewrote the paragraph to name the verbs shared and the two `run_policy` signatures that diverge, so the reader learns the mismatch before the sim's "Robot '\<...MockPolicy object at 0x...\>' not found" error names it for them.

Behaviour unchanged; a targeted refusal that reads "run_policy: the first positional is `robot_name` on sim and `policy_object` on real hardware — did you mean `policy_object=`?" is deliberately left as a follow-up, so the docs change and the behaviour change land on separate PRs.
