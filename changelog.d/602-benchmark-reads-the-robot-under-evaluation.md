### Fixed: benchmark clauses evaluate the robot under evaluation

In a multi-robot scene, unnamed `base_*` predicates (benchmark success /
failure / dense_reward, `stop_when`) read the first registered robot instead
of the one `robot_name` named, and the benchmark compatibility check refused
for any bystander robot. `run_policy` / `eval_policy` / `evaluate_benchmark`
now bind the robot they resolved (`SimEngine.bind_predicate_robot`); unnamed
clauses and the compat check read that robot. `evaluate_benchmark(
benchmark_name='go2_walk_forward', robot_name='go2')` runs with an arm in the
scene, and the refusal names the robot under evaluation.
