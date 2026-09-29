### Fixed: a benchmark accepts its robot added under a registry alias

`add_robot("go2")` then `evaluate_benchmark("go2_walk_forward")` was refused
because the compatibility check compared the alias against
`supported_robots=["unitree_go2"]` literally. Both check sites now fold each
side through the registry's `resolve_name`, so every documented spelling of the
robot runs the benchmark written for it while an unrelated robot is still
refused. Closes #4160.
