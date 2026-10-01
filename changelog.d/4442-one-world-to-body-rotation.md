### Changed: Newton and Isaac body-frame `base_ang_vel` through one rotation

The Newton and Isaac backends now rotate a floating base's world-frame angular
velocity into the body frame with the same helper the `base_velocity` reward
term uses (`strands_robots.simulation.predicates`), instead of one private copy
each. The reported values are unchanged; a future fix to the rotation now lands
on both backends and the reward DSL at once.
