### Fixed: the benchmark-authoring example scores a folded Go2 as the fall it is

`examples/11_author_a_benchmark.py` is the page a reader copies to write their
own task, and its Go2 spec still carried the two numbers the shipped specs
dropped in #3918: `base_below_z(0.18)`, a line no collapse reaches, and a
`base_height` target of 0.32 m, a stance this asset does not have. Run as
shipped, both episodes of a folded Go2 scored `failure: false` over 800/800
steps at reward 387.7 while the trunk lay at 0.20 m for 97% of the frames.
Grounded in the asset like the built-ins (fall line 0.22 m, height target the
0.27 m `home` keyframe), the same rollout is scored 2/2 failures at step 21,
reward 10.1. The pin that caught this now reads every Go2 spec the tree ships -
the built-ins plus the example and notebook copies - so a fourth copy cannot
drift.
