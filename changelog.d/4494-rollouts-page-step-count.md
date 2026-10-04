### Fixed: the predicates-and-rollouts page shows the step count its sketch prints

The "You should see" block on `docs/learn/simulation/predicates-and-rollouts.md`
said the `joint_above` stop fired after 20 applied actions; the sketch above it
prints 19. A test now runs the page's sketch and compares its output to the block
line for line.
