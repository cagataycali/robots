### Fixed: `annotate_episode` / `filter_episodes` / `measure_agreement` typo refusals print a `Did you mean 'X'?` hint

The three `episode_labels` writers (`annotate_episode`, `filter_episodes`,
`measure_agreement`) refuse a `quality` or `failure_mode` outside the fixed
vocabulary by naming the vocabulary — but used to stop there, while 17+ sibling
refusals across the codebase (`robot.py`, `policies/factory.py`,
`hardware_robot.py`, `simulation/base.py`, `simulation/predicates.py`,
`training/factory.py`, ...) all follow the same enumerate-and-arrow convention
(`difflib.get_close_matches(cutoff=0.6, n=1)`). A user following the documented
`docs/learn/data/label-and-judge.md` recipe who lands on a one-letter typo
(`quality="hi"`), a hyphen vs. underscore (`failure_mode="near-miss"`), or a
short form (`failure_mode="occlusion"`) now gets the same one-token nudge
every other refusal in the project ships. Non-matches (`"excellent"`,
`"sloppy"`) keep their byte-exact historical message — the suffix is additive.
