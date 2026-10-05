### Fixed: a misspelled keyword on `randomize`, `set_obs_noise` or `add_object` gets a suggestion

The sim methods that take `**kwargs` refused an unknown key as
`Unknown parameter(s) ['randomize_colours'] for action 'randomize'. Valid: [...]`,
with no hint, while every other action already answered the same typo with
`Did you mean: randomize_colors?`. Both now give one sentence, suggestion included,
and a refusal naming several keys still names each of them.
