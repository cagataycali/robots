### Fixed: `create_policy()` names each suggested provider once, by the name it routes to

The "Did you mean" hint for an unknown provider searched provider names and
their aliases together and printed the raw matches, so one typo could list two
spellings of the same provider (`'gtp', 'gtp_g1'` for `GTP`) or hide a
different provider behind a shorthand (`'text2motion'`, which builds `kimodo`,
offered for a typo of `protomotions`). Each match is now named by its canonical
provider, once, in near-match order: `GTP` suggests `'protomotions'`, and
`protomotion` suggests `'protomotions', 'kimodo'`.
