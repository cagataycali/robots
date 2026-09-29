### Fixed: the mock refuses a misspelled `policy_config` keyword like every other provider

`MockPolicy` took `**kwargs` and declared nothing, so
`policy_config={"amplitud": 0.5}` ran the sinusoid on its default without a
word while the docs promise a misspelled keyword is refused before anything
runs. It now declares `amplitude` (the sinusoid's peak, finite and
non-negative) and `seed`, the factory's did-you-mean screen applies to it, and a
test types one typo into each of the 13 registered providers. Closes #4165.
