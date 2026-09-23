### Fixed: a hardware task runs the provider's preflight before it builds the policy

`Policy.preflight` judges a configuration on the class, before `create_policy`
constructs anything and therefore before any weight download. Only the sim path
called it, so on the physical arm a configuration the provider refuses without
constructing - a chunk count the consumer cannot execute, an `image_keys` list
that withholds a feature the embodiment feeds, camera names that cannot be routed
to a VLA's declared image inputs - energized the arm, was approved by the
operator, commanded a test motion and was refused only after the download.
`Robot._get_policy` now runs the hook against the arm's own observation first,
and the refusal is reported with the arm disconnected again. The observation is
read only when the resolved class overrides the hook, and a read the arm cannot
serve stays `_initialize_policy`'s to report rather than becoming a verdict on
the policy configuration.
