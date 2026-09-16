### Fixed: the policy surface can be found from the schema and from a wrong first guess

An unknown parameter is now answered with the nearest valid keys, and the
"Valid" list names only parameters a tool call can carry (no callbacks or
live objects). The `policy_provider` schema entry lists every registered
provider and says the model itself goes in `policy_config`; `instruction`
says it is the task the policy is conditioned on.
