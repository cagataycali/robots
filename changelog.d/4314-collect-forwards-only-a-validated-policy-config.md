### Fixed: `/api/collect` forwards only a `policy_config` the wire would accept

The dashboard's collect route now runs `policy_provider` and `policy_config`
through the same validator the mesh applies to an `execute`: the provider,
policy type, policy host, Hub reference and model path are allowlisted, a
`model_path` is contained under the checkpoint homes, and a key the wire does
not carry is refused by name. Before, both fields reached `create_policy` in the
child process untouched. (f028)
