### Fixed: the README and the policies index name both providers behind the `trust_remote_code` gate

The README Security section and `docs/learn/policies/index.md` named only
`lerobot_local` as needing `STRANDS_TRUST_REMOTE_CODE=1`. The gate also covers
`kimodo`; both pages now say so.
