### Fixed: an in-process GR00T load refuses with the fact that stopped it

`Gr00tPolicy(model_path=...)` answered three different failures with one
sentence - `ImportError("Isaac-GR00T not installed. Use service mode
(host/port).")` - and one of them with no sentence at all.

`groot_version=` selects which loader runs, so a value naming no release matched
no dispatch branch and fell through to that message. Measured against a `gr00t`
package that *is* importable and auto-detected as `n1.7`,
`groot_version="N1.7"` reported the environment as lacking Isaac-GR00T: a false
statement about the machine, answered with an install instruction for a package
the caller already had, and never naming the parameter that caused it. The
domain is now `strands_robots.utils.SUPPORTED_GROOT_VERSIONS`, graded by
`groot_version_error` on the branch that reads it - service mode loads no
checkpoint, so it is validated where `port` is not, and vice versa.

Forcing a release also skipped the missing-package check outright: with no gr00t
installed, `groot_version="n1.7"` reached `_load_n17`'s own `from gr00t...` line
and surfaced `ModuleNotFoundError: No module named 'gr00t'`, while leaving it
unset answered the identical missing package with the actionable message. The
caller who supplied more information got the worse error. Detection now gates
the load for both spellings of the request.

And the actionable message named one route where two are open. "Install
Isaac-GR00T" is the one instruction a `strands-robots[all]` caller cannot
follow: no extra declares `gr00t` - it installs from
`github.com/NVIDIA/Isaac-GR00T` - and it pins `transformers==4.57.3` while
lerobot needs `transformers>=5`, so the two cannot be imported in the same
Python process. Both refusals now name the routes reachable from a declared
extra: `create_policy("groot", host=..., port=...)` for the Isaac-GR00T
container, and `create_policy("lerobot_local", policy_type="groot",
pretrained_name_or_path=...)` for lerobot's own GR00T N1.7. `docs/policies/groot.md`
documents both under "In-process inference", where it previously showed
`model_path=` as available from `[groot-service]` alone.

Falling off the end of the dispatch is now reported as the release being
unidentifiable rather than the package being absent - which is what it is, for a
gr00t install whose layout none of the three probes recognise - and names
`groot_version=` as what resolves it.
