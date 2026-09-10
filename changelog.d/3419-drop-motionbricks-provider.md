### Removed: the `motionbricks` policy provider

`create_policy("motionbricks")` now raises the registry's normal
unknown-provider `ValueError` naming the 13 providers that remain. The provider
wrapped NVlabs' `GR00T-WholeBodyControl/motionbricks` generative motion model,
which is not published on PyPI: the `[motionbricks]` extra installed only the
PyPI support libraries its model code imports, and the model itself had to be
cloned and `pip install -e`'d from the upstream checkout with its checkpoints
pulled through git-LFS. Nothing in the package imported it.

Removed with it: `strands_robots/policies/motionbricks/`, its tests, the
`registry/policies.json` entry, the `[motionbricks]` extra and its `[all]`
membership, the mypy `ignore_missing_imports` rows, the `motionbricks` pytest
marker, `docs/policies/motionbricks.md` and its mkdocs nav entry, the
`examples/wbc/motionbricks_g1_mujoco.py` demo and the `tests_integ` live suite.
Dropping the extra also drops `adam-atan2-pytorch`, `hydra-core`, `omegaconf`,
`pytorch-lightning` and `vector-quantize-pytorch` from `[all]`; no other module
imports any of them, and `uv.lock` loses 10 entries (those five plus five
transitive).

Four shared rules used the departed provider as their second witness and are
re-anchored on a shipped one rather than dropped:

* `MAX_TARGET_VELOCITY_COMPONENTS`'s reason for not owning the arity verdict is
  now WBC (needs at least three components, reads the first three) against
  `microduck` (accepts exactly two or three) - the same disagreement, so the
  wire still caps for DoS only rather than deciding a width.
* `Policy.set_robot_state_keys`'s "already total without the shared domain by
  resolving every joint by NAME" carve-out is `WBCPolicy` alone, and
  `name_list_error` and `download_robots` state it that way too.
* The `int()`-normalisation ordering witness in the ProtoMotions body-index
  tests is `WBCConfig.__post_init__`, which states the reason in as many words.
* The goal-kwargs single-definition negative control keeps three members:
  `target_heading` lost its only reader, so it is re-pointed at `command`
  (`microduck` alone), which sits in the same `get_actions` body as
  `target_velocity` in one of the two families that make that key shared.

`target_velocity` remains shared vocabulary on the ABC - `wbc` and `microduck`
are the two families that read it - so the ABC's well-known goal-key block is
unchanged.
