---
description: Runnable example scripts — links to the repo's examples/ directory.
---

# Examples

The repository's `examples/` directory contains runnable scripts. Use them as
copy-paste starting points.

## In the repo

Browse them on GitHub:
[`strands-labs/robots/tree/main/examples`](https://github.com/strands-labs/robots/tree/main/examples)

| File | What it does |
|------|--------------|
| `cosmos3_sim_rollout.py` | Full Cosmos 3 sim rollout: spawn SO-100, connect to a Cosmos 3 action-policy server, run episodes, and save a LeRobot v3 recording. |
| `molmoact2_so101_pickplace.py` | SO-101 pick-and-place using MolmoAct2 via `LerobotLocalPolicy` with `norm_tag` / `image_keys` / `inference_action_mode`. Requires hardware + GPU. |
| `mesh_acl_example.json5` | Mesh ACL configuration example: per-peer allow/deny rules for the Zenoh mesh used by Robot mesh wiring. |

## Run them

```bash
git clone https://github.com/strands-labs/robots
cd robots
pip install -e ".[all]"

# Cosmos 3 sim rollout (sim + cosmos3-service extras, Cosmos server running on :8000)
python examples/cosmos3_sim_rollout.py

# MolmoAct2 pick-and-place (requires hardware + GPU)    # requires hardware
python examples/molmoact2_so101_pickplace.py            # requires GPU

# Inspect the mesh ACL config
cat examples/mesh_acl_example.json5
```

Each Python script has a top-level docstring documenting its requirements.

## See also

- [Quickstart](../getting-started/quickstart.md) — minimal copy-paste starter.
- [Cosmos3Policy](../policies/cosmos3.md) — Cosmos 3 provider details.
- [LerobotLocalPolicy](../policies/lerobot-local.md) — MolmoAct2 and other local models.
- [GitHub source](https://github.com/strands-labs/robots) — issues, discussions,
  releases.
