---
description: Runnable example scripts — links to the repo's examples/ directory and the tutorial chapters.
---

# Examples

The repository's `examples/` directory hosts runnable scripts that mirror the tutorial
chapters. Use them as copy-paste starting points.

## In the repo

Browse them on GitHub:
[`strands-labs/robots/tree/main/examples`](https://github.com/strands-labs/robots/tree/main/examples)

| Script | What it does |
|--------|--------------|
| `01_sim_quickstart.py` | The 5-line sim quickstart from the README. |
| `02_sim_agent.py` | Wire `Robot()` into a Strands `Agent`. |
| `03_sim_recording.py` | Record a LeRobot v3 dataset. |
| `04_real_hardware.py` | Bring up a real arm with cameras. |
| `05_real_groot_policy.py` | Drive a real arm with GR00T. |
| `06_list_robots.py` | Walk the registry. |
| `act_policy_simulation.py` | ACT in MuJoCo with video export. |
| `physics_agent.py` | Natural-language physics introspection. |

The exact set evolves; check the repo for the current list.

## Run them

```bash
git clone https://github.com/strands-labs/robots
cd robots
pip install -e ".[all]"
python examples/01_sim_quickstart.py
```

Each script has a top-level docstring documenting its requirements (sim only, GPU,
real hardware, etc.).

## Tutorial alignment

Most examples have a sibling tutorial chapter:

| Tutorial chapter | Example |
|------------------|---------|
| [1 — Your first robot](../tutorial/01-your-first-robot.md) | `01_sim_quickstart.py` |
| [4 — AI agents](../tutorial/04-agents.md) | `02_sim_agent.py` |
| [6 — Recording](../tutorial/06-recording.md) | `03_sim_recording.py` |
| [8 — Real hardware](../tutorial/08-real-hardware.md) | `04_real_hardware.py`, `05_real_groot_policy.py` |
| [Robot catalog](../robots/index.md) | `06_list_robots.py` |

## Beyond the examples

For extended demos — full notebooks, third-party integrations — track the
[GitHub Discussions board](https://github.com/strands-labs/robots/discussions). We
don't bundle large notebook galleries in the repo to keep `pip install` lean.

## See also

- [Tutorial](../tutorial/index.md) — chapters 1–9 with concept commentary.
- [Quickstart](../getting-started/quickstart.md) — minimal copy-paste starter.
- [GitHub source](https://github.com/strands-labs/robots) — issues, discussions,
  releases.
