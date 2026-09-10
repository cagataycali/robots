### Fixed:
- `Robot(..., mode="real")` on an install without the `[lerobot]` extra now raises the tree's standard install hint (`pip install 'strands-robots[lerobot]'`) instead of a bare `ModuleNotFoundError: No module named 'lerobot'` from inside `hardware_robot._initialize_robot`.
