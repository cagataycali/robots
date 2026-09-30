### Changed: the Isaac backend imports Isaac Sim's deprecated core APIs from one module

Isaac Sim 6.1 moved the core-API extensions this backend uses into `extsDeprecated`. Every such import now goes through `strands_robots/simulation/isaac/_deprecated_api.py`, and the dead `omni.isaac.*` fallbacks for Isaac Sim 4.x are removed, so migrating off the deprecated APIs is a one-module change.
