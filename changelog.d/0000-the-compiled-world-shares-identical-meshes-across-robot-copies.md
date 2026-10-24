### Fixed: robots of the same type share one copy of each mesh in a MuJoCo world

The MuJoCo backend merges identical meshes and textures (`MjSpec.dedup_assets`) before it compiles the world, so N
so101 arms compile to the 13 meshes of one so101 instead of 13 per arm. Each added so101 now costs 13-32 MB resident,
from 65-93 MB. Robots of different types keep their own meshes. The `mujoco` extra now needs MuJoCo 3.17 or newer, the
first release with `dedup_assets`.
