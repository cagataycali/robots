### Fixed: N robots of the same type share one mesh in the compiled world

The MuJoCo backend now calls `spec.dedup_assets()` before compile, collapsing
identical mesh, texture, and material assets that would otherwise be
re-materialised once per attached robot. The compiled `mjModel` now holds one
copy of each so101 mesh (and its BVH) instead of N — not just one copy of each
spec, but one copy end-to-end. On top of the earlier fix that dropped the
retained `MjSpec`, five so101 arms now take 643 MB resident with
14 MB per added arm (down from 894 MB total / 78 MB per added). `nmesh` falls
from 65 to 13, and the per-world BVH drops from 148 MB to 29 MB. Mixed-type
worlds dedup within each type (3 so101 + 1 g1 → 48 meshes shared vs 174).
