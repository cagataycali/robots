### Docs: the ProtoMotions page names the install extra and the reference MJCF it needs

`docs/policies/protomotions.md` opened with `pip install
"strands-robots[protomotions]"`, an extra that declares no MuJoCo, so both of the
page's own examples refused: the simulation fence raised `ImportError: 'mujoco'
is required for MuJoCo simulation`, and `qpos_to_motion_data` parses the
reference MJCF through the same package. `proto_mjcf_path` then required a
33-body ProtoMotions G1 model that this package bundles none of, while naming no
file to fetch. Of the five upstream candidates only
`g1_bm_no_mesh_box_feet.xml` loads from a single download: three more reference
the Git LFS `mesh/G1` tree (a `raw.githubusercontent.com` copy of those meshes is
a pointer file, which MuJoCo refuses with `decoder failed for mesh file`), and
`g1_holo.xml` is refused for the three bodies it lacks. All four that load give
byte-identical `body_pos`, so the no-mesh variant costs the bridge nothing, and
the page now says so with the `curl` line beside it.
