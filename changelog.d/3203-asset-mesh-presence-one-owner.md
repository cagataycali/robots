### Fixed: mesh presence is answered by the owner of that question

"Are this model's meshes on disk?" has one owner,
`strands_robots.assets.download._mjcf_missing_meshes`, which resolves each
`file=` reference against the model's `<compiler meshdir>` the way MuJoCo does,
read once across every `<include>`d fragment. Two readers ask it: the download
decision, and the check that gates `add_robot`.

`resolve_model_path` is the third reader and documents the same question as its
own second download trigger -- "XML is found but mesh files are missing,
downloads the asset". It answered that by walking the model directory for files
with a mesh extension, a different reading, and the two disagree in both
directions:

| layout | owner | directory walk | resolver fetched | MuJoCo |
|---|---|---|---|---|
| meshes beside the model | complete | complete | no | loads |
| meshes one level down (`meshdir="assets"`) | complete | complete | no | loads |
| **meshes one level up (`meshdir="../meshes/"`)** | **complete** | **none found** | **yes, every call** | **loads** |
| **declares no meshes** | **nothing to fetch** | **none found** | **yes, every call** | **loads** |
| **one of two declared absent** | **1 missing** | **complete** | **no** | **fails** |
| the only declared one absent | 1 missing | none found | yes | fails |

The upward `meshdir` is a shipped layout, not a hypothetical: `aliengo`,
`unitree_a1`, `jvrc` and `asimov_v0` all ship as `<robot>/xml/<model>.xml`
declaring `<compiler meshdir="../meshes/"/>`, with the meshes in
`<robot>/meshes/`. All four load in MuJoCo (5, 5, 70 and 15 meshes). Every one
of them drew a `robot_descriptions` fetch on *every* call to
`resolve_model_path`, and the fetch could never satisfy the condition it was
reaching for, because a download does not move meshes into the directory being
walked. Over the 61 registry robots resolvable on a warm cache, the resolver and
the download decision disagreed about exactly those four.

The other direction is the documented trigger silently not firing: a model
missing one of the meshes it declares still has its other meshes on disk, so the
walk reported it as fine, no fetch was attempted, and the caller was handed a
path MuJoCo refuses to load. `MuJoCoSimEngine.add_robot` re-asks the owner and
recovers; `NewtonSimEngine`, whose only mesh reading is this resolver, does not.

The resolver now asks the owner. The walker and its mesh-extension frozenset are
removed rather than corrected: a second copy of the rule is what let the readings
drift apart, and the correct predicate is the one MuJoCo itself applies. An
unreadable model is read as "fetch", which is the reading the download decision
already takes of the same failure. Cost is ~1 ms per candidate against the
network fetch it removes.

Three existing cells named the second trigger while their fixture wrote a model
declaring *no* meshes -- the case the walk and the owner happen to agree about,
and the exact conflation the defect lived in. They now declare a mesh that is
absent, so their ids mean what they say. The suite that reasoned about a
`<compiler meshdir>` never declared one either; it does now.
