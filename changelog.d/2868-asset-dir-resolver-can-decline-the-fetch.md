### Fixed: the asset directory resolver can decline the fetch it cannot use

`resolve_model_dir` reads the filesystem. It returns a directory that already exists on a search path
and never downloads the asset, so the only call in it that can reach the network is the shared
registry lookup - which falls back to `robot_descriptions` discovery for any name the curated
registry does not know, and that import *is* the fetch, because `robot_descriptions` calls
`clone_to_cache` at module scope.

That left the resolver that cannot download as the one whose caller could not decline a download.
Asking where a discovery-only robot's directory is clones the upstream asset corpus, while the same
question about a curated robot clones nothing, and the resolver reaches the asset downloader zero
times where its sibling `resolve_model_path` reaches it once for the same absent robot.
`discover_robot` is documented "Call only from asset-resolution paths that are allowed to download",
and this is not one of them.

`resolve_model_dir` now takes `allow_download`, spelled and forwarded exactly as `resolve_model_path`
has done since the fetch was first made declinable, so `allow_download=False` is a pure filesystem
lookup with no network and no `robot_descriptions` import. The default stays open, so the long tail
still resolves for a caller about to load a model and no existing answer changes.

The one in-tree consumer documented the opposite. The cuRobo planner example reported that
`resolve_model_dir` "triggers the same auto-download the MuJoCo path uses, so a clean box with
internet populates the cache here too"; it triggers none, so on a cold cache the helper returned no
URDF and sent the reader back to the manual `--curobo-urdf` flag its own docstring says it removed.
It now declines the network for its best-effort probe and, on a miss, downloads through the capable
sibling and re-resolves the directory - rather than taking the model XML's parent, which is not the
asset directory for the robots that nest their XML in a subdirectory.
