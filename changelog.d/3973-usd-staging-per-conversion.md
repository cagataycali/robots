### Fixed: two concurrent MJCF conversions each get their own staging directory

`convert_mjcf_to_usd` (strands_robots/simulation/isaac/mjcf_assets.py) staged a
conversion in `.<key>.<pid>.tmp` beside the cache entry. The pid alone does not
identify a conversion: two THREADS of one process converting the same
description derive the same name and convert into the same directory. That is
not a benign sharing of identical output -- publishing is
`os.rename(staging, target_root)`, so the winner moves the shared directory onto
the key and the loser's importer output disappears mid-call. The loser then
raises

    the Isaac MJCF importer reported success for '...' but wrote no USD file
    under '...' (it returned '.../probe/probe.usda')

for a conversion that in fact succeeded, naming the vendor importer for a race
in the cache. One process is enough, so a parallel test session or a host
launching several sims from one process hits it; the first conversion of a cold
cache is the only one that can.

`tempfile.mkdtemp` now names the staging directory, so no concurrent caller can
derive it, and the pre-clean the derived name needed goes away. The loser still
finds the winner's published entry and defers to it, which is the behaviour
`_install_entry` already documents.
