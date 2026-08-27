### Fixed: resolving an asset path no longer has to fetch the ones that are absent

`resolve_model_path` downloads whatever it cannot find on disk. That is the right
default for a caller about to load the model - a path whose meshes are absent is
useless to MuJoCo, so fetching them is the helpful thing. It is the wrong default
for a caller that reports on assets rather than loading them, and there was no way
to ask for the other one: `is_robot_asset_present` answers *whether* an asset is on
disk without a download, and nothing answered *where* under the same terms.

Two callers wanted it, and both had already noticed. `list_available_robots`
hand-rolled a guard for the first of the resolver's two download triggers - "Only
resolve full path when asset is present - avoids download attempts" - and still
fetched for the second, because a cached XML whose meshes are missing passes the
presence check. Building a status listing on this machine's cache downloaded 4
robots. And `tests/registry/test_asset_family_joint_counts.py` re-derives the
asset-family grouping by walking all 72 registry entries, so it fetched every
asset the machine did not already have: 63 on a fresh checkout, 58 recorded
`attempting auto-download` lines and 11 real clones in one CI run, which took it
past its 120s `pytest-timeout` budget.

`resolve_model_path` now takes a keyword-only `allow_download`, defined as exactly
a download that declines: an absent XML still returns `None`, and a mesh-less XML
still returns its first candidate - the two outcomes a failed download already
produces. It introduces no third outcome, so the 46 existing call sites keep the
downloading default unchanged, and a caller that only reports can opt out of the
network entirely.

Both reporting callers now do. The family re-derivation also reads the machine's
real asset cache rather than the per-test temp directory `tests/registry/conftest.py`
points `STRANDS_ASSETS_DIR` at for user-registry isolation. Under that directory the
grouping was unconfirmable on every machine, and the downloading default hid it by
fetching the corpus into the temp directory instead of reporting that there was
nothing to read; the class now confirms all six families where the assets are
present and skips where they are not, which is what its own docstring said it did.

Declining changes no answer. The same 58 models compile and the same six families
are derived either way, and every registry robot still resolves and loads through
the unchanged default.
