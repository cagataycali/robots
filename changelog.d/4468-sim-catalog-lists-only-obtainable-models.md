### Fixed: `list_robots(mode="sim")` lists only robots `Robot(name)` can spawn

`google_robot` and `trossen_wxai` declare `auto_download: false` and ship no
model, so no documented step makes them spawnable, yet the sim catalog listed
them. `has_sim(name)`, the `has_sim` field of `list_robots()` and
`list_robots(mode="sim")` now count such an entry only once its model file is on
disk; every other asset entry is unchanged, and `Robot(name)` still answers a
missing asset with the path to place it at.
