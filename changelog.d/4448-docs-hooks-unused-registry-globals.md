### Fixed: the two docs hooks no longer carry a registry path neither of them reads

`docs/hooks/robot_pages.py` and `docs/hooks/manifest.py` each defined a
module-level `_REGISTRY` path and never used it (CodeQL
`py/unused-global-variable`, alerts #1371 and #1372). The dead assignments
are gone; both hooks build the same pages and manifest as before.
