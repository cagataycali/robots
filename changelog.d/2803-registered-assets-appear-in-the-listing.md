### Fixed: a registered asset is named by the listing that advertises it

`list_urdfs` is the discovery entry point an agent uses to learn what it can spawn without guessing
a model name, and its docstring promises "built-in *and* user-registered". It returns
`list_available_models`, which promises "Menagerie + custom" and returned the asset-manager table
alone whenever the asset manager was importable -- which is every normal install, so the two halves
were an either/or rather than a union. The half that was dropped was the one the caller had just
written. `register_urdf("widget_arm", path)` answers `Registered 'widget_arm' -> <path>` followed by
`Resolved: <path>`, `resolve_urdf` returns that path, `list_registered_urdfs` maps the name to it,
and `add_robot` spawns it -- and `list_urdfs` reported 76 lines of built-in robots with no mention of
`widget_arm`. `add_robot`'s own unresolved-model message sends the caller to `list_urdfs` to "pick a
registered model", so the documented recovery path pointed at the one surface that denied the
registration existed.

Both halves are now reported: the built-in table leads, then a `Registered URDFs:` section naming
each registration and whether it currently resolves (`[OK]` / `[MISSING]`). A dangling registration
is reported rather than dropped, since a path typo is the likelier mistake and the listing is where
it is visible. The section is omitted entirely when nothing has been registered, so a default
install's listing is byte-for-byte what it was. The row format is written down once, in a helper both
branches call, so the asset-manager branch and the asset-manager-absent branch cannot drift into two
vocabularies for the same rows -- the shipped fallback test now grades the format for both.

The custom half was only ever exercised through the asset-manager-*absent* branch, reached by
monkeypatching `_HAS_ASSET_MANAGER` to `False`, which no real install takes; the present-branch test
asserted only that the built-in columns appear. That is why the suite was green, and reverting this
change fails none of the surrounding model-registry, foundation, MuJoCo or Newton discovery suites.
