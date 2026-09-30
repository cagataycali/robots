### Fixed: `lekiwi` and `spot` are listed as mobile manipulators

Both are a base that carries an arm, but the registry filed them under
`mobile`, so `list_robots_by_category()["mobile_manip"]`, the *Mobile
manipulators* docs page and the catalog filter left them out while their peers
(`lekiwi_client`, `stretch`, `tiago_dual`, `yahboom_m3pro`) were listed. They
now declare `category: "mobile_manip"`. Code that matched `category == "mobile"`
to find a moving base should also accept `"mobile_manip"`, as
`examples/fleet/01_skill_dispatch_multi_vendor.py` now does.
