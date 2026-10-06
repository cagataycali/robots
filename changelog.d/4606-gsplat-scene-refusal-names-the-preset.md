### Fixed: `download_gsplat_scene` points a near-miss at the preset it meant

An unknown scene name now gets one `Did you mean '<preset>'?` before the
listing. The bare slug (`"tabletop"`, `"bonsai"`) - the name the cache file is
saved under - is offered back as its full preset name; an unrelated name gets
no guess.
