### Fixed: `Robot("so1000")` suggests the nearest registered names

The refusal read `Unknown robot 'so1000' (resolved to 'so1000'). Pass a
registered name (see list_robots())` - the resolved spelling repeated the
input and no candidate was offered, although `add_robot(data_config=...)`
already answers "Did you mean". `Robot()` now says `Did you mean: so100,
so101?`, shows the resolved spelling only when it differs, and qualifies the
pointers as `strands_robots.list_robots()` / `strands_robots.list_discoverable()`.
