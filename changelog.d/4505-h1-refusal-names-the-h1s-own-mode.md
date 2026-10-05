### Fixed: an H1 write refusal names the H1's motion mode, not the Go2's sport mode

`Robot("h1", mode="real")` refused a write made before the onboard mode was
released with "sport mode is not released", a Go2 service the H1 does not have.
It now says "the H1's onboard motion mode is not released". The gate is
unchanged: both robots release the same motion-switcher mode, as the SDK's H1
low-level example does, and `release_sport_mode()` performs it on each. The
Unitree hardware page's gate table names the H1.
