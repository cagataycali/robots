### Fixed: spawning a remembered USB profile checks the chip and USB location it remembered, and never rebinds them

`POST /api/devices/spawn-remembered` found the profile by the serial the board
reports about itself and spawned it as that real robot, then re-saved the
profile with the chip of whatever board had just confirmed. The scan now
carries pyserial's USB `location`; the profile records it next to `usb`
(`vid:pid`), and the route refuses a board whose chip or location differs from
(or cannot be read against) what was recorded with a 409 naming both, unless
the body sends `"accept_different_board": true`, which is written to the
activity trail and rebinds the profile to that board. `ProfileStore.save` keeps
recorded anchors, so no spawn rewrites them. The auto-spawn watcher holds such a
board for every serial, not only allowlisted ones, and its entries show the
remembered and seen chip and location side by side.
