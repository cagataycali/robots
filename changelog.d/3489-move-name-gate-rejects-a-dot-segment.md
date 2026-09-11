### Security: a recorded-move name is one bare path segment, so a dot segment is refused

`move_name` is interpolated into the Reachy Mini daemon's
`/api/move/play/recorded-move-dataset/{dataset}/{move}`, and both gates on that
path (`ReachyDriver.play_move`, `ReachyMiniDriver.playMove`) admitted any 1-128
characters of `[A-Za-z0-9._-]`. `.` and `..` are spelled entirely from that
alphabet, so the charset alone admitted the two tokens a URL path resolves
relative to its parent: `move_name=".."` was sent, and the request it builds
resolves to `.../recorded-move-dataset/pollen-robotics` - a daemon endpoint the
caller named nothing about. Both patterns now require an alphanumeric first
character, which is what makes them one bare path segment rather than only a
safe charset, the way the calibration path-segment gate in
`strands_robots.drivers.feetech.bus` already does.

The two hardening rows that were supposed to catch this could not see it: they
drive `playMove` on a double whose caller is unauthorized, and `playMove`
authorizes before it reads `move_name`, so a clean name earned the same refusal.
They now name the caller and assert the request list, and the table they became
carries the dot-segment rows.
