### Fixed: pose_tool and serial_tool no longer spend an approval on a call they would refuse

Both tools asked the operator to approve a motion or write, then refused it
on its inputs - a missing position or delta, an empty positions dict, a pose
not in the library, a motor the arm does not have, a write with no payload or
with `hex_data` that is not hex (a `ValueError` after the port was opened).
Those checks now run before the operator is asked, with the branch's own
wording; a sound call asks exactly as before, and the branch still checks.
