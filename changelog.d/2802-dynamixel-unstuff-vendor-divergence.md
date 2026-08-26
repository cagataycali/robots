### Fixed: the Dynamixel codec no longer cites a vendor unstuffer that loses a byte

`strands_robots.drivers.dynamixel.protocol` opened with the claim that every
function in it is verifiable against Robotis' `dynamixel_sdk` byte-for-byte, and
`_unstuff` described itself as the mirror of the SDK's `removeStuffing`. Measured
over every payload that can carry a Protocol 2.0 reserved run - 21760 cases,
lengths four through seven over `{FF, FD, 00, 01}` - the two stuffers agree
byte-for-byte, and the vendor's unstuffer is lossy on twelve of them. Its loop
compacts the packet in place, so the two-byte look-back that decides whether an
`FD` is an escape reads positions the loop has already overwritten; on a payload
carrying two reserved runs it matches an `FF FF` that is no longer there and
drops a data byte. The shortest witness is `FF FF FD FF FD FD`, which comes back
from `removeStuffing` as `FF FF FD FF FD`.

The shipped implementation is correct and is unchanged. What the docstrings did
was point a future reader at the broken side as the thing to match: substituting
the vendor algorithm for the shipped one leaves this module's own suite at 68
passed, so the byte loss would have landed silently. Both docstrings now state
the divergence and its cause, and the round-trip identity is graded over the
whole reserved-run corpus rather than a hand-picked list - an escape and
unescape pair that is not the identity is broken whichever implementation is
older, which is the adjudicator the SDK cannot serve as. Two further cells
consult the SDK to grade the divergence claim itself and skip where it is
absent, since it is not a declared dependency.
