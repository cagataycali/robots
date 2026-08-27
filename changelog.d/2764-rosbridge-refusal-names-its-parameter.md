### Fixed: a rosbridge port refusal names the parameter it was given

`use_rosbridge._transport_port_error` refuses a port that is legal TCP but outside the range the
rosbridge WebSocket transport can address. It takes a `param` argument documented as "the parameter
name it came from, used in the message" -- the same sentence eleven sibling helpers in this package
carry -- and hard-coded the word `port` into the message instead, so the argument was accepted and
never read. Eleven of the twelve helpers making that promise keep it; this one did not.

That matters because the helper is shared by design. Its own docstring says it is applied by both
the `use_rosbridge` tool and `RosbridgeRobot` so the two "cannot disagree about which ports it can
carry", and the shared 16-bit domain that runs one line earlier on the same value
(`utils.tcp_port_error`) does interpolate the name. A caller whose parameter is spelled anything
other than `port` would therefore be told the wide domain refused its `bridge_port` and the
transport refused a `port` it never passed.

The message now interpolates `param`. Both shipping call sites pass the literal `"port"`, so every
refusal text a caller can currently reach is byte-identical; the change is what makes the
documented contract true for the third caller the docstring anticipates. A derived test reads the
promise sentence out of the package and holds every helper carrying it to interpolating its
`param`, so a twelfth helper that copies the sentence is graded when it lands.
