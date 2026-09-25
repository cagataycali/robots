### Docs: the robot factory page is 1,396 words, with the driver contract on its own page

`docs/getting-started/robot-factory.md` was 3,614 words, and 2.2k of them were
not the factory signature: what a native driver is and the contract one
satisfies - the `baud_rate=` and `timeout=` domains, the two halt verbs, the
telemetry coercion every decoder shares - plus the Reachy Mini daemon bring-up,
from the transport it cannot import through the handshake it gives up on to the
read-only hardware check.

The page is split at its H2s. The factory page keeps `Robot(...)`, the parameter
table, name resolution, the `driver=` selection table and mesh; the new
`docs/hardware/native-drivers.md` carries the contract at 1,428 words and the
new `docs/hardware/reachy-mini.md` the bring-up at 892. No fact is dropped, and
`getting-started/robot-factory.md` leaves the word-budget exemption list, so the
ratchet grades it from now on.
