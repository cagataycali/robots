### Fixed: docs/learn/hardware/feetech-arms.md LeKiwi sketch names the wrong class and the wrong address kwarg

`docs/learn/hardware/feetech-arms.md:73` told a laptop-side user to run
`Robot("lekiwi", mode="real", robot_ip="192.168.1.50")` and said this "builds
the client". Two silent failures in one line:

1. **Wrong class.** `Robot("lekiwi", mode="real", ...)` resolves to lerobot's
   `lerobot.robots.lekiwi.lekiwi.LeKiwi` -- the *host* process that runs on
   the Pi and dials serial motors. The **client** class is
   `LeKiwiClient`, reached via `Robot("lekiwi_client", ...)`. A user copies
   the sketch on their laptop, gets no error, and their laptop code silently
   holds a Pi-side host object that opens no ZMQ socket to the address they
   named.
2. **Wrong kwarg.** `robot_ip` is on the cross-robot passthrough allowlist
   (`strands_robots/hardware_robot.py:159`) but LeKiwi's dataclass does not
   declare it, so the value is silently dropped. The correct address kwarg
   for the client is `remote_ip` (`hardware_robot.py:220` `_ADDRESS_FIELDS`,
   and the `LeKiwiClientConfig` refusal at `hardware_robot.py:1406`).

The sketch is now the two-line pair a user actually needs -- `lekiwi_client`
with `remote_ip=` for the laptop side, `lekiwi` with `port=` for the on-Pi
host. Cross-referenced by
`docs/robots/lekiwi.md` and `docs/robots/lekiwi_client.md`, which already
spell both correctly; only this page carried the mismatch.
