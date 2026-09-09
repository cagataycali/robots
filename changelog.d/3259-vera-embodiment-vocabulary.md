### Fixed: `policies/vera` - an unknown embodiment is refused, not resolved as mimicgen

`VeraConfig.embodiment` is the key every other per-embodiment default is looked
up by, and it was the one field on the dataclass with no domain. Both default
ports, the per-view `render_width`, the checkpoint-root variable that is probed,
the container name and the server's own `--embodiment` flag are all derived from
it, so a spelling no table has an entry for was not one wrong value but a whole
configuration assembled from what each of those six readers does with an unknown
key.

The two table lookups made that silent rather than merely wrong. Each carried its
own fallback - `_DEFAULT_PORTS.get(self.embodiment, (8800, 8801))` and
`_DEFAULT_RENDER_WIDTH.get(self.embodiment, 128)` - and those literals are
byte-for-byte `mimicgen`'s entries in the very tables they were fallbacks for. So
`embodiment="PushT"` did not degrade, it resolved to a configuration
indistinguishable from a deliberate `embodiment="mimicgen"`, and because
`VeraServerRunner.start` opens with a port probe and returns early on a hit
("Already serving (ours or someone else's) - reuse it"), a mistyped `pusht`
dialed 8800, found a running mimicgen server, completed the metadata handshake
and rolled out against the wrong embodiment's planner/IDM pair - reporting
success throughout. The `--embodiment` flag that would have carried the typo to a
server able to object was never used, because no server was launched. Measured
over eight plausible misspellings, all eight resolved to `(8800, 8801, 128)`;
`PushT` and `pushT` additionally probed `VERA_PUSHT_CKPT_ROOT`, so the readers
did not even agree with each other on which embodiment had been asked for.

`docker/entrypoint.sh` already refuses exactly this vocabulary
(`ERROR: unknown embodiment`, `exit 2`), but that arm is reached only under
`server_mode="docker"`, only once an image has started, and never at all for the
subprocess runner or for `auto_launch_server=False`. The vocabulary was stated in
shell and enforced in shell; the Python that computes the ports, the width and
the `-e VERA_EMBODIMENT=` value it passes in did not hold it. It is now held at
the config funnel - the one surface the `VeraPolicy` keyword, a pre-built config
and the `VERA_*` overrides all pass through - and derived from `get_args`
of the `Embodiment` alias, so an embodiment added there participates on arrival
instead of being absorbed by a fallback.

With the vocabulary held, both lookups index their tables directly: the second
copy of a default is what made "not a known embodiment" and "mimicgen" the same
request. A regression test also pins the shell `case` and the Python alias
against each other, so the two enforcers of one vocabulary cannot drift.
