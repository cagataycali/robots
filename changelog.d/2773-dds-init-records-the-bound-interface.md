### Fixed: ensure_dds records the interface it bound, not the one it was asked for

The G1 DDS helper's first stated requirement is that `ChannelFactoryInitialize`
runs "exactly once per process, with a known network interface", and it keeps a
module-level record of that interface so a later caller asking for a different
one is refused rather than silently re-bound onto the wrong NIC. Its own comment
on that record says a second call with a different interface "is a bug worth
catching, not a silent no-op".

The record was only sometimes true. `ChannelFactory.Init` short-circuits on
`if __initialized: return True` and never reads its `networkInterface` argument,
so a second `ChannelFactoryInitialize` returns normally without binding
anything - a no-op indistinguishable, at the call site, from a successful bind.
`ensure_dds` read that quiet return as a bind and recorded the interface it had
asked for. Measured against `unitree_sdk2py` 1.0.1, with the bus brought up on
`lo` by anything other than `ensure_dds`:

```
ensure_dds("eth-does-not-exist")  ->  None          # reported success
recorded interface                ->  'eth-does-not-exist'
factory actually bound to         ->  'lo'
ensure_dds("another-nic")         ->  "...was called with interface
                                       'eth-does-not-exist'; refusing to
                                       re-initialise on 'another-nic'"
```

So every subscriber and publisher attached to whichever NIC the first caller
chose while the helper reported the one this caller asked for - the silent
re-bind, producing empty topics with no obvious cause, that the refusal exists
to prevent - and the refusal that did eventually fire quoted an interface the
bus was never on, sending a reader after the wrong fault.

`ensure_dds` now asks the factory whether it is already bound before calling
init. A bus this process did not bind is refused by name, giving the cause and
the call to drop, and nothing is recorded - so the interface a later refusal
compares against is only ever one `ensure_dds` actually bound. The SDK build
that raises "already initialized" instead of short-circuiting reaches the same
answer for the same reason, which also stops that branch's substring match
turning a genuine init failure whose message contains "initialized" into a
reported success. A build that does not publish its factory state keeps the
behaviour it had before the probe existed.

Nothing public distinguishes a bind from a no-op: `Init` returns `True` for
both and `ChannelFactoryInitialize` discards that bool, so the probe reads the
attribute the factory records its state on. That is a narrower coupling to the
SDK than matching its exception text, which this function already does. The
module docstring claimed a second init raises from `unitree_sdk2py`; it does
not, and that claim is corrected to what the SDK does.
