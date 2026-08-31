### Fixed: the Booster T1 driver opens its channels under the shared DDS lock

`BoosterDriver.connect_eagerly` built ten DDS endpoints in a row - the channel
factory, an RPC client and four subscriber/publisher channels - while holding no
lock, in a process where `DDSSubscriberSet` constructs every subscriber under
`_DDS_INIT_LOCK` on its own streaming, rollout and mesh-telemetry threads. The
CycloneDDS bindings segfault on concurrent endpoint construction, and a segfault
is not catchable by the driver's "return a reason and stay usable for reads"
boundary: the process dies, possibly while a 1.2 m biped stands under its own
controller.

The whole construction block now runs under the shared lock, which is the shape
`Go2Driver` and `G1Driver` already use. Measured on the engine with 40 subscribes
against 40 opens, concurrent endpoint constructions go from 79 to 0, at a 9%
wall-clock cost that is the serialisation rather than a regression.

Two things stay outside the lock and are pinned there: the lazy SDK import, since
an import creates no endpoint and holding the shared lock across one would stall
every endpoint construction in the process for its duration; and the partial-set
release, since a close creates no endpoint either and must not stall the process
while the driver discards a set it is abandoning.

Only two of the ten sites were reachable by the rule that grades endpoint
construction over the source. That rule derives its vocabulary from the Unitree
infrastructure modules, so it recognises `Init` and not the Booster SDK's
`InitWithName`, `InitChannel`, `InitChannelWithName`, or its
`B1LowStateSubscriber` / `B1LowCmdPublisher` / `B1BatteryStateSubscriber` /
`B1FallDownStateSubscriber` constructors. Wrapping only the reported pair would
have turned the rule green over eight identical hazards, so the regression pin
grades the behaviour - was the shared lock held while each endpoint was built -
which no vendor's choice of name can evade.
