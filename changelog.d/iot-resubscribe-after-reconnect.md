### Fixed: an IoT peer re-subscribes after a reconnect; a robot that dropped its MQTT session no longer goes deaf to the fleet e-stop

`IotMqttTransport` builds the MQTT5 client without a session request, so
every awscrt reconnect (a keep-alive miss, a network blip, the broker ending
the session after an ungranted publish) is a clean session and the broker
forgets the previous one's subscriptions. The transport kept its handler
table and reported `is_alive()`, so the peer still answered direct `cmd`
(routed without a subscription) while hearing no `safety/estop`,
`safety/resume`, `broadcast` or presence for the rest of its life, with
nothing above DEBUG to say so. Measured on a live Thing: 3/3 e-stops received
before a broker DISCONNECT, awscrt reconnected in 1.35 s, 0/3 after.

`_on_connection_success` now reads the CONNACK's `session_present`; when the
broker did not resume the session it re-issues every registered topic filter
from a worker thread (the lifecycle callback runs on the awscrt event loop,
where a blocking subscribe would deadlock), WARNs once with the list of
filters re-subscribed and ERRORs per filter the broker refused, naming it.
`wait_for_resubscribe(timeout)` lets a caller block until the filters are
back. A missing CONNACK or flag is read as a fresh session, the safe
direction.
