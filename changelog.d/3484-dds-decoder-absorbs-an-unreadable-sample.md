### Fixed: a DDS decoder that could not read a delivered sample killed its own subscription

`Go2Driver._on_lowstate`, `Go2Driver._on_sportmode` and
`BoosterDriver._on_fall_state` read a member off the delivered sample outside
any guard, so a member the compiled IDL binding could not produce raised on the
thread the vendor SDK owns - where no caller is on the stack to report it and
nothing re-subscribes. The cache then stopped advancing while the gates that
read it kept answering from the last frame that happened to decode: on the Go2
that is the battery floor and the measured pose `send_action` holds uncommanded
joints at, and on the T1 the fall state `send_action` refuses writes on. All
three now guard every read, as the other eight decoders on these drivers
already did, and a derived test holds every present and future decoder to it.
