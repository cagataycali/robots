### Fixed: a peer's continuable refusal crosses the wire with its code

A command a peer refuses for a reason its operator can lift (the trust-remote-code gate, an allowlist) answers with `code`, `grant` (the variable that lifts it) and `subject` next to a remedy sentence, instead of only `dispatch error`. The exception text still never leaves the peer. (#4173)
