### Fixed: the dashboard's first passkey enrollment is admitted on proof, never on where the connection appears to come from

`begin_registration` granted the ownership-sealing first enrollment, when no
`STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN` was set, to a request whose socket peer was
loopback and which carried no proxy header. A same-host Layer-4 forwarder -
`socat`, `ssh -L`, nginx `stream{}`, HAProxy `mode tcp`, an iptables DNAT rule,
`kubectl port-forward` - relays raw bytes, adds no HTTP header and hands every
remote client a `127.0.0.1` peer, so both checks passed and a stranger could
enroll the owner passkey (F-007 follow-up, CWE-290 / CWE-348). No header roster
closes that class, because "at this machine" is not a property a request can
assert. The first enrollment now always needs a bootstrap token: the configured
one, or - when none is set - a token the module mints into a `0600` file beside
the credential store (`~/.strands_dashboard/enroll_token`, relocatable with
`STRANDS_DASH_AUTH_ENROLL_TOKEN_FILE`), logged by path at first demand and
deleted once a passkey exists. Reading that file is the local act a remote
peer cannot perform. The peer address and proxy evidence are still read, but
only to word the refusal. `status()` reports `bootstrap_required` for every
fresh install and a new `bootstrap_source` (`env` | `file`) so a login screen
knows what to ask for; the token itself and its path are never in the response.
