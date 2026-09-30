### Fixed: the dashboard's fresh-install open posture admits a caller on the bootstrap proof, never on a loopback peer alone

Before the owner enrolled a passkey, `access.open_posture` admitted the whole
API to any request whose socket peer was loopback, that carried no forwarding
header, a loopback `Host` and no `Origin`. A second local account, any local
process, or a remote client behind a same-host port forward (`ssh -L`, `socat`,
`docker -p`, a DNAT rule) sends exactly that, and could spawn a real arm, start
a task and grant standing agent motion, while enrolling the owner passkey was
the one thing it could not do (f002, CWE-290 / CWE-348). The open posture now
requires the same proof the first enrollment does: the configured
`STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN`, or the token the server mints into a
`0600` file beside the credential store, presented as the bearer and compared
in constant time. The topology checks stay, the posture still closes on its own
once a passkey exists, and the login screen already asks for the token.
