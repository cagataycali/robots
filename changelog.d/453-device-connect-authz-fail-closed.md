### Fixed: a Device Connect device with no caller allowlist refuses to move

`is_authorized_caller` treated an unset `DEVICE_CONNECT_RPC_ALLOW` as "allow
all": a device that was simply never configured executed every peer's
`execute` / `stop` / `step` / `reset`, and a one-time warning was the only sign
(F-003, CWE-862). An unset allowlist now authorizes nobody on the RPC scope; the
refusal is logged once and names the variable to set, and `*` is the development
spelling of "allow every named caller" (it still warns). The emergency-stop scope
keeps honouring a *named* caller when no allowlist is set anywhere - stopping
must never get harder than moving - but refuses an anonymous one, since a caller
that carries no id cannot be told apart from one that forged none. Docs and the
module docstring no longer say that unset means allow all.
