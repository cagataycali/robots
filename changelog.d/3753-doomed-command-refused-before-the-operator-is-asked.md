### Fixed: a real-robot command the dispatcher would refuse no longer costs an approval

`Robot(mode="real")` asked the operator to approve an `execute`/`start` that
`execute_task`/`start_task` then refused on its inputs alone - a negative
duration, a port outside 1-65535, a provider with no port. Those checks now
run before the operator is asked, with the dispatcher's wording; a sound
command asks exactly as before, and the dispatcher still checks again.
