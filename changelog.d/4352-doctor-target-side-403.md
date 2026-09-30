### Fixed: `doctor` no longer fails the IoT Direct row for a running operator

The Direct Messaging API answers 403 with an empty body when the caller has
no `iot:SendDirectMessage` grant, and with a body naming the target client
when the sender's grant passed but the session under the target id may not
receive on that topic. The doctor read both as "may send a direct message
neither as a robot nor as an operator" and told the user to re-provision. A
403 that names the target is now a PASS that says where the broker stopped;
the empty-body 403 keeps its FAIL and remedy.
