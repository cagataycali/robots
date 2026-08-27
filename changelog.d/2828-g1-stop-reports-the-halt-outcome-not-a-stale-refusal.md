### Docs: g1 driver's ``stop`` verb reports the halt outcome, not a stale #358 refusal

``G1Driver.stream({"action": "stop"})`` mapped to a text envelope claiming
"no motion path wired yet (issue #358)". That claim was stale: issue #361
already landed the transport primitive - :class:`DDSPublisher` on
``rt/lowcmd``, :meth:`send_action` publishing ``LowCmd_``, the 500 Hz
control loop, and :meth:`stop_task` publishing a zero-torque frame on the
way out. The verb was calling the wired :meth:`stop` and lying about it.

Now the envelope names what the code does: "control loop halted; a running
task publishes a zero-torque frame on exit". The ``inputSchema`` description
for the ``stop`` action is updated to match, and
``test_stream_stop_action_calls_stop`` pins both the new text and a guard
against the pre-#361 refusal text sneaking back through a rebase.

No behaviour change - the wire path was already live. This is a docs and
observability fix so an agent that reads the tool spec sees the surface it
gets.
