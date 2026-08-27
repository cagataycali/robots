### Fixed: g1 driver's ``stop`` verb reports the halt outcome instead of asserting one

``G1Driver.stream({"action": "stop"})`` built its envelope beside
``await self.stop()`` and hardcoded ``status="success"``. Two things were wrong
with the text it carried. It named issue #358 and claimed no motion path was
wired, which is stale - #361 landed :class:`DDSPublisher` on ``rt/lowcmd``,
:meth:`send_action`, the 500 Hz control loop and the zero-torque frame the loop
publishes on exit. And the envelope could not report what the stop achieved,
because ``stop`` is the protocol's shutdown hook and returns ``None``: an
envelope written next to it can only restate the intent.

That mattered on the case :meth:`_ControlLoop.stop`'s own docstring names. It
returns whether the thread joined, and a caller-supplied policy that outlasts
the join budget - a remote inference call is the ordinary case - leaves the loop
publishing frames. ``stop_task`` already reads that verdict and answers
``status="error"`` with ``stopped=False`` and ``running=True``; ``cleanup`` and
``stop`` were taught to read it too. The verb an agent reaches was the one
surface left that could not say the stop had failed:

```text
policy parked past the 2.0s join budget, same driver, same wedge

  stream({"action": "stop"})   status="success"   text only, loop still running
  stop_task()                  status="error"     stopped=False, running=True
```

The verb now returns ``stop_task``'s envelope, so the verdict has one owner
rather than two: a joined stop is still ``success`` and carries
``stopped=True``, an unjoined one is an ``error`` naming the timeout, and a
driver with nothing running says so instead of claiming a halt it did not
perform. The ``inputSchema`` description promises the report the verb now
delivers, and ``tests/drivers/test_g1_stream_stop_reports_the_halt_outcome.py``
parks a policy past the budget to grade it - the fast path the shipped suite
drives cannot distinguish a reported outcome from an asserted one.
