### Tests: the 2F-85's degraded Modbus link is graded, not just its happy path

`RobotiqDriver`'s command path was pinned against a gripper that answers; the
half that matters when a cable comes loose was uncovered. Four ways the link
fails are now graded, and the three distinctions the driver draws between them
are pinned rather than left to drift:

* **A connection that is refused** yields a reason naming host and port -
  `connect_eagerly` is declared `-> str | None` so an unreachable gripper is a
  value the caller reads, and the one mistake that reason diagnoses is a driver
  pointed at the wrong address. The reason survives the call, so `get_status`
  reports it as `connect_error` instead of only saying "not connected".
* **A controller that starts refusing** (a Modbus exception reply) is reported
  by `send_action`, `read_status` and `stop_task` with the exception code
  intact. The frame reached the controller and was rejected: the code is the
  actionable part, and letting `ProtocolError` escape would put a traceback
  where the mesh command path and the agent tool dispatch both read an envelope.
* **A controller that stops answering** - connection held open, no reply - is a
  *wire* failure and is spelled as one ("writing to the gripper failed",
  "reading the gripper failed"), with no exception code, because there is
  nothing in the manual to look up. Collapsing it into the refusal spelling
  sends an operator chasing a dead cable into the exception-code table.
* **A controller that closed its side** is decoded from the short frame, and the
  reason says how much of the header arrived. Modbus TCP is a byte stream, so a
  reply must be read by its declared length: a driver trusting one `recv` would
  decode a position out of a truncated frame and report a gripper as open
  because the bytes ran out.

`get_observation` is graded against all three live failures. It is the mesh's
joint source for this driver and is annotated `-> dict[str, float]`, so it has
no envelope to refuse into: it must degrade to no joints and log why. An
exception there does not fail one gripper, it takes the state publication for
every robot on that peer with it.

Also pinned: `get_task_status` *succeeds* with `in_flight: False` where its
`start_task` / `run_policy` siblings refuse - a gripper having no rollout is a
true answer to that question, not a fault, and a task poller reading an error
envelope would report the gripper as broken - and `get_status` renders the
cached reading with its enums named (`object: "CONTACT_CLOSING"`, not `2`)
while reporting `battery_pct` as `None` rather than inventing a charge for a
gripper powered from the arm.

The scripted gripper grows two failure modes to support this - `stall` (reads a
request and never answers) and `half_close()` (sends FIN on the live
connection) - so both are exercised over a real socket rather than by patching
the driver's internals. `strands_robots/drivers/robotiq/driver.py` coverage:
90% -> 97%.
