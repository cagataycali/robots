### Tests: the g1 per-joint gain wire cells are graded in CI, and the gating rule now holds for the module that states it

``tests/drivers/test_g1_per_joint_gains`` is where this project writes the rule
down: ``unitree-sdk2`` is not a declared dependency, so a contract asserted only
behind ``skipif(not _HAS_SDK)`` is asserted by nothing in CI. Five of that
module's own cells sat behind that marker, and they are the whole wire half of
the gain contract -- every slot's gains on a full frame, a knee and an arm
carrying different gains in one frame, the partial fallback where a caller tunes
``kp`` and leaves ``kd``, a supplied gain overriding the table, and the stop
frame staying soft. The module reported ``35 passed, 5 skipped`` where the SDK is
absent, which is every runner ``call-test-lint`` uses, against ``40 passed``
where it happens to be present.

None of the five needs the real SDK. The module's own layering note explains why:
the vendor gains are stated locally there precisely so the value cells are an
independent oracle rather than a tautology, so what the cells read back is
compared against those local tuples and not against anything the SDK computes.
What they need is a ``LowCmd_``-*shaped* object to write into. They now take that
shape from the stub ``tests.drivers.test_g1_control_loop`` installs, which is
what puts them in front of CI. The fixture is opt-in per class rather than
autouse, so an SDK-absent refusal cell added to this module later is not quietly
made unreachable.

The class had been closed one file at a time twice, so a derived guard is added
beside the cells: every node gated on the SDK probe has to import the SDK inside
itself. That is what an independent-oracle cell does to obtain its reference
value, so the import is the evidence the marker is load-bearing rather than
inherited. The two ``crc`` cells in ``tests/drivers/test_g1_driver`` keep their
marker and satisfy the rule -- a stub whose ``Crc`` returns a constant would
compare that constant against itself. The scan walks every suite, so the next
gated cell is graded wherever it lands.

Five real gain regressions were measured against the tree CI runs. Three of them
were previously undetectable there: both tables read one slot's gains for every
joint (the flat-default failure mode, which keeps the subscript syntax the
source-reading contract cell looks for), the gain table leaking into the
zero-torque stop frame so a stop lands stiff, and a supplied ``kp`` no longer
winning over the table. The other two -- ``kp`` and ``kd`` each collapsed to a
single value -- went from one cell to three. The firmware validates ``crc``,
``mode_machine`` and the Enable byte, not gains, so each of those publishes with
a success envelope and leaves the joints at whatever stiffness was sent.
