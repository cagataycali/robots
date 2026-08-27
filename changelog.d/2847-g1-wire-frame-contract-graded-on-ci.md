### Tests: the g1's ``rt/lowcmd`` wire-frame contract is graded in CI, not only where the SDK is installed

Twenty cells in ``tests/drivers/test_g1_driver.py`` covered ``send_action``'s
write path and the two ``LowCmd_`` builders, and every one of them sat behind
``skipif(not _HAS_SDK)``. ``unitree-sdk2`` is in no extra and in no core
dependency, and ``hatch``'s default env resolves ``features = ["all"]``, so
``call-test-lint`` skipped all twenty: the CRC, the ``mode_machine`` echo,
``mode_pr`` and the Enable byte -- the four fields the G1 firmware validates,
and the four a review of the write path asked for by name -- were asserted by
nothing on the runner that gates a merge. The suite reported ``53 passed, 20
skipped`` there while reporting ``73 passed`` on a box that happens to carry
the SDK.

``tests/drivers/test_g1_per_joint_gains`` already states the rule for the same
builder: a contract asserted only behind ``skipif(not _HAS_SDK)`` is asserted
by nothing in CI. Four sibling suites act on it by installing a
``unitree_sdk2py`` stub on ``sys.modules`` and driving the production lane
against it. This module was the one that skipped instead, and its section
comment gave the reason as "a missing SDK short-circuits the whole class" --
true of the driver, and an argument for supplying a ``LowCmd_``-shaped object
rather than for leaving the contract ungraded.

Eighteen of the twenty need only that shape to write into, so they now take the
stub ``tests.drivers.test_g1_control_loop`` installs, imported rather than
copied so every suite grading this builder writes into the same object. The two
cells that recompute the SDK's own CRC as an independent oracle keep the marker:
under a stub whose ``Crc`` returns a constant, ``cmd.crc == CRC().Crc(cmd)``
compares that constant against itself and grades nothing. One added cell takes
the half of the stop-frame oracle that is not a CRC question -- the Enable bound
over the 29 named slots and the reserved tail -- so a stop frame that lands as a
wire-side no-op is refused by CI rather than by a box with the SDK.

The fixture is opt-in per cell rather than autouse because this module also pins
the SDK-*absent* refusals (``test_ensure_dds_reports_missing_sdk``,
``test_connect_eagerly_reports_reason_without_sdk``); a module-wide stub would
have made those unreachable while leaving them green.

On the tree CI runs, seven wire-frame regressions were undetectable before and
are detected now: a dropped CRC stamp, ``mode_pr = 1`` (which silently remaps
four ankle indices), a dropped ``mode_machine`` echo, an Enable byte left at
Disable in either builder, the per-joint gain table collapsed to a scalar, and a
stop frame that keeps its stiffness. The suite is unchanged where the SDK is
present (73 passed, now 74 with the added cell) and moves from ``53 passed, 20
skipped`` to ``72 passed, 2 skipped`` where it is not.
