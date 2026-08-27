### Fixed: the FSM-set docstrings name the topic ``send_action`` actually writes

Two docstrings said :data:`HANDSHAKE_FSMS` gated writes to ``rt/armsdk``:
the ``#:`` block above ``HANDSHAKE_FSMS`` in ``strands_robots/tools/g1/_g1_common.py``
and the ``_check_motion_gates`` docstring in ``strands_robots/drivers/g1.py``.
Since #2767 landed, ``G1Driver.send_action`` publishes on ``rt/lowcmd`` -
``_TOPIC_LOWCMD = "rt/lowcmd"`` and ``_pubs.publish(_TOPIC_LOWCMD, LowCmd_, cmd)``
is the one write site.  The 500 Hz control loop wired by #2779 writes the same
topic on every step and on stop.  A reader following ``HANDSHAKE_FSMS`` from
either docstring was pointed at a topic string this file does not appear in and
could not confirm by grep, and the two mentions of the real topic in the same
module (``send_action``'s own docstring and its scope classification) already
said "``rt/lowcmd``" - so the module contradicted itself.

The arm-SDK-*shape* (the FSM set) and the arm-SDK-*topic* (``rt/armsdk``, which
the ``g1_tools`` client for issue #358 will write) are not the same thing.  The
docstrings now name the FSM set on its own terms (motion-switcher state) and
cite ``rt/lowcmd`` as the topic the driver writes today; the ``rt/armsdk`` topic
name is left where it belongs - inside the SDK error-code table for firmware
response 7400, which is out of scope.

The rule is pinned by
``tests/drivers/test_g1_handshake_fsms_docstring_names_the_topic_it_gates``:
two defect cells, both failed on ``main @ 5ded625b`` and pass on this branch;
two premise cells (the constant is still declared, the write site still reads
``_TOPIC_LOWCMD``); and one scope-boundary cell that pins the SDK error-table
mention as legitimate so the rule cannot creep to a blanket ban.
