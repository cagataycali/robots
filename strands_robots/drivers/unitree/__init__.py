"""Unitree DDS transport, shared by every Unitree-protocol driver.

The G1, the Go2 and the Booster T1 all speak raw Unitree IDL over CycloneDDS,
so the subscriber/publisher layer, the SDK-absence refusal, the return-code
catalogue and the motion-switcher FSM read are one transport with three
consumers rather than three copies:

* :mod:`._common` - ``ensure_dds``, the shared init lock, ``sdk_missing``,
  ``ERR_CODES``/``decode_code`` and the ``HANDSHAKE_FSMS``/``WALK_FSMS`` gates;
* :mod:`._dds_engine` - ``DDSPublisher`` and ``DDSSubscriberSet``;
* :mod:`._motion_switcher` - the FSM id off ``MotionSwitcherClient.CheckMode()``.

``unitree_sdk2py`` is lazy-imported throughout: importing this package pulls no
SDK submodule, so a machine without the SDK - every headless CI runner - builds
the drivers, lists them in the registry and runs every test against a mocked
bus. Nothing is re-exported here; a consumer names the submodule it reads, so
each symbol has exactly one import path.
"""
