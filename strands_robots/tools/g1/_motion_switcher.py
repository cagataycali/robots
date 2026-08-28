"""Motion-switcher decoder for the G1 driver's FSM gate.

The G1 driver's ``_check_motion_gates`` refuses every arm-SDK write while
``_fsm_id`` is ``None``. On the shipped tree that attribute has exactly one
writer (the ``None`` initialiser in ``__init__``); the ``rt/lowstate`` topic
carries ``mode_machine`` but not the high-level FSM state, so the missing
producer is the motion-switcher API, not the DDS bus.

Issue #2765 lists the wire-format decisions the write path depends on and
names this decoder as the FSM producer's home. This module is the *checkable*
half of that answer: it owns the read-side of the API, decodes the FSM id from
``MotionSwitcherClient.CheckMode()`` in exactly one place, and refuses a shape
the SDK does not name rather than guessing. Wiring the decoder onto the driver
is a separate step -- see #2765. A pin in the driver's test tree asserts that
``_fsm_id`` has exactly one assignment today (the ``None`` initialiser in
``G1Driver.__init__``) and that a healthy driver still refuses with ``FSM id
unknown``; both flip on the day a second writer lands, so the wire and that
pin's replacement belong in one change.

What the SDK reports.  ``MotionSwitcherClient.CheckMode`` returns
``(status, result)``: ``status`` is a Unitree response code (``0`` is OK,
``ERR_CODES`` render the rest) and ``result`` is a dict whose ``name`` key
carries the current motion mode as a string -- ``""`` when no mode is
selected, otherwise a mode label such as ``"ai"`` or ``"normal"``. The
mode label alone does not identify the FSM state the gate reads, so this
decoder also reads an integer FSM id from the ``form`` key.

``form`` is *not* evidenced by the SDK. ``CheckMode`` returns
``json.loads(data)`` straight from the robot, so the Python package cannot
tell us which keys that payload carries, and the string ``"form"`` appears
nowhere in ``unitree_sdk2py``: every SDK example reads ``result['name']``
and nothing else (the G1 low-level example loops
``while result['name']: self.msc.ReleaseMode()``). Which key carries the
FSM id is therefore one of the wire-format questions #2765 tracks, and
answering it needs a robot. Until then this decoder refuses an active mode
whose payload has no integer ``form`` rather than defaulting -- so a wrong
guess here surfaces as a named refusal, never as an FSM id the gate might
open on. Both values are surfaced so a caller sees the reason a decode
declined rather than only the outcome.

Wire-side invariants this decoder enforces.

* ``unitree_sdk2py`` is **not** imported at module load. Every SDK read goes
  through :func:`_load_motion_switcher_client`, which lazy-imports the class
  and is the seam every unit test mocks. This keeps the module importable
  on Thor and CI (mirrors the invariant :mod:`._dds_engine` and
  :mod:`._g1_common` already carry).
* The mapping from ``CheckMode()`` return-shape to ``_fsm_id`` value is
  spelled once, here, in :func:`decode_fsm_id`. A driver-side wire (the
  step #2765 defers) reads the return of this function; there is no second
  copy of "which key carries the FSM" to disagree with.
* A shape the SDK never returns (``None`` for the result dict, an integer
  under ``name``, a missing ``form`` on an active mode) refuses with a
  message naming the received shape, rather than defaulting to a value
  the gate might silently open on.

Related: :mod:`._g1_common` carries ``HANDSHAKE_FSMS = {500, 501, 801}`` and
``WALK_FSMS = {501, 801}``; membership in those sets is the gate's admission
question. This decoder does not evaluate membership -- that is the driver's
job, and stays there -- it only produces the value the gate compares.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from typing import Any

from strands_robots.tools.g1._g1_common import decode_code
from strands_robots.utils import sequence_length

logger = logging.getLogger(__name__)


# The exact keys ``MotionSwitcherClient.CheckMode`` populates on its result
# dict. Named as module constants so a spelling drift lands here rather than
# in every caller.
_RESULT_NAME_KEY = "name"
_RESULT_FORM_KEY = "form"

# ``MotionSwitcherClient`` lives under ``comm/``, not ``g1/``: the motion
# switcher is shared across every Unitree platform (the SDK's own
# ``example/g1``, ``example/h1`` and ``example/go2`` low-level examples all
# import it from the same place), so the package is not robot-scoped. Named
# here so the path is a reviewable constant rather than a string buried in a
# call: a seam naming a module the SDK does not ship raises
# ``ModuleNotFoundError`` the first time a caller opens a real client, and the
# decode tests all hand in an already-open client so none of them reaches it.
_SDK_MODULE = "unitree_sdk2py.comm.motion_switcher.motion_switcher_client"


@dataclass(frozen=True)
class FSMReading:
    """A single ``CheckMode()`` reading, decoded for the driver's gate.

    :attr:`fsm_id` is what the gate compares against
    :data:`~strands_robots.tools.g1._g1_common.HANDSHAKE_FSMS` and
    :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS`. When it is
    ``None`` the reason is on :attr:`refusal` -- either the SDK reported a
    non-OK status code, or the result dict was a shape ``CheckMode`` does
    not return on this SDK version.

    :attr:`mode_name` is preserved separately from :attr:`fsm_id` because
    the two carry different questions: a caller inspecting *why* the FSM
    is what it is (an operator handing off from the high-level motion
    service, for example) reads the mode label; a caller writing
    ``rt/lowcmd`` reads the integer. The two are decoded from the same
    return and returning both keeps the driver's diagnostic message
    honest about what the SDK actually said.
    """

    fsm_id: int | None
    mode_name: str
    refusal: str | None


def _load_motion_switcher_client() -> Any:
    """Lazy-import :class:`MotionSwitcherClient`.

    The SDK module is imported on first call and never at import time, so
    ``strands_robots.tools.g1`` still loads on hosts without
    ``unitree_sdk2py`` installed -- same invariant as
    :mod:`._dds_engine` and :mod:`._g1_common`. Every test in this file
    mocks the returned class rather than the module attribute, so no test
    depends on the SDK's presence.
    """
    module = importlib.import_module(_SDK_MODULE)
    return module.MotionSwitcherClient


def decode_fsm_id(check_mode_return: Any) -> FSMReading:
    """Decode one ``MotionSwitcherClient.CheckMode()`` return into an FSM id.

    ``CheckMode`` returns a ``(status, result)`` pair. Both values need to
    be checked before the FSM id is read: a non-zero status means the RPC
    itself failed, and a well-formed result on a failed RPC is not a
    truthful reading. The refusal path names the code via :func:`decode_code`
    so a firmware update surfaces at least as its integer.

    On the OK branch, the result dict carries a ``name`` (the mode label)
    and, when a mode is active, a ``form`` (the FSM id as an integer).
    ``name == ""`` is the SDK's "no motion mode selected" reading and is
    reported as-is rather than treated as an error, because that is the
    same information the driver's gate carries today (``_fsm_id = None``).

    Refuses with a message naming the received shape when the input is
    not a ``(status, result)`` pair, when ``result`` is not a dict, when
    ``name`` is missing or is not a string, or when ``form`` is present
    but is not an ``int``. Refusing on shape rather than defaulting keeps
    the gate from opening on a silent zero.
    """
    if not isinstance(check_mode_return, tuple) or len(check_mode_return) != 2:
        return FSMReading(
            fsm_id=None,
            mode_name="",
            refusal=(
                "CheckMode() return must be a (status, result) tuple; "
                f"got {type(check_mode_return).__name__} of length "
                f"{sequence_length(check_mode_return) if sequence_length(check_mode_return) is not None else '?'}"
            ),
        )
    status, result = check_mode_return
    if not isinstance(status, int):
        return FSMReading(
            fsm_id=None,
            mode_name="",
            refusal=(f"CheckMode() status must be an int response code; got {type(status).__name__}"),
        )
    if status != 0:
        # Named refusal, mirroring the driver's other refusals against
        # ``ERR_CODES``. ``mode_name`` stays empty because the result dict
        # cannot be trusted on a failed RPC.
        return FSMReading(
            fsm_id=None,
            mode_name="",
            refusal=f"CheckMode() reported {decode_code(status)}",
        )
    if not isinstance(result, dict):
        return FSMReading(
            fsm_id=None,
            mode_name="",
            refusal=(f"CheckMode() result must be a dict; got {type(result).__name__}"),
        )
    if _RESULT_NAME_KEY not in result:
        return FSMReading(
            fsm_id=None,
            mode_name="",
            refusal=(f"CheckMode() result missing {_RESULT_NAME_KEY!r} key; got keys {sorted(result.keys())}"),
        )
    mode_name = result[_RESULT_NAME_KEY]
    if not isinstance(mode_name, str):
        return FSMReading(
            fsm_id=None,
            mode_name="",
            refusal=(f"CheckMode() result {_RESULT_NAME_KEY!r} must be a string; got {type(mode_name).__name__}"),
        )
    if mode_name == "":
        # No motion mode selected -- the SDK's own "high-level released"
        # reading. Preserve the string as observed rather than mapping to
        # ``None`` here, because a caller inspecting the reading should
        # see the same value ``CheckMode`` reported.
        return FSMReading(fsm_id=None, mode_name="", refusal=None)
    # Active mode: ``form`` carries the FSM id. Missing or non-int refuses
    # so the gate never opens on a defaulted value.
    if _RESULT_FORM_KEY not in result:
        return FSMReading(
            fsm_id=None,
            mode_name=mode_name,
            refusal=(
                f"CheckMode() result names mode {mode_name!r} but is missing "
                f"{_RESULT_FORM_KEY!r}; got keys {sorted(result.keys())}"
            ),
        )
    form = result[_RESULT_FORM_KEY]
    if not isinstance(form, int) or isinstance(form, bool):
        # ``bool`` is a subclass of ``int``; refusing it explicitly stops
        # a ``True`` from decoding as FSM id ``1``, which is not a value
        # ``HANDSHAKE_FSMS`` or ``WALK_FSMS`` names anyway but would silently
        # pass this checkpoint.
        return FSMReading(
            fsm_id=None,
            mode_name=mode_name,
            refusal=(f"CheckMode() result {_RESULT_FORM_KEY!r} must be an int FSM id; got {type(form).__name__}"),
        )
    return FSMReading(fsm_id=form, mode_name=mode_name, refusal=None)


def read_fsm_id(client: Any) -> FSMReading:
    """Call ``client.CheckMode()`` and decode its return.

    ``client`` is either a real ``MotionSwitcherClient`` (opened elsewhere,
    since :class:`MotionSwitcherClient` requires the DDS bus and this
    module does not own that bring-up) or a test double with a
    ``CheckMode`` method. The two paths share this function so a wire
    change lands in one place: the driver's future ``_refresh_fsm_id``
    calls :func:`read_fsm_id` and takes the :class:`FSMReading`.

    Refuses if ``client`` has no ``CheckMode`` attribute rather than
    raising ``AttributeError`` at the caller, so a mis-passed object is
    reported the same way as a mis-shaped return.
    """
    check_mode = getattr(client, "CheckMode", None)
    if not callable(check_mode):
        return FSMReading(
            fsm_id=None,
            mode_name="",
            refusal=(f"motion-switcher client has no callable ``CheckMode`` attribute; got {type(client).__name__}"),
        )
    try:
        return decode_fsm_id(check_mode())
    except Exception as exc:  # noqa: BLE001 - the SDK is opaque; report the class
        # The SDK's RPC path can raise on transport failures. Surface the
        # exception class and message rather than propagating, so the
        # driver's gate can refuse with a reason rather than crash a
        # control loop mid-step.
        #
        # This is the only place in the module where information is absorbed
        # rather than returned: every other failure is already a ``refusal``
        # string the caller reads. The traceback is not, and an FSM read that
        # fails once a second inside a control loop is exactly the case an
        # operator needs a log for -- so it is reported at WARNING, the same
        # spelling ``_dds_engine`` uses at its own absorbed-exception
        # boundaries.
        logger.warning("CheckMode() raised %s: %s", type(exc).__name__, exc)
        return FSMReading(
            fsm_id=None,
            mode_name="",
            refusal=f"CheckMode() raised {type(exc).__name__}: {exc}",
        )


__all__ = [
    "FSMReading",
    "decode_fsm_id",
    "read_fsm_id",
]
