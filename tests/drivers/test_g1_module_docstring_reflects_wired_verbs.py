"""The G1 driver's module docstring names the verbs that are wired today.

The module-level docstring of :mod:`strands_robots.drivers.g1` describes what
the driver does: what it subscribes, what it gates, what it publishes. A
reader who trusts the docstring writes their code around the shapes it
promises, so a stale line about a verb's behaviour sends them to write a
transport layer around a driver that already has one -- or to skip the
driver entirely as "not ready".

Three of the four task/policy verbs (``run_policy``, ``stop_task``,
``get_task_status``) publish real work on the 500 Hz control loop today;
only ``start_task`` still returns a "not wired yet" refusal, and it does
so precisely because the provider registry lives behind issue #358 whose
vendoring decision is separate from harness#361. This file pins the
docstring to that split so a caller reading the module can trust the line
in it about which verb refuses.

What is graded, and by which cell:

1. **The refusal is attributed only to a verb that refuses.**
   :func:`test_the_docstring_never_attributes_a_refusal_to_a_wired_verb`
   derives the refusing set by *driving* all four verbs and reading which
   responses carry the idiom, then reads every refusal-claiming sentence in
   the docstring and requires the verbs named on its subject side to be a
   subset of that set. This is the cell that grades the drift: the stale
   text named four verbs in one sentence carrying the idiom once, so a rule
   that counts occurrences, or that asks whether the sentence mentions
   ``start_task``, is satisfied by it. Only comparing the *named* set
   against the *measured* set refuses it.

   Attribution is subject-side on purpose. The corrected text names
   ``run_policy`` after the idiom, as the verb a caller should use instead;
   a whole-sentence rule reads that as a second refusal claim and refuses
   the very wording this file ships with.

2. **The wired verbs do not carry the idiom in their own responses.**
   ``run_policy`` returns ``status="success"`` on a callable policy (the
   loop starts); ``stop_task`` returns a success envelope naming ``no task
   is running`` when idle; ``get_task_status`` returns a success envelope
   with ``running=False`` reason ``no task has been started on this
   driver``. These are what makes the derived set in cell 1 a measurement
   rather than a restatement of the docstring.

3. **The idiom, where the docstring still uses it, names the one verb it
   applies to.** Two text-shape cells cover this. They are deliberately
   weak: the shipped docstring describes ``start_task``'s refusal without
   using the ``not wired yet`` phrase at all, so both hold trivially today
   and only bite if a future edit reintroduces the phrase.

On the wiring day -- when #358's provider registry replaces
``start_task``'s refusal -- the cell that fires is
:func:`test_start_task_still_returns_the_not_wired_yet_refusal`, measured
by planting that change. The text-shape cells do not fire, because the
wiring commit does not touch the docstring; that is exactly why cell 1
grades the docstring against driven behaviour instead of against itself.
"""

from __future__ import annotations

import re

import strands_robots.drivers.g1 as g1_module
from strands_robots.drivers.g1 import G1Driver


class _Callable:
    """A callable policy that returns an empty joint dict.

    ``run_policy`` accepts either a ``Policy`` or a bare callable that
    returns a joint-name-keyed action dict. The empty dict is a valid
    (refused-per-step) action for this test's purpose, which is admission:
    the loop must start and be seen running, not command a real posture.
    """

    def __call__(self, _snapshot: object) -> dict[str, object]:
        return {}


def _make_admitted_driver() -> G1Driver:
    """Build a driver whose gates would clear for ``run_policy`` admission.

    ``run_policy`` calls :meth:`_check_motion_gates` with scope ``"motion"``
    before starting the loop; the gates test ``_connected``,
    ``_mode_machine``, ``_fsm_id`` (against the walk/handshake union) and
    the battery floor. ``get_task_status`` and ``stop_task`` do not go
    through the gates, so their success paths hold with just the
    ``_last_task_snapshot`` slot ``None``.

    The driver is left unconnected on the DDS side; the loop does not
    actually publish because there is no publisher, and the test does not
    wait for it to. Admission is what the docstring pin needs to observe.
    """
    driver = G1Driver(network_interface="lo")
    # Bypass ``connect_eagerly`` because it opens a real DDS participant.
    # Populate the slots ``_check_motion_gates`` reads directly so
    # ``run_policy``'s admission can pass without touching the bus.
    driver._connected = True
    driver._mode_machine = 1  # any non-``None`` uint8 clears the mode-machine gate
    driver._fsm_id = 500  # inside HANDSHAKE_FSMS | WALK_FSMS; see g1._check_motion_gates
    driver._battery = {"pct": 90.0}
    return driver


#: The four verbs the docstring's "task and policy paths" bullet describes.
_TASK_VERBS = ("start_task", "run_policy", "stop_task", "get_task_status")

#: The vocabulary a reader greps for when asking "does this verb work yet".
#: Wider than the literal ``not wired yet`` phrase so a reworded claim -- "task
#: and policy paths still refuse" -- is graded by the same rule.
_REFUSAL_IDIOM = re.compile(r"not wired yet|not wired|empty stub|still refuses", re.IGNORECASE)


def _verbs_that_refuse_with_the_idiom() -> frozenset[str]:
    """Drive each task/policy verb and report which responses carry the idiom.

    Derived rather than listed, so the day a verb is wired -- or un-wired --
    the docstring rule below moves with it instead of grading a stale tuple.
    Every verb is driven on an admitted driver so the response seen is the
    verb's own, not the FSM/battery gate wall.
    """
    refusing = set()
    for verb in _TASK_VERBS:
        driver = _make_admitted_driver()
        try:
            if verb == "start_task":
                result = driver.start_task(instruction="stand")
            elif verb == "run_policy":
                result = driver.run_policy(policy_object=_Callable(), duration=0.05)
            else:
                result = getattr(driver, verb)()
            if _REFUSAL_IDIOM.search(str(result)):
                refusing.add(verb)
        finally:
            driver.stop_task()
    return frozenset(refusing)


def _refusal_claims(doc: str) -> list[tuple[frozenset[str], str]]:
    """Return each refusal-claiming sentence with the verbs on its subject side.

    A sentence is a refusal claim when it uses the idiom vocabulary. The verbs
    it *attributes* the refusal to are those named before the idiom; a verb
    named after it is being offered as the alternative, not described as
    refusing.
    """
    flat = " ".join(doc.split())
    claims = []
    for sentence in re.split(r"(?<=[.])\s+", flat):
        match = _REFUSAL_IDIOM.search(sentence)
        if match is None:
            continue
        subject = sentence[: match.start()]
        claims.append((frozenset(v for v in _TASK_VERBS if v in subject), sentence))
    return claims


def test_the_driven_refusal_set_is_not_empty() -> None:
    """Premise: driving the verbs finds a refusal, so the rule below has a subject.

    If every verb were wired the derived set would be empty and the rule would
    hold for any docstring at all, including one that calls all four un-wired.
    """
    assert _verbs_that_refuse_with_the_idiom(), (
        "no task/policy verb returns the refusal idiom, so the docstring rule "
        "below cannot distinguish a correct docstring from a stale one. If "
        "every verb is now wired, replace this file with one grading that."
    )


def test_the_docstring_carries_the_task_and_policy_prose_it_is_graded_on() -> None:
    """Premise: the docstring names the verbs, so the rule reads real prose."""
    doc = g1_module.__doc__ or ""
    named = [verb for verb in _TASK_VERBS if verb in doc]
    assert len(named) == len(_TASK_VERBS), (
        f"module docstring names only {named} of the four task/policy verbs; "
        f"the attribution rule below has nothing to read for the rest."
    )


def test_the_docstring_never_attributes_a_refusal_to_a_wired_verb() -> None:
    """Every verb a refusal sentence names must be one that actually refuses.

    This is the cell that grades the drift the file exists for. The stale text
    read ``Task and policy paths (start_task, run_policy, stop_task,
    get_task_status) return a named "not wired yet" envelope`` -- one sentence,
    one occurrence of the idiom, naming four verbs where one refuses. Counting
    occurrences passes it; asking whether the sentence mentions ``start_task``
    passes it. Comparing the named set against the driven set does not.
    """
    refusing = _verbs_that_refuse_with_the_idiom()
    doc = g1_module.__doc__ or ""
    for named, sentence in _refusal_claims(doc):
        overclaimed = sorted(named - refusing)
        assert not overclaimed, (
            f"module docstring attributes a refusal to {overclaimed}, which "
            f"return a success envelope when driven (only {sorted(refusing)} "
            f"carries the idiom). Sentence was:\n{sentence!r}"
        )


def test_the_docstring_uses_the_not_wired_yet_idiom_at_most_once() -> None:
    """The docstring must not tell three shipped verbs they are un-wired."""
    doc = g1_module.__doc__ or ""
    occurrences = doc.count("not wired yet")
    assert occurrences <= 1, (
        f"module docstring contains 'not wired yet' {occurrences} times; "
        f"three of four task/policy verbs are wired today. The idiom must "
        f"name at most one verb (``start_task``), because that is the one "
        f"verb whose response still returns that string."
    )


def test_the_docstring_names_start_task_when_it_names_the_refusal() -> None:
    """The single 'not wired yet' occurrence, if any, must name ``start_task``.

    Reading the docstring's own words is the load-bearing check: a reader
    scanning for the idiom must land on the one verb it applies to, so the
    surrounding sentence must contain the verb name where the idiom lives.
    """
    doc = g1_module.__doc__ or ""
    if "not wired yet" not in doc:
        # The docstring dropped the idiom entirely; that is fine, because
        # ``start_task``'s own refusal text still carries it (checked below).
        return
    # Read the sentence containing the idiom -- the ``.`` before the idiom
    # to the ``.`` after it -- and require ``start_task`` names itself in it.
    idx = doc.index("not wired yet")
    left = doc.rfind(".", 0, idx)
    right = doc.find(".", idx)
    if left == -1:
        left = 0
    if right == -1:
        right = len(doc)
    sentence = doc[left:right]
    assert "start_task" in sentence, (
        f"'not wired yet' appears in the module docstring in a sentence that "
        f"does not name ``start_task``. Sentence was:\n{sentence!r}"
    )


def test_run_policy_admits_a_callable_policy_and_does_not_return_not_wired_yet() -> None:
    """``run_policy`` is wired: on a callable policy it returns success.

    The gates are set up so the admission check passes; the loop starts
    and the driver reports ``task_running=True``. Reading the response
    payload for the ``not wired yet`` idiom refuses the docstring if this
    verb is ever again described as returning that refusal.
    """
    driver = _make_admitted_driver()
    try:
        result = driver.run_policy(policy_object=_Callable(), duration=0.1)
        assert result["status"] == "success", f"run_policy refused an admitted callable policy: {result}"
        # The wired shape reports the loop is running; the refusal shape
        # would carry a text-only envelope with 'not wired yet' in it.
        text = str(result)
        assert "not wired yet" not in text, f"run_policy's response contains 'not wired yet': {text}"
    finally:
        # Halt the loop we started so this test does not leak a thread.
        driver.stop_task()


def test_get_task_status_reports_no_task_when_idle_and_does_not_refuse() -> None:
    """``get_task_status`` returns a success envelope on an idle driver."""
    driver = G1Driver(network_interface="lo")
    result = driver.get_task_status()
    assert result["status"] == "success", f"get_task_status refused an idle driver: {result}"
    text = str(result)
    assert "not wired yet" not in text, f"get_task_status's response contains 'not wired yet': {text}"


def test_stop_task_reports_no_task_when_idle_and_does_not_refuse() -> None:
    """``stop_task`` returns a success envelope naming the idle state."""
    driver = G1Driver(network_interface="lo")
    result = driver.stop_task()
    assert result["status"] == "success", f"stop_task refused an idle driver: {result}"
    text = str(result)
    assert "not wired yet" not in text, f"stop_task's response contains 'not wired yet': {text}"


def test_start_task_still_returns_the_not_wired_yet_refusal() -> None:
    """``start_task`` is the one verb the docstring may still name.

    The provider registry vendoring decision is tracked separately from
    the transport this driver owns. Until it lands, ``start_task`` returns a
    named refusal identifying the missing registry and containing the ``not
    wired yet`` idiom -- so a caller reading the module docstring for that
    phrase lands here.

    The reason is asserted, not a tracker number: a bare number in a refusal
    resolves against *this* repository, and the one this assertion used to
    require ("#358") resolves to a merged pull request about zenoh mock
    isolation. See tests/test_deferral_strings_do_not_cite_a_landed_change.

    The gates are set up to pass so the refusal seen is the
    verb-specific one, not the FSM/battery gate wall.
    """
    driver = _make_admitted_driver()
    result = driver.start_task(instruction="stand")
    assert result["status"] == "error", f"start_task returned success unexpectedly: {result}"
    text = str(result)
    assert "not wired yet" in text, (
        f"start_task's refusal no longer contains 'not wired yet': {text}. "
        f"If this verb has been wired, the module docstring's mention of "
        f"the idiom must move to whatever verb (if any) still carries it, "
        f"or be removed entirely."
    )
    assert "provider registry" in text, f"start_task's refusal no longer names the missing provider registry: {text}"
