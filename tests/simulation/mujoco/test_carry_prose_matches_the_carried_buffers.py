"""No ``scene_ops`` docstring says a state buffer is uncarried that the code carries.

A scene rebuild has to move the dynamic state across a fresh ``MjData`` by name,
and which buffers it moves is decided in code and explained in prose. The prose
is what the next contributor reads before adding a buffer, so a stale claim there
is not cosmetic: it tells them to add a carry that already exists, or to leave one
out that is needed.

That drift has happened once. #2376 documented ``qfrc_applied`` as "deliberately
NOT carried here" on the grow path, reasoning that nothing in the package writes
it. #2380 then measured what that reasoning missed -- the eject path snapshots the
buffer with the rest of the joint state, so a latched torque survived
``remove_robot`` and vanished on ``add_object``, and a hinge held at 0.5 N m
stopped turning the moment anything entered the scene under a
``"status": "success"`` -- and shipped
:func:`~strands_robots.simulation.mujoco.scene_ops._snapshot_joint_forces` with
its restore. It left #2376's bullet in place, so one docstring then said three
things: that both force buffers are snapshotted by name (true), that
``xfrc_applied`` is carried (true), and that ``qfrc_applied`` is deliberately not
(false, and followed by an instruction to add the carry that had just landed).

Nothing graded the prose against the code, because #2380's own suite is entirely
behavioural: it measures the joint's angle, which cannot see a sentence. This
grades the sentence.

Both halves are DERIVED from the module. The functions checked are those that
call a ``_snapshot_*`` helper, and the buffers each one carries are the
``data.<buffer>`` fields those helpers read -- so a buffer or a carrying function
added later is held to the rule the hour it lands, rather than inheriting an
exemption by being absent from a list. The derived-vs-hardcoded pair in
:class:`TestTheRuleIsNotVacuous` measures that difference.

Scope. Only the NEGATIVE claim is graded: a carried buffer must not be documented
as uncarried. The positive form -- every carrying helper must be named in its
caller's docstring -- is deliberately not required. ``_snapshot_scene_state``
carries ``xfrc_applied`` under a one-line docstring, and three spec-rollback
callers reach ``_snapshot_spec`` without naming it; demanding a helper name from
each would be a style rule about docstring length rather than a check on whether
the prose is true.

The module needs no ``mujoco`` for any of this, so this file does not gate on it:
every cell here reads source text.
"""

from __future__ import annotations

import ast
import inspect
import re

from strands_robots.simulation.mujoco import scene_ops

_REBUILD_FUNCTION = "_recompile_preserving_state"

# A claim that the documented function itself does not carry a buffer.
#
# Deliberately narrow. The same docstring correctly says ``spec.recompile``
# "transfers neither applied-force buffer at all" and that both "come back
# entirely zero" -- true statements about the COMPILER, and the reason the carry
# exists. A pattern that read "not transferred" or "dropped" as a denial would
# flag those, so the phrasing graded here is the one that speaks about carrying.
_DENIES_CARRY = re.compile(r"\bnot carried\b|\bnever carried\b", re.IGNORECASE)

# A floor under the derived inventory. A discovery pass that silently matched
# nothing would report a clean module for the same reason a clean module does.
_MINIMUM_CARRYING_FUNCTIONS = 3


def _flat(text: str) -> str:
    """Collapse a docstring's wrapping so a claim spanning two lines still reads."""
    return " ".join(text.split())


def _functions(source: str) -> dict[str, ast.FunctionDef]:
    return {n.name: n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef)}


def _snapshot_helpers_called(function: ast.FunctionDef) -> set[str]:
    """Names of the ``_snapshot_*`` helpers ``function`` calls, excluding itself."""
    return {
        call.func.id
        for call in ast.walk(function)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id.startswith("_snapshot_")
        and call.func.id != function.name
    }


def carried_buffers(source: str, function_name: str) -> dict[str, set[str]]:
    """The ``MjData`` buffers ``function_name`` carries, and the helper that reads each.

    Derived rather than listed: a snapshot helper's subject is whichever
    ``data.<buffer>`` fields it reads, so the carried set follows the code.

    Args:
        source: Module source text to parse.
        function_name: The function whose carries are being derived.

    Returns:
        ``buffer name -> {helper names that read it}``, empty when the function
        calls no snapshot helper or when its helpers read no ``data`` field.
    """
    functions = _functions(source)
    function = functions.get(function_name)
    if function is None:
        return {}
    buffers: dict[str, set[str]] = {}
    for helper_name in _snapshot_helpers_called(function):
        helper = functions.get(helper_name)
        if helper is None:
            continue
        for node in ast.walk(helper):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "data":
                buffers.setdefault(node.attr, set()).add(helper_name)
    return buffers


def carrying_functions(source: str) -> list[str]:
    """Every function in ``source`` that carries at least one ``MjData`` buffer."""
    return sorted(name for name in _functions(source) if carried_buffers(source, name))


def buffers_documented_as_uncarried(source: str, function_name: str) -> list[tuple[str, str]]:
    """Carried buffers whose own docstring says they are not carried.

    A sentence is the unit rather than a bullet: the docstring's bullets run into
    the prose that follows them, so a denial in a trailing paragraph would be
    attributed to the bullet above it.

    Args:
        source: Module source text to parse.
        function_name: The function whose docstring is graded.

    Returns:
        ``(buffer, sentence)`` for each denial, empty when the prose agrees with
        the code.
    """
    function = _functions(source).get(function_name)
    if function is None:
        return []
    docstring = _flat(ast.get_docstring(function) or "")
    buffers = carried_buffers(source, function_name)
    denials: list[tuple[str, str]] = []
    for sentence in re.split(r"(?<=[.:]) ", docstring):
        if not _DENIES_CARRY.search(sentence):
            continue
        denials.extend((buffer, sentence) for buffer in buffers if buffer in sentence)
    return denials


# ── Constructed exemplars ─────────────────────────────────────
# The module is clean once the prose is corrected, so it can no longer exercise a
# rejection. These four drive the predicate directly.

_CARRIES_AND_DENIES = '''
def _snapshot_forces(model, data, mj):
    """Read the buffer."""
    return {0: data.qfrc_applied[0]}

def _rebuild(world, spec):
    """Rebuild the scene.

    * ``qfrc_applied`` is deliberately NOT carried here: nothing writes it.
    """
    return _snapshot_forces(None, None, None)
'''

_DENIES_A_BUFFER_IT_DOES_NOT_CARRY = '''
def _snapshot_forces(model, data, mj):
    """Read the buffer."""
    return {0: data.qfrc_applied[0]}

def _rebuild(world, spec):
    """Rebuild the scene.

    * ``xfrc_applied`` is deliberately NOT carried here: nothing writes it.
    """
    return _snapshot_forces(None, None, None)
'''

_CARRIES_AND_SAYS_SO = '''
def _snapshot_forces(model, data, mj):
    """Read the buffer."""
    return {0: data.qfrc_applied[0]}

def _rebuild(world, spec):
    """Rebuild the scene.

    * ``qfrc_applied`` IS carried here, by :func:`_snapshot_forces`.
    """
    return _snapshot_forces(None, None, None)
'''

# The true claim about the compiler, in the shape the real docstring uses. It must
# not read as a claim about the function's own carrying.
_DESCRIBES_WHAT_THE_COMPILER_DROPS = '''
def _snapshot_forces(model, data, mj):
    """Read the buffer."""
    return {0: data.qfrc_applied[0]}

def _rebuild(world, spec):
    """Rebuild the scene.

    ``spec.recompile`` transfers neither applied-force buffer at all, so
    ``qfrc_applied`` and ``xfrc_applied`` are dropped by it and both come back
    entirely zero.
    """
    return _snapshot_forces(None, None, None)
'''

# A denial that is TRUE, sharing a docstring with a carried buffer. Reading the
# docstring whole rather than sentence by sentence would attribute this bullet's
# "not carried" to the ``qfrc_applied`` named in the bullet above it.
_DENIES_ONE_BUFFER_BESIDE_A_CARRIED_ONE = '''
def _snapshot_forces(model, data, mj):
    """Read the buffer."""
    return {0: data.qfrc_applied[0]}

def _rebuild(world, spec):
    """Rebuild the scene.

    * ``qfrc_applied`` IS carried here, by :func:`_snapshot_forces`.
    * ``plugin_state`` is deliberately NOT carried here: nothing writes it.
    """
    return _snapshot_forces(None, None, None)
'''


class TestTheCarriedSetIsDerivedFromTheCode:
    """Premise: the inventory both halves read is the module's own, not a list."""

    def test_the_rebuild_carries_both_applied_force_buffers(self):
        buffers = carried_buffers(inspect.getsource(scene_ops), _REBUILD_FUNCTION)
        assert set(buffers) == {"qfrc_applied", "xfrc_applied"}, (
            "the grow path's carried set should be derived as exactly the two applied-force "
            f"buffers its snapshot helpers read, got {sorted(buffers)}"
        )

    def test_each_buffer_names_the_helper_that_reads_it(self):
        buffers = carried_buffers(inspect.getsource(scene_ops), _REBUILD_FUNCTION)
        assert buffers["qfrc_applied"] == {"_snapshot_joint_forces"}
        assert buffers["xfrc_applied"] == {"_snapshot_body_wrenches"}

    def test_the_discovery_pass_reaches_more_than_the_rebuild(self):
        functions = carrying_functions(inspect.getsource(scene_ops))
        assert _REBUILD_FUNCTION in functions
        assert len(functions) >= _MINIMUM_CARRYING_FUNCTIONS, (
            "a discovery pass that matched almost nothing would report a clean module for "
            f"the same reason a clean module does, got {functions}"
        )


class TestNoCarriedBufferIsDocumentedAsUncarried:
    """The regression: prose that contradicts the carry the code performs."""

    def test_the_rebuild_prose_agrees_with_the_code(self):
        denials = buffers_documented_as_uncarried(inspect.getsource(scene_ops), _REBUILD_FUNCTION)
        assert denials == [], (
            "the grow path documents a buffer as uncarried that it carries, so a reader "
            f"deciding whether to add the carry is told to add one that exists: {denials}"
        )

    def test_no_carrying_function_in_the_module_denies_a_buffer_it_carries(self):
        source = inspect.getsource(scene_ops)
        offenders = {name: buffers_documented_as_uncarried(source, name) for name in carrying_functions(source)}
        assert {name: found for name, found in offenders.items() if found} == {}


class TestTheRuleIsNotVacuous:
    """Constructed exemplars: the module cannot exercise a rejection once clean."""

    def test_a_denied_carry_is_reported(self):
        denials = buffers_documented_as_uncarried(_CARRIES_AND_DENIES, "_rebuild")
        assert [buffer for buffer, _ in denials] == ["qfrc_applied"]

    def test_denying_a_buffer_the_function_does_not_carry_is_accepted(self):
        assert buffers_documented_as_uncarried(_DENIES_A_BUFFER_IT_DOES_NOT_CARRY, "_rebuild") == []

    def test_documenting_the_carry_is_accepted(self):
        assert buffers_documented_as_uncarried(_CARRIES_AND_SAYS_SO, "_rebuild") == []

    def test_naming_what_the_compiler_drops_is_not_a_denial_of_the_carry(self):
        assert buffers_documented_as_uncarried(_DESCRIBES_WHAT_THE_COMPILER_DROPS, "_rebuild") == [], (
            "the real docstring explains that spec.recompile transfers neither buffer, which is "
            "why the carry exists; reading that as a denial would flag a true sentence"
        )

    def test_a_true_denial_is_not_pinned_on_the_carried_buffer_beside_it(self):
        assert buffers_documented_as_uncarried(_DENIES_ONE_BUFFER_BESIDE_A_CARRIED_ONE, "_rebuild") == [], (
            "a docstring may say one buffer is not carried while carrying another; reading it "
            "whole rather than by sentence would blame the carried buffer for its neighbour's claim"
        )

    def test_the_exemplars_reach_both_outcomes(self):
        outcomes = {
            bool(buffers_documented_as_uncarried(source, "_rebuild"))
            for source in (
                _CARRIES_AND_DENIES,
                _DENIES_A_BUFFER_IT_DOES_NOT_CARRY,
                _CARRIES_AND_SAYS_SO,
                _DESCRIBES_WHAT_THE_COMPILER_DROPS,
                _DENIES_ONE_BUFFER_BESIDE_A_CARRIED_ONE,
            )
        }
        assert outcomes == {True, False}
