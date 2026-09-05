"""A refused primitive in the SO-101 reference pick is reported, not carried past.

``examples/18_so101_pick_and_lift.py`` composes ten simulation primitives, each
of which answers with a tool envelope. Measured on ``ca57d58d`` with the
``[sim-mujoco]`` IK solver absent, the example discarded all ten envelopes and
returned a hard-coded ``status="success"``::

    move_to hover   -> error   move_to: IK bridge unavailable: The mink IK bridge
                               needs 'mink' + 'mujoco' + a qpsolvers backend...
                               uv pip install 'strands-robots[sim-mujoco]'
    move_to descend -> error   (same)
    move_to lift    -> error   (same)

    run_pick() -> {'status': 'success', 'lifted_mm': 0.0, 'success': False}
    printed    -> "PICK FAILED - cube lifted only 0.0 mm"

Three refusals, each naming the install that fixes them, and the run reported a
completed pick that merely failed to lift. That is the *one* outcome the module
docstring spends three paragraphs saying to expect - a friction pinch holds
nothing, "0 mm lift with the fingers in contact" - so a reader who hits the
dependency gap concludes the reference itself is the known-broken friction case
and never sees the remedy the library already produced.

``lifted_mm >= 80`` in the sibling behavioural pin
``tests/test_examples_so101_pick_lifts.py`` cannot catch this: a refused run
lifts 0 mm, which is exactly what that test's own failure message attributes to
"a friction-only regression". And its first assertion, ``result["status"] ==
"success"``, was vacuous while ``status`` was a literal - it could not fail. This
module grades the envelope handling, which is the half that reads the refusals;
that module remains the half that pins the lift.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mujoco")

_EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "18_so101_pick_and_lift.py"

# The refusal text the absent-IK envelope actually carries, abbreviated. The
# remedy is the part a discarded envelope costs the reader, so it is what the
# summary is checked for.
_REMEDY = "uv pip install 'strands-robots[sim-mujoco]'"
_REFUSAL = {
    "status": "error",
    "content": [{"text": f"move_to: IK bridge unavailable: needs 'mink' + a qpsolvers backend. {_REMEDY}"}],
}

# One primitive per stage of the sequence: the first call, the mid-sequence IK
# step that the absent solver really refuses, and the grasp-assist step after
# it. A guard that only checked the first call would pass on the first row
# alone.
_REFUSABLE_STEPS = ["add_object", "move_to", "attach_bodies"]


def _load_example() -> Any:
    """Load the example by path (a leading digit makes it unimportable by name)."""
    spec = importlib.util.spec_from_file_location("so101_pick_and_lift", _EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestARefusedPrimitiveReachesTheSummary:
    """The summary names the refused step and carries the refusal's own text."""

    @pytest.mark.parametrize("step", _REFUSABLE_STEPS)
    def test_the_summary_reports_the_refusal_instead_of_a_completed_pick(
        self, step: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A refused step is an error summary, not ``success`` with a 0 mm lift."""
        from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine

        monkeypatch.setattr(MuJoCoSimEngine, step, lambda *a, **k: _REFUSAL, raising=True)

        result = _load_example().run_pick()

        assert result["status"] == "error", result
        assert step in result["step"], result
        assert _REMEDY in result["detail"], result
        assert result["success"] is False, result

    def test_a_success_envelope_passes_through_unchanged(self) -> None:
        """Over-reach control: the guard must not refuse a healthy envelope.

        Keeps the parametrized rows above from passing for the wrong reason - a
        guard that treated every envelope as a refusal would satisfy them all
        while breaking the pick entirely. Unit-level on the helper so it needs
        no IK solver, which is the dependency the rows above stand in for.
        """
        module = _load_example()
        envelope = {"status": "success", "content": [{"text": "'cube' added"}]}

        assert module._ok("add_object(cube)", envelope) is envelope

    def test_the_refusal_detail_is_the_envelope_text_and_not_the_step_label(self) -> None:
        """The remedy survives into ``detail``; the label alone would not help."""
        module = _load_example()

        with pytest.raises(module._Refused) as caught:
            module._ok("move_to(lift)", _REFUSAL)

        assert caught.value.step == "move_to(lift)"
        assert _REMEDY in caught.value.detail
