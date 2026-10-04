"""The ``SimEngine.robot_joint_names`` docstring must not promise a width it
doesn't deliver.

Pre-fix the ABC docstring said:

    "This order is the one a LeRobotDataset recording writes the
     observation.state columns in, so it is the order a policy must read
     that vector back in."

That claim is false for any robot with a floating base: the first joint is a
6-DoF free joint (`nq=7`, `nv=6`), and every scalar surface on every backend
already skips it. For `g1` the roster is 30 names and the recorded
`observation.state` vector is 29-wide; a policy keyed off the roster has one
too many columns and the mismatch is silent.

The actual rule is already encoded by three siblings:
- ``NewtonSimEngine.robot_action_keys`` (newton/simulation.py:762-790) is the
  policy-facing roster and documents the free-base narrowing in long form;
- ``test_rl_action_head_binds_action_keys.py`` pins the width-differ invariant;
- ``test_ros_sim_bridge.py`` pins the symmetric name/value filter the ROS
  bridge needs because of the same gap.

This pin keeps the ABC docstring honest about that: it has to point policy
callers at ``robot_action_keys``, not at ``robot_joint_names``, and it has to
name the floating-base case where the two widths disagree.
"""

from __future__ import annotations

import inspect

from strands_robots.simulation.base import SimEngine


class TestRobotJointNamesDocstring:
    """The abstract declaration has to describe what the method actually is."""

    def _doc(self) -> str:
        doc = SimEngine.robot_joint_names.__doc__
        assert doc is not None, "SimEngine.robot_joint_names must carry a docstring"
        return doc

    def test_does_not_claim_observation_state_binding(self) -> None:
        """A docstring that says the opposite of the method's behaviour is a trap.

        The pre-fix text wasn't ambiguous: it said this list orders
        ``observation.state`` and so a policy must read it. For a floating-base
        robot neither clause is true. Any wording that puts a reader back on
        that path re-opens the footgun; this pin fires on the exact pre-fix
        phrasing.
        """
        text = " ".join(self._doc().split())
        # Pre-fix text, modulo whitespace. Backticks preserved.
        banned = (
            "This order is the one a LeRobotDataset recording writes the "
            "``observation.state`` columns in, so it is the order a policy "
            "must read that vector back in"
        )
        assert banned not in text, (
            "robot_joint_names docstring must not promise observation.state ordering "
            "(a floating-base robot's roster is one wider than that vector); "
            "point policy callers at robot_action_keys instead."
        )

    def test_mentions_robot_action_keys_for_policy_binding(self) -> None:
        """A reader who wants the policy-side list has to be redirected explicitly."""
        text = self._doc().lower()
        assert "robot_action_keys" in text, (
            "robot_joint_names docstring must name robot_action_keys as the "
            "method policies/Policy.set_robot_state_keys/PolicyRunner.replay bind."
        )

    def test_names_the_floating_base_case(self) -> None:
        """The failure mode has to be named by its handle so a grep lands here."""
        text = self._doc().lower()
        assert "floating" in text and ("free joint" in text or "free-base" in text or "6-dof" in text), (
            "robot_joint_names docstring must mention the floating-base / free-joint "
            "case where its width exceeds observation.state and robot_action_keys."
        )

    def test_is_still_abstract(self) -> None:
        """The method is a backend contract; a docstring edit must not change that."""
        assert getattr(SimEngine.robot_joint_names, "__isabstractmethod__", False), (
            "SimEngine.robot_joint_names must remain abstract."
        )
        # And its signature is still (self, robot_name).
        sig = inspect.signature(SimEngine.robot_joint_names)
        params = list(sig.parameters.keys())
        assert params == ["self", "robot_name"], params
