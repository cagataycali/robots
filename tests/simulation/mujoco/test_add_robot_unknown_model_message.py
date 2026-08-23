"""add_robot surfaces an actionable error for an unknown/unresolvable model.

Contract: when a caller names a robot that resolves to no model (a mistyped
registry key, or an instance label given without a model source), the engine
returns the same actionable "no model found (did you mean ...?) / list_urdfs /
pass data_config= or urdf_path=" error that the ``data_config`` exit and the
top-level ``Robot()`` factory give - not a dead-end "Either urdf_path or
data_config is required" that never names the robot nor points at discovery.

The bare "supply a model source" message is preserved for the genuine no-name
case (``add_robot()`` with nothing to resolve), and the deprecated positional
name-as-registry-key short form still resolves a VALID name.

A model source that is SUPPLIED but empty is a fourth condition, and it is not
the "given without a model source" one above: read by truthiness the two were
indistinguishable, so ``urdf_path=""`` fell through to the name lookup and got
byte-identically what passing no source returns - a name diagnosis, close-match
suggestions for a name the caller never asked to resolve, and advice to pass the
kwarg they had just passed. It is refused naming that parameter instead, the way
``register_urdf`` already refuses an empty ``urdf_path``. Whitespace stays a
path ("File not found"), which is the boundary.
"""

import os

import pytest

os.environ.setdefault("MUJOCO_GL", "egl")


@pytest.fixture
def world():
    from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine

    sim = MuJoCoSimEngine(tool_name="test_add_robot_unknown_model")
    sim.create_world()
    yield sim
    sim.cleanup()


def _msg(result):
    assert result["status"] == "error", f"expected error, got: {result}"
    return result["content"][0]["text"]


class TestUnknownModelMessage:
    def test_positional_typo_is_actionable(self, world):
        """A mistyped positional robot name names the robot + offers discovery."""
        msg = _msg(world.add_robot("panda_typo"))
        assert "panda_typo" in msg, msg
        assert "list_urdfs" in msg, msg
        # dead-end message must NOT be what a named typo gets
        assert "Either urdf_path or data_config is required" not in msg, msg

    def test_typo_offers_close_match_suggestion(self, world):
        """A near-miss of a real robot suggests the correct name (difflib)."""
        msg = _msg(world.add_robot("panda_typo"))
        assert "panda" in msg, msg  # 'Did you mean: panda, ...'

    def test_instance_label_without_model_points_at_both_options(self, world):
        """A name given without a model source explains BOTH interpretations:
        pick a registered model (list_urdfs) OR supply data_config=/urdf_path=."""
        msg = _msg(world.add_robot(name="myarm"))
        assert "myarm" in msg, msg
        assert "list_urdfs" in msg, msg
        assert "data_config" in msg and "urdf_path" in msg, msg

    def test_no_args_keeps_generic_model_source_message(self, world):
        """No caller-provided name -> the generic 'supply a model source' error
        is preserved (the actionable unknown-robot message is name-specific)."""
        msg = _msg(world.add_robot())
        assert msg == "Either urdf_path or data_config is required.", msg

    def test_deprecated_positional_valid_name_still_resolves(self, world):
        """The deprecated name-as-registry-key short form still resolves a VALID
        name past the unknown-model gate (regression guard on the fallback)."""
        result = world.add_robot("so100")
        txt = result["content"][0]["text"] if result.get("content") else ""
        assert "No model found" not in txt, txt


class TestASuppliedEmptyModelSourceIsNotAnOmittedOne:
    """``urdf_path=""`` / ``data_config=""`` name the empty parameter."""

    @pytest.mark.parametrize("param", ["urdf_path", "data_config"])
    def test_a_supplied_empty_source_names_that_parameter(self, world, param):
        """The refusal names the parameter the caller left empty."""
        msg = _msg(world.add_robot(name="probe", **{param: ""}))
        assert f"'{param}'" in msg, msg
        assert "non-empty string" in msg, msg

    @pytest.mark.parametrize("param", ["urdf_path", "data_config"])
    def test_the_refusal_does_not_advise_passing_what_was_passed(self, world, param):
        """A remedy the caller already applied is a dead end, not advice.

        Pre-fix the message ended "Or pass data_config=<registered model> or
        urdf_path=<file>" for a call that had passed exactly that.
        """
        msg = _msg(world.add_robot(name="probe", **{param: ""}))
        assert "pass data_config=<registered model> or urdf_path=<file>" not in msg, msg

    @pytest.mark.parametrize("param", ["urdf_path", "data_config"])
    def test_a_supplied_empty_source_does_not_diagnose_the_name(self, world, param):
        """The problem is the empty value, not the instance label.

        Spelling suggestions for a name the caller never asked to resolve are
        the wrong advice here for the same reason they are wrong for a
        hardware-only entry and a missing asset: the name is not what failed.
        """
        msg = _msg(world.add_robot(name="probe", **{param: ""}))
        assert "No model found for" not in msg, msg
        assert "Did you mean" not in msg, msg

    def test_a_supplied_empty_source_is_distinguishable_from_supplying_none(self, world):
        """Supplying an empty source must not report as supplying no source.

        This is the headline: pre-fix both calls returned the same bytes, so
        the report could not tell "you passed an empty path" from "you passed
        no path".
        """
        # The SAME label in both calls: the label is interpolated into the
        # pre-fix message, so two different names would differ for an
        # incidental reason rather than because the reports say different
        # things. Neither call adds a robot, so the label stays free.
        empty = _msg(world.add_robot(name="probe", urdf_path=""))
        omitted = _msg(world.add_robot(name="probe"))
        assert empty != omitted, f"indistinguishable: {empty!r}"

    def test_an_empty_source_without_a_name_names_the_parameter(self, world):
        """No name + an empty source still names the parameter.

        Pre-fix this said the parameter "is required" to a caller who had
        supplied it.
        """
        msg = _msg(world.add_robot(urdf_path=""))
        assert "'urdf_path'" in msg, msg
        assert "is required" not in msg, msg

    def test_the_rule_covers_every_model_source_the_signature_declares(self, world):
        """The graded set is the model-source parameters ``add_robot`` accepts.

        Derived from the signature so a third model source added later is
        refused-when-empty too, or this fails until it is graded.
        """
        import inspect

        params = set(inspect.signature(type(world).add_robot).parameters)
        declared = {p for p in params if p == "data_config" or p.endswith("_path")}
        assert declared == {"urdf_path", "data_config"}, declared
        for param in sorted(declared):
            msg = _msg(world.add_robot(name=f"probe_{param}", **{param: ""}))
            assert f"'{param}'" in msg, msg


class TestTheBoundaryAndTheShortFormsAreUntouched:
    """Controls: what an empty source must NOT change."""

    def test_whitespace_is_a_path_not_an_empty_value(self, world):
        """``urdf_path=" "`` stays a path, so it reports the missing file.

        Mirrors ``register_urdf``'s ``if not urdf_path`` boundary exactly -
        this rule refuses the empty value, not an unusable one.
        """
        msg = _msg(world.add_robot(name="probe", urdf_path=" "))
        assert "File not found" in msg, msg

    def test_an_omitted_source_still_diagnoses_the_name(self, world):
        """An unresolvable label with NO source keeps the name diagnosis."""
        msg = _msg(world.add_robot(name="panda_typo"))
        assert "No model found for 'panda_typo'" in msg, msg
        assert "list_urdfs" in msg, msg

    def test_a_taken_label_still_reports_the_collision_first(self, world):
        """The empty-source rule does not displace the name-collision report.

        The guard sits after the existing name-taken check, so a caller who
        made both mistakes is told about the label they cannot reuse - the
        precedence this rule was added behind, not in front of.
        """
        # Hoisted out of the assert: this call IS the premise (it takes the
        # label), so under `python -O` an asserted call would be stripped and
        # the collision below would never be set up.
        added = world.add_robot(name="arm", data_config="so101")
        assert added["status"] == "success", added
        msg = _msg(world.add_robot(name="arm", urdf_path=""))
        assert "already exists" in msg, msg

    def test_an_empty_name_still_derives_a_label(self, world):
        """``name=""`` is the documented derive-a-label short form, not an error.

        The falsy-value rule applies to the model SOURCE; ``name`` documents
        ``None``/``""`` as the short form that auto-derives a label.
        """
        result = world.add_robot(name="", data_config="so101")
        assert result["status"] == "success", result
        assert "so101" in world.list_robots()
