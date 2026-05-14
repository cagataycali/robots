"""Tests for the LIBERO BDDL parser.

Covers:

* Tokenizer handling of comments, quoted strings, nested parens.
* S-expression parsing - depth, arity, EOF errors.
* Top-level ``(define ...)`` structure + section extraction
  (``:domain``, ``:objects``, ``:init``, ``:goal``, ``:language``).
* Predicate compilation for every entry in ``PREDICATE_VOCABULARY``.
* Boolean combinators (``and`` / ``or`` / ``not``) with short-circuit behaviour.
* Rejection of unknown predicates / wrong arities.
* Round-trip on a curated 5-task subset covering each predicate family.

The compiled callables are executed against the same fake sims used by
``tests/simulation/test_benchmark_predicates.py`` - no LIBERO / MuJoCo
dependency required.
"""

from __future__ import annotations

from typing import Any

import pytest

from strands_robots.benchmarks.libero.bddl_parser import (
    PREDICATE_VOCABULARY,
    And,
    BDDLParseError,
    Not,
    Or,
    Pred,
    _tokenize,
    compile_goal,
    parse_bddl,
    parse_bddl_file,
)

# Fake sims


class _BodyStateSim:
    def __init__(self, bodies: dict[str, dict[str, Any]]):
        self._bodies = bodies

    def get_body_state(self, body_name: str) -> dict[str, Any]:
        if body_name not in self._bodies:
            return {"status": "error", "content": [{"text": "missing"}]}
        return {
            "status": "success",
            "content": [
                {"text": body_name},
                {
                    "json": {
                        "position": self._bodies[body_name].get("position", [0, 0, 0]),
                        "quaternion": self._bodies[body_name].get("quaternion", [1, 0, 0, 0]),
                        "mass": 1.0,
                    }
                },
            ],
        }

    def get_observation(self, *_, **__) -> dict[str, Any]:
        return self._bodies.get("_joints", {})


class _ContactSim:
    def __init__(self, contacts: list[dict[str, str]]):
        self._contacts = contacts

    def get_contacts(self) -> dict[str, Any]:
        return {
            "status": "success",
            "content": [
                {"text": f"{len(self._contacts)} contacts"},
                {"json": {"contacts": self._contacts, "n_contacts": len(self._contacts)}},
            ],
        }


class _CombinedSim(_BodyStateSim, _ContactSim):
    """Both body state and contacts for multi-predicate goals."""

    def __init__(
        self,
        bodies: dict[str, dict[str, Any]] | None = None,
        contacts: list[dict[str, str]] | None = None,
    ):
        _BodyStateSim.__init__(self, bodies or {})
        _ContactSim.__init__(self, contacts or [])


# Tokenizer


class TestTokenize:
    def test_basic(self):
        assert _tokenize("(and a b)") == ["(", "and", "a", "b", ")"]

    def test_comments_stripped(self):
        assert _tokenize("(foo) ; trailing comment\n(bar)") == ["(", "foo", ")", "(", "bar", ")"]

    def test_quoted_strings_preserved(self):
        toks = _tokenize('(:language "pick the red cube")')
        # The quoted region is a single token, including the quotes.
        assert '"pick the red cube"' in toks

    def test_unterminated_quote_errors(self):
        with pytest.raises(BDDLParseError, match="unterminated quoted string"):
            _tokenize('(:language "unterminated')


# Top-level parser


class TestParseBDDL:
    def test_minimal(self):
        text = """
            (define (problem libero_pick)
              (:domain kitchen)
              (:goal (on cube_1 plate_1)))
        """
        problem = parse_bddl(text)
        assert problem.name == "libero_pick"
        assert problem.domain == "kitchen"
        assert isinstance(problem.goal, Pred)
        assert problem.goal.name == "on"
        assert problem.goal.args == ("cube_1", "plate_1")

    def test_extracts_language(self):
        text = """
            (define (problem p1)
              (:language "pick up the red cube")
              (:goal (grasped cube_1)))
        """
        problem = parse_bddl(text)
        assert problem.language == "pick up the red cube"

    def test_extracts_objects_flattening_typed_syntax(self):
        """PDDL-style ``obj1 obj2 - type`` annotations are flattened to symbols."""
        text = """
            (define (problem p)
              (:objects cube_1 plate_1 - object table_1 - fixture)
              (:goal (on cube_1 plate_1)))
        """
        problem = parse_bddl(text)
        assert problem.objects == ["cube_1", "plate_1", "object", "table_1", "fixture"]

    def test_extracts_init_clauses(self):
        text = """
            (define (problem p)
              (:init (on cube_1 table_1) (upright bottle_1))
              (:goal (on cube_1 plate_1)))
        """
        problem = parse_bddl(text)
        assert len(problem.init) == 2
        # Each init clause is a compiled Pred.
        assert all(isinstance(n, Pred) for n in problem.init)

    def test_goal_with_and(self):
        text = """
            (define (problem p)
              (:goal (and (on cube_1 plate_1) (upright cube_1))))
        """
        problem = parse_bddl(text)
        assert isinstance(problem.goal, And)
        assert len(problem.goal.clauses) == 2

    def test_goal_with_or_and_not(self):
        text = """
            (define (problem p)
              (:goal (or (grasped cube_1) (not (on cube_1 table_1)))))
        """
        problem = parse_bddl(text)
        assert isinstance(problem.goal, Or)
        inner = problem.goal.clauses[1]
        assert isinstance(inner, Not)

    def test_missing_define_rejected(self):
        with pytest.raises(BDDLParseError, match="top-level"):
            parse_bddl("(problem foo)")

    def test_empty_input_rejected(self):
        with pytest.raises(BDDLParseError):
            parse_bddl("")

    def test_missing_paren_rejected(self):
        with pytest.raises(BDDLParseError, match="closing"):
            parse_bddl("(define (problem p)")

    def test_trailing_tokens_rejected(self):
        with pytest.raises(BDDLParseError, match="trailing"):
            parse_bddl("(define (problem p) (:goal (on a b))) (extra)")


# Predicate vocabulary


class TestPredicateVocabulary:
    @pytest.mark.parametrize("bddl_name", sorted(PREDICATE_VOCABULARY.keys()))
    def test_every_predicate_compiles(self, bddl_name: str):
        """Each BDDL predicate must produce a compilable goal with a valid argc."""
        sample_args = {
            "on": "cube_1 table_1",
            "near": "cube_1 gripper_1",
            "inside": "cube_1 basket_1",
            "open": "drawer_joint",
            "closed": "drawer_joint",
            "grasped": "cube_1",
            "upright": "bottle_1",
        }
        args = sample_args[bddl_name]
        text = f"(define (problem p) (:goal ({bddl_name} {args})))"
        problem = parse_bddl(text)
        # Must compile without error.
        fn = compile_goal(problem.goal)  # type: ignore[arg-type]
        assert callable(fn)

    def test_unknown_predicate_rejected_with_list(self):
        text = "(define (problem p) (:goal (telekinesis cube_1)))"
        with pytest.raises(BDDLParseError) as exc:
            parse_bddl(text)
        assert "unknown predicate" in str(exc.value).lower()
        # Error must list the valid vocabulary so the author can fix it.
        for expected in ("on", "grasped", "upright"):
            assert expected in str(exc.value)

    @pytest.mark.parametrize(
        "expr,reason",
        [
            ("(on cube_1)", "wrong arity"),
            ("(on cube_1 plate_1 extra)", "extra arg"),
            ("(grasped)", "no arg"),
            ("(upright a b)", "extra arg"),
        ],
    )
    def test_wrong_arity_rejected(self, expr: str, reason: str):
        with pytest.raises(BDDLParseError):
            parse_bddl(f"(define (problem p) (:goal {expr}))")

    def test_not_with_wrong_arity(self):
        with pytest.raises(BDDLParseError, match="not"):
            parse_bddl("(define (problem p) (:goal (not (on a b) (on c d))))")


# Compiled goal evaluation


class TestCompileGoal:
    def test_and_short_circuits(self):
        """``and`` must evaluate to False as soon as one clause fails."""
        text = """
            (define (problem p)
              (:goal (and
                (on cube_1 table_1)
                (upright bottle_1))))
        """
        problem = parse_bddl(text)
        fn = compile_goal(problem.goal)  # type: ignore[arg-type]
        sim_hit = _BodyStateSim(
            {
                "cube_1": {"position": [0, 0, 0.2]},
                "table_1": {"position": [0, 0, 0.0]},
                "bottle_1": {"quaternion": [1.0, 0.0, 0.0, 0.0]},
            }
        )
        sim_miss_upright = _BodyStateSim(
            {
                "cube_1": {"position": [0, 0, 0.2]},
                "table_1": {"position": [0, 0, 0.0]},
                "bottle_1": {"quaternion": [0.707, 0.707, 0, 0]},
            }
        )
        assert fn(sim_hit) is True
        assert fn(sim_miss_upright) is False

    def test_or(self):
        text = "(define (problem p) (:goal (or (upright a) (upright b))))"
        problem = parse_bddl(text)
        fn = compile_goal(problem.goal)  # type: ignore[arg-type]
        only_b = _BodyStateSim(
            {
                "a": {"quaternion": [0.707, 0.707, 0, 0]},  # tipped
                "b": {"quaternion": [1.0, 0, 0, 0]},  # upright
            }
        )
        assert fn(only_b) is True

    def test_not(self):
        text = "(define (problem p) (:goal (not (grasped cube_1))))"
        problem = parse_bddl(text)
        fn = compile_goal(problem.goal)  # type: ignore[arg-type]
        # Without any contacts, grasped is False, so (not grasped) is True.
        no_contacts = _ContactSim([])
        assert fn(no_contacts) is True
        # With a gripper contact, grasped is True, so (not grasped) is False.
        with_grip = _ContactSim([{"geom1": "robot0_gripper_finger_r", "geom2": "cube_1_geom"}])
        assert fn(with_grip) is False


# Representative LIBERO-style round-trip


class TestRoundTrip:
    """One example per predicate family so a regression in any predicate is caught here."""

    def test_pick_task_on(self):
        text = """
            (define (problem libero_spatial_pick_up_the_red_cube)
              (:language "pick up the red cube and place it on the plate")
              (:objects cube_1 plate_1 table_1 - object)
              (:init (on cube_1 table_1))
              (:goal (on cube_1 plate_1)))
        """
        problem = parse_bddl(text)
        assert problem.language == "pick up the red cube and place it on the plate"
        fn = compile_goal(problem.goal)  # type: ignore[arg-type]
        sim_success = _BodyStateSim({"cube_1": {"position": [0, 0, 0.25]}, "plate_1": {"position": [0, 0, 0.1]}})
        assert fn(sim_success) is True

    def test_open_task(self):
        text = "(define (problem libero_open_drawer) (:goal (open drawer_slide)))"
        problem = parse_bddl(text)
        fn = compile_goal(problem.goal)  # type: ignore[arg-type]
        sim = _BodyStateSim({"_joints": {"drawer_slide": 0.2}})
        assert fn(sim) is True
        sim2 = _BodyStateSim({"_joints": {"drawer_slide": 0.02}})
        assert fn(sim2) is False

    def test_grasp_task(self):
        text = "(define (problem libero_grasp_cube) (:goal (grasped cube_1)))"
        problem = parse_bddl(text)
        fn = compile_goal(problem.goal)  # type: ignore[arg-type]
        sim = _ContactSim([{"geom1": "robot0_gripper_finger_l", "geom2": "cube_1"}])
        assert fn(sim) is True

    def test_upright_task(self):
        text = "(define (problem libero_keep_upright) (:goal (upright bottle_1)))"
        problem = parse_bddl(text)
        fn = compile_goal(problem.goal)  # type: ignore[arg-type]
        sim = _BodyStateSim({"bottle_1": {"quaternion": [1.0, 0, 0, 0]}})
        assert fn(sim) is True

    def test_inside_task(self):
        text = """
            (define (problem libero_put_inside)
              (:goal (inside cube_1 basket_1)))
        """
        problem = parse_bddl(text)
        fn = compile_goal(problem.goal)  # type: ignore[arg-type]
        # Approximate-inside uses default tolerances (0.15, 0.15).
        sim = _BodyStateSim({"cube_1": {"position": [0.05, 0.02, 0.1]}, "basket_1": {"position": [0, 0, 0.1]}})
        assert fn(sim) is True


# File loader


class TestParseBDDLFile:
    def test_happy_path(self, tmp_path):
        p = tmp_path / "task.bddl"
        p.write_text("(define (problem p) (:goal (grasped cube_1)))")
        problem = parse_bddl_file(p)
        assert problem.name == "p"

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_bddl_file(tmp_path / "nope.bddl")

    def test_not_a_file(self, tmp_path):
        with pytest.raises(ValueError):
            parse_bddl_file(tmp_path)
