"""``rotation_dim`` names a rotation encoding, so it takes the decoder's enumeration.

:func:`~strands_robots.policies.vera.sim_ik.delta_to_matrix` implements exactly
two rotation-delta encodings - axis-angle (3) and rot6d (6) - dispatching on the
width and raising for anything else. So the accepted set is declared, closed and
not a judgement call. Two public surfaces take that width and neither held it to
the enumeration:
:meth:`~strands_robots.policies.vera.provider.VeraPolicy.set_ik_target` stored
``int(rotation_dim)``, and
:func:`~strands_robots.policies.vera.sim_ik.decode_vera_delta_chunk_to_targets`
took it verbatim.

Its sibling in *both* signatures was already checked. ``set_ik_target`` opens with
"Validate before mutating any state, so a refused call leaves the policy
untouched (guard-before-mutation discipline)" and applies that to
``translation_scale`` only, while ``rotation_dim`` was written into the policy two
statements later with no check at all; the decoder guards ``translation_scale`` in
its first statement and explains at length in its ``Raises`` section why an
applied multiplier cannot be left to the arithmetic that consumes it.

Measured on ``363de01``, one ``VeraPolicy(auto_launch_server=False)`` per row: the
setter, then a real 7-wide ``eef_delta`` chunk through
``decode_vera_delta_chunk_to_targets`` against a MuJoCo Panda:

| ``rotation_dim=`` | setter | stored | where it was refused |
| --- | --- | --- | --- |
| ``3`` / ``3.0`` / ``6`` | accepted | 3 / 3 / 6 | honored (unchanged) |
| ``0`` / ``-3`` / ``2`` / ``4`` | accepted | as given | mid-rollout, ``unsupported rotation_dim`` |
| ``2.7`` | accepted | ``2`` | mid-rollout, naming **2** |
| ``True`` | accepted | ``1`` | mid-rollout, naming **1** |
| ``nan`` | ``ValueError`` | - | ``cannot convert float NaN to integer`` |
| ``inf`` | ``OverflowError`` | - | not the method's ``ValueError`` channel |
| ``[6]`` | ``TypeError`` | - | ``int() argument must be ...`` |

Three things that table shows, in increasing order of cost.

**The refusal arrived mid-rollout.** Six widths were stored and then raised from
inside ``get_actions`` on the first inference - after the policy-server
handshake, after the IK bridge was built - rather than at the call that supplied
them. The setter had already written ``_mj_model``, ``_ee_frame_name``,
``_ee_frame_type`` and reset ``_ik_bridge`` by then, so the discipline its own
comment claims did not hold for this parameter.

**Two refusals named a width the caller never supplied.** ``int()`` truncates
before storage, so ``2.7`` was reported as ``unsupported rotation_dim 2`` and
``True`` as ``1``.

**Three escaped the documented channel.** ``set_ik_target`` raises ``ValueError``
for ``translation_scale``, which is what teaches a caller to wrap it; ``inf``
raised ``OverflowError`` and ``[6]`` ``TypeError``, so an ``except ValueError``
did not catch them.

The decoder was worse on the type axis, and it disagreed with the setter about
the same value. Passing the width straight to ``decode_vera_delta_chunk_to_targets``:
``3.0``, ``2.7``, ``nan`` and a numeric string reached the per-step rotation slice
``step[3 : 3 + rotation_dim]`` and raised ``TypeError: slice indices must be
integers`` - naming neither the parameter nor the function - and ``inf`` reported
needing ``>= inf`` pose dims. So ``3.0`` and ``"6"`` were *honored* through the
setter (which coerced them with ``int()``) and *refused* by the decoder: one
parameter, two public surfaces, opposite verdicts.

Both now route through one owner,
:func:`~strands_robots.policies.vera.sim_ik.coerce_rotation_dim` - beside the
dispatch that defines the enumeration - which delegates numeric-ness, ``bool``
rejection and finiteness to
:func:`~strands_robots.utils.finite_number_error` and decides only membership.
Integral floats stay accepted - ``3.0`` is what a config read produces, and the
dispatch accepts it - and the conversion to ``int`` is kept because the width
indexes a slice. A numeric string is now refused at both surfaces rather than
coerced at one, which is the divergence above being removed.
"""

from __future__ import annotations

import inspect
import subprocess
import sys
from typing import Any

import numpy as np
import pytest

from strands_robots.policies.vera.sim_ik import (
    ROTATION_DIMS,
    coerce_rotation_dim,
    decode_vera_delta_chunk_to_targets,
    delta_to_matrix,
)

#: Widths the decoder implements, in the spellings a caller can reach them by.
#: Integral floats are included deliberately: a width read from JSON or a YAML
#: config arrives as ``3.0``, the dispatch accepts it, and the setter honored it
#: before this change.
USABLE: list[Any] = [3, 6, 3.0, 6.0, np.int64(3), np.float64(6.0)]

#: Widths no encoding exists for. Split by which half of the domain refuses
#: them: membership for the numbers, and the shared numeric domain for the rest.
NOT_A_MEMBER: list[Any] = [0, -3, 1, 2, 4, 7, 9, 2.7, -6]
NOT_A_NUMBER: list[Any] = [True, False, "6", "abc", [6], {}, float("nan"), float("inf"), float("-inf"), 10**400]
UNUSABLE: list[Any] = NOT_A_MEMBER + NOT_A_NUMBER

#: ``None`` is deliberately absent from :data:`UNUSABLE`. It is refused by the
#: rule itself - it names no width - but the *setter* documents it as "keep the
#: embodiment's convention", so it is the one value on which the two surfaces
#: legitimately differ. Pinned on its own in
#: :class:`TestSetIkTargetRefusesAWidthTheDecoderCannotRead`.


class _StubBridge:
    """Duck-typed IK bridge: the decoder touches only these three members."""

    def __init__(self, nq: int = 3) -> None:
        self.model = type("_M", (), {"nq": nq})()

    def ee_pose(self, q: Any) -> np.ndarray:
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = np.asarray(q, dtype=np.float64)[:3]
        return pose

    def solve(self, target: Any, q: Any) -> np.ndarray:
        return np.asarray(target, dtype=np.float64)[:3, 3].copy()


def _policy() -> Any:
    """A VeraPolicy that launches nothing and needs no ``vera`` package.

    ``Any`` because these tests deliberately supply out-of-domain widths, and
    because they read the private ``_rotation_dim`` the setter writes - the
    stored width is the observable this change is about.
    """

    class FakeClient:
        def get_server_metadata(self) -> dict[str, Any]:
            return {"action_space": "eef_delta", "view_keys": ["image"]}

        def reset(self, *_a: Any, **_k: Any) -> None:
            return None

        def configure(self, *_a: Any, **_k: Any) -> dict[str, Any]:
            return {}

        def close(self) -> None:
            return None

        def infer(self, *_a: Any, **_k: Any) -> dict[str, Any]:
            return {"action": np.zeros((1, 7), np.float32)}

    from strands_robots.policies.vera.provider import VeraPolicy

    # One ``Any``-typed local rather than a suppression: the stub is duck-typed on
    # the handful of members the policy calls, while the signature names the
    # concrete websocket client.
    client: Any = FakeClient()
    return VeraPolicy(client=client, auto_launch_server=False)


def _decode(rotation_dim: Any, *, width: int = 7) -> Any:
    """Run the real decoder over a ``width``-wide descend chunk.

    ``Any`` for the width because these tests deliberately supply values outside
    the declared ``int``; the bridge is duck-typed, so the decode path needs
    neither ``mink`` nor ``mujoco``.
    """
    chunk = np.tile(np.array([0.02, 0.0, -0.01, 0.0, 0.05, 0.0, 1.0][:width], np.float64), (3, 1))
    bridge: Any = _StubBridge()
    return decode_vera_delta_chunk_to_targets(
        chunk, bridge, np.zeros(3, dtype=np.float64), rotation_dim=rotation_dim, has_gripper=True
    )


class TestTheEncodingWidthDomain:
    """The enumeration itself, with no policy and no sim stack."""

    @pytest.mark.parametrize("value", USABLE)
    def test_a_width_the_decoder_implements_is_accepted(self, value: Any) -> None:
        width, err = coerce_rotation_dim(value, "rotation_dim", "ctx")
        assert err is None
        assert width in ROTATION_DIMS

    @pytest.mark.parametrize("value", USABLE)
    def test_an_accepted_width_is_normalized_to_an_int(self, value: Any) -> None:
        """The width indexes a slice, so it has to arrive there as an index."""
        width, _err = coerce_rotation_dim(value, "rotation_dim", "ctx")
        assert type(width) is int

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_a_width_no_encoding_exists_for_is_refused(self, value: Any) -> None:
        width, err = coerce_rotation_dim(value, "rotation_dim", "ctx")
        assert width is None
        assert err is not None

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_the_refusal_names_the_surface_the_parameter_and_the_value(self, value: Any) -> None:
        """A refusal a caller can act on: not an internal coerced width."""
        _width, err = coerce_rotation_dim(value, "rotation_dim", "set_ik_target")
        assert err is not None
        assert err.startswith("set_ik_target: rotation_dim ")
        assert repr(value) in err

    @pytest.mark.parametrize("value", NOT_A_MEMBER)
    def test_a_number_outside_the_enumeration_is_told_which_widths_exist(self, value: Any) -> None:
        _width, err = coerce_rotation_dim(value, "rotation_dim", "ctx")
        assert err is not None
        assert "3 (axis-angle) or 6 (rot6d)" in err

    def test_only_membership_is_decided_here(self) -> None:
        """Everything but the enumeration is the shared numeric domain's.

        Pinned as an equivalence rather than assumed, so the two cannot drift:
        for every value outside the enumeration the shared guard already refuses,
        this one refuses with that guard's own wording.
        """
        from strands_robots.utils import finite_number_error

        for value in UNUSABLE + USABLE:
            shared = finite_number_error(value, "rotation_dim", "ctx")
            _width, err = coerce_rotation_dim(value, "rotation_dim", "ctx")
            if shared is not None:
                assert err == shared, value


class TestSetIkTargetRefusesAWidthTheDecoderCannotRead:
    """The setter is where the width is supplied, so it is where it is refused."""

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_an_unusable_width_is_refused_as_a_value_error(self, value: Any) -> None:
        """One channel, matching the sibling ``translation_scale`` guard.

        ``inf`` and ``[6]`` used to raise ``OverflowError`` / ``TypeError`` out
        of the ``int()`` coercion, so a caller wrapping the call the way the
        sibling guard teaches did not catch them.
        """
        policy = _policy()
        with pytest.raises(ValueError, match="rotation_dim"):
            policy.set_ik_target(object(), "hand", "body", rotation_dim=value)

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_a_refused_call_mutates_nothing(self, value: Any) -> None:
        """The discipline the method's own opening comment claims."""
        policy = _policy()
        policy.set_ik_target(object(), "hand", "body", rotation_dim=6)
        sentinel = object()
        policy._ik_bridge = sentinel
        with pytest.raises(ValueError):
            policy.set_ik_target(object(), "wrist", "site", rotation_dim=value)
        assert policy._rotation_dim == 6
        assert policy._ee_frame_name == "hand"
        assert policy._ee_frame_type == "body"
        assert policy._ik_bridge is sentinel, "a refused call must not force a bridge rebuild"

    @pytest.mark.parametrize("value", USABLE)
    def test_a_usable_width_is_stored_as_an_int(self, value: Any) -> None:
        policy = _policy()
        policy.set_ik_target(object(), "hand", "body", rotation_dim=value)
        assert policy._rotation_dim == int(value)
        assert type(policy._rotation_dim) is int

    def test_none_still_leaves_the_current_width_alone(self) -> None:
        """``None`` is the documented opt-out, not a refused value.

        The shared numeric domain refuses ``None``, so the asymmetry is
        deliberate: on this surface it means "keep the embodiment's convention",
        which is a different request from naming a width.
        """
        policy = _policy()
        policy.set_ik_target(object(), "hand", "body", rotation_dim=6)
        policy.set_ik_target(object(), "hand", "body", rotation_dim=None)
        assert policy._rotation_dim == 6
        assert coerce_rotation_dim(None, "rotation_dim", "ctx")[1] is not None


class TestTheDecoderRefusesAWidthItCannotRead:
    """The other surface that takes the width, held to the same enumeration."""

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_an_unusable_width_is_refused_as_a_value_error(self, value: Any) -> None:
        """No bare ``TypeError`` out of the per-step slice."""
        with pytest.raises(ValueError, match="rotation_dim"):
            _decode(value)

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_the_refusal_names_this_function(self, value: Any) -> None:
        with pytest.raises(ValueError, match=r"decode_vera_delta_chunk_to_targets: rotation_dim"):
            _decode(value)

    def test_a_usable_width_still_decodes_the_chunk(self) -> None:
        out = _decode(3)
        assert np.asarray(out["qpos"]).shape[0] == 3

    def test_an_integral_float_width_now_decodes_instead_of_failing_to_slice(self) -> None:
        """``3.0`` raised ``TypeError: slice indices must be integers`` here.

        The setter honored the same value, so this is the divergence between the
        two surfaces closing rather than a new capability.
        """
        out = _decode(3.0)
        assert np.asarray(out["qpos"]).shape[0] == 3

    def test_the_pose_width_check_still_reports_a_server_mismatch(self) -> None:
        """A legal width the chunk is too narrow for keeps its own message."""
        with pytest.raises(ValueError, match="needs >= 9 pose dims"):
            _decode(6)


class TestTheTwoSurfacesAgree:
    """One parameter, two public surfaces, one accepted set."""

    @pytest.mark.parametrize("value", USABLE + UNUSABLE)
    def test_neither_surface_accepts_what_the_other_refuses(self, value: Any) -> None:
        """``None`` is excluded: see the note on :data:`UNUSABLE`."""
        policy = _policy()
        try:
            policy.set_ik_target(object(), "hand", "body", rotation_dim=value)
            setter_refused = False
        except ValueError:
            setter_refused = True
        try:
            _decode(value)
            decoder_refused = False
        except ValueError as exc:
            decoder_refused = "rotation_dim must be" in str(exc)
        assert setter_refused == decoder_refused, f"verdicts differ for rotation_dim={value!r}"


class TestTheDispatchAndTheGuardDescribeTheSameSet:
    """The enumeration has one owner, pinned against the dispatch that defines it."""

    @pytest.mark.parametrize("value", USABLE + UNUSABLE)
    def test_the_guard_accepts_exactly_what_the_dispatch_implements(self, value: Any) -> None:
        _width, err = coerce_rotation_dim(value, "rotation_dim", "ctx")
        try:
            delta_to_matrix(np.zeros(3, dtype=np.float64), value)
            dispatch_refused = False
        except ValueError as exc:
            dispatch_refused = "unsupported rotation_dim" in str(exc)
        except Exception:
            dispatch_refused = False
        assert (err is not None) == dispatch_refused, f"guard and dispatch differ for {value!r}"

    def test_the_dispatch_builds_its_refusal_from_the_shared_enumeration(self) -> None:
        """Its wording is unchanged; the numbers in it are no longer literals."""
        source = inspect.getsource(delta_to_matrix)
        assert "ROTATION_DIMS[0]" in source
        with pytest.raises(ValueError, match=r"unsupported rotation_dim 4; use 3 \(axis-angle\) or 6 \(rot6d\)"):
            delta_to_matrix(np.zeros(3, dtype=np.float64), 4)


class TestTheRuleCostsNoDependency:
    """The enumeration lives with the dispatch, and that stays cheap to reach.

    ``set_ik_target`` is reachable - and tested elsewhere in this package - with
    no real ``mujoco``, so the guard it now calls must not drag the simulation
    stack in behind it. Measured rather than asserted in a comment, because the
    guard's home was chosen on this property.
    """

    def test_reaching_the_rule_loads_no_heavy_module(self) -> None:
        """A fresh interpreter: import the rule, answer with it, stay light.

        Asserted on ``sys.modules`` rather than by blocking the imports: the
        package itself probes ``find_spec("mujoco")`` to decide what to export,
        so a blocking finder measures that probe instead of this import graph.
        """
        code = (
            "import sys\n"
            "from strands_robots.policies.vera.sim_ik import coerce_rotation_dim\n"
            "assert coerce_rotation_dim(3, 'rotation_dim', 'ctx') == (3, None)\n"
            "assert coerce_rotation_dim(4, 'rotation_dim', 'ctx')[0] is None\n"
            "heavy = sorted(m for m in ('mujoco', 'mink', 'qpsolvers', 'torch') if m in sys.modules)\n"
            "assert not heavy, heavy\n"
            "print('OK')\n"
        )
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180)
        assert done.returncode == 0, done.stderr[-900:]
        assert "OK" in done.stdout

    def test_the_setter_refuses_a_width_with_no_heavy_module_loaded(self) -> None:
        """The surface that has to stay light, refusing end to end while light."""
        code = (
            "import sys\n"
            "import numpy as np\n"
            "from strands_robots.policies.vera.provider import VeraPolicy\n"
            "class C:\n"
            "    def get_server_metadata(self): return {'action_space': 'eef_delta', 'view_keys': ['image']}\n"
            "    def reset(self, *a, **k): return None\n"
            "    def configure(self, *a, **k): return {}\n"
            "    def close(self): return None\n"
            "    def infer(self, *a, **k): return {'action': np.zeros((1, 7), np.float32)}\n"
            "p = VeraPolicy(client=C(), auto_launch_server=False)\n"
            "try:\n"
            "    p.set_ik_target(object(), 'hand', 'body', rotation_dim=4)\n"
            "    raise SystemExit('accepted an unusable width')\n"
            "except ValueError as e:\n"
            "    assert 'rotation_dim' in str(e), e\n"
            "p.set_ik_target(object(), 'hand', 'body', rotation_dim=6)\n"
            "assert p._rotation_dim == 6\n"
            "heavy = sorted(m for m in ('mujoco', 'mink', 'qpsolvers', 'torch') if m in sys.modules)\n"
            "assert not heavy, heavy\n"
            "print('OK')\n"
        )
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180)
        assert done.returncode == 0, done.stderr[-900:]
        assert "OK" in done.stdout
