# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""The state-key-mismatch warning must not interpolate the observation.

CodeQL's ``py/log-injection`` alert (rule id ``py/log-injection``, error
severity, alert 949 on ``main = 10ed6913``) fires on
:meth:`~strands_robots.policies.lerobot_local.policy.LerobotLocalPolicy._resolve_state_order`
because the warning message it emits is built by concatenating strings that
originate in the live observation dict (``scalar_keys`` is the observation's
own scalar keys, and ``_state_key_cause`` and ``state_key_remedy`` are read
from configured/observed keys respectively). The mitigation the ``logging``
module documents is to keep the untrusted value in the ``args`` tuple:
``logger.warning("%s", msg)`` rather than ``logger.warning(msg)``. The two
call shapes render *identically* through any handler that has a formatter,
because ``LogRecord.getMessage`` computes ``msg % args`` when ``args`` is
non-empty, so the observable warning text is byte-identical. The difference
CodeQL grades is *where in the record the untrusted content lives*: in
``msg`` (a taint sink) or in ``args`` (not a sink).

This module pins the format-sink shape.

Two cells that grade the shape, one over-reach control:

  * *args carries the payload* -- ``LogRecord.msg == "%s"`` and
    ``LogRecord.args == (msg,)``. A revert to ``logger.warning(msg)`` fails
    this cell because ``msg`` would then hold the full string and ``args``
    would be empty.

  * *the rendered text is unchanged* -- ``LogRecord.getMessage()`` still
    contains every observed joint name and every configured (generic) key,
    so a downstream reader that inspects ``.message`` reads identical bytes
    under either shape. Grading this here makes sure the shape edit did not
    accidentally regress the message content (which is what
    ``test_state_key_mismatch.py`` grades separately -- this cell is the
    local guard, so a future edit sees the two constraints together).

  * *the one-shot spam guard still fires exactly once per policy* -- the
    format-sink edit is co-located with the ``_state_key_mismatch_warned``
    flag, so a rewrite that moves the ``logger.warning`` call outside the
    guard would regress a rollout's log volume from ``O(1)`` to ``O(steps)``.
    Grading it here surfaces that regression at the same point the shape
    is graded.
"""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy

NAMED_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def _visual(shape=(3, 224, 224)):
    return SimpleNamespace(type=SimpleNamespace(name="VISUAL"), shape=shape)


def _state(dim=6):
    return SimpleNamespace(type=SimpleNamespace(name="STATE"), shape=(dim,))


def _policy_with_generic_keys():
    """A policy whose configured keys match no observation key.

    The generic ``joint_0..N`` names are what the policy auto-fills when a
    caller does not supply ``robot_state_keys`` and no embodiment hint
    resolves; a NAMED observation (``shoulder_pan`` etc.) then triggers the
    mismatch path. ``_load_model`` is patched so the constructor does not
    reach the network.
    """
    with patch.object(LerobotLocalPolicy, "_load_model"):
        policy = LerobotLocalPolicy(
            pretrained_name_or_path=None,
            policy_type="molmoact2",
            strict_keys=False,
        )
    policy._input_features = {
        "observation.images.base": _visual(),
        "observation.state": _state(6),
    }
    policy._device = torch.device("cpu")
    policy.robot_state_keys = [f"joint_{i}" for i in range(6)]
    return policy


def _named_obs():
    obs: dict[str, object] = {"base": np.zeros((224, 224, 3), np.uint8)}
    obs.update({name: 0.5 for name in NAMED_JOINTS})
    return obs


class TestArgsCarriesThePayload:
    """The warning is emitted through the ``%s`` format sink."""

    def test_msg_is_format_string_and_args_is_the_payload(self, caplog):
        """``LogRecord.msg == "%s"`` and the payload lives in ``args[0]``.

        This is the shape CodeQL's ``py/log-injection`` grades as clean: the
        untrusted content never reaches the format string, so an adversarial
        key of the form ``%(sensitive)s`` cannot re-enter the interpolation
        under any downstream handler that iterates ``args`` (nothing in the
        stdlib does, but the sink model applies at the record level, not the
        handler level).

        A revert to ``logger.warning(msg)`` would set ``msg`` to the full
        payload and leave ``args`` empty -- this cell fails immediately, and
        the alert re-opens.
        """
        policy = _policy_with_generic_keys()
        with caplog.at_level("WARNING", logger="strands_robots.policies.lerobot_local.policy"):
            keys = policy._resolve_state_order(_named_obs(), NAMED_JOINTS)

        assert set(keys) == set(NAMED_JOINTS), (
            "the fallback should return the observation's own keys under strict_keys=False"
        )

        mismatches = [r for r in caplog.records if "robot_state_keys" in r.getMessage()]
        assert len(mismatches) == 1, (
            f"expected exactly one mismatch warning across a single resolve call, got {len(mismatches)}: "
            f"{[r.getMessage()[:80] for r in mismatches]}"
        )
        rec = mismatches[0]

        # The taint boundary: the untrusted content must live in args, not msg.
        assert rec.msg == "%s", (
            f"expected LogRecord.msg == '%s' (format sink shape), got {rec.msg!r}. "
            "A direct logger.warning(msg) would put the payload in msg -- "
            "that shape is what CodeQL py/log-injection grades."
        )
        assert isinstance(rec.args, tuple) and len(rec.args) == 1, (
            f"expected args to be a 1-tuple carrying the payload, got {rec.args!r}"
        )
        assert "robot_state_keys" in rec.args[0], (
            f"the payload should be the mismatch message; got args[0]={rec.args[0]!r}"
        )

    def test_the_rendered_message_still_names_the_configured_and_observed_keys(self, caplog):
        """Over-reach control: ``.getMessage()`` reads identically to the pre-fix shape.

        Without this cell, a rewrite that keeps ``msg == "%s"`` but drops the
        payload from ``args`` (or replaces it with a redacted placeholder)
        would pass the shape cell above while silently erasing the diagnostic
        content the fallback exists to surface. Grading the fully-formatted
        message here pins that the payload still names both halves of the
        mismatch -- the configured (generic) keys and the observed (named)
        keys -- exactly as ``test_state_key_mismatch.py`` grades on the
        wider path.
        """
        policy = _policy_with_generic_keys()
        with caplog.at_level("WARNING", logger="strands_robots.policies.lerobot_local.policy"):
            policy._resolve_state_order(_named_obs(), NAMED_JOINTS)

        mismatches = [r for r in caplog.records if "robot_state_keys" in r.getMessage()]
        assert len(mismatches) == 1
        rendered = mismatches[0].getMessage()
        for name in NAMED_JOINTS:
            assert name in rendered, (
                f"expected the observed joint {name!r} to appear in the rendered warning; got: {rendered!r}"
            )
        assert "joint_0" in rendered, (
            f"expected the configured (generic) key 'joint_0' to appear in the rendered warning; got: {rendered!r}"
        )

    def test_the_one_shot_spam_guard_still_fires_exactly_once(self, caplog):
        """The ``_state_key_mismatch_warned`` flag survives the format-sink edit.

        A rollout that keeps invoking ``_resolve_state_order`` on the same
        policy instance sees exactly one warning across all calls; grading
        this on the same instance the shape cell uses makes sure the edit
        did not accidentally move the ``logger.warning`` call outside the
        guard. A regression here would be a log-volume regression from
        ``O(1)`` to ``O(steps)`` on a long rollout.
        """
        policy = _policy_with_generic_keys()
        with caplog.at_level("WARNING", logger="strands_robots.policies.lerobot_local.policy"):
            for _ in range(5):
                policy._resolve_state_order(_named_obs(), NAMED_JOINTS)

        mismatches = [r for r in caplog.records if "robot_state_keys" in r.getMessage()]
        assert len(mismatches) == 1, (
            f"expected exactly one warning across 5 resolve calls (one-shot guard), got {len(mismatches)}"
        )
