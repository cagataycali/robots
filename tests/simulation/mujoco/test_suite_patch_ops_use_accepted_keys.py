"""No test sends a ``patch_scene_mjcf`` op key that op does not read.

Every field of every structured op is read with a fallback default, so a key
outside an op's vocabulary is refused rather than silently applied (see
``test_patch_scene_mjcf_unknown_op_keys.py`` for that contract). A test that
sends such a key therefore fails loudly - *as long as it expected the op to
succeed*.

When the test expects the batch to be **rejected**, the same mistake is silent.
The batch is refused during the first op's key check, nothing is applied, and
every assertion about the rejected-and-rolled-back scene holds trivially. The
test keeps passing while exercising none of the path it was written for. That is
not hypothetical: ``test_a_refused_patch_batch_leaves_the_scene_mutable`` sent
``"position"`` where ``add_body`` reads ``"pos"``, so the batch died on op #1 and
the mid-batch rollback under test never ran.

This guard closes that gap statically: it reads the op vocabulary from its single
source of truth and refuses any op-dict literal in the suite whose keys fall
outside it. Only literal ``str`` keys are inspected, which is what a hand-written
op dict uses.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.scene_ops import _PATCH_OP_KEYS  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
TEST_ROOTS = ("tests", "tests_integ")

# The one module whose subject *is* rejection: it sends misspelled and
# out-of-vocabulary keys on purpose and asserts each is refused. Exempting it by
# name keeps the guard's intent readable - anywhere else, an unread key means the
# author believed the op reads it.
REJECTION_MODULE = "test_patch_scene_mjcf_unknown_op_keys.py"


def _unread_keys_in(source: str, filename: str) -> list[tuple[int, str, list[str]]]:
    """Locate op-dict literals in ``source`` whose keys are outside the vocabulary.

    Args:
        source: Python source text to scan.
        filename: Name used in parse errors.

    Returns:
        One ``(lineno, op, sorted unread keys)`` tuple per offending literal.
    """
    findings: list[tuple[int, str, list[str]]] = []
    for node in ast.walk(ast.parse(source, filename=filename)):
        if not isinstance(node, ast.Dict):
            continue
        keys = {key.value for key in node.keys if isinstance(key, ast.Constant) and isinstance(key.value, str)}
        op = next(
            (
                value.value
                for key, value in zip(node.keys, node.values, strict=True)
                if isinstance(key, ast.Constant)
                and key.value == "op"
                and isinstance(value, ast.Constant)
                and isinstance(value.value, str)
            ),
            None,
        )
        if op is None or op not in _PATCH_OP_KEYS:
            continue
        if unread := sorted(keys - _PATCH_OP_KEYS[op]):
            findings.append((node.lineno, op, unread))
    return findings


def _suite_modules() -> list[Path]:
    return sorted(
        path for root in TEST_ROOTS if (REPO_ROOT / root).is_dir() for path in (REPO_ROOT / root).rglob("test_*.py")
    )


def test_no_test_module_sends_an_unread_patch_op_key():
    offenders = []
    for path in _suite_modules():
        if path.name == REJECTION_MODULE:
            continue
        for lineno, op, unread in _unread_keys_in(path.read_text(encoding="utf-8"), path.name):
            accepted = ", ".join(sorted(_PATCH_OP_KEYS[op]))
            offenders.append(
                f"{path.relative_to(REPO_ROOT)}:{lineno}: {op} does not read {unread} (accepted: {accepted})"
            )
    assert not offenders, (
        "patch_scene_mjcf op literal(s) use keys the op does not read. Such an op "
        "is refused, so a test that expects an error still passes while testing "
        "nothing:\n  " + "\n  ".join(offenders)
    )


def test_the_scanner_detects_a_planted_unread_key():
    # Without this, a scanner that silently matched nothing would look like a
    # clean suite.
    planted = 'sim.patch_scene_mjcf([{"op": "add_body", "name": "rig", "position": [0, 0, 1]}])\n'
    assert _unread_keys_in(planted, "planted.py") == [(1, "add_body", ["position"])]


def test_a_correctly_spelled_op_is_not_flagged():
    ok = 'sim.patch_scene_mjcf([{"op": "add_body", "name": "rig", "pos": [0, 0, 1]}])\n'
    assert _unread_keys_in(ok, "ok.py") == []


def test_the_exempted_module_still_earns_its_exemption():
    # If the deliberate-typo module ever stops carrying unread keys, the
    # exemption is stale and should be deleted rather than left as a hole.
    exempt = [p for p in _suite_modules() if p.name == REJECTION_MODULE]
    assert exempt, f"{REJECTION_MODULE} not found - drop the exemption"
    assert _unread_keys_in(exempt[0].read_text(encoding="utf-8"), REJECTION_MODULE)
