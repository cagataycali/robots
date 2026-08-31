# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every public class across ``strands_robots.policies`` must document its public members.

:mod:`tests.policies.test_provider_policy_docstrings` already pins the docstring
contract for the backend *provider* Policy classes (cuRobo, cosmos3, the
two lerobot providers, MotionBricks, MoveIt2, VERA, the two WBC controllers) and
:mod:`tests.policies.test_builtin_policy_docstrings` pins the dependency-free
built-ins. Neither reaches the *ancillary* public classes the providers lean on:
the MoveIt2 ZMQ wire codec, the VERA websocket client / config /
server runners, and the LeRobot :class:`ProcessorStep` subclasses that pack and
normalize observations. An agent reading the tree to drive a provider blind
still lands on those helpers, so each of their public methods and properties
needs its own docstring rather than leaning silently on an inherited one (a
``transform_features`` override that is a deliberate passthrough has to say so).

This guard walks every module across the policies tree by AST (no import, so it
needs none of the optional policy backends -- ``[groot]`` / ``[cosmos3]`` /
``[vera]`` / ``[moveit2]`` / ``[wbc]`` / ``[lerobot]`` -- installed) and fails
if any public method or property of a public class defines no docstring. It
mirrors the tools-package guard (``tests/tools/test_public_member_docstrings.py``).
Property setters and deleters are exempt: a ``@x.setter`` mirrors its documented
getter and a docstring on it is noise, matching the convention elsewhere.
"""

from __future__ import annotations

import ast
from pathlib import Path

import strands_robots.policies as policies_pkg

_PACKAGE_DIR = Path(policies_pkg.__file__).parent


def _is_setter_or_deleter(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """True if the function is a ``@<name>.setter`` or ``@<name>.deleter``."""
    for dec in node.decorator_list:
        if isinstance(dec, ast.Attribute) and dec.attr in ("setter", "deleter"):
            return True
    return False


def _public_members_without_docstring(class_node: ast.ClassDef) -> list[str]:
    """Return names of public methods/properties in the class body lacking a docstring."""
    offenders: list[str] = []
    for node in class_node.body:
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if node.name.startswith("_"):
            continue
        if _is_setter_or_deleter(node):
            continue
        if ast.get_docstring(node) is None:
            offenders.append(node.name)
    return offenders


def _public_classes() -> dict[str, ast.ClassDef]:
    """Map ``relpath::ClassName`` -> ClassDef for every public class across the policies tree.

    ``ast.walk`` reaches classes defined inside ``if _HAVE_<dep>:`` / ``if
    TYPE_CHECKING:`` guards (the LeRobot ProcessorStep subclasses live inside a
    factory guarded that way) as well as at module top level.
    """
    classes: dict[str, ast.ClassDef] = {}
    for source_file in sorted(_PACKAGE_DIR.rglob("*.py")):
        tree = ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))
        rel = source_file.relative_to(_PACKAGE_DIR)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                classes[f"{rel}::{node.name}"] = node
    return classes


def test_policies_tree_has_ancillary_public_classes() -> None:
    """Guard: the scan walked the tree and reached the helper classes the provider guard misses."""
    found = set(_public_classes())
    for expected in (
        "moveit2/client.py::MoveIt2InferenceClient",
        "vera/client.py::VeraWebsocketClient",
        "vera/server_runner.py::VeraServerRunner",
    ):
        assert expected in found, (expected, found)


def test_policies_public_members_have_docstrings() -> None:
    offenders = {
        qualname: missing
        for qualname, node in _public_classes().items()
        if (missing := _public_members_without_docstring(node))
    }
    assert not offenders, (
        "Every public method/property of a public class in strands_robots.policies "
        "must have a docstring describing what it returns or does (a rich class "
        "docstring does not document its individual accessors). Undocumented "
        "members: " + repr(offenders)
    )
