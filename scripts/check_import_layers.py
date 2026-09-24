#!/usr/bin/env python3
"""Grade ``strands_robots`` against the layered import DAG.

The package is meant to be readable top to bottom as seven layers, each
importing only downward::

    core -> registry -> drivers|mesh -> sim|policies -> app -> tools -> dashboard

Two properties make that claim checkable, and this script measures both from
the source with :mod:`ast` alone - no import of the package, so a machine
without a vendor SDK or a GPU grades the same graph CI does.

*No cycles.* The import graph is built three ways, because the three kinds of
import fail differently. A **runtime** module-scope import is the one that can
deadlock an interpreter, so its graph must be acyclic. A **typing-only** import
(inside ``if TYPE_CHECKING:``) and a **late** import (inside a function) cost
nothing at import time and are the two sanctioned ways to break a cycle, so
they are reported and excluded from the acyclicity requirement.

*No inversions.* Every edge that points at a higher layer is an inversion, and
direction is graded on two of the three kinds - because direction is a claim
about who depends on whom, which a deferred import does not change: a function
that imports the dashboard on its first call still cannot do its job without
the dashboard. So a runtime inversion is enumerated in
:data:`KNOWN_UPWARD_EDGES` and a deferred one in
:data:`KNOWN_DEFERRED_UPWARD_EDGES`, module pair by module pair. Both rosters
are ratchets: removing an inversion means deleting its line, and adding one
fails the grader until someone writes it down. Typing-only imports are reported
and not graded - an annotation is not a dependency at any point in the run.

Usage::

    python scripts/check_import_layers.py            # report + exit status
    python scripts/check_import_layers.py --verbose  # also list every inversion
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

#: The package under grading.
PACKAGE = "strands_robots"

#: The layers, lowest first, each naming the top-level members it owns. A
#: module's layer is its first path component under the package, so a subpackage
#: never disagrees with its parent.
#:
#: The placements that are a judgement rather than a reading of the tree:
#: ``assets`` sits with ``registry`` because it resolves the asset paths the
#: registry declares; the five dataset modules sit in ``core`` because a dataset
#: is a contract rather than a host - ``dataset_recorder`` writes one, and
#: reading what it recorded (``dataset_metadata``), resolving the directory a
#: ``repo_id`` names (``dataset_source``), streaming the frames back out
#: (``streaming_dataset``) and uploading the finished directory
#: (``dataset_transfer``) are the other four. The writer is the one that had to
#: be argued: no ``app`` module imports it and none holds a recording session -
#: ``start_recording`` exists only on the three sim backends a layer below -
#: while its own five imports are all ``core``, so keeping it in ``app`` inverted
#: the layering for its only caller and made three core modules look like they
#: had an ``app`` reader when that reader was the recorder. And
#: ``audit`` sits in ``core`` for the same reason the dataset modules do: the
#: append-only safety log is a contract, not a host. It imports nothing from the
#: package at all, and its writers are in three layers - the mesh that names the
#: file, the ``robot_mesh`` tool, and the ``_hitl_audit`` row every
#: human-in-the-loop gate owes - so under ``mesh`` it was a ``core`` module
#: reaching two layers up for a JSONL appender, which is the one inversion this
#: roster carried that nothing forced. And
#: ``teleop_mixin`` sits with ``drivers|mesh`` because it is an input-device
#: concern shared by three hosts in three layers - the hardware ``Robot``, the
#: MuJoCo ``Simulation`` and the Device Connect sim driver - so it belongs under
#: the lowest of them, which is where its own module-scope imports already put
#: it (``utils`` alone). ``teleoperator`` sits there for the same reason and
#: reads the same way: it is the factory for the input device that mixin
#: attaches, its own only in-package import is ``utils``, and its two readers
#: are that mixin and the hardware ``Robot`` a layer above.
#: ``__main__`` sits in the top layer because a console entry point is the one
#: module nothing can import: it is the process, not a part of the library, and
#: it reaches for whatever the command a reader typed needs - the doctor, the
#: dataset verifier, the dashboard CLI. Placing it in ``app``, under ``tools``
#: and ``dashboard``, made every command it dispatches look like an inversion,
#: and the dashboard one had to be declared as such. Named with the top layer
#: rather than given an eighth of its own, the way ``assets`` is named with the
#: ``registry`` it resolves paths for.
LAYERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "core",
        (
            "_async_utils",
            "_command_gate",
            "_dyld",
            "_hitl_audit",
            "_mesh_switch",
            "_motion_grants",
            "_mujoco_gl",
            "_path_validation",
            "_serial_discovery",
            "audit",
            "bus_access",
            "dataset_metadata",
            "dataset_recorder",
            "dataset_source",
            "dataset_transfer",
            "episode_labels",
            "locomotion_envelope",
            "recording_errors",
            "refusal_codes",
            "rendering",
            "streaming_dataset",
            "utils",
        ),
    ),
    ("registry", ("assets", "registry")),
    (
        "drivers|mesh",
        (
            "device_connect",
            "drivers",
            "mesh",
            "ros",
            "rosbridge",
            "ros_telemetry",
            "rtps",
            "teleop_mixin",
            "teleoperator",
        ),
    ),
    ("sim|policies", ("inference", "policies", "simulation", "training")),
    (
        "app",
        (
            "doctor",
            "hardware_observe",
            "hardware_robot",
            "hardware_ros_bridge",
            "hardware_rtps_bridge",
            "robot",
            "verify_dataset",
        ),
    ),
    ("tools", ("tools",)),
    ("dashboard", ("__main__", "dashboard")),
)

#: Layer index by top-level member name, derived from :data:`LAYERS`.
LAYER_OF_MEMBER: dict[str, int] = {member: index for index, (_name, members) in enumerate(LAYERS) for member in members}

#: Layer names by index, derived from :data:`LAYERS`.
LAYER_NAMES: tuple[str, ...] = tuple(name for name, _members in LAYERS)

#: The runtime imports that still point upward, ``(importer, imported)``. Each
#: line would be a cut this lane has not made yet; the grader fails on an edge
#: that is not here, and on an entry here that no longer exists, so the roster
#: can only shrink deliberately. It is empty: every layer imports downward only.
KNOWN_UPWARD_EDGES: tuple[tuple[str, str], ...] = ()

#: The deferred imports that point upward, ``(importer, imported)``. A late
#: import is exempt from the acyclicity requirement and not from the layering
#: one, so these are the inversions that survive: a module that reaches up from
#: inside a function body, once, on first call.
#:
#: Sanctioned, and staying: ``_hitl_audit`` writes the operator's answer through
#: the mesh safety log. Each is pinned individually in
#: ``tests/test_import_layers_are_a_dag.py``.
#:
#: Every entry is an inversion the code forces. An inversion that only the
#: placement forced belongs in :data:`LAYERS` instead, and
#: :func:`misplaced_members` refuses it here.
#:
#: The cuts left here reach up from a driver: ``drivers.ur`` builds a policy,
#: and the twin transports (``drivers.feetech.twin``,
#: ``drivers.yahboom_m3pro_twin``) build the MuJoCo engine they step through
#: ``simulation.create_simulation`` - deferred to the first ``connect`` rather
#: than imported, and taken from the simulation package rather than the driver
#: factory, which imports the driver registry and would close a cycle around
#: the driver each one twins.
KNOWN_DEFERRED_UPWARD_EDGES: tuple[tuple[str, str], ...] = (
    ("strands_robots.drivers.rollout", "strands_robots.policies"),
    ("strands_robots.drivers.feetech.twin", "strands_robots.simulation"),
    ("strands_robots.drivers.yahboom_m3pro_twin", "strands_robots.simulation"),
)


@dataclass(frozen=True)
class ImportGraph:
    """The package's internal import edges, split by the kind of import.

    :param modules: Every module in the package, dotted name to source path.
    :param runtime: Module-scope imports executed on import.
    :param typing_only: Imports inside an ``if TYPE_CHECKING:`` block.
    :param late: Imports inside a function or method body.
    """

    modules: dict[str, Path]
    runtime: dict[str, frozenset[str]] = field(default_factory=dict)
    typing_only: dict[str, frozenset[str]] = field(default_factory=dict)
    late: dict[str, frozenset[str]] = field(default_factory=dict)

    def edge_count(self, kind: str) -> int:
        """Return the number of edges of one kind.

        :param kind: ``"runtime"``, ``"typing_only"`` or ``"late"``.
        """
        return sum(len(targets) for targets in getattr(self, kind).values())


def module_name(path: Path, package_root: Path) -> str:
    """Return the dotted module name of a source file.

    A package's ``__init__.py`` names the package itself, so an import of
    ``strands_robots.mesh`` and one of ``strands_robots.mesh.__init__`` are the
    same node.

    :param path: The ``.py`` file.
    :param package_root: The package directory, e.g. ``<repo>/strands_robots``.
    """
    parts = list(path.relative_to(package_root).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join([PACKAGE, *parts])


def _owning_module(target: str, modules: dict[str, Path]) -> str | None:
    """Return the module a dotted import target resolves to.

    Trailing components that name an attribute rather than a module are dropped,
    so ``strands_robots.utils.finite_number_error`` resolves to
    ``strands_robots.utils``.

    :param target: A dotted name inside the package.
    :param modules: Every module in the package.
    """
    parts = target.split(".")
    while parts:
        candidate = ".".join(parts)
        if candidate in modules:
            return candidate
        parts.pop()
    return None


def _import_targets(module: str, node: ast.Import | ast.ImportFrom, *, is_package: bool) -> list[str]:
    """Return the dotted names one import statement reaches for.

    ``from pkg import name`` yields ``pkg.name`` only - never ``pkg`` as well.
    :func:`_owning_module` then truncates that to ``pkg`` when ``name`` is an
    attribute of ``pkg/__init__.py`` rather than a submodule, which is the whole
    of the distinction: a module that reads ``strands_robots.refusal_codes``
    depends on that module, not on whatever the package root happens to
    re-export, and adding the parent edge too would make every leaf's import of
    a core module look like a cycle through the root ``__init__``.

    :param module: The importing module's dotted name.
    :param node: The import statement.
    :param is_package: Whether the importing module is a package ``__init__``.
    """
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names if alias.name == PACKAGE or alias.name.startswith(f"{PACKAGE}.")]
    base = module.split(".") if is_package else module.split(".")[:-1]
    if node.level:
        ascend = node.level - 1
        if ascend:
            base = base[: len(base) - ascend]
        prefix = ".".join([*base, *([node.module] if node.module else [])])
    else:
        absolute = node.module or ""
        if not (absolute == PACKAGE or absolute.startswith(f"{PACKAGE}.")):
            return []
        prefix = absolute
    return [f"{prefix}.{alias.name}" for alias in node.names]


def _classify(tree: ast.Module) -> tuple[set[int], set[int]]:
    """Return the ids of the import nodes that are late, and typing-only.

    :param tree: A parsed module.
    """
    late: set[int] = set()
    typing_only: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            late.update(id(child) for child in ast.walk(node) if isinstance(child, ast.Import | ast.ImportFrom))
        elif isinstance(node, ast.If):
            names = {name.id for name in ast.walk(node.test) if isinstance(name, ast.Name)}
            names |= {attr.attr for attr in ast.walk(node.test) if isinstance(attr, ast.Attribute)}
            if "TYPE_CHECKING" in names:
                typing_only.update(
                    id(child) for child in ast.walk(node) if isinstance(child, ast.Import | ast.ImportFrom)
                )
    return late, typing_only


def build_graph(package_root: Path) -> ImportGraph:
    """Build the package's internal import graph from source.

    :param package_root: The package directory, e.g. ``<repo>/strands_robots``.
    """
    modules = {module_name(path, package_root): path for path in sorted(package_root.rglob("*.py"))}
    packages = {name for name, path in modules.items() if path.name == "__init__.py"}
    collected: dict[str, dict[str, set[str]]] = {kind: defaultdict(set) for kind in ("runtime", "typing_only", "late")}
    for module, path in modules.items():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        late, typing_only = _classify(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Import | ast.ImportFrom):
                continue
            kind = "late" if id(node) in late else ("typing_only" if id(node) in typing_only else "runtime")
            for target in _import_targets(module, node, is_package=module in packages):
                owner = _owning_module(target, modules)
                if owner is not None and owner != module:
                    collected[kind][module].add(owner)
    return ImportGraph(
        modules=modules,
        **{kind: {src: frozenset(dst) for src, dst in edges.items()} for kind, edges in collected.items()},
    )


def cycles(adjacency: dict[str, frozenset[str]], nodes: frozenset[str]) -> list[list[str]]:
    """Return the strongly connected components larger than one module.

    Tarjan's algorithm, iterative so a deep package cannot exhaust the stack.

    :param adjacency: The graph, importer to imported.
    :param nodes: The nodes to consider; edges leaving the set are ignored.
    """
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: dict[str, bool] = {}
    stack: list[str] = []
    found: list[list[str]] = []
    counter = 0
    for root in sorted(nodes):
        if root in index:
            continue
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack[root] = True
        work: list[tuple[str, list[str]]] = [(root, sorted(adjacency.get(root, ())))]
        while work:
            node, pending = work[-1]
            descended = False
            while pending:
                target = pending.pop(0)
                if target not in nodes:
                    continue
                if target not in index:
                    index[target] = low[target] = counter
                    counter += 1
                    stack.append(target)
                    on_stack[target] = True
                    work.append((target, sorted(adjacency.get(target, ()))))
                    descended = True
                    break
                if on_stack.get(target):
                    low[node] = min(low[node], index[target])
            if descended:
                continue
            work.pop()
            if work:
                low[work[-1][0]] = min(low[work[-1][0]], low[node])
            if low[node] == index[node]:
                component = []
                while True:
                    member = stack.pop()
                    on_stack[member] = False
                    component.append(member)
                    if member == node:
                        break
                if len(component) > 1:
                    found.append(sorted(component))
    return sorted(found, key=len, reverse=True)


def layer_of(module: str) -> int | None:
    """Return a module's layer index, or ``None`` for the package root.

    :param module: A dotted module name inside the package.
    """
    parts = module.split(".")
    if len(parts) < 2:
        return None
    return LAYER_OF_MEMBER.get(parts[1])


def unassigned_members(graph: ImportGraph) -> tuple[str, ...]:
    """Return the top-level members :data:`LAYERS` does not place.

    A member missing from the map would be graded against nothing, so the
    completeness of the map is itself a checked property.

    :param graph: The graph to read the module list from.
    """
    members = {name.split(".")[1] for name in graph.modules if len(name.split(".")) > 1}
    return tuple(sorted(members - set(LAYER_OF_MEMBER)))


def member_of(module: str) -> str | None:
    """Return the top-level member a module belongs to, or ``None`` for the root.

    :param module: A dotted module name inside the package.
    """
    parts = module.split(".")
    return parts[1] if len(parts) > 1 else None


def misplaced_members(graph: ImportGraph) -> tuple[tuple[str, int, int], ...]:
    """Return the members :data:`LAYERS` places below their own imports.

    An inversion is meant to say something about the code: this module cannot do
    its job without one above it. A member placed lower than it belongs says
    nothing of the kind - every downward import it makes is reported as an
    inversion, and declaring those in a roster records the placement rather than
    a dependency. So a declared inversion has to be one no placement could
    remove, and that is checkable: a member may move up to the highest layer it
    imports, as long as nothing that imports it sits at or below there.

    Both dependency directions are read from the runtime and late graphs and not
    the typing-only one, for the same reason :func:`upward_edges` grades those
    two: deferring an import moves when a dependency is paid, not whether it
    exists, while an annotation is not a dependency at all. Edges inside the
    member are ignored - a member moves whole.

    :param graph: The graph to read.
    :returns: ``(member, declared layer, lowest layer it could move to)`` per
        misplaced member, sorted by member.
    """
    modules_of: dict[str, list[str]] = defaultdict(list)
    for module in graph.modules:
        member = member_of(module)
        if member is not None:
            modules_of[member].append(module)
    depends_on: dict[str, set[str]] = defaultdict(set)
    imported_by: dict[str, set[str]] = defaultdict(set)
    for kind in ("runtime", "late"):
        for importer, targets in getattr(graph, kind).items():
            source = member_of(importer)
            for target in targets:
                sink = member_of(target)
                if source is None or sink is None or source == sink:
                    continue
                depends_on[source].add(sink)
                imported_by[sink].add(source)
    top = len(LAYERS) - 1
    found: list[tuple[str, int, int]] = []
    for member in sorted(modules_of):
        declared = LAYER_OF_MEMBER.get(member)
        if declared is None:
            continue
        needs = max(
            (layer for name in depends_on[member] if (layer := LAYER_OF_MEMBER.get(name)) is not None),
            default=0,
        )
        allowed = min(
            (layer for name in imported_by[member] if (layer := LAYER_OF_MEMBER.get(name)) is not None),
            default=top,
        )
        if declared < needs <= allowed:
            found.append((member, declared, needs))
    return tuple(found)


def upward_edges(graph: ImportGraph, kind: str = "runtime") -> tuple[tuple[str, str], ...]:
    """Return every import of one kind that points at a higher layer.

    :param graph: The graph to read.
    :param kind: ``"runtime"``, ``"typing_only"`` or ``"late"``.
    """
    found = []
    for importer, targets in getattr(graph, kind).items():
        source_layer = layer_of(importer)
        if source_layer is None:
            continue
        for target in targets:
            target_layer = layer_of(target)
            if target_layer is not None and target_layer > source_layer:
                found.append((importer, target))
    return tuple(sorted(found))


def _pair_label(edge: tuple[str, str]) -> str:
    """Return ``"<source layer> -> <target layer>"`` for one edge.

    A module whose layer is unknown is labelled ``?``; only :func:`upward_edges`
    output reaches here, and that has already resolved both ends.

    :param edge: An ``(importer, imported)`` pair.
    """
    names = ["?" if (index := layer_of(name)) is None else LAYER_NAMES[index] for name in edge]
    return f"{names[0]} -> {names[1]}"


def main(argv: list[str] | None = None) -> int:
    """Report the layer graph and fail on a runtime cycle or a new inversion.

    :param argv: Command-line arguments; ``None`` reads :data:`sys.argv`.
    :returns: ``0`` when the package matches the declared shape.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verbose", action="store_true", help="list every upward edge, not just the counts")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parent.parent / PACKAGE
    graph = build_graph(root)
    runtime_cycles = cycles(graph.runtime, frozenset(graph.modules))
    graded = (
        ("runtime", upward_edges(graph), KNOWN_UPWARD_EDGES),
        ("deferred", upward_edges(graph, "late"), KNOWN_DEFERRED_UPWARD_EDGES),
    )
    orphans = unassigned_members(graph)
    misplaced = misplaced_members(graph)

    print(f"{PACKAGE}: {len(graph.modules)} modules, {len(LAYERS)} layers")
    for kind in ("runtime", "typing_only", "late"):
        print(f"  {kind:12s} edges: {graph.edge_count(kind)}")
    print(f"  runtime cycles: {len(runtime_cycles)}")
    print(f"  misplaced members: {len(misplaced)}")
    for name, inversions, declared in graded:
        counts: dict[str, int] = defaultdict(int)
        for edge in inversions:
            counts[_pair_label(edge)] += 1
        print(f"  upward {name} edges: {len(inversions)} (declared {len(declared)})")
        for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            print(f"    {label:30s} {count}")
        if args.verbose:
            for importer, target in inversions:
                print(f"    {importer} -> {target}")

    failed = False
    for component in runtime_cycles:
        failed = True
        print(f"FAIL: runtime import cycle over {len(component)} modules: {', '.join(component)}")
    for name, inversions, declared in graded:
        for importer, target in sorted(set(inversions) - set(declared)):
            failed = True
            print(f"FAIL: undeclared upward {name} import {importer} -> {target} ({_pair_label((importer, target))})")
        for importer, target in sorted(set(declared) - set(inversions)):
            failed = True
            print(f"FAIL: declared upward {name} import no longer exists, delete it: {importer} -> {target}")
    for member in orphans:
        failed = True
        print(f"FAIL: {PACKAGE}.{member} is in no layer; add it to LAYERS")
    for misplaced_member, sits_in, reads in misplaced:
        failed = True
        print(
            f"FAIL: {PACKAGE}.{misplaced_member} is in layer {LAYER_NAMES[sits_in]} but imports "
            f"{LAYER_NAMES[reads]}, and nothing that imports it sits below {LAYER_NAMES[reads]}; "
            "move it in LAYERS rather than declaring the inversion"
        )
    if not failed:
        print("OK: no runtime cycle, no undeclared inversion")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
