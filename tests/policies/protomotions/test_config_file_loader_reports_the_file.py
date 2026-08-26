# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every policy config file loader reports a file it cannot use, by name.

``load_config_from_yaml`` is the fourth policy config file loader and carried
none of the four guards the other three share. Measured against the shipped
loader before the fix, through the public
:class:`~strands_robots.policies.protomotions.policy.ProtoMotionsPolicy`
constructor, which forwards a caller's ``yaml_path`` straight into it:

* An **empty** sidecar - a truncated download, a ``touch``ed placeholder, a
  file holding only comments - raised ``AttributeError: 'NoneType' object has
  no attribute 'get'``. This is the one that decides the disposition rather
  than only the message: ``yaml.safe_load("")`` is ``None``, and the loader
  documents fields absent from the yaml as falling back to the dataclass
  defaults, "so a missing block is not an error". An empty document is the
  limit of that - every field absent - and ``{}`` already returns the
  all-defaults config. Two spellings of the same information, and one of them
  dead-ended.
* ``~`` was never expanded, so a sidecar at ``~/unified_pipeline.yaml`` was
  reported *missing* while it existed, quoting the literal tilde.
* The check was ``exists()`` rather than ``is_file()``, so a directory passed
  it and surfaced as ``IsADirectoryError`` from the read.
* Malformed yaml escaped as a bare ``yaml.YAMLError`` (a ``ParserError``), and
  a yaml document holding a list or a scalar reached ``data.get(...)`` and
  surfaced as ``AttributeError: 'list' object has no attribute 'get'`` - a
  message naming a method of the parsed value rather than the file that could
  not supply fields. The loader documents ``ValueError`` for a bad payload.

None of this is a new rule. ``KimodoConfig.from_json``,
``MotionBricksConfig.from_file`` and ``WBCConfig.from_file`` each expand ``~``,
each check ``is_file()``, each wrap their decode into a ``ValueError`` naming
the path, and each refuse a non-mapping payload by type name - and
``from_json``'s docstring already calls that "the reporting the sibling
policy-config file loaders ... already give". So the guards graded here are the
three loaders' own, applied to the fourth.

The survey below is derived from the source rather than listed, and it is
deliberately keyed on *reading a path from disk* rather than on being a
``from_*`` classmethod: ``tests/policies/kimodo/test_config_file_loader_refusals.py``
grades the JSON ``from_dict`` family and, by keying discovery on the
classmethod shape, structurally cannot see a module-level loader. That is why a
fourth loader landed unheld. Keyed on the read instead, all four are in scope
however they are spelled.

One deliberate divergence stays, for the reason ``from_json``'s docstring gives
for its own: two of the loaders refuse a file whose extension they do not know
and these two do not. A yaml document stored under any name loads today, and
refusing it would stop a payload that currently works, so it is pinned as a
control below rather than closed.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("yaml", reason="the yaml sidecar loader needs pyyaml (the [protomotions] extra)")

import strands_robots.policies.protomotions.config as pm_config  # noqa: E402
from strands_robots.policies.protomotions.config import (  # noqa: E402
    ProtoMotionsConfig,
    load_config_from_yaml,
)

_POLICY_CONFIG_ROOT = Path(inspect.getsourcefile(pm_config) or "").resolve().parents[2]

#: The four guards a config file loader shares. Each is the presence of a
#: construct in the loader's own source, so the same predicate grades a shipped
#: loader and a constructed exemplar.
_GUARDS = ("expands_user", "reads_a_file", "wraps_the_decode", "refuses_a_non_mapping")

#: A loader reads its path from disk if it calls one of these.
_READS = ("read_text", "safe_load", "loads", "open")


def _guards_of(function: ast.FunctionDef) -> frozenset[str]:
    """Which of :data:`_GUARDS` a loader's body carries.

    Args:
        function: The loader's parsed definition.

    Returns:
        The guard names present, as a frozenset.
    """
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
    attrs = {node.func.attr for node in calls if isinstance(node.func, ast.Attribute)}
    present: set[str] = set()
    if "expanduser" in attrs:
        present.add("expands_user")
    if "is_file" in attrs:
        present.add("reads_a_file")
    # A decode is wrapped when some handler turns a parse failure into a
    # ValueError, rather than letting the format library's own class escape.
    for handler in [node for node in ast.walk(function) if isinstance(node, ast.ExceptHandler)]:
        if any(
            isinstance(raised, ast.Raise)
            and isinstance(raised.exc, ast.Call)
            and isinstance(raised.exc.func, ast.Name)
            and raised.exc.func.id == "ValueError"
            for raised in ast.walk(handler)
        ):
            present.add("wraps_the_decode")
    # A non-mapping payload is refused when an isinstance(..., dict) test leads
    # to a ValueError, rather than the payload reaching the field lookups.
    for branch in [node for node in ast.walk(function) if isinstance(node, ast.If)]:
        tests = [node for node in ast.walk(branch.test) if isinstance(node, ast.Call)]
        names = {node.func.id for node in tests if isinstance(node.func, ast.Name)}
        if "isinstance" not in names:
            continue
        if any(
            isinstance(raised, ast.Raise)
            and isinstance(raised.exc, ast.Call)
            and isinstance(raised.exc.func, ast.Name)
            and raised.exc.func.id == "ValueError"
            for raised in ast.walk(branch)
        ):
            present.add("refuses_a_non_mapping")
    return frozenset(present)


def _shipped_loaders() -> list[tuple[str, ast.FunctionDef]]:
    """Every policy config loader that reads a caller-named path from disk.

    Returns:
        ``(label, definition)`` pairs sorted by label, one per loader.
    """
    found: list[tuple[str, ast.FunctionDef]] = []
    for module_path in sorted(_POLICY_CONFIG_ROOT.glob("policies/*/config.py")):
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        for function in [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]:
            if "path" not in {argument.arg for argument in function.args.args}:
                continue
            body = ast.unparse(ast.Module(body=function.body, type_ignores=[]))
            if not any(marker in body for marker in _READS):
                continue
            found.append((f"{module_path.parent.name}.{function.name}", function))
    return sorted(found)


def _labels() -> list[str]:
    return [label for label, _ in _shipped_loaders()]


def _sidecar(directory: Path, text: str, name: str = "unified_pipeline.yaml") -> Path:
    """Write ``text`` as a sidecar and return its path."""
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


class TestAnEmptySidecarIsEveryFieldAbsent:
    """The regression: two spellings of "no fields" resolve to one config."""

    @pytest.mark.parametrize(
        ("label", "text"),
        [("empty-mapping", "{}\n"), ("empty-file", ""), ("comments-only", "# no fields here\n")],
    )
    def test_a_document_carrying_no_fields_loads_the_defaults(self, tmp_path: Path, label: str, text: str) -> None:
        assert load_config_from_yaml(_sidecar(tmp_path, text)) == ProtoMotionsConfig()

    def test_an_empty_file_and_an_empty_mapping_agree(self, tmp_path: Path) -> None:
        empty_file = load_config_from_yaml(_sidecar(tmp_path, "", name="a.yaml"))
        empty_mapping = load_config_from_yaml(_sidecar(tmp_path, "{}\n", name="b.yaml"))
        assert empty_file == empty_mapping

    def test_the_public_policy_constructor_accepts_an_empty_sidecar(self, tmp_path: Path) -> None:
        from strands_robots.policies.protomotions.policy import ProtoMotionsPolicy

        class _Session:
            def get_inputs(self) -> list[Any]:
                return []

            def get_outputs(self) -> list[Any]:
                return []

            def run(self, *args: Any, **kwargs: Any) -> list[Any]:
                return []

        session: Any = _Session()
        policy = ProtoMotionsPolicy(session=session, yaml_path=_sidecar(tmp_path, ""))
        assert policy.config == ProtoMotionsConfig()


class TestTheSidecarIsReadAsAFileTheCallerNamed:
    """The regression: the path is resolved, and a directory is not a config."""

    def test_a_home_relative_path_is_expanded(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        _sidecar(tmp_path, "{}\n")
        assert load_config_from_yaml("~/unified_pipeline.yaml") == ProtoMotionsConfig()

    def test_a_directory_is_refused_as_a_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="ProtoMotions yaml not found"):
            load_config_from_yaml(tmp_path)


class TestAPayloadThatCannotSupplyFieldsIsReportedByName:
    """The regression: the file is named, not a method of what was parsed."""

    def test_malformed_yaml_is_reported_as_a_value_error(self, tmp_path: Path) -> None:
        path = _sidecar(tmp_path, "control: {unclosed\n")
        with pytest.raises(ValueError, match="is not valid YAML") as caught:
            load_config_from_yaml(path)
        assert str(path) in str(caught.value)

    @pytest.mark.parametrize(
        ("label", "text", "type_name"),
        [("list", "- 1\n- 2\n", "list"), ("scalar", "42\n", "int"), ("string", '"control_dt"\n', "str")],
    )
    def test_a_non_mapping_document_names_its_type(self, tmp_path: Path, label: str, text: str, type_name: str) -> None:
        path = _sidecar(tmp_path, text)
        with pytest.raises(ValueError, match=f"must contain a mapping, got {type_name}") as caught:
            load_config_from_yaml(path)
        assert str(path) in str(caught.value)


class TestEveryPolicyConfigFileLoaderCarriesTheSameGuards:
    """The structural rule, derived over the loaders rather than listed."""

    def test_the_survey_finds_every_shipped_loader(self) -> None:
        assert _labels() == [
            "kimodo.from_json",
            "motionbricks.from_file",
            "protomotions.load_config_from_yaml",
            "wbc.from_file",
        ]

    @pytest.mark.parametrize("label", _labels())
    def test_the_loader_carries_all_four_guards(self, label: str) -> None:
        function = dict(_shipped_loaders())[label]
        missing = sorted(set(_GUARDS) - _guards_of(function))
        assert not missing, f"{label} is missing {missing}; its three siblings carry all four"


class TestTheGuardPredicateIsNotVacuous:
    """Constructed exemplars: the shipped loaders are all compliant now, so the
    survey cannot exercise its own failing branch and the predicate is graded
    against sources written to miss exactly one guard each."""

    _COMPLIANT = """
def loader(path):
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"not found: {p}")
    try:
        data = parse(p.read_text())
    except ParseError as e:
        raise ValueError(f"{p} is not valid: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"{p} must contain a mapping, got {type(data).__name__}")
    return build(data)
"""

    @staticmethod
    def _parse(source: str) -> ast.FunctionDef:
        module = ast.parse(source)
        function = module.body[0]
        assert isinstance(function, ast.FunctionDef)
        return function

    def test_a_compliant_loader_carries_every_guard(self) -> None:
        assert _guards_of(self._parse(self._COMPLIANT)) == frozenset(_GUARDS)

    @pytest.mark.parametrize(
        ("guard", "old", "new"),
        [
            ("expands_user", "Path(path).expanduser()", "Path(path)"),
            ("reads_a_file", "not p.is_file()", "not p.exists()"),
            ("wraps_the_decode", 'raise ValueError(f"{p} is not valid: {e}") from e', "raise"),
            ("refuses_a_non_mapping", "if not isinstance(data, dict):", "if False:"),
        ],
    )
    def test_dropping_one_guard_is_seen_as_exactly_that_guard(self, guard: str, old: str, new: str) -> None:
        assert self._COMPLIANT.count(old) == 1
        weakened = self._parse(self._COMPLIANT.replace(old, new))
        assert _guards_of(weakened) == frozenset(_GUARDS) - {guard}

    def test_the_predicate_reports_both_outcomes(self) -> None:
        compliant = _guards_of(self._parse(self._COMPLIANT)) == frozenset(_GUARDS)
        bare = _guards_of(self._parse("def loader(path):\n    return build(parse(open(path).read()))\n"))
        assert {compliant, bare == frozenset(_GUARDS)} == {True, False}


class TestNothingThatLoadedBeforeStopsLoading:
    """Over-reach controls: every expectation here held before the change."""

    def test_a_populated_sidecar_still_supplies_its_fields(self, tmp_path: Path) -> None:
        path = _sidecar(tmp_path, "timing:\n  control_dt: 0.04\n  decimation: 10\n")
        config = load_config_from_yaml(path)
        assert (config.control_dt, config.decimation) == (0.04, 10)
        assert config != ProtoMotionsConfig()

    def test_a_yaml_document_under_any_name_still_loads(self, tmp_path: Path) -> None:
        assert load_config_from_yaml(_sidecar(tmp_path, "{}\n", name="sidecar.txt")) == ProtoMotionsConfig()

    def test_a_missing_file_is_still_a_file_not_found_error(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="ProtoMotions yaml not found"):
            load_config_from_yaml(tmp_path / "absent.yaml")

    def test_the_config_domain_still_grades_a_value_the_yaml_supplied(self, tmp_path: Path) -> None:
        path = _sidecar(tmp_path, "robot:\n  anchor_body_index: 9999\n")
        with pytest.raises(ValueError):
            load_config_from_yaml(path)
