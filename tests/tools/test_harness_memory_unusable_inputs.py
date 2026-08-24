"""Every branch of ``harness_memory`` that reports an unusable input or store.

The module has five such branches, and together they were its entire uncovered
set: four refuse and one degrades.

* the action vocabulary cannot be read -- the simulation tool spec is present
  but does not carry an ``action`` enum (``get_valid_actions``);
* a trace entry cannot be serialized (``_validate_trace``);
* a summary cannot be serialized (``_validate_summary``);
* the distribution metadata is absent, so trace provenance records an unknown
  version instead of failing the save (``_version_string``) -- the one
  degradation rather than a refusal;
* a global-rule store is not valid UTF-8 (``HarnessMemory._read_rules``).

Each one is a documented contract that nothing exercised. The two that reach a
caller through :class:`HarnessMemory` rather than through the tool -- an
unreadable store seen from ``load_rules`` and from ``append_rule`` -- had no
``Raises:`` entry at all, so their all-or-nothing read was undocumented as well
as unpinned; this module pins the behaviour those entries now describe.

The tool converts all four refusals into the ``{"status": "error"}`` envelope
rather than raising, so both the library-level reason and the tool-level report
are asserted: the reason has to name the input or the file, because the remedy
is to repair the thing it names.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import strands_robots.tools.harness_memory as hm
from tests.tool_result_contract import assert_strands_tool_result, tool_json

TRACE: list[dict[str, Any]] = [{"action": "run_policy", "instruction": "grasp the bowl"}]
SUMMARY: dict[str, Any] = {"strategy": "approach, grasp, lift", "avoid": ["empty grasp"]}


@pytest.fixture
def memory_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point STRANDS_MEMORY_DIR at a temp dir so the store is isolated."""
    d = tmp_path / "memory"
    monkeypatch.setenv("STRANDS_MEMORY_DIR", str(d))
    return d


def _texts(result: dict[str, Any]) -> str:
    """Concatenate every content ``text`` field of a tool result."""
    return "\n".join(item.get("text", "") for item in result.get("content", []))


def _circular() -> dict[str, Any]:
    """A dict json.dumps refuses with ValueError rather than TypeError."""
    d: dict[str, Any] = {"action": "run_policy"}
    d["self"] = d
    return d


def _foreign_object() -> dict[str, Any]:
    """A dict json.dumps refuses with TypeError."""
    return {"action": "run_policy", "handle": object()}


# Both members of the ``except (TypeError, ValueError)`` tuple the two payload
# validators share: an unencodable value raises TypeError, a cycle ValueError.
UNSERIALIZABLE = [
    pytest.param(_foreign_object, id="foreign-object-TypeError"),
    pytest.param(_circular, id="circular-reference-ValueError"),
]


class TestTheActionVocabularyCannotBeRead:
    """``get_valid_actions`` reads the simulation tool spec for the enum.

    A spec that parses as JSON but does not carry ``properties.action.enum``
    leaves the trace vocabulary undefined, so no trace can be validated. The
    refusal has to name the file, since that file is what has to be repaired.
    """

    def _spec(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload: object) -> Path:
        spec = tmp_path / "tool_spec.json"
        spec.write_text(json.dumps(payload), encoding="utf-8")
        monkeypatch.setattr(hm, "_sim_tool_spec_path", lambda: spec)
        monkeypatch.setattr(hm, "_valid_actions_cache", None)
        return spec

    def test_a_spec_without_an_action_enum_is_refused_naming_the_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spec = self._spec(tmp_path, monkeypatch, {"properties": {"action": {}}})
        with pytest.raises(ValueError, match="malformed simulation tool spec") as excinfo:
            hm.get_valid_actions()
        assert str(spec) in str(excinfo.value)

    def test_a_spec_whose_properties_is_not_a_mapping_is_refused_the_same_way(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The TypeError arm of the same handler, not the KeyError arm."""
        self._spec(tmp_path, monkeypatch, {"properties": ["action"]})
        with pytest.raises(ValueError, match="malformed simulation tool spec"):
            hm.get_valid_actions()

    def test_a_refused_read_leaves_the_vocabulary_cache_unpoisoned(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The vocabulary is cached in a module global; a failure must not cache.

        Repairing the same file and asking again has to return the real
        vocabulary, or one malformed read would disable trace validation for
        the rest of the process.
        """
        spec = self._spec(tmp_path, monkeypatch, {"properties": {"action": {}}})
        with pytest.raises(ValueError, match="malformed simulation tool spec"):
            hm.get_valid_actions()
        spec.write_text(json.dumps({"properties": {"action": {"enum": ["step", "render"]}}}), encoding="utf-8")
        actions = hm.get_valid_actions()
        assert {"step", "render"} <= actions

    def test_the_tool_reports_it_instead_of_raising(
        self, memory_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._spec(tmp_path, monkeypatch, {"properties": {"action": {}}})
        result = hm.harness_memory(action="save_trace", task="t0", trace=TRACE, summary=SUMMARY)
        assert_strands_tool_result(result)
        assert result["status"] == "error"
        assert "malformed simulation tool spec" in _texts(result)


class TestAPayloadThatCannotBeSerialized:
    """A trace entry or summary that ``json.dumps`` refuses.

    Both validators size the payload by serializing it, so an unencodable value
    is caught where the size is measured. The reason names which entry, because
    a trace is a list and the index is the only way to find the offender.
    """

    @pytest.mark.parametrize("factory", UNSERIALIZABLE)
    def test_a_trace_entry_is_refused_naming_its_index(self, factory: Any) -> None:
        with pytest.raises(ValueError, match=r"trace\[1\] is not JSON-serializable"):
            hm._validate_trace([{"action": "run_policy"}, factory()])

    @pytest.mark.parametrize("factory", UNSERIALIZABLE)
    def test_a_summary_is_refused(self, factory: Any) -> None:
        with pytest.raises(ValueError, match="summary is not JSON-serializable"):
            hm._validate_summary(factory())

    def test_an_unserializable_trace_writes_no_file(self, memory_dir: Path) -> None:
        """The refusal precedes the write, so no torn pair is left behind."""
        result = hm.harness_memory(action="save_trace", task="t0", trace=[_foreign_object()], summary=SUMMARY)
        assert_strands_tool_result(result)
        assert result["status"] == "error"
        assert "not JSON-serializable" in _texts(result)
        assert not list(memory_dir.rglob("t0*")), "a refused save wrote a file"

    def test_an_unserializable_summary_writes_no_file(self, memory_dir: Path) -> None:
        result = hm.harness_memory(action="save_trace", task="t0", trace=TRACE, summary={"why": object()})
        assert_strands_tool_result(result)
        assert result["status"] == "error"
        assert "not JSON-serializable" in _texts(result)
        assert not list(memory_dir.rglob("t0*")), "a refused save wrote a file"


class TestProvenanceWhenTheDistributionMetadataIsAbsent:
    """A running-from-source checkout has no installed distribution metadata.

    Provenance is recorded beside every trace, so this has to degrade rather
    than refuse: an unknown version is still a saveable, loadable trace.
    """

    def test_the_version_falls_back_to_unknown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def absent(name: str) -> str:
            raise hm._importlib_metadata.PackageNotFoundError(name)

        monkeypatch.setattr(hm._importlib_metadata, "version", absent)
        assert hm._version_string() == "unknown"

    def test_the_trace_still_saves_and_loads(self, memory_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def absent(name: str) -> str:
            raise hm._importlib_metadata.PackageNotFoundError(name)

        monkeypatch.setattr(hm._importlib_metadata, "version", absent)
        saved = hm.harness_memory(action="save_trace", task="t0", trace=TRACE, summary=SUMMARY)
        assert saved["status"] == "success"
        assert tool_json(saved)["provenance"]["strands_robots_version"] == "unknown"
        loaded = hm.harness_memory(action="load_trace", task="t0")
        assert loaded["status"] == "success"
        assert tool_json(loaded)["trace"] == TRACE


class TestARuleStoreThatIsNotUtf8:
    """A global-rule store whose bytes are not valid UTF-8.

    Rules are plain text appended a line at a time, so a store can be edited
    or truncated outside the tool. Reading one is all-or-nothing on purpose:
    ``load_rules`` feeds one prompt, and a per-kind fallback would present an
    unreadable store as a kind with no rules.
    """

    def _corrupt(self, memory_dir: Path, kind: str = "success_rule") -> Path:
        memory = hm.HarnessMemory()
        memory._ensure_dirs()
        memory.append_rule("failure_model", "a grasp that does not move the object is empty")
        store = memory.global_dir / hm._RULE_FILES[kind]
        store.write_bytes(b"a readable line\n\xff\xfe not utf-8\n")
        return store

    def test_the_refusal_names_the_file(self, memory_dir: Path) -> None:
        store = self._corrupt(memory_dir)
        with pytest.raises(ValueError, match="is not valid UTF-8") as excinfo:
            hm.HarnessMemory().load_rules()
        assert store.name in str(excinfo.value)

    def test_the_read_is_all_or_nothing(self, memory_dir: Path) -> None:
        """The readable kind is available, and the aggregate read still refuses.

        A per-kind fallback would return this kind and present the unreadable
        one as empty. Refusing instead is what keeps "no rules of that kind"
        from meaning "that store could not be read".
        """
        self._corrupt(memory_dir)
        memory = hm.HarnessMemory()
        healthy = memory.global_dir / hm._RULE_FILES["failure_model"]
        assert hm.HarnessMemory._read_rules(healthy) == ["a grasp that does not move the object is empty"]
        with pytest.raises(ValueError, match="is not valid UTF-8"):
            memory.load_rules()

    def test_a_healthy_store_still_reads_and_appends(self, memory_dir: Path) -> None:
        """Non-vacuity for the two tests above: the fixture's healthy kind works.

        Each kind has its own file, so the corruption is scoped to writes on the
        corrupt kind only.
        """
        self._corrupt(memory_dir)
        memory = hm.HarnessMemory()
        assert memory.append_rule("failure_model", "re-localize after every reset") == 2
        with pytest.raises(ValueError, match="is not valid UTF-8"):
            memory.append_rule("success_rule", "verify placement before declaring done")

    def test_the_tool_reports_it_instead_of_raising(self, memory_dir: Path) -> None:
        store = self._corrupt(memory_dir)
        result = hm.harness_memory(action="load_rules")
        assert_strands_tool_result(result)
        assert result["status"] == "error"
        assert store.name in _texts(result)
