"""A policy config file loader reports a payload it cannot use, by name.

``KimodoConfig.from_json`` read a JSON file and handed whatever it parsed
straight to :meth:`KimodoConfig.from_dict`, which immediately calls
``data.items()``. A file holding a JSON array, string, number, boolean or
``null`` therefore surfaced as ``AttributeError: 'list' object has no attribute
'items'`` - a message naming a method of the parsed value rather than the file
that could not supply fields. A file that was not JSON at all escaped as a bare
``json.JSONDecodeError``, and ``~`` in the path was never expanded, so a config
at ``~/kimodo.json`` was reported missing while it existed.

The two sibling policy-config file loaders
(:mod:`strands_robots.policies.motionbricks.config` and
:mod:`strands_robots.policies.wbc.config`) already refuse a non-object payload
with a message naming the class, the resolved path and the JSON type they got,
already expand ``~``, and already wrap a decode failure. So the rule graded here
is not new: it is the reporting two of the three loaders shipped, applied to the
third.

The survey is derived rather than listed - every public class in a
``strands_robots/policies/*/config.py`` module that exposes a ``from_*(path)``
classmethod is held to it, so a fourth policy config's loader is graded the hour
it lands rather than inheriting an exemption by being absent from a tuple.

One deliberate divergence stays: the two siblings refuse a file whose extension
is not ``.json``, and ``from_json`` does not. A JSON object stored under another
name loads today, and refusing it would stop a payload that currently works,
so it is pinned as a control below rather than closed.
"""

from __future__ import annotations

import importlib
import inspect
import json
import warnings
from pathlib import Path
from typing import Any

import pytest

_POLICY_CONFIG_ROOT = Path(__file__).resolve().parents[3] / "strands_robots" / "policies"
_DOCS_ROOT = Path(__file__).resolve().parents[3] / "docs"

# A minimal payload each loader accepts. Two of the three configs require an
# entry, so a shared "this builds" fixture cannot be derived from the class
# alone; the non-vacuity test below fails if a newly discovered loader has no
# entry here, rather than skipping it.
_MINIMAL_PAYLOAD: dict[str, dict[str, Any]] = {
    "KimodoConfig": {"diffusion_steps": 50},
    "MotionBricksConfig": {"result_dir": "results"},
    "WBCConfig": {"policy_path": "policy.onnx"},
}

# Every JSON value that is not an object. ``from_dict`` needs ``.items()``, so
# each of these is a payload no config loader can consume.
_NON_OBJECT_PAYLOADS = (
    ("array", "[1, 2, 3]", "list"),
    ("string", '"diffusion_steps"', "str"),
    ("number", "42", "int"),
    ("boolean", "true", "bool"),
    ("null", "null", "NoneType"),
)

# A misspelled knob, held as a mapping so the two cases below can unpack it.
# Spelling it as a literal keyword argument is itself a static type error
# ("Unexpected keyword argument ... did you mean"), so a type checker rejects
# the call before it runs and the runtime refusal these two grade never gets
# measured. Unpacking keeps the call a runtime construct, which is the layer
# the documented ``TypeError`` belongs to.
_MISSPELLED_KNOB: dict[str, Any] = {"diffusion_stpes": 25}

# Unpacking is what makes the call a runtime construct, and the cost is that
# no static check reads the keyword any more. So the two cases below match
# Python's own phrasing rather than the bare key - a constructor that grew a
# ``**kwargs`` raising its own ``TypeError`` naming the key would satisfy
# ``match="diffusion_stpes"`` while the documented mechanism had changed - and
# ``TestNeitherConstructorAcceptsVarKeyword`` grades the signature promise
# that makes Python's phrasing the right one to expect.
_UNEXPECTED_KEYWORD = "unexpected keyword argument 'diffusion_stpes'"


def _config_modules() -> list[Any]:
    """Import every ``strands_robots/policies/*/config.py`` module."""
    return [
        importlib.import_module(f"strands_robots.policies.{path.parent.name}.config")
        for path in sorted(_POLICY_CONFIG_ROOT.glob("*/config.py"))
    ]


def _own_public_classes(module: Any) -> list[type]:
    return [
        obj
        for name in dir(module)
        if not name.startswith("_")
        for obj in [getattr(module, name)]
        if isinstance(obj, type) and getattr(obj, "__module__", None) == module.__name__
    ]


def _classmethods_taking(param: str) -> list[tuple[type, str]]:
    """Find ``from_*`` classmethods whose only parameter is named ``param``."""
    found: list[tuple[type, str]] = []
    for module in _config_modules():
        for cls in _own_public_classes(module):
            for name in dir(cls):
                if not name.startswith("from_"):
                    continue
                candidate = getattr(cls, name, None)
                if not callable(candidate):
                    continue
                try:
                    parameters = list(inspect.signature(candidate).parameters)
                except (TypeError, ValueError):
                    continue
                if parameters == [param]:
                    found.append((cls, name))
    return sorted(found, key=lambda entry: (entry[0].__name__, entry[1]))


def _path_loaders() -> list[tuple[type, str]]:
    return _classmethods_taking("path")


def _dict_loaders() -> list[tuple[type, str]]:
    return _classmethods_taking("data")


def _ids(loaders: list[tuple[type, str]]) -> list[str]:
    return [f"{cls.__name__}.{name}" for cls, name in loaders]


def _misspellable_constructors() -> list[type]:
    """The two keyword forms the config reference offers for setting a field."""
    from strands_robots.policies.kimodo import KimodoConfig, KimodoPolicy

    return [KimodoPolicy, KimodoConfig]


def _constructor_ids() -> list[str]:
    return [cls.__name__ for cls in _misspellable_constructors()]


def _write_non_object(directory: Path) -> Path:
    """Write a JSON array - a payload no config loader can consume."""
    path = directory / "config.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    return path


class TestTheSurveyReachesEveryPolicyConfigLoader:
    """Non-vacuity: the derived sets are populated and cover the known loaders."""

    def test_the_three_shipped_file_loaders_are_discovered(self) -> None:
        assert _ids(_path_loaders()) == [
            "KimodoConfig.from_json",
            "MotionBricksConfig.from_file",
            "WBCConfig.from_file",
        ]

    def test_the_three_shipped_dict_loaders_are_discovered(self) -> None:
        assert _ids(_dict_loaders()) == [
            "KimodoConfig.from_dict",
            "MotionBricksConfig.from_dict",
            "WBCConfig.from_dict",
        ]

    def test_every_discovered_loader_has_a_payload_that_builds(self) -> None:
        discovered = {cls.__name__ for cls, _ in _path_loaders() + _dict_loaders()}
        missing = discovered - set(_MINIMAL_PAYLOAD)
        assert not missing, f"add a minimal payload for {sorted(missing)} so it is graded rather than skipped"

    @pytest.mark.parametrize(
        ("label", "text", "type_name"), _NON_OBJECT_PAYLOADS, ids=[p[0] for p in _NON_OBJECT_PAYLOADS]
    )
    def test_each_probe_payload_is_valid_json_that_is_not_an_object(
        self, label: str, text: str, type_name: str
    ) -> None:
        parsed = json.loads(text)
        assert not isinstance(parsed, dict)
        assert type(parsed).__name__ == type_name


class TestANonObjectPayloadIsRefusedByName:
    """The regression: a file that cannot supply fields names itself, not ``.items()``."""

    @pytest.mark.parametrize(("cls", "loader"), _path_loaders(), ids=_ids(_path_loaders()))
    @pytest.mark.parametrize(
        ("label", "text", "type_name"), _NON_OBJECT_PAYLOADS, ids=[p[0] for p in _NON_OBJECT_PAYLOADS]
    )
    def test_every_json_value_that_is_not_an_object_is_refused(
        self, tmp_path: Path, cls: type, loader: str, label: str, text: str, type_name: str
    ) -> None:
        path = tmp_path / "config.json"
        path.write_text(text, encoding="utf-8")

        with pytest.raises(ValueError):
            getattr(cls, loader)(path)

    # The three message components are graded separately so a refusal that stops
    # naming one of them is distinguishable from a refusal that stops happening.

    @pytest.mark.parametrize(("cls", "loader"), _path_loaders(), ids=_ids(_path_loaders()))
    def test_the_refusal_names_the_config_class(self, tmp_path: Path, cls: type, loader: str) -> None:
        path = _write_non_object(tmp_path)

        with pytest.raises(ValueError) as caught:
            getattr(cls, loader)(path)

        assert cls.__name__ in str(caught.value)

    @pytest.mark.parametrize(("cls", "loader"), _path_loaders(), ids=_ids(_path_loaders()))
    def test_the_refusal_names_the_file_it_read(self, tmp_path: Path, cls: type, loader: str) -> None:
        path = _write_non_object(tmp_path)

        with pytest.raises(ValueError) as caught:
            getattr(cls, loader)(path)

        assert str(path) in str(caught.value)

    @pytest.mark.parametrize(("cls", "loader"), _path_loaders(), ids=_ids(_path_loaders()))
    def test_the_refusal_names_the_json_type_it_got(self, tmp_path: Path, cls: type, loader: str) -> None:
        path = _write_non_object(tmp_path)

        with pytest.raises(ValueError) as caught:
            getattr(cls, loader)(path)

        assert "list" in str(caught.value)

    @pytest.mark.parametrize(("cls", "loader"), _path_loaders(), ids=_ids(_path_loaders()))
    def test_a_file_that_is_not_json_is_reported_as_a_value_error(self, tmp_path: Path, cls: type, loader: str) -> None:
        path = tmp_path / "config.json"
        path.write_text("{not json", encoding="utf-8")

        with pytest.raises(ValueError) as caught:
            getattr(cls, loader)(path)

        message = str(caught.value)
        assert cls.__name__ in message
        assert str(path) in message
        assert "not valid JSON" in message

    @pytest.mark.parametrize(("cls", "loader"), _path_loaders(), ids=_ids(_path_loaders()))
    def test_a_missing_file_is_reported_with_the_path_it_looked_for(
        self, tmp_path: Path, cls: type, loader: str
    ) -> None:
        path = tmp_path / "absent.json"

        with pytest.raises(FileNotFoundError) as caught:
            getattr(cls, loader)(path)

        message = str(caught.value)
        assert cls.__name__ in message
        assert str(path) in message

    @pytest.mark.parametrize(("cls", "loader"), _path_loaders(), ids=_ids(_path_loaders()))
    def test_a_directory_is_reported_as_a_missing_file_not_a_read_error(
        self, tmp_path: Path, cls: type, loader: str
    ) -> None:
        """Why the check is ``is_file()`` and not ``exists()``."""
        directory = tmp_path / "config.json"
        directory.mkdir()

        with pytest.raises(FileNotFoundError) as caught:
            getattr(cls, loader)(directory)

        assert cls.__name__ in str(caught.value)

    @pytest.mark.parametrize(("cls", "loader"), _path_loaders(), ids=_ids(_path_loaders()))
    def test_a_home_relative_path_is_expanded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cls: type, loader: str
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        (tmp_path / "config.json").write_text(json.dumps(_MINIMAL_PAYLOAD[cls.__name__]), encoding="utf-8")

        assert isinstance(getattr(cls, loader)("~/config.json"), cls)


class TestNoPayloadThatLoadsTodayStopsLoading:
    """Over-reach controls: the refusals only reach inputs that already failed."""

    def test_an_object_still_loads_through_the_json_loader(self, tmp_path: Path) -> None:
        from strands_robots.policies.kimodo.config import KimodoConfig

        path = tmp_path / "config.json"
        path.write_text(json.dumps({"diffusion_steps": 50, "guidance_scale": 3.0}), encoding="utf-8")

        config = KimodoConfig.from_json(path)

        assert (config.diffusion_steps, config.guidance_scale) == (50, 3.0)

    def test_a_json_object_under_a_non_json_extension_still_loads(self, tmp_path: Path) -> None:
        """The one divergence from the sibling loaders, pinned so it is deliberate."""
        from strands_robots.policies.kimodo.config import KimodoConfig

        path = tmp_path / "config.yaml"
        path.write_text(json.dumps({"diffusion_steps": 7}), encoding="utf-8")

        assert KimodoConfig.from_json(path).diffusion_steps == 7

    def test_an_unrecognised_key_in_the_file_is_still_dropped(self, tmp_path: Path) -> None:
        from strands_robots.policies.kimodo.config import KimodoConfig

        path = tmp_path / "config.json"
        path.write_text(json.dumps({"diffusion_stpes": 999, "diffusion_steps": 50}), encoding="utf-8")

        assert KimodoConfig.from_json(path).diffusion_steps == 50

    def test_a_value_inside_the_object_keeps_the_domain_the_constructor_applies(self, tmp_path: Path) -> None:
        """The loader adds no second domain: the field's own refusal still fires."""
        from strands_robots.policies.kimodo.config import KimodoConfig

        path = tmp_path / "config.json"
        path.write_text(json.dumps({"model_id": "   "}), encoding="utf-8")

        with pytest.raises(ValueError, match="model_id must be a non-empty string"):
            KimodoConfig.from_json(path)

        path.write_text(json.dumps({"diffusion_steps": 0}), encoding="utf-8")
        with pytest.raises(ValueError, match="diffusion_steps"):
            KimodoConfig.from_json(path)


class TestAnUnrecognisedKeyIsDroppedWithoutWarning:
    """The documented forward-compatibility policy, graded across the three configs.

    ``KimodoConfig.from_dict`` documented the drop as happening "with a warning"
    while its two siblings documented it as silent. Measured, none of the three
    warns, so the docstring was the outlier and now states what the code does.
    """

    @pytest.mark.parametrize(("cls", "loader"), _dict_loaders(), ids=_ids(_dict_loaders()))
    def test_no_warning_is_emitted_for_a_dropped_key(self, cls: type, loader: str) -> None:
        payload = dict(_MINIMAL_PAYLOAD[cls.__name__])
        payload["definitely_not_a_field"] = "x"

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            built = getattr(cls, loader)(payload)

        assert isinstance(built, cls)
        assert [str(w.message) for w in caught] == []

    @pytest.mark.parametrize(("cls", "loader"), _dict_loaders(), ids=_ids(_dict_loaders()))
    def test_the_drop_is_not_documented_as_warning(self, cls: type, loader: str) -> None:
        doc = inspect.getdoc(getattr(cls, loader)) or ""
        assert "with a warning" not in doc, f"{cls.__name__}.{loader} documents a warning it does not emit"


class TestEachConstructionFormHandlesAMisspelledKnobAsDocumented:
    """The config reference presents three interchangeable ways to set a field.

    It claimed a misspelled knob "raises ``TypeError`` at construction instead of
    being silently ignored", which held for the two keyword forms and not for the
    ``config`` dict - that one is read by :meth:`KimodoConfig.from_dict`, whose
    documented drop policy is exactly to ignore it. The page now says which form
    does which, and these grade the three claims it makes.
    """

    def test_a_misspelled_keyword_on_the_policy_is_refused(self) -> None:
        from strands_robots.policies.kimodo import KimodoPolicy

        with pytest.raises(TypeError, match=_UNEXPECTED_KEYWORD):
            KimodoPolicy(**_MISSPELLED_KNOB)

    def test_a_misspelled_keyword_on_the_config_is_refused(self) -> None:
        from strands_robots.policies.kimodo import KimodoConfig

        with pytest.raises(TypeError, match=_UNEXPECTED_KEYWORD):
            KimodoConfig(**_MISSPELLED_KNOB)

    def test_a_misspelled_key_in_a_config_dict_is_dropped_for_the_default(self) -> None:
        from strands_robots.policies.kimodo import KimodoPolicy

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            policy = KimodoPolicy(config={"diffusion_stpes": 25})

        assert policy.config.diffusion_steps == 100
        assert [str(w.message) for w in caught] == []

    def test_a_correctly_spelled_key_in_a_config_dict_still_arrives(self) -> None:
        from strands_robots.policies.kimodo import KimodoPolicy

        assert KimodoPolicy(config={"diffusion_steps": 25}).config.diffusion_steps == 25

    def test_the_page_no_longer_claims_a_typo_is_never_ignored(self) -> None:
        page = _DOCS_ROOT / "policies" / "kimodo.md"

        assert "instead of being silently ignored" not in page.read_text(encoding="utf-8"), (
            f"{page.name} claims a misspelled knob is never silently ignored, "
            "which the config dict form does not honour"
        )

    def test_the_page_names_the_reader_that_drops_the_key(self) -> None:
        page = _DOCS_ROOT / "policies" / "kimodo.md"

        assert "KimodoConfig.from_dict" in page.read_text(encoding="utf-8"), (
            f"{page.name} does not name the reader that drops an unrecognised key"
        )


class TestNeitherConstructorAcceptsVarKeyword:
    """The refusal above is Python's, and both constructors promise it stays so.

    ``KimodoPolicy.__init__`` states in its own docstring that there is
    "deliberately no ``**kwargs``: an unknown knob raises ``TypeError`` at
    construction rather than being swallowed by a parameter nothing reads". That
    promise is what makes the two refusals above Python's signature enforcement
    rather than a hand-rolled check, and it was graded only as a side effect of
    how those calls were spelled: a literal ``diffusion_stpes=25`` is a call a
    static checker resolves against the signature. The keyword is unpacked now,
    so no checker reads it, and the promise is graded here directly instead.

    A constructor that grew a ``**kwargs`` swallowing the key would leave both
    refusals above failing outright. One that grew a ``**kwargs`` raising its own
    ``TypeError`` naming the key would leave them passing while the documented
    mechanism had silently changed, which is why they match Python's phrasing and
    why this class exists beside them.
    """

    @pytest.mark.parametrize("cls", _misspellable_constructors(), ids=_constructor_ids())
    def test_the_constructor_declares_no_var_keyword(self, cls: type) -> None:
        variadic = [
            parameter.name
            for parameter in inspect.signature(cls).parameters.values()
            if parameter.kind is inspect.Parameter.VAR_KEYWORD
        ]

        assert variadic == [], (
            f"{cls.__name__} accepts **{variadic[0] if variadic else ''}, so an unknown knob is "
            "swallowed by a parameter nothing reads instead of raising at construction"
        )

    @pytest.mark.parametrize("cls", _misspellable_constructors(), ids=_constructor_ids())
    def test_the_misspelled_knob_is_a_near_miss_of_a_real_parameter(self, cls: type) -> None:
        """Non-vacuity: any unknown keyword raises, so the name must be a typo.

        Were ``diffusion_steps`` renamed, the refusals above would still pass on
        the misspelling while no longer grading a near miss of a field a caller
        would plausibly reach for.
        """
        misspelled = next(iter(_MISSPELLED_KNOB))
        parameters = inspect.signature(cls).parameters

        assert misspelled not in parameters, f"{misspelled!r} is a real parameter of {cls.__name__}"
        assert "diffusion_steps" in parameters, (
            f"{cls.__name__} has no 'diffusion_steps' parameter, so {misspelled!r} is not a typo of one"
        )
