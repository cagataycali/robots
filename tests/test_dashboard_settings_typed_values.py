"""Q15 remainder: a wrong-TYPED settings value is reported, never transmuted.

982f4df6 made temperature/camera_hz/port/max_tokens strict; three families were
still silently transformed on the UI/API path:

* ``cors_origins: 5`` became ``[]`` - silently REPLACING a security posture,
* ``trust_remote_code: "banana"`` became ``False`` - an operator who believed
  they enabled something got it disabled with a success toast,
* ``auth_token: {"a": 1}`` became the literal string ``"{'a': 1}"`` - which is
  then REQUIRED on every request, locking the operator out of the UI that set it.

The strict path (update_strict - what /api/config uses) now refuses each with a
reason naming the key; the lenient path (env/CLI/file) still degrades, but to
the key's own SHAPE (a list key falls back to [], never to a scalar).
"""

from __future__ import annotations

import pytest

from strands_robots.dashboard.settings import CoercionError, _coerce, _coerce_strict


class TestListKeysRefuseScalars:
    @pytest.mark.parametrize("bad", [5, 1.5, True, {"a": 1}])
    def test_cors_origins_refuses_a_non_list_on_the_strict_path(self, bad: object) -> None:
        with pytest.raises(CoercionError, match="cors_origins"):
            _coerce_strict("security", "cors_origins", bad)

    def test_cors_origins_still_takes_a_comma_string_and_a_list(self) -> None:
        assert _coerce_strict("security", "cors_origins", "http://a, http://b") == ["http://a", "http://b"]
        assert _coerce_strict("security", "cors_origins", ["http://a"]) == ["http://a"]
        assert _coerce_strict("security", "cors_origins", None) == []

    def test_the_lenient_path_degrades_to_the_keys_own_shape(self) -> None:
        # env/CLI writes must not kill startup, but a list key degrading to a
        # SCALAR poisons every consumer that iterates it.
        assert _coerce("security", "cors_origins", 5, strict=False) == []


class TestBooleansRefuseTypos:
    @pytest.mark.parametrize("bad", ["banana", "yess", "enabled", 2])
    def test_a_spelling_that_is_neither_true_nor_false_is_reported(self, bad: object) -> None:
        with pytest.raises(CoercionError, match="boolean"):
            _coerce_strict("runtime", "trust_remote_code", bad)

    @pytest.mark.parametrize(
        ("spelled", "expected"),
        [(True, True), (False, False), ("true", True), ("ON", True), ("0", False), ("no", False), ("", False)],
    )
    def test_every_honest_spelling_still_works(self, spelled: object, expected: bool) -> None:
        assert _coerce_strict("runtime", "trust_remote_code", spelled) is expected


class TestStringKeysRefuseContainers:
    @pytest.mark.parametrize("bad", [{"a": 1}, ["x"], ("y",), {"z"}])
    def test_a_container_is_never_repred_into_a_string(self, bad: object) -> None:
        # str({'a': 1}) as auth_token = the operator locked out by a value
        # nobody can retype.
        with pytest.raises(CoercionError, match="auth_token"):
            _coerce_strict("security", "auth_token", bad)

    def test_numbers_are_still_stringified(self) -> None:
        # Ids get typed unquoted; that is not a bug worth a 422.
        assert _coerce_strict("agent", "model_id", 42) == "42"

    def test_a_plain_string_passes_untouched(self) -> None:
        assert _coerce_strict("security", "auth_token", "s3cret") == "s3cret"


class TestTheErrorReachesTheCaller:
    def test_update_strict_reports_the_refusal_and_stores_nothing(self, tmp_path, monkeypatch) -> None:
        from strands_robots.dashboard import settings as mod

        monkeypatch.setattr(mod, "SETTINGS_FILE", tmp_path / "settings.json")
        monkeypatch.setattr(mod, "_cache", None)
        changed, errors = mod.update_strict({"security": {"cors_origins": 5}})
        assert changed == []
        assert errors and "cors_origins" in errors[0]
        assert not (tmp_path / "settings.json").exists(), "a refused value must not be persisted"
