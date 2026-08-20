"""The HF auth verdict is about ONE token, so its cache must know which.

``hf_auth_state()`` answers "can this machine reach gated repos, and as whom".
That answer is a property of the token on disk, but the cache used to be keyed on
nothing at all — a bare ``{at, value}`` — so for ten minutes after a token changed
the dashboard kept reporting the previous verdict: the wrong username, or "token
present but rejected" for a token that had just been fixed.

The second half matters more than it looks. The user who reads "rejected" goes and
fixes the token immediately; a ten-minute memory of the rejection then tells them
their fix did not work, which is the cache actively obstructing the recovery it
should be reporting.
"""

from __future__ import annotations

from unittest import mock

from strands_robots.dashboard import checkpoints


def _cold() -> None:
    checkpoints._WHOAMI.update(at=0.0, value=None, token=None)


def _state_with(token: str, *, whoami=None, boom: Exception | None = None) -> dict:
    with mock.patch("huggingface_hub.get_token", return_value=token), \
         mock.patch("huggingface_hub.HfApi") as api:
        if boom is not None:
            api.return_value.whoami.side_effect = boom
        else:
            api.return_value.whoami.return_value = whoami or {"name": "u"}
        return checkpoints.hf_auth_state()


class TestTheCacheIsKeyedToTheToken:
    def test_a_different_token_is_re_checked_immediately(self) -> None:
        _cold()
        first = _state_with("hf_alice", whoami={"name": "alice"})
        second = _state_with("hf_bob", whoami={"name": "bob"})
        assert first["user"] == "alice"
        assert second["user"] == "bob", "the new token inherited the old token's identity"

    def test_a_fixed_token_is_believed_without_waiting(self) -> None:
        _cold()
        rejected = _state_with("hf_dead", boom=PermissionError("401"))
        assert rejected["authenticated"] is False
        fixed = _state_with("hf_fresh", whoami={"name": "cagataycali"})
        assert fixed == {"authenticated": True, "user": "cagataycali", "detail": None}

    def test_the_same_token_is_not_re_checked(self) -> None:
        _cold()
        with mock.patch("huggingface_hub.get_token", return_value="hf_same"), \
             mock.patch("huggingface_hub.HfApi") as api:
            api.return_value.whoami.return_value = {"name": "u"}
            checkpoints.hf_auth_state()
            checkpoints.hf_auth_state()
            assert api.return_value.whoami.call_count == 1

    def test_the_token_itself_is_never_stored(self) -> None:
        _cold()
        _state_with("hf_supersecret_value", whoami={"name": "u"})
        assert "hf_supersecret_value" not in repr(checkpoints._WHOAMI), (
            "module state shows up in tracebacks and reprs; keep a fingerprint, not the secret"
        )
        assert checkpoints._WHOAMI["token"]


class TestVerdictBudgets:
    """The pure half, so the time arithmetic is pinned without sleeping."""

    def test_a_rejection_expires_much_sooner_than_a_success(self) -> None:
        good = {"at": 0.0, "value": {"authenticated": True, "user": "u", "detail": None}, "token": "fp"}
        bad = {"at": 0.0, "value": {"authenticated": False, "user": None, "detail": "rejected"}, "token": "fp"}
        at_30s = 30.0
        assert checkpoints.whoami_cache_verdict(good, "fp", at_30s) is not None
        assert checkpoints.whoami_cache_verdict(bad, "fp", at_30s) is None

    def test_an_entry_with_no_recorded_token_is_not_trusted(self) -> None:
        legacy = {"at": 0.0, "value": {"authenticated": True, "user": "u", "detail": None}}
        assert checkpoints.whoami_cache_verdict(legacy, "fp", 1.0) is None

    def test_a_clock_that_went_backwards_is_not_a_valid_cache_hit(self) -> None:
        entry = {"at": 100.0, "value": {"authenticated": True, "user": "u", "detail": None}, "token": "fp"}
        assert checkpoints.whoami_cache_verdict(entry, "fp", 10.0) is None

    def test_a_success_survives_its_whole_budget(self) -> None:
        entry = {"at": 0.0, "value": {"authenticated": True, "user": "u", "detail": None}, "token": "fp"}
        assert checkpoints.whoami_cache_verdict(entry, "fp", checkpoints._WHOAMI_TTL_S - 1) is not None
        assert checkpoints.whoami_cache_verdict(entry, "fp", checkpoints._WHOAMI_TTL_S + 1) is None
