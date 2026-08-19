"""Checkpoint search must NAME its failure modes instead of rendering silence.

Before this contract, hub_search logged failures at DEBUG and returned [],
so three different worlds looked identical in the UI as "no results": a Hub
outage, a machine with no network, and a fine query with zero matches. And
nothing anywhere reported whether the machine's HF token (which decides if
gated/private checkpoints are reachable) existed or worked.

Run with --no-cov (single-file runs trip the global coverage gate).
"""

from unittest import mock

from strands_robots.dashboard import checkpoints


def _no_cache():
    """Each test starts with a cold query cache and whoami cache."""
    checkpoints._CACHE.clear()
    checkpoints._WHOAMI.update(at=0.0, value=None)


class TestHubSearchNamesItsFailure:
    def test_hub_failure_returns_problem_sentence(self):
        _no_cache()
        with mock.patch("huggingface_hub.HfApi") as api:
            api.return_value.list_models.side_effect = ConnectionError("nope")
            rows, problem = checkpoints.hub_search("smolvla", limit=3)
        assert rows == []
        assert problem is not None
        assert "ConnectionError" in problem
        assert "local cache" in problem

    def test_failure_is_not_cached(self):
        # a transient outage must not pin "unavailable" for the 5-min TTL -
        # the very next keystroke should retry
        _no_cache()
        with mock.patch("huggingface_hub.HfApi") as api:
            api.return_value.list_models.side_effect = ConnectionError("blip")
            checkpoints.hub_search("act", limit=3)
        assert not checkpoints._CACHE  # nothing pinned

    def test_success_returns_none_problem_and_caches(self):
        _no_cache()
        m = mock.Mock(id="org/policy-act", tags=["lerobot"], downloads=42)
        with mock.patch("huggingface_hub.HfApi") as api:
            api.return_value.list_models.return_value = [m]
            rows, problem = checkpoints.hub_search("act", limit=3)
        assert problem is None
        assert rows and rows[0]["repo_id"] == "org/policy-act"
        assert checkpoints._CACHE  # cached for the TTL


class TestHfAuthState:
    def test_no_token_is_anonymous_with_reason(self):
        _no_cache()
        with mock.patch("huggingface_hub.get_token", return_value=None):
            st = checkpoints.hf_auth_state()
        assert st["authenticated"] is False
        assert "no HF token" in st["detail"]

    def test_valid_token_names_the_user(self):
        _no_cache()
        with mock.patch("huggingface_hub.get_token", return_value="hf_x"), \
             mock.patch("huggingface_hub.HfApi") as api:
            api.return_value.whoami.return_value = {"name": "cagataycali"}
            st = checkpoints.hf_auth_state()
        assert st == {"authenticated": True, "user": "cagataycali", "detail": None}

    def test_rejected_token_is_distinct_from_no_token(self):
        # revoked token != anonymity: gated downloads will 401, not public-only
        _no_cache()
        with mock.patch("huggingface_hub.get_token", return_value="hf_dead"), \
             mock.patch("huggingface_hub.HfApi") as api:
            api.return_value.whoami.side_effect = PermissionError("401")
            st = checkpoints.hf_auth_state()
        assert st["authenticated"] is False
        assert "rejected" in st["detail"]

    def test_whoami_answer_is_cached(self):
        _no_cache()
        with mock.patch("huggingface_hub.get_token", return_value="hf_x"), \
             mock.patch("huggingface_hub.HfApi") as api:
            api.return_value.whoami.return_value = {"name": "u"}
            checkpoints.hf_auth_state()
            checkpoints.hf_auth_state()
        assert api.return_value.whoami.call_count == 1


class TestSearchEnvelope:
    def test_search_carries_hub_problem_and_auth(self):
        _no_cache()
        with mock.patch.object(checkpoints, "local_checkpoints", return_value=[]), \
             mock.patch.object(checkpoints, "hub_search", return_value=([], "Hub search unavailable (X) - showing local cache only")), \
             mock.patch.object(checkpoints, "hf_auth_state", return_value={"authenticated": False, "user": None, "detail": "no HF token on this machine"}):
            out = checkpoints.search("q", limit=5)
        assert out["hub_problem"].startswith("Hub search unavailable")
        assert out["hf_auth"]["authenticated"] is False
        assert out["results"] == []
