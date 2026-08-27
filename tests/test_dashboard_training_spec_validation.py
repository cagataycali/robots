"""Q6: /api/training/validate + submit must refuse unknown fields as JSON, never 500.

The request body used to be splatted into train_policy as **kwargs, so any
unexpected key (a typo, an old client, a curl experiment) became a TypeError
that FastAPI rendered as a bare-HTML 500 - which the UI's res.json() then
choked on, compounding the failure. The contract now: a bad body comes back
as a structured {"status": "error", "text": "unknown field(s): ..."} that
names the offenders AND the valid vocabulary, so the error teaches the fix.

Run with --no-cov (single-file runs trip the global coverage gate).
"""

from strands_robots.dashboard import training


class TestSpecKwargs:
    def test_clean_body_passes_through(self):
        kwargs, err = training._spec_kwargs({"provider": "lerobot_local", "steps": 500, "dataset_root": "/tmp/d"})
        assert err is None
        assert kwargs == {"provider": "lerobot_local", "steps": 500, "dataset_root": "/tmp/d"}

    def test_unknown_keys_named_and_sorted(self):
        kwargs, err = training._spec_kwargs({"policy": "x", "dataset": "y"})
        assert kwargs is None
        assert err["status"] == "error"
        # both offenders named, alphabetical, so the message is deterministic
        assert "unknown field(s): dataset, policy" in err["text"]

    def test_error_teaches_the_valid_vocabulary(self):
        _, err = training._spec_kwargs({"nope": 1})
        for field in ("dataset_root", "base_model", "steps", "method"):
            assert field in err["text"]

    def test_none_values_dropped_not_forwarded(self):
        kwargs, err = training._spec_kwargs({"provider": "lerobot_local", "seed": None})
        assert err is None
        assert "seed" not in kwargs

    def test_action_key_tolerated(self):
        # callers historically included action; it is routing, not spec
        kwargs, err = training._spec_kwargs({"action": "validate", "steps": 10})
        assert err is None
        assert kwargs == {"steps": 10}


class TestValidateAndSubmitRefuseBadBodies:
    def test_validate_bad_body_is_structured_error(self):
        # the exact body from the BUGS.md Q6 repro
        res = training.validate({"dataset": "nobody/nothing-zz", "policy": "nope"})
        assert res["status"] == "error"
        assert "unknown field(s)" in res["text"]

    def test_submit_bad_body_is_structured_error_and_records_no_job(self, tmp_path, monkeypatch):
        monkeypatch.setattr(training, "JOBS_FILE", tmp_path / "jobs.json")
        res = training.submit({"stepz": 100})
        assert res["status"] == "error"
        assert "stepz" in res["text"]
        assert training.jobs() == []

    def test_validate_and_submit_share_one_vocabulary(self):
        # SPEC_KEYS drifting between the two would let a field validate but
        # silently vanish on submit - the worst kind of "it worked in check"
        v = training.validate({"definitely_not_a_field": 1})
        s = training.submit({"definitely_not_a_field": 1})
        assert v["text"] == s["text"]
