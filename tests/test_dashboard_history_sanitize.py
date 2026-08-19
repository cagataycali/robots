"""BUGS.md #15 - a persisted conversation must never come back poisoned."""

import json

import pytest

from strands_robots.dashboard import agent_bridge as ab


def user(text="hi"):
    return {"role": "user", "content": [{"text": text}]}


def assistant_tool_use(tid="t1"):
    return {"role": "assistant", "content": [{"toolUse": {"toolUseId": tid, "name": "fleet", "input": {}}}]}


def user_tool_result(tid="t1"):
    return {"role": "user", "content": [{"toolResult": {"toolUseId": tid, "content": [{"text": "ok"}]}}]}


def assistant(text="sure"):
    return {"role": "assistant", "content": [{"text": text}]}


def test_valid_history_is_untouched():
    good = [user(), assistant_tool_use(), user_tool_result(), assistant()]
    assert ab.sanitize_history(good) == good
    assert ab._history_problem(good) is None


def test_leading_assistant_prefix_is_dropped():
    out = ab.sanitize_history([assistant("orphan opener"), user("real"), assistant()])
    assert out[0] == user("real")
    assert ab._history_problem(out) is None


def test_dangling_tool_use_is_pruned_not_restored():
    out = ab.sanitize_history([user(), assistant_tool_use("zz"), assistant("after")])
    assert ab._history_problem(out) is None
    assert "zz" not in json.dumps(out)
    assert out[0]["role"] == "user"


def test_dangling_tool_result_is_pruned():
    out = ab.sanitize_history([user(), user_tool_result("nope"), assistant()])
    assert ab._history_problem(out) is None
    assert "nope" not in json.dumps(out)


def test_garbage_input_yields_empty_history():
    assert ab.sanitize_history("not a list") == []
    assert ab.sanitize_history([{"role": "assistant", "content": "str"}]) == []


def test_unrestorable_file_is_backed_up_and_not_restored(tmp_path, monkeypatch, caplog):
    hist = tmp_path / "chat_history.json"
    hist.write_text(json.dumps([assistant_tool_use("dangling")]))
    monkeypatch.setattr(ab, "HISTORY_FILE", hist)

    raw = ab._load_history()
    assert raw, "precondition: the poisoned file loads"
    assert ab.sanitize_history(raw) == []

    with caplog.at_level("WARNING"):
        backup = ab._backup_history_file()
    assert backup is not None and backup.exists()
    assert not hist.exists()
    assert json.loads(backup.read_text())  # evidence preserved


def test_save_round_trips_through_sanitizer(tmp_path, monkeypatch):
    hist = tmp_path / "h.json"
    monkeypatch.setattr(ab, "HISTORY_FILE", hist)
    ab._save_history([assistant("leading junk"), user("keep me"), assistant_tool_use("x")])
    saved = json.loads(hist.read_text())
    assert ab._history_problem(saved) is None
    assert saved[0] == user("keep me")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
