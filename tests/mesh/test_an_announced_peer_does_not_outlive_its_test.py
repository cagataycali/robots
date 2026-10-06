"""A peer one test announces is not in the roster the next test reads.

The roster is module-global and ``Mesh._on_presence`` never removes a sender,
so without ``tests/mesh/conftest.py::_restore_peer_roster`` a test announcing
``leader-1`` made a later gateway test count a remote peer it never
discovered. The two cells run in file order: the first leaks, the second reads.
"""

from __future__ import annotations

from strands_robots.mesh import session


def test_a_test_announces_a_peer() -> None:
    session.update_peer("leaked-by-an-earlier-test", "operator", "", {})
    assert any(p["peer_id"] == "leaked-by-an-earlier-test" for p in session.get_peers())


def test_the_next_test_does_not_see_it() -> None:
    assert not any(p["peer_id"] == "leaked-by-an-earlier-test" for p in session.get_peers())
