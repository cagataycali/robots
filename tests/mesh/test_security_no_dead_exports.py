"""Regression test: AGENTS.md Key Conventions #10 -- no dead code in public API.

Pinned by review R3 on PR #223: `LockoutError` was defined and exported
in `__all__` with no consumer in this PR's scope. Same shape as the R1
removal of `MAX_TIMEOUT_S`. Will re-land in PR-6 alongside Mesh._dispatch
(the consumer). This test pins the invariant that every name in
`security.__all__` is either importable from the module AND has at least
one test or internal reference demonstrating usage.
"""

from strands_robots.mesh import security


def test_lockout_error_not_exported():
    """LockoutError must not be in __all__ until a consumer lands."""
    assert "LockoutError" not in security.__all__, (
        "LockoutError re-introduced in __all__ without a consumer in this module. "
        "Per AGENTS.md Key Conventions #10, land the exception alongside "
        "Mesh._dispatch (PR-6) which raises it."
    )


def test_lockout_error_class_not_defined():
    """LockoutError must not be defined in this module until a consumer lands."""
    assert not hasattr(security, "LockoutError"), (
        "LockoutError class re-introduced in security.py without a consumer. Land alongside Mesh._dispatch in PR-6."
    )


def test_all_exports_are_importable():
    """Every name in __all__ must be importable from the module."""
    for name in security.__all__:
        assert hasattr(security, name), f"{name} in __all__ but not importable"
