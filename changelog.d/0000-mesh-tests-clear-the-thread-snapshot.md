### Tests: mesh tests clear the thread-local ACL snapshot between cases

`tests/mesh/conftest.py` clears the thread-local ACL snapshot around every mesh test, so a test that calls `Mesh._refuse_under_permissive_default_acl()` directly no longer leaves `auth_mode="mtls"` on the main thread for the tests that run after it. `tests/mesh/test_acl_thread_snapshot_isolation.py` pins the invariant.
