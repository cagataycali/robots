### Fixed: the mesh rate-limit window a test spends is refunded before the next test

`robot_mesh`'s sliding window is process-global, so an accepted tool call holds
its slot for the life of the process. A test that spent the window left the next
one's call answered "rate limit exceeded" instead of doing the thing it asserts:
with `tests/test_hitl_operator_response_audit.py` (which drains `tell` to its
limit of 30 to make the post-approval re-check deterministic) ahead of
`tests/mesh/test_robot_mesh_tool.py`, four cells there failed on the refusal.
One autouse fixture at the session root now clears the window, replacing the
copies ten modules kept; resets inside a test, where two accepted calls of one
action are the subject, are unchanged. Test-suite isolation only - no change to
the limiter or to any shipped behaviour.
