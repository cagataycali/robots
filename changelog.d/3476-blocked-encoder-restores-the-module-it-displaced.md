### Fixed: blocking an optional module in a test puts back the module it displaced

Two test helpers made ``imageio`` read as absent by assigning ``None`` over its
``sys.modules`` entry and ended by *deleting* the key. A deleted entry does not
undo an import - the next import returns a different module object - so a later
test's double, applied to the module it imported at collection time, was
installed on an orphan and the code under test reached the real encoder. The
rollout video writer's cleanup test reported a leaked writer whenever it ran
behind either helper, and passed alone. Both now share one owner that restores
both registries through ``monkeypatch``, and the tree-wide rule no longer reads
a ``finally`` that only removes the key as restoration.
