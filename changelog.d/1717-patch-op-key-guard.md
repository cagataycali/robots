### Quality: a `patch_scene_mjcf` op key a test misspells can no longer pass for the wrong reason

Two rollback tests sent `"position"` to an `add_body` op, which reads `"pos"`.
Before the op vocabulary was enforced the key was silently dropped and the body
was authored at the origin; both tests still passed, because each asserted only
that the body was *present by name* -- an assertion that cannot distinguish a
honored pose from a defaulted one. Once the vocabulary began refusing unread
keys, one of the two started failing and the other kept passing while testing
nothing: its batch was refused on op #1, so nothing was applied and the
mid-batch rollback it exists to exercise never ran.

Both ops now use the keys they are read under, and both tests assert the
authored pose rather than just the name. The refused-batch case additionally
asserts the failure came from op #2, which is the observable proof that op #1
was applied and then rolled back.

`tests/simulation/mujoco/test_suite_patch_ops_use_accepted_keys.py` closes the
gap that let this hide. It reads the op vocabulary from its single source of
truth and refuses any op-dict literal in the suite whose keys fall outside it,
exempting only the module whose subject is rejection -- and pins that the
exemption is still earned, plus that the scanner detects a planted misspelling
rather than silently matching nothing.
