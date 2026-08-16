### Fixed

Repaired `test_key_naming_no_compiled_camera_is_absent`, which reached its case
through a camera double-registration that has since been fixed and so failed its
own premise assert on `main`. The unanswerable camera key is now registered
directly against the camera registry the render loop resolves against, keeping
the contract pinned (an observation key that names a camera carries that
camera's view, or is absent) without depending on another module's defect.
