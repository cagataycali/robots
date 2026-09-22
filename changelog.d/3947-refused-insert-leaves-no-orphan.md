### Fixed: a refused object or camera insert leaves no orphan in the spec

`SpecBuilder.add_object` / `add_camera` rolled a refused insert back by name,
which cannot see what mujoco 3.12 and later leave behind: the repeated name is
refused and the appended element is left UNNAMED, so an anonymous body reached
every later compile and an anonymous camera shifted the camera indices after it
(measured on 3.14.0: `nbody` 2 -> 3, `ncam` 1 -> 2 per refused add). The rollback
is now by position - new elements are appended, so the ones a call added are the
tail beyond the length taken before the insert - and a refused insert leaves the
compiled model exactly as it was on every supported build.
