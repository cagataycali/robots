### Fixed

- `list_objects` (MuJoCo) reports each object's live position from the physics instead of the pose
  `add_object`/`move_object` requested. The scene record is never written by the simulation, so an
  object that settled under gravity reported its spawn height for the rest of the session and one the
  robot pushed reported no motion at all -- the reading a caller grounds "did the object move" on. An
  object the compiled model carries no body for is now reported as an error naming it, rather than as
  its last requested placement.
