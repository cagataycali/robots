### Fixed

- **Remote inference: a body name dropped from an advertised declaration is a pose
  that never arrives.** `Policy.required_bodies` is the "policy declares, runtime
  supplies" contract, and `collect_required_bodies` is its single owner - the
  simulation runtime and `PolicyServer` both ask it, and its docstring is explicit
  that the two "must not disagree". The `RemotePolicy` handshake mirrored the
  advertised list through a filter instead: it kept the entries it could use and
  dropped the rest, so a peer advertising `["torso_link", 42]` produced a proxy
  declaring `("torso_link",)` - a declaration nobody made. Nothing downstream can
  tell that from a peer that really asked for one body: the robot host resolves the
  shorter set against its scene, merges poses for it, and reports a successful
  rollout. Measured over a real `PolicyServer` driving a real MuJoCo rollout, a
  policy declaring two anchor bodies reported `status="success"` with the second
  body's `quat` arriving on 0 of 12 control steps - and a whole-body mimic tracker
  whose anchor link goes unsupplied does not fail, it reads `base_quat`, the pelvis,
  and silently tracks the wrong frame.

  The shape rule now lives in `required_bodies_error`, the shared owner both
  surfaces ask, and an unmirrorable declaration is refused in the handshake beside
  the chunk counts and capability flags - `required_bodies` was the last field there
  still coerced rather than checked, after the `int()` and `bool()` coercions were
  removed for the same reason. A refusal names the field, quotes the offending entry
  and its index, and leaves neither the mirror half-applied nor the connection
  cached. A repeated name and an empty list stay accepted, because the local owner
  accepts them - refusing those would be the same disagreement in the other
  direction.

  The local half gains attribution it lacked: a declaration that is not a sequence
  raised a bare `'int' object is not iterable` naming no policy, and a `Mapping` was
  read as its keys with its values silently discarded.

  The regression grades
  the advertised body list against `collect_required_bodies` itself rather than
  against a list of values the test picked, so the two halves cannot drift apart, and
  the accepting rows pin that the refusal is not an over-refusal. The pin that had
  accepted the filter as intended posed only wholly-unusable values, never the mixed
  list the filter answered with a shorter declaration; it is retargeted to the
  refusal with that row added.
