### Fixed

- **Isaac `remove_robot` no longer drops another robot's prim from the teardown
  registry.** The prune is documented as removing "any prims rooted at the
  robot's prim path", and tested that with a bare `startswith` on the path
  string. A prim path is interpolated from the robot's name
  (`{stage_path}/Robots/{name}`), so a robot whose name merely *extends* the
  removed one's counted as a prim beneath it: with `arm` and `arm_left` both
  live, `remove_robot("arm")` dropped `/World/Robots/arm_left` while `arm_left`
  stayed registered, leaving a live robot with no prim for `destroy` to release
  and a `prims_released` count one short. The prune is now bounded at `/`, USD's
  path separator, so it keeps the subtree it promises and stops at a sibling.
