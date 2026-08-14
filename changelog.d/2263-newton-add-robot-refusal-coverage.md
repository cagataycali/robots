### Quality: pin Newton `add_robot`'s five caller-input refusals

`NewtonSimEngine.add_robot` refuses five caller-input shapes -- no world, a
duplicate robot name, an unknown `source`, a `keyframe` the backend does not
support yet, and an asset it cannot resolve -- and on an install without Newton
or Warp every one of those refusal returns was unexecuted. The modules that pin
them are gated on the solver being importable, so they skip on exactly the
install where a caller-input refusal is all a caller gets. The method's three
other refusals, the shared entity-name and pose-vector domains, are already
pinned by un-gated cross-backend modules; this covers the remaining five.

All five run before the solver is touched, so a `__new__` skeleton with a fatal
rebuild drives them with neither Newton nor Warp installed. Each refusal now
pins the token a caller has to act on, the five stay pairwise distinguishable,
and a rebuild that would raise proves the refusal precedes it.
