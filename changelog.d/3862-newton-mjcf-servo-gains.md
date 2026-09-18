### Fixed

- **Newton backend: an MJCF position servo keeps its damping and torque ceiling.**
  Newton's MJCF importer reads a `<position>` actuator's `kp` into
  `joint_target_ke` and drops the rest of the servo - the `dampratio` MuJoCo
  compiles into a velocity gain and the `forcerange` that caps the torque - so a
  joint arrived P-only with a 1e6 N m ceiling and a held position command
  oscillated instead of settling (on the shipped `so100`, a constant
  `Rotation = 0.5` swung between 0.054 and 0.956 rad indefinitely where the
  MuJoCo backend settles on 0.4997). Both values are now read off the compiled
  model and written onto the builder before `finalize`; a model MuJoCo cannot
  compile keeps the gains Newton did carry and logs the reason.
