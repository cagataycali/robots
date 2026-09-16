### Fixed: `evaluate_benchmark` names a benchmark written for another robot first

Asking a fixed-base arm to run a locomotion benchmark used to answer with a
floating-base lecture and a `<the sole robot>` placeholder; it now says which
robots the benchmark is written for, what the scene's robot is, and the two
ways out - ahead of the clause probe. The probe spells the robot's name, and
the runner's compatibility refusal names the benchmark by its registered id
instead of `DeclarativeBenchmark`.
