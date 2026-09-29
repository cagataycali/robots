### Tests: Isaac stand-in engines start from the real constructor

Fifty-two test modules built an `IsaacSimulation` skeleton through `__new__` and
then restated `__init__` field by field - 420 assignments, each a stale copy
waiting for the constructor to gain a field. They now start from
`tests.simulation._isaac_engine.isaac_engine()`, which runs the real (Kit-free)
constructor with the finalizer disarmed, and override only what they model. A
grader refuses a skeleton that restates a constructor default; a bare `__new__`
stays legal where partial construction is the subject.
