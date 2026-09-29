### Fixed: the microduck hardware doc's "Sim to real" sketch ran MockPolicy

`docs/learn/hardware/microduck.md` followed the paragraph titled *Sim to real*
- the one that stakes the doc's whole claim on "the on-robot policy is the
same `alpha_walking.onnx` the `[microduck]` extra runs in MuJoCo
(byte-compatible, difference 0.0), so a sim rollout with equal observations
predicts the hardware" - with this sketch:

```python
sim = Robot("microduck")
sim.run_policy("microduck", instruction="walk forward")   # the provider self-configures from the ONNX metadata
```

`Robot("microduck")` returns a `MuJoCoSimEngine`, and its `run_policy`
signature is
`run_policy(robot_name=None, policy_provider="mock", policy_config=None, ...)`,
so the positional `"microduck"` bound to `robot_name` and `policy_provider`
kept its default. The rollout that ran was `MockPolicy`, not
`MicroduckPolicy`; the trailing content read `MockPolicy | walk forward` and
the "provider self-configures from the ONNX metadata" comment described a
provider that was never constructed. A reader who copied the sketch to
verify the sim-to-real claim two lines above verified only that
`MuJoCoSimEngine.run_policy` accepts a string and reports `success`.

Every other `run_policy` call in `docs/` uses keyword arguments; the sibling
`docs/learn/policies/microduck.md` spells the same call correctly on lines 58
and 91. This sketch is now aligned with those:

```python
sim = Robot("microduck")
sim.run_policy(
    robot_name="microduck",
    policy_provider="microduck",
    policy_config={"onnx_path": "alpha_walking.onnx"},
    instruction="walk forward",
)   # the provider self-configures from the ONNX metadata
```

which drives `MicroduckPolicy` with the same `alpha_walking.onnx` the
paragraph names, so what the sketch runs is what the surrounding prose
promises it runs.
