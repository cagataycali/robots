### Deprecated: a driver's `start_task(policy_provider=)` is removed in 0.8

The Feetech and UR drivers now raise a `DeprecationWarning` when `start_task`
builds a policy from a provider name. In 0.8 the `Driver` contract keeps one
policy verb, `run_policy(policy_object=...)`, which takes a built policy:

```python
policy = create_policy("lerobot_local", pretrained_name_or_path=...)
driver.run_policy(policy_object=policy, instruction="pick up the cube")
```

Nothing else changes in 0.6: `start_task` still builds and runs.
