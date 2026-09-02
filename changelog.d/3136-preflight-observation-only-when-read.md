### Fixed: the policy preflight builds the runtime observation only for a provider that reads it

`SimEngine._preflight_policy_config` ran a provider's class-level `Policy.preflight` hook
against the keys the runtime observation will carry, and built that observation before asking
whether the hook exists. `preflight_policy` is a no-op for any provider that does not override
the hook - every shipped provider except `lerobot_local` - so on that path a full
`get_observation` rendered every model camera plus every python camera in the scene and the
frames were discarded. In a lekiwi + unitree_go2 + so101 scene with one added camera that
observation renders 6 image keys in 1.283s on a software rasterizer, against 0.00026s for the
joint state alone.

The cost is not only wasted work: the preflight runs before the rollout loop, so it also
delays the loop's cooperative-stop check. Fleet-wide, three `mock` rollouts stopped
immediately after `start_policy` went quiet 2.430s after the stop was acknowledged, and
`examples/fleet/04_emergency_evacuation.py` exceeded its 10s abort deadline and exited 1 on a
software-GL host.

The keys are now passed to `preflight_policy` as a supplier, so the one place that decides
whether the hook consumes them is the place that invokes it. `mock` renders nothing;
`lerobot_local`, whose hook validates camera routing, still receives the full observation with
its camera keys. The same seam serves `run_policy`, `eval_policy` and `start_policy`. Time to
fleet quiet falls to 0.050s and the example passes (abort 1.50s). A supplier returning `None`
reports an observation that is not available yet, preserving the existing disposition that
such a run is not blocked by the check.
