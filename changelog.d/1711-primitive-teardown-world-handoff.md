### Fixed: tearing the world down during a motion primitive aborts it instead of raising

`move_to`, `set_gripper` and `rotate_wrist` each document "Never raises.", and
`_primitive_abort_reason` documents the intended outcome for exactly this case:
"the world can legitimately be destroyed, the model recompiled, or a policy
started while a primitive runs. Each of those aborts the primitive with a
structured error rather than stepping a stale/contended model."

A concurrent `destroy()` / `cleanup()` did neither. `cleanup` handed the world
off (`self._world = None`) without holding `self._lock`, so the handoff could
land inside a primitive's control tick - between the tick's own world check and
its write-back of `sim_time` / `step_count`. All three primitives then raised
`AttributeError: 'NoneType' object has no attribute 'sim_time'` from
`_primitive_tick`, on the caller's thread:

| teardown during | before | after |
| --- | --- | --- |
| `move_to` | `AttributeError` | structured "world was destroyed ... aborting" |
| `set_gripper` | `AttributeError` | structured "world was destroyed ... aborting" |
| `rotate_wrist` | `AttributeError` | structured "world was destroyed ... aborting" |

`cleanup`'s existing reasoning covers the workers it can await: it signals
`policy_running = False` and joins every outstanding Future before nulling the
world. A motion primitive is neither. It runs on its CALLER's thread, so no
Future awaits it, and the cooperative-stop flag never applies to it - a
primitive refuses to start while a policy runs, so it never sets that flag.
`self._lock`, which it takes per control tick, was its only synchronisation, and
teardown was the one world-lifecycle operation that did not take it: `load_scene`
already brackets its own world handoff with the lock for the same reason,
`reset` does its work under the lock, and `create_world` refuses outright when a
world already exists.

The handoff now happens under `self._lock`. The Future join stays outside it - a
live policy worker takes the lock per step, so awaiting one while holding the
lock would deadlock - and the acquire is bounded (`_WORLD_HANDOFF_LOCK_TIMEOUT`),
making the same tradeoff the bounded Future join already makes: warn and tear
down regardless rather than hang the host process on exit.
