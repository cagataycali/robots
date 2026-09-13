### Changed: private instance state must have a reader

Six private attributes were assigned and never read anywhere in the package:
`_recent_returns` (the FastSAC and FastTD3 trainers, five assignments -
`collect_rollout` already reports the same numbers as `mean_episode_return`),
`_renderer_model` (the MuJoCo renderer cache invalidates per thread through
`_renderer_tls`; the per-instance marker was kept only "for compatibility with
any remaining read paths" and there were none), `_pb2_grpc` (the lerobot async
client builds its stub inline, unlike `_pb2` and `_grpc`, which are read),
`_viewer_thread` (`_viewer_handle` is the real viewer state), `_provider_arg`
(`provider_name` reports the wrapped policy's own provider) and
`_dds_security_config` (the participant's QoS is built from the argument).

All twelve assignments are removed. `_renderer_model` also came out of
`RenderingMixin`'s documented coupling list and its `TYPE_CHECKING` stub, so
hosting the mixin no longer implies providing an attribute nothing consults.
`PersistentPolicy`'s docstring no longer claims `provider` is "recorded for
identification" when a `policy_object` is given, because it was not.

An assignment with no reader is either state that documents an abandoned intent
or a value collected for something that never happens, and only reading the
whole class tells you which. `tests/test_no_private_state_is_written_and_never_read.py`
now pins the invariant across `strands_robots/`, counting a load anywhere in the
package, a string literal for the `getattr(self, "_name", None)` shape, and any
mention under `tests/` as readers. Two attributes stay, each with a reason:
`_background_task`, which is the strong reference that stops asyncio collecting
the Device Connect runtime task, and `_record_converge`, where
`SO101_RECORD_CONVERGE` has no consumer yet (#3578).
