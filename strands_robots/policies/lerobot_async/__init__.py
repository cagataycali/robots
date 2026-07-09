"""LeRobot async-inference client policy (gRPC bridge to a lerobot PolicyServer).

Exposes :class:`LerobotAsyncPolicy`, a :class:`~strands_robots.policies.base.Policy`
that offloads inference to a remote lerobot ``PolicyServer`` over gRPC. The heavy
gRPC / lerobot transport imports are deferred to first use, so importing this
package is cheap and does not require ``lerobot[async]`` to be installed until
the policy is actually run.
"""

from strands_robots.policies.lerobot_async.policy import LerobotAsyncPolicy

__all__ = ["LerobotAsyncPolicy"]
