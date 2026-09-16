### Fixed: a `stop_policy` that stopped nothing names the rollouts it did not stop

`stop_policy` on a robot with nothing running stays an idempotent success, but its
sentence was the same whether the world was idle or another arm was mid-rollout, so
a stop aimed at the wrong robot read as a stop that had nothing to do. It now names
the rollouts still in flight and ends with a remedy the tool accepts; a backend that
keeps no rollout registry keeps the bare verdict.
