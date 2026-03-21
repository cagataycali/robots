"""Quick script for memray profiling."""
import strands_robots
from strands_robots.registry import list_robots, list_aliases, get_robot, resolve_policy
from strands_robots.policies.factory import list_providers
from strands_robots.policies.mock import MockPolicy
from strands_robots.policies.groot.data_config import load_data_config
import numpy as np

list_robots(); list_aliases(); list_providers()
get_robot("so100"); resolve_policy("groot")
load_data_config("so100")

mock = MockPolicy(action_space={f"j{i}": 0.0 for i in range(6)})
obs = {"observation.state": np.zeros(6, dtype=np.float32)}
for _ in range(100):
    mock.get_actions_sync(obs, "test")
