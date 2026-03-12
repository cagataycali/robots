# Custom Policies

Build your own policy provider and plug it into the Strands Robots ecosystem.

---

## The Policy Interface

Every policy implements one method:

```python
from strands_robots.policies import Policy

class MyPolicy(Policy):
    def get_actions(self, observation, instruction):
        """
        Args:
            observation: dict with 'joint_positions', 'cameras', etc.
            instruction: str like "pick up the red cube"

        Returns:
            numpy array of joint target positions
        """
        # Your logic here
        return np.zeros(6)

    @property
    def provider_name(self):
        return "my_policy"
```

That's the entire contract. Everything else is up to you.

---

## Register It

```python
from strands_robots.policies import register_policy

register_policy("my_policy", lambda: MyPolicy, aliases=["custom", "my"])
```

Now it works everywhere:

```python
policy = create_policy("my_policy")
# or
policy = create_policy("custom")
```

---

## Examples

### PID Controller

```python
class PIDPolicy(Policy):
    def __init__(self, target_position):
        self.target = np.array(target_position)
        self.kp = 1.0

    def get_actions(self, observation, instruction):
        current = observation['joint_positions']
        error = self.target - current
        return current + self.kp * error

    @property
    def provider_name(self):
        return "pid"
```

### Remote API

```python
class RemotePolicy(Policy):
    def __init__(self, api_url):
        self.url = api_url

    def get_actions(self, observation, instruction):
        response = requests.post(self.url, json={
            "observation": observation,
            "instruction": instruction,
        })
        return np.array(response.json()["action"])

    @property
    def provider_name(self):
        return "remote"
```

---

## Testing

Use the mock policy pattern to test your integration:

```python
from strands_robots import Robot, create_policy

robot = Robot("so100")
policy = create_policy("my_policy")

obs = robot.get_observation()
action = policy.get_actions(obs, "test instruction")
assert action.shape == (6,), f"Expected 6 actions, got {action.shape}"
```
