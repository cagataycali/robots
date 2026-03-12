# Examples

Practical examples showing how to use Strands Robots for real tasks.

---

## Minimal: 3 Lines

```python
from strands import Agent
from strands_robots import Robot

robot = Robot("so100")
agent = Agent(tools=[robot])
agent("Pick up the red cube")
```

---

## Simulation Loop

```python
from strands_robots import Simulation

sim = Simulation("so100")
sim.reset()

for step in range(1000):
    obs = sim.get_observation()
    sim.apply_action([0.1, -0.2, 0.0, 0.0, 0.0, 0.5])
    sim.step()

sim.render("final.png")
```

---

## Policy Control

```python
from strands_robots import Robot, create_policy

robot = Robot("so100")
policy = create_policy("mock")

for step in range(200):
    obs = robot.get_observation()
    action = policy.get_actions(obs, "pick up cube")
    robot.apply_action(action)
```

---

## Multi-Tool Agent

```python
from strands import Agent
from strands_robots import Robot, gr00t_inference, lerobot_camera, pose_tool

agent = Agent(tools=[
    Robot("so100"),
    gr00t_inference,
    lerobot_camera,
    pose_tool,
])

agent("Discover cameras")
agent("Save current position as home")
agent("Start GR00T on port 8000")
agent("Pick up the red block using GR00T")
agent("Go to home position")
agent("Stop GR00T")
```

---

## Training Pipeline

```python
from strands_robots import Robot, create_policy, create_trainer

# 1. Record
robot = Robot("so100")
policy = create_policy("mock")
robot.start_recording("demo_data")
for ep in range(10):
    robot.reset()
    for step in range(200):
        obs = robot.get_observation()
        action = policy.get_actions(obs, "wipe table")
        robot.apply_action(action)
robot.stop_recording()

# 2. Train
trainer = create_trainer("lerobot",
    policy_type="act",
    dataset_repo_id="demo_data",
)
trainer.train()

# 3. Deploy
trained_policy = create_policy("/data/checkpoints/act_demo")
agent = Agent(tools=[Robot("so100")])
agent("Wipe the table")
```

---

## Interactive Control

```python
from strands import Agent
from strands_robots import Robot, pose_tool

agent = Agent(tools=[Robot("so100"), pose_tool])

while True:
    command = input("🤖 > ")
    if command in ("exit", "quit"):
        break
    agent(command)
```
