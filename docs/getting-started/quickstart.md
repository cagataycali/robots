# Quickstart

Get a robot agent running in 5 minutes.

---

## 1. Install

```bash
pip install "strands-robots[mujoco]"
```

## 2. Run

```python
from strands import Agent
from strands_robots import Robot

robot = Robot("so100")
agent = Agent(tools=[robot])
agent("Pick up the red cube")
```

## 3. That's it

The robot auto-detected no hardware, started a MuJoCo simulation, and the agent used the Robot tool to execute your instruction.

<div class="robot-gallery" markdown>
<figure markdown>
  ![SO-100 Simulation](../assets/so100_act_demo.gif){ width="400" }
  <figcaption><code>Robot("so100")</code> running ACT policy in MuJoCo</figcaption>
</figure>
</div>

---

## What Just Happened?

1. `Robot("so100")` looked up "so100" in the registry → found the SO-100 arm
2. No USB hardware detected → launched MuJoCo simulation
3. `Agent(tools=[robot])` registered the robot as a callable tool
4. `agent("Pick up the red cube")` → the LLM reasoned about the task and called the robot tool

---

## Next Steps

| Want to... | Go to... |
|---|---|
| Follow the full tutorial | [Learning Path](../learning-path.md) |
| See all 38 robots | [Robot Catalog](../robots/index.md) |
| Understand policies | [Policy Providers](../policies/overview.md) |
| Use real hardware | [Real Hardware](../hardware/robot-control.md) |

---

## Model Provider

By default, Strands uses Amazon Bedrock. You can use any provider:

=== "Bedrock (default)"
    ```bash
    # Configure AWS credentials
    aws configure
    ```

=== "Ollama (local)"
    ```python
    from strands.models.ollama import OllamaModel
    agent = Agent(model=OllamaModel(model_id="llama3"), tools=[robot])
    ```

=== "OpenAI"
    ```python
    from strands.models.openai import OpenAIModel
    agent = Agent(model=OpenAIModel(model_id="gpt-4o"), tools=[robot])
    ```

See [Strands docs](https://strandsagents.com/) for all providers.
