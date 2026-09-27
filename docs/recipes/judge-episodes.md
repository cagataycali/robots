---
description: Label episodes, then train on the ones worth keeping.
---

# Keep only the good episodes

You score recorded episodes, keep the successful ones and train on that subset alone.

![Frames from a judged episode](../assets/recipes/judge_episodes.png)

```python title="examples/17_judge_recorded_episodes.py"
--8<-- "examples/17_judge_recorded_episodes.py"
```

```console
$ MUJOCO_GL=egl python examples/17_judge_recorded_episodes.py
selected episodes: [1, 2]
labels on disk   : {"0": "low", "1": "high", "2": "high"}
training status: success
```

Go deeper: [Episode judge](../reference/data/episode-judge.md) · [Episode labels](../reference/data/episode-labels.md)
