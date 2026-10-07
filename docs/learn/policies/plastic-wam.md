---
description: Self-learning VLA with a bounded plastic LoRA.
---

# plastic_wam

Qwen3.5-4B System 2, flow-matching System 1, plastic LoRA learning from corrections.

```bash
pip install 'plastic-wam @ git+https://github.com/cagataycali/plastic-model'
```

{{providers:kwargs:plastic_wam}}

```python title="sketch"
policy = create_policy("plastic_wam", plastic=True)
policy.learn_from_correction(chunk)
policy.reset_plastic()
```
