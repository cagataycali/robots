# Qwen-VLA reference model + end-to-end runner

A **small but genuine** runnable Qwen-VLA implementation used to exercise the
full inference + training pipeline locally on a single GPU, before the upstream
Qwen-VLA package / checkpoint is public.

- `reference_model.py` — VLM-style conditioning encoder + AdaLN DiT
  flow-matching action expert + stop-gradient value head. Implements exactly the
  model interface the stage runners (`run_t2a/cpt/sft/rl`) and the ZMQ server
  expect.
- `reference_server.py` — ZMQ server speaking the `QwenVlaInferenceClient`
  msgpack envelope (`ping` / `get_action` / `reset` / `reload`).
- `sim_env.py` — seeded, success-scored rollout env for Stage-4 PPO.
- `run_end_to_end.py` — runs T2A → CPT → SFT → RL → SERVICE+LOCAL inference →
  hot-swap redeploy, with assertions.

```bash
pip install -e '.[qwen-vla-train]'
python examples/qwen_vla_reference/run_end_to_end.py
```

This is a **reference for testing**, not the production model. When the upstream
Qwen-VLA ships, `policies/qwen_vla/policy.py` LOCAL mode loads it instead;
SERVICE mode already works against any server speaking the documented envelope.
