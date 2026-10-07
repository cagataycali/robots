Added a `plastic_wam` policy provider: a self-learning VLA (frozen Qwen3.5-4B System 2 + flow-matching System 1 with
test-time-training fast weights) whose bounded, reversible plastic LoRA learns online from corrections
(`learn_from_correction`, `reset_plastic`, `save_brain`/`load_brain`). Model code: github.com/cagataycali/plastic-model.
