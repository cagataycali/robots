# lerobot_train publication gate — measurement

`capture.py` drives `lerobot_train(action="start", ...)` with `subprocess.Popen`
replaced by a recorder, so "launched" means "the tool would have started
training with this argv". It is run once per tree; each dump records the tree it
imported so the two arms cannot be confused.

    PYTHONPATH=<tree> python3 capture.py measured_<tree>.json
    python3 compose.py measured_main.json measured_branch.json publication_gate.png

`mutate.py` re-applies the five regressions in the figure's third row, each
anchor AST-scoped to its enclosing function, and restores the source
byte-identically.

No GPU, no network, no Hugging Face Hub access, no training.
