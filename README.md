# robot_state_keys refusal coverage

`capture.py` measures, per provider, whether the `raise ValueError(error)` line in
`set_robot_state_keys` is executed by the suite, and re-runs both halves of
`tests/policies/test_state_key_name_list_contract.py` against three mutations of
each guard. `compose.py` renders the figure from that JSON and asserts every
number it prints. Run from a checkout root:

    PYTHONPATH=. python3 capture.py && python3 compose.py
