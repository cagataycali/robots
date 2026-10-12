### Tests: RL `device` joins the RL refusal table, and an array device is reported instead of raised

`RLTrainSpec.device` is now a row of the RL field table in
`tests/training/test_rl_spec_field_domains.py`, replacing
`tests/training/test_rl_trainer_device_domain.py`; 95 cells become 66. Running
the row over an array probe found that the PPO, FastSAC and FastTD3 preflights
raised `ValueError` for a device such as `np.array(["cuda", "cpu"])` (the
"is it unstated" check read it by truthiness); `validate()` now reports it as
not a torch device string.
