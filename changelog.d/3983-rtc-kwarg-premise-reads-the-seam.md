### Tests: the RTC kwarg premise reads lerobot's contract off the seam, not by name

`test_rtc_guidance_ceiling_reaches_the_config.py` grades one lerobot fact: the
RTC guidance ceiling is *not* a `predict_action_chunk` keyword, which is why
`_init_rtc` writes a caller's override onto `rtc_config` instead of forwarding
it. It read that fact as `modeling_smolvla.ActionSelectKwargs.__annotations__`
- a module attribute whose name belongs to lerobot. lerobot moved the RTC
`TypedDict` to `pretrained.RTCActionSelectKwargs` and gave the old name a
different meaning (`{noise}`), so on lerobot main the cell raised
`AttributeError` and reported nothing about the contract; the ceiling could be
added upstream without the canary ever firing.

The contract is now resolved from the seam `_predict_with_rtc` actually
forwards into - the `TypedDict` that `SmolVLAPolicy.predict_action_chunk`
unpacks - so it reads the same fact under either spelling. The cell also gained
the other direction: the forwarded keyword names are derived from
`_predict_with_rtc`'s own source, so every keyword the method sends is graded
against what lerobot declares, and a newly forwarded one is covered without
editing the test.
