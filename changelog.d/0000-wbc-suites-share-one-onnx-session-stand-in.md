### Tests: the WBC suites share one ONNX session stand-in

Nine WBC test modules each carried their own copy of the stand-in for the
`onnxruntime.InferenceSession` that `WBCPolicy` reads through `policy_session`
and `walk_session`, and two of them also carried the same G1 key list, config
and policy builders. They now import one copy from
`tests/policies/wbc/_stand_ins.py`. The 730 WBC cells and the package lines they
execute are unchanged; the suite is 139 lines shorter.
