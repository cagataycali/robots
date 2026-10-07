### Docs: the Microduck page names the walking and standing weights that run on it

`docs/robots/microduck.md` said no checkpoint had been verified on the robot and
sent the reader off to record and train one, while the policy page, the driver
and the `microduck` provider all use Pollen's shipped `alpha_walking.onnx` and
`alpha_stand.onnx`. The page now lists both with what a 5 s MuJoCo rollout did,
and a robot page that names a provider written for its body may no longer say
that no checkpoint ran on it.
