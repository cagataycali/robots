### Fixed: a wrong-family WBC checkpoint is refused where it is opened

The two decoupled-WBC G1 checkpoint families differ in exactly one number -- the
ONNX input width (Balance/Walk declare `[batch, 516]` = 86 x 6, the gait-clock
family `[batch, 570]` = 95 x 6) -- while both declare a 15-wide output. Only the
output width was checked, so loading the wrong family constructed fine and then
failed on the first rollout tick with an onnxruntime `Got invalid dimensions for
input ... Please fix either the inputs/outputs or the model`, naming neither the
policy nor the fix, after the world had been built and the G1 had begun to fall.
Both sessions now state their own width at load and a mismatch names both widths
plus the provider each family belongs to.
