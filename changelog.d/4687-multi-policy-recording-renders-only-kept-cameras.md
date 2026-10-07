### Fixed: a `run_multi_policy` recording scoped to no camera renders no frame

`run_multi_policy` on MuJoCo and Isaac rendered every scene camera on every
step while a dataset recording was open, even one started with
`start_recording(cameras=[])` whose dataset declares no image column, and then
dropped the pixels. The loop now leaves that choice to the recording, as
`run_policy` and `step` already do: a recording that keeps cameras still gets
its images, and one scoped to no camera, driven by policies that read no
pixels, renders nothing.
