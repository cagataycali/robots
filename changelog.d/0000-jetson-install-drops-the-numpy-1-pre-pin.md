### Docs: the Jetson install path no longer pins `numpy<2` first

`docs/getting-started/installation.md` told JetPack users to
`uv pip install "numpy<2" "pandas==2.1.4"` before installing
`strands-robots[sim-mujoco,lerobot]`. `lerobot >= 0.6` requires `numpy >= 2`,
so the second line replaced the first (numpy 1.26.4 -> 2.2.6, measured on a
Thor devkit), and the pin only suggested a numpy-1 requirement that does not
exist - `strands-robots doctor` passes there on numpy 2.2.6 with torch 2.11
`+cu130`. The pre-pin is gone; the troubleshooting row for a numpy ABI
mismatch now names the actual cause (a wheel built against numpy 1.x imported
under numpy 2) and the fix that survives the next install.
