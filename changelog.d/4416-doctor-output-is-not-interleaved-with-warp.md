### Fixed: `strands-robots doctor` no longer prints Warp's init banner in the middle of its report

The Warp architecture check queried Warp's devices, which runs `wp.init()` and
prints Warp's 8-line "Warp 1.x initialized: ... Devices ... Kernel cache" block
between the torch and Warp checks. The probe now silences it first - `config.quiet`
on Warp before 1.19, `config.log_level` raised to warnings on 1.19+ - and the
check's own PASS line still carries the architecture and the CUDA build.
