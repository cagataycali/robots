### Fixed: the dashboard's Hub search works on the documented install, and says what is missing when it cannot

The dashboard page installs `strands-robots[dashboard,sim-mujoco]`, and neither extra
shipped `huggingface_hub`, so on exactly that install the Train tab read "Hub search
unavailable (ModuleNotFoundError) - showing local datasets only" and the Policies tab
the same for checkpoints: an exception class, with no module and no remedy. The
`[dashboard]` extra now ships `huggingface_hub` (same `>=1.5,<2.0.0` floor as every
other extra), and a missing module is reported by name with the extra that supplies
it ("huggingface_hub is not installed: pip install 'strands-robots[dashboard]'");
any other failure keeps its own message.
