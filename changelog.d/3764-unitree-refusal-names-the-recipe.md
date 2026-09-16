### Fixed: a missing `unitree_sdk2py` is refused with the install recipe, not only its module name

Every lazy import of Unitree's SDK in the G1 and Go2 drivers, the DDS engine
and `use_unitree` answered `unitree_sdk2py is not installed: <exc>` and
stopped, and no extra of this project declares it - nor can one: the PyPI
`unitree-sdk2` wheel ships no `g1` package and pins `cyclonedds==0.10.2`, which
has no wheel for Python 3.12. The refusal now comes from one place,
`strands_robots.tools.g1._g1_common.sdk_missing`, and names the diagnosis, the
install line (`cyclonedds` wheel + the upstream checkout with `--no-deps`), the
Jetson/aarch64 caveat (no `cyclonedds` wheel exists there; build 0.10.2 and set
`CYCLONEDDS_HOME`) and the docs section. `docs/robots/humanoids.md` gains the
G1 native-driver section with that recipe per platform, proven in fresh venvs
on macOS arm64, x86_64 Linux and Jetson aarch64; a source-level test keeps
every `unitree_sdk2py` import site on the shared text.

That includes the sites a partial install reaches, which are the ones the wheel
produces: with `comm` absent but the bus bindings present, `connect_eagerly()`
succeeds and the motion-switcher open is the only failure, so the G1's
`get_status()` and the Go2's `release_sport_mode()` are where the remedy has to
appear. The source-level test reads three kinds of site, because a handler
catching `Exception` answers the `ImportError` too, and an import reached
through a propagating helper is answered by its caller - in a file that names
`unitree_sdk2py` nowhere.
