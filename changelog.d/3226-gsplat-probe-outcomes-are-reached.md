### Tests: the gsplat capability probe's three outcomes are reached, not assumed

`gsplat_rasterizer_available` performs a one-gaussian trial rasterization
rather than reporting on an import, because a plain `pip install gsplat` is
importable on a host whose CUDA kernels can never build: gsplat JIT-compiles
them through `nvcc` on first use, and a GPU image with the CUDA runtime but no
toolkit disables the backend silently, so the first `rasterization` call raises
`AttributeError: 'NoneType' object has no attribute 'CameraModelType'`.

On any host without the `sim-gs` extra the probe answers from its import guard,
so everything after that guard - the CUDA-device check, the trial
rasterization, and the `except` that turns a disabled backend into a reason -
was never executed by the suite. The one existing pin asserts the probe returns
`(bool, non-empty str)` on whatever host it runs on, which a probe that
returned `True` straight off the import would satisfy just as well.

Standing `torch` and `gsplat` in makes all three outcomes reachable: no CUDA
device, an importable rasterizer that cannot rasterize, and a working one. The
`ok` case additionally asserts that a rasterization really ran - on the CUDA
device, at a non-zero size - which is the half no import check can supply and
the reason the function is not a `hasattr` probe.

`strands_robots/rendering/backgrounds.py` goes from 97% to 100% statement
coverage (15 uncovered lines to 0).
