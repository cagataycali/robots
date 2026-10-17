### Added: the dashboard, the world in a browser tab

`Dashboard(sim).start()` serves one page on a loopback port that draws the very Simulation it is given (MuJoCo
WASM in the page, state frames over a websocket): every robot's joints live, an e-stop per robot that shows the
latch, every world camera's picture (`GET /api/cameras/<name>.png`, what a policy sees), robots added at runtime, the mesh's devices next to the world's, a chat over `robot.tool` / `sim.tool`, and
a policy run from the page. Every write from the page is the operator gate's (a real body needs `X-Confirm`), and
cross-site requests are refused. With no server the page runs standalone: the browser steps the world and runs the
ONNX skills itself (onnxruntime-web); `python -m strands_robots.dashboard.standalone` builds that bundle. The MuJoCo
and onnxruntime WASM builds are vendored, pinned by sha256, so the page works on a robot LAN with no internet.
`[dashboard]` extra; without it every entry point (`app`, `start`, `serve`) names the extra. The page changes the
world with `sim.tool`'s own verbs (`POST /api/world`).
