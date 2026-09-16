### Added: `render_dir=` on `Simulation()` / `Robot(name, mode="sim")`

`render(output_path=...)` confines the model-supplied path to a render sandbox
(`~/.strands_robots/renders`, or `STRANDS_ROBOTS_RENDER_ROOT`). Until now the
only way to point it somewhere else was an environment variable set before the
process started, so an agent asked to "save the frame to my project folder"
could only refuse or write into the sandbox and leave the file there.

`Simulation(render_dir="./shots")` - or `Robot("so101", render_dir="./shots")`
through the factory's `**kwargs` - makes that directory THIS Simulation's
sandbox: a bare filename lands in it, an absolute path under it is accepted,
one outside it is refused as before (the refusal names the directory), and it
is created on first use. `None` keeps the process-wide sandbox. A value that
cannot name a directory (a non-path type, an empty string) is refused by the
constructor; a misspelling is refused like every other constructor keyword.
