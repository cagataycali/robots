### Fixed: `sim.add_robot("so101")` is the supported short form, not a deprecated one

The MuJoCo backend answered the call every quickstart teaches with a
`Warning: Hint: ... resolved via deprecated name-as-registry-key fallback`
line, while Isaac and Newton accepted the same call silently and the
simulation guide documents it as the way to add a registry robot. The notice
is gone: `add_robot(name)` with no model source resolves `name` in the model
registry on every backend, and `add_robot(name="arm", data_config="so101")`
remains the way to give an instance its own label. No removal is planned.
