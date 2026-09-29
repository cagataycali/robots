### Fixed: the Isaac Franka GPU integration tests run on Isaac Sim 6.0.1 and 6.1

The two Franka USD integration tests used a pre-6.0 asset path, resolved the assets root before Kit had booted, and expected observation keys without the `<joint>.vel` entries the backend reports, so both errored before exercising the backend. They now pass on 6.0.1 and 6.1.
