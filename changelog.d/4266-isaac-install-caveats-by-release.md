### Docs: the Isaac Sim install caveats name the release they apply to

The pip caveats now say that 6.0.x pins `coverage==7.4.4` (6.1 does not) and that both releases pin `numpy==2.3.1` and `torch==2.11.0` exactly, so a fresh venv is the safe route. 6.0.1.0 and 6.1.0.0 are documented as the verified wheels.
