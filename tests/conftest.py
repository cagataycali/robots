import pytest

from strands_robots.simulation import Simulation


@pytest.fixture(autouse=True)
def _fresh_default_world():
    """Every test starts without a process-global default world and leaves none behind."""
    if Simulation._default is not None:
        Simulation._default.close()
    Simulation._default = None
    yield
    if Simulation._default is not None:
        Simulation._default.close()
    Simulation._default = None
