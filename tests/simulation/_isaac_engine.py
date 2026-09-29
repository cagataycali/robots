"""An :class:`~strands_robots.simulation.isaac.simulation.IsaacSimulation` with no Kit.

``IsaacSimulation.__init__`` launches nothing - Isaac Sim starts in
``create_world`` - so a test that drives one method against stand-in handles
starts from the state the real constructor builds and overrides only what it
models. Restating the constructor attribute by attribute on a ``__new__``
skeleton falls behind the moment ``__init__`` gains a field, and
``tests/simulation/test_isaac_stand_ins_start_from_the_constructor.py`` refuses it.

The finalizer is disarmed: ``SimEngine.__del__`` runs ``cleanup()`` - and so
``destroy()`` - on a fully constructed engine, which at garbage collection would
drive this test's stand-ins from inside whichever test happens to be running.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from strands_robots.simulation.isaac.config import IsaacConfig
    from strands_robots.simulation.isaac.simulation import IsaacSimulation


def isaac_engine(config: IsaacConfig | None = None) -> IsaacSimulation:
    """Return an engine carrying every attribute ``__init__`` sets, finalizer off."""
    # Imported here so a module that only imports this helper pays for the Isaac
    # backend when it builds an engine, as it did when it imported it lazily.
    from strands_robots.simulation.isaac.simulation import IsaacSimulation

    engine = IsaacSimulation(config)
    engine._init_complete = False
    return engine
