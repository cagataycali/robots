"""Simulation ABC — backend-agnostic interface for all simulation backends.

Every simulation backend (MuJoCo, Isaac, Newton) implements this interface.
Agent tools and the Robot() factory interact through these methods only —
they never touch backend-specific APIs directly.

Usage::

    from strands_robots.simulation import Simulation  # returns MuJoCo by default

    # Or explicitly:
    from strands_robots.simulation.mujoco import MuJoCoSimulation

    # Future:
    from strands_robots.simulation.isaac import IsaacSimulation
    from strands_robots.simulation.newton import NewtonSimulation
"""

from abc import ABC, abstractmethod
from typing import Any


class SimulationBackend(ABC):
    """Abstract base class for simulation backends.

    Defines the contract that all backends (MuJoCo, Isaac, Newton) must
    implement. This is the *programmatic* API — the AgentTool layer
    wraps it with tool_spec/stream for LLM access.

    Lifecycle::

        sim = SomeBackend()
        sim.create_world()
        sim.add_robot("so100", data_config="so100")
        sim.add_object("cube", shape="box", position=[0.3, 0, 0.05])

        # Control loop
        obs = sim.get_observation("so100")
        sim.send_action({"joint_0": 0.5}, robot_name="so100")
        sim.step(n_steps=10)

        # Render
        result = sim.render(camera_name="default")

        # Cleanup
        sim.destroy()
    """

    # --- World lifecycle ---

    @abstractmethod
    def create_world(
        self,
        timestep: float = None,
        gravity: list[float] = None,
        ground_plane: bool = True,
    ) -> dict[str, Any]:
        """Create a new simulation world."""
        ...

    @abstractmethod
    def destroy(self) -> dict[str, Any]:
        """Destroy the simulation world and release resources."""
        ...

    @abstractmethod
    def reset(self) -> dict[str, Any]:
        """Reset simulation to initial state."""
        ...

    @abstractmethod
    def step(self, n_steps: int = 1) -> dict[str, Any]:
        """Advance simulation by n physics steps."""
        ...

    @abstractmethod
    def get_state(self) -> dict[str, Any]:
        """Get full simulation state summary."""
        ...

    # --- Robot management ---

    @abstractmethod
    def add_robot(
        self,
        name: str,
        urdf_path: str = None,
        data_config: str = None,
        position: list[float] = None,
        orientation: list[float] = None,
    ) -> dict[str, Any]:
        """Add a robot to the simulation."""
        ...

    @abstractmethod
    def remove_robot(self, name: str) -> dict[str, Any]:
        """Remove a robot from the simulation."""
        ...

    # --- Object management ---

    @abstractmethod
    def add_object(
        self,
        name: str,
        shape: str = "box",
        position: list[float] = None,
        size: list[float] = None,
        color: list[float] = None,
        mass: float = 0.1,
        is_static: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """Add an object to the scene."""
        ...

    @abstractmethod
    def remove_object(self, name: str) -> dict[str, Any]:
        """Remove an object from the scene."""
        ...

    # --- Observation / Action ---

    @abstractmethod
    def get_observation(self, robot_name: str = None, camera_name: str = None) -> dict[str, Any]:
        """Get observation from simulation (Robot ABC compatible)."""
        ...

    @abstractmethod
    def send_action(self, action: dict[str, Any], robot_name: str = None, n_substeps: int = 1) -> None:
        """Apply action to simulation (Robot ABC compatible)."""
        ...

    # --- Rendering ---

    @abstractmethod
    def render(self, camera_name: str = "default", width: int = None, height: int = None) -> dict[str, Any]:
        """Render a camera view."""
        ...

    # --- Optional overrides (have default no-op implementations) ---

    def load_scene(self, scene_path: str) -> dict[str, Any]:
        """Load a complete scene from file. Override per backend."""
        return {"status": "error", "content": [{"text": "load_scene not supported by this backend"}]}

    def run_policy(self, robot_name: str, policy_provider: str = "mock", **kwargs) -> dict[str, Any]:
        """Run a policy loop. Override per backend."""
        return {"status": "error", "content": [{"text": "run_policy not supported by this backend"}]}

    def randomize(self, **kwargs) -> dict[str, Any]:
        """Apply domain randomization. Override per backend."""
        return {"status": "error", "content": [{"text": "randomize not supported by this backend"}]}

    def get_contacts(self) -> dict[str, Any]:
        """Get contact information. Override per backend."""
        return {"status": "success", "content": [{"text": "No contact support in this backend"}]}

    def cleanup(self):
        """Release all resources. Called on __del__ / context exit."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.cleanup()

    def __del__(self):
        try:
            self.cleanup()
        except (FileNotFoundError, ValueError, KeyError):
            pass  # Best-effort cleanup during object destruction
