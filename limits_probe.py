"""Empirically verify _articulation_dof_limits' two documented sources and every
'no usable bounds' outcome, plus the reader/writer failure surfaces."""
import pathlib, sys, types, json
import numpy as np
import strands_robots
print("TREE:", pathlib.Path(strands_robots.__file__).parents[1])

# The one isaacsim type the adapter lazily imports.
for n in ("isaacsim", "isaacsim.core", "isaacsim.core.utils", "isaacsim.core.utils.types"):
    sys.modules.setdefault(n, types.ModuleType(n))
class _AA:
    def __init__(self, joint_positions=None, joint_indices=None):
        self.joint_positions, self.joint_indices = joint_positions, joint_indices
sys.modules["isaacsim.core.utils.types"].ArticulationAction = _AA

from strands_robots.simulation.isaac.motion_primitives import IsaacMotionPrimitivesMixin as M

dof_limits = M._articulation_dof_limits
read_q = M._read_joint_positions

def structured(spans, with_has_limits=True):
    fields = [("hasLimits", "?"), ("lower", "f8"), ("upper", "f8")] if with_has_limits \
             else [("lower", "f8"), ("upper", "f8")]
    a = np.zeros(len(spans), dtype=fields)
    for i, s in enumerate(spans):
        if s is None:
            if with_has_limits: a["hasLimits"][i] = False
        else:
            if with_has_limits: a["hasLimits"][i] = True
            a["lower"][i], a["upper"][i] = s
    return a

class A:  # bare articulation; attributes added per case
    pass

class TorchLike:
    def __init__(self, arr): self._a = arr
    def cpu(self): return self
    def numpy(self): return self._a

out = {}
def cell(label, art, n=3, note=""):
    try:
        r = dof_limits(art, n)
    except Exception as e:
        r = f"RAISED {type(e).__name__}: {e}"
    out[label] = {"result": str(r), "note": note}
    print(f"  {label:38s} -> {r}   {note}")

print("\n[_articulation_dof_limits: the two documented SOURCES]")
a = A(); a.dof_properties = structured([(-1.0, 1.0), (-2.0, 2.0), (0.0, 0.5)])
cell("props authoritative (happy)", a)

a = A(); a.get_dof_limits = lambda: np.array([[-1.0, 1.0], [-2.0, 2.0], [0.0, 0.5]])
cell("FALLBACK get_dof_limits (no props)", a, note="whole fallback surface")

a = A(); a.get_dof_limits = lambda: TorchLike(np.array([[-1.0, 1.0], [-2.0, 2.0], [0.0, 0.5]]))
cell("FALLBACK torch-like .cpu()", a)

a = A(); a.get_dof_limits = lambda: np.array([[[-1.0, 1.0], [-2.0, 2.0], [0.0, 0.5]]])
cell("FALLBACK view-shaped (1,n,2)", a, note="Isaac ArticulationView shape")

a = A(); a.dof_properties = np.zeros(3)   # plain array: props['lower'] raises
a.get_dof_limits = lambda: np.array([[-1.0, 1.0], [-2.0, 2.0], [0.0, 0.5]])
cell("props UNREADABLE -> fallback", a, note="lines 237-238 then 245-251")

a = A(); a.dof_properties = structured([(-1.0, 1.0)]*3, with_has_limits=False)
cell("props without hasLimits field", a, note="lines 242-243")

print("\n[_articulation_dof_limits: every 'no usable bounds' outcome]")
a = A(); a.get_dof_limits = lambda: (_ for _ in ()).throw(RuntimeError("stage torn down"))
cell("fallback RAISES", a, note="252-253 then 256-258")
cell("NEITHER source", A(), note="256-258")
a = A(); a.get_dof_limits = lambda: np.array([[-1.0, 1.0]])
cell("bounds SHORTER than n_dofs", a, note="256-258 via dof >= size")
a = A(); a.dof_properties = structured([(-1.0, 1.0), (0.0, float("inf")), (float("nan"), 1.0)])
cell("non-finite bound", a, note="263-265")
a = A(); a.dof_properties = structured([(-1.0, 1.0), (1.0, 1.0), (2.0, 0.5)])
cell("degenerate hi <= lo", a, note="263-265")
a = A(); a.dof_properties = structured([(-1.0, 1.0), None, (0.0, 0.5)])
cell("hasLimits False (already covered)", a, note="259-261")

print("\n[_read_joint_positions]")
class RaiseQ:
    def get_joint_positions(self): raise RuntimeError("articulation torn down")
class NoneQ:
    def get_joint_positions(self): return None
class TorchQ:
    def get_joint_positions(self): return TorchLike(np.array([0.1, 0.2, 0.3]))
for label, art in (("raises -> None", RaiseQ()), ("returns None -> None", NoneQ()),
                   ("torch tensor .cpu().numpy()", TorchQ())):
    r = read_q(art)
    out["read: " + label] = str(r)
    print(f"  {label:38s} -> {r}")

pathlib.Path("/tmp/limits-probe.json").write_text(json.dumps(out, indent=2))
print("\nOK")
