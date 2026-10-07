"""strands-robots policy provider for plastic-wam.

    from strands_robots.policies.plastic_wam import PlasticWAMPolicy
    policy = PlasticWAMPolicy("cagataydev/plastic-wam-small-r1", plastic=True)
    actions = policy.get_actions_sync(observation, "reach the red cube")    # list of {joint_key: target}

Joint units: the checkpoint speaks LeRobot SO-101 units (deg ×5 + gripper %); `joint_units="rad"` (MuJoCo sim)
converts with strands-robots' flux3 UnitAdapter (same calibration as that provider). Self-learning hooks:
`learn_from_correction(chunk)`, `reset_plastic()` (bit-exact base), `save_brain()/load_brain()`.
"""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import torch

from strands_robots.policies.base import Policy as _Base

_INSTALL = "plastic-wam provider needs the `plastic_wam` package: pip install 'git+https://github.com/cagataycali/plastic-model'"


class PlasticWAMPolicy(_Base):
    def __init__(self, pretrained_name_or_path: str = "cagataydev/plastic-wam-small-r1", device: str = "cuda",
                 camera_map: list[str] | None = None, joint_units: str = "rad", embodiment: str = "so101",
                 execute_steps: int = 16, plastic: bool = False, plastic_lr: float = 1e-2, token: str | None = None,
                 **_: Any):
        from huggingface_hub import snapshot_download
        try:
            import plastic_wam  # noqa: F401
        except ImportError as e:
            raise ImportError(_INSTALL) from e
        from plastic_wam.data.soup import SoupIndex
        from plastic_wam.features import S2Featurizer
        from plastic_wam.runtime import WAMPolicy, soup_median_norm
        tok = token or os.environ.get("HF_TOKEN")
        ck = snapshot_download(pretrained_name_or_path, token=tok)
        cfg = json.load(open(os.path.join(ck, "config.json")))
        soup = snapshot_download(cfg["soup"], repo_type="dataset", token=tok)
        norm = soup_median_norm(SoupIndex(soup, horizon=cfg["s1"]["horizon"], repos=cfg["repos"]))
        self._w = WAMPolicy(ck, S2Featurizer(cfg["s2"], token=tok, device=device), norm, device=device,
                            execute=execute_steps, embodiment=embodiment)
        self.camera_map = camera_map or ["scene", "wrist"]
        self.units = None
        if joint_units == "rad":
            from strands_robots.policies.flux3_action.units import UnitAdapter
            self.units = UnitAdapter()
        self.state_keys: list[str] = [str(i) for i in range(1, 7)]
        self.learner = None
        if plastic:
            from plastic_wam.plastic import LearnerConfig, PlasticLearner
            self.learner = PlasticLearner(self._w.m, LearnerConfig(rank=16, lr=plastic_lr))

    # ------------------------------------------------------------------ strands-robots Policy API
    @property
    def provider_name(self) -> str:
        return "plastic_wam"

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        self.state_keys = [k for k in robot_state_keys if not k.endswith(".vel")][:6]

    def reset(self, seed: int | None = None) -> None:
        self._w.reset()                       # episode start: TTT fast weights → W0 (plastic LoRA persists)
        if seed is not None:
            self._w.g.manual_seed(int(seed))

    def _image(self, obs, key):
        for k in (key, f"observation.images.{key}"):
            if k in obs:
                return np.asarray(obs[k])
        raise KeyError(f"camera {key!r} not in observation (have {sorted(k for k in obs if not isinstance(obs[k], float))})")

    async def get_actions(self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any) -> list[dict[str, Any]]:
        joints = [float(observation_dict[k]) for k in self.state_keys]
        state = np.array(self.units.robot_to_model(joints) if self.units else joints, np.float32)
        obs = {"images": [self._image(observation_dict, c) for c in self.camera_map[:2]], "state_lerobot": state,
               "instruction": instruction}
        chunk = self._w(obs)
        out = []
        for a in chunk:
            q = self.units.model_to_robot([float(v) for v in a]) if self.units else [float(v) for v in a]
            out.append({k: q[i] for i, k in enumerate(self.state_keys)})
        return out

    # ------------------------------------------------------------------ self-learning hooks
    def learn_from_correction(self, corrected_chunk_robot_units: list[list[float]], updates: int = 2) -> dict | None:
        """Corrected chunk for the LAST observation (robot units) → bounded plastic update."""
        if self.learner is None:
            raise RuntimeError("construct with plastic=True")
        from plastic_wam.plastic import Record
        conv = [self.units.robot_to_model(c) if self.units else c for c in corrected_chunk_robot_units]
        e = self._w.emb; aq, asp = self._w.aq, self._w.asp
        canon = np.stack([e.to_canonical(np.asarray(c, np.float32)) for c in conv])
        tgt = torch.from_numpy(np.clip((2 * (canon - aq) / asp - 1) * e.mask, -1.5, 1.5).astype(np.float32))
        last = self._w.last
        rec = Record(state=last["state"], emb_id=e.id, ctx=last["ctx"], ctx_mask=last["mask"], target=tgt,
                     mask=torch.from_numpy(e.mask), label="correction")
        out = None
        for i in range(updates):
            out = self.learner.step(rec) if i == 0 else self.learner.learn(rec)
        return out

    def reset_plastic(self):
        if self.learner is not None:
            self.learner.reset()

    def save_brain(self, path: str):
        self.learner.save_brain(path)

    def load_brain(self, path: str):
        self.learner.load_brain(path)
