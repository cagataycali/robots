"""Measure the prefix the RTC denoiser is actually fed, per fallback, per tree."""
import json, logging, os, pathlib, sys
import torch
from unittest.mock import MagicMock, patch
from lerobot.processor import AbsoluteActionsProcessorStep, RelativeActionsProcessorStep
from lerobot.processor.converters import create_transition
from lerobot.processor.pipeline import DataProcessorPipeline
from lerobot.utils.constants import OBS_STATE
import strands_robots.policies.lerobot_local.policy as pol_mod
from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy
from strands_robots.policies.lerobot_local.processor import ProcessorBridge

TREE = str(pathlib.Path(pol_mod.__file__).parents[3])
A, T, EH = 4, 6, 2
S1 = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
S2 = torch.tensor([[10.0, 20.0, 30.0, 40.0]])


def chunk(): return torch.arange(T * A, dtype=torch.float32).reshape(1, T, A)
def leftover(): return chunk().squeeze(0)[EH:]


class Rec(logging.Handler):
    def __init__(self): super().__init__(); self.recs = []
    def emit(self, r): self.recs.append((r.levelname, r.getMessage()))


def build(post_kind):
    names = [f"j{i}.pos" for i in range(A)]
    rel = RelativeActionsProcessorStep(enabled=True, action_names=names)
    pre = DataProcessorPipeline(steps=[rel])
    post = (DataProcessorPipeline(steps=[AbsoluteActionsProcessorStep(enabled=True, relative_step=rel)])
            if post_kind == "absolute" else None)
    with patch.object(LerobotLocalPolicy, "_load_model"):
        p = LerobotLocalPolicy(pretrained_name_or_path="test/model")
    p._loaded = True; p._device = torch.device("cpu")
    inner = MagicMock(); inner.config = MagicMock()
    inner.config.action_feature_names = names
    cap = []
    def _pred(_b, **kw):
        cap.append(kw.get("prev_chunk_left_over")); return chunk()
    inner.predict_action_chunk.side_effect = _pred
    p._policy = inner
    p._rtc_enabled = True; p._rtc_execution_horizon = EH; p.rtc_observed_delay_steps = 0
    p._processor_bridge = ProcessorBridge(preprocessor=pre, postprocessor=post)
    return p, rel, cap


def run(label, post_kind, *, non_tensor=False, absolute_action=False):
    p, rel, cap = build(post_kind)
    if non_tensor:
        p._processor_bridge.postprocess = lambda a: {"action": a}
    h = Rec(); pol_mod.logger.addHandler(h)
    lvl = pol_mod.logger.level; pol_mod.logger.setLevel(logging.DEBUG)
    try:
        rel(create_transition(observation={OBS_STATE: S1}))
        if absolute_action:
            p._rtc_rebase_resolved = True  # skip detection: behave as absolute-action
        with torch.inference_mode(): p._predict_with_rtc({})
        rel(create_transition(observation={OBS_STATE: S2}))
        with torch.inference_mode(): p._predict_with_rtc({})
    finally:
        pol_mod.logger.removeHandler(h); pol_mod.logger.setLevel(lvl)
    lm = leftover()
    prefix = cap[1]
    warns = [m for lv, m in h.recs if lv == "WARNING"]
    return {
        "label": label,
        "relative_detected": p._rtc_relative_step is not None,
        "prefix": [[round(v, 3) for v in row] for row in prefix.tolist()] if prefix is not None else None,
        "expected_reanchored": [[round(v, 3) for v in row] for row in (lm + S1 - S2).tolist()],
        "stale_model_space": [[round(v, 3) for v in row] for row in lm.tolist()],
        "is_stale": bool(prefix is not None and torch.allclose(prefix, lm)),
        "is_reanchored": bool(prefix is not None and torch.allclose(prefix, lm + S1 - S2, atol=1e-5)),
        "n_warnings": len(warns),
        "warning": warns[0] if warns else None,
        "info_enabled": any("re-anchoring enabled" in m for lv, m in h.recs if lv == "INFO"),
    }


rows = [
    run("healthy (postprocessor converts)", "absolute"),
    run("no postprocessor", None),
    run("postprocessor yields a dict", "absolute", non_tensor=True),
    run("absolute-action policy (benign)", "absolute", absolute_action=True),
]
out = {"tree": TREE, "rows": rows,
       "state_shift": [round(v, 3) for v in (S2 - S1).squeeze(0).tolist()]}
dest = pathlib.Path(sys.argv[1]); dest.write_text(json.dumps(out, indent=2))
print("TREE:", TREE, "->", dest)
for r in rows:
    print(f"  {r['label']:36s} stale={r['is_stale']!s:5} reanch={r['is_reanchored']!s:5} warns={r['n_warnings']}")
