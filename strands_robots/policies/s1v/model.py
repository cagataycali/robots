"""The S1V decider: a small non-autoregressive typed-decision transformer.

One forward pass over ``[camera tokens, proprio, task, candidate, questions]``
answers every question in the fixed typed set at once::

    joint      choice over 7   (5 arm joints, gripper, none = hold)
    direction  choice over 2   (+, -)
    size       choice over 3   (small, medium, large)
    cube_in_jaws, progress_if_executed, safe   noul (sigmoid), conditioned on a
                                               candidate primitive token

The choice questions are masked away from the candidate token, so the same
forward pass gives an unconditioned decision and a conditioned gate for any
candidate. Camera tokens come from a frozen DINOv2-small (CLS + pooled patch
grid, see :mod:`dataset`), so the decider itself is ~3M parameters and runs in
a few milliseconds.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .primitives import SIZE_LABELS, SO101_ARM_LABELS, SO101_PRIMITIVES

CHOICE_HEADS = {"joint": len(SO101_ARM_LABELS) + 2, "direction": 2, "size": len(SIZE_LABELS)}
NOUL_HEADS = ("cube_in_jaws", "progress_if_executed", "safe")
QUESTIONS = tuple(CHOICE_HEADS) + NOUL_HEADS
N_PRIMITIVES = len(SO101_PRIMITIVES)
CANDIDATE_NONE = N_PRIMITIVES  # "no candidate" embedding used by the choice heads
STATE_SCALE = torch.tensor([180.0, 180.0, 180.0, 180.0, 180.0, 100.0])


@dataclass
class S1VConfig:
    """Architecture and calibration knobs stored next to the weights as ``config.json``."""

    feat_dim: int = 384
    tokens_per_camera: int = 17
    cameras: tuple[str, ...] = ("scene", "wrist")
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 3
    ffn: int = 512
    dropout: float = 0.1
    use_proprio: bool = True
    n_tasks: int = 2
    temperatures: dict[str, float] = field(default_factory=dict)
    backbone: str = "facebook/dinov2-small"
    grid: int = 4

    def save(self, path: Path) -> None:
        """Write the config as JSON."""
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> S1VConfig:
        """Read a config written by :meth:`save`."""
        raw = json.loads(Path(path).read_text())
        raw["cameras"] = tuple(raw.get("cameras", ("scene", "wrist")))
        return cls(**raw)


class Block(nn.Module):
    """Pre-norm transformer block on ``F.scaled_dot_product_attention`` (CUDA-graph capturable)."""

    def __init__(self, d: int, n_heads: int, ffn: int, dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Linear(ffn, d))
        self.drop = nn.Dropout(dropout)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, blocked: torch.Tensor) -> torch.Tensor:
        """One masked self-attention + MLP step; ``blocked`` is a bool ``(S, S)`` mask, True = may not attend."""
        b, s, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).reshape(b, s, 3, self.n_heads, d // self.n_heads).permute(2, 0, 3, 1, 4)
        # SDPA bool mask: True = may attend
        a = F.scaled_dot_product_attention(
            q, k, v, attn_mask=~blocked, dropout_p=self.dropout if self.training else 0.0
        )
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(b, s, d)))
        return x + self.drop(self.mlp(self.ln2(x)))


class S1VDecider(nn.Module):
    """Typed-decision transformer over camera tokens, proprio, task, candidate and question embeddings."""

    def __init__(self, cfg: S1VConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.cam_proj = nn.Linear(cfg.feat_dim, d)
        self.cam_pos = nn.Parameter(torch.zeros(len(cfg.cameras), cfg.tokens_per_camera, d))
        self.proprio = nn.Sequential(nn.Linear(6, d), nn.GELU(), nn.Linear(d, d))
        self.proprio_blank = nn.Parameter(torch.zeros(1, 1, d))
        self.task_emb = nn.Embedding(cfg.n_tasks, d)
        self.cand_emb = nn.Embedding(N_PRIMITIVES + 1, d)
        self.question_emb = nn.Parameter(torch.zeros(len(QUESTIONS), d))
        self.type_emb = nn.Embedding(5, d)  # camera, proprio, task, candidate, question
        self.blocks = nn.ModuleList([Block(d, cfg.n_heads, cfg.ffn, cfg.dropout) for _ in range(cfg.n_layers)])
        self.norm = nn.LayerNorm(d)
        self.heads = nn.ModuleDict({k: nn.Linear(d, n) for k, n in CHOICE_HEADS.items()})
        self.heads.update({k: nn.Linear(d, 1) for k in NOUL_HEADS})
        nn.init.normal_(self.cam_pos, std=0.02)
        nn.init.normal_(self.question_emb, std=0.02)
        nn.init.normal_(self.proprio_blank, std=0.02)
        self.register_buffer("state_scale", STATE_SCALE.clone(), persistent=False)
        self.register_buffer("attn_mask", self._build_mask(), persistent=False)
        types = torch.cat(
            [
                torch.zeros(self.n_cam_tokens, dtype=torch.long),
                torch.tensor([1, 2, 3]),
                torch.full((len(QUESTIONS),), 4),
            ]
        )
        self.register_buffer("token_types", types, persistent=False)

    @property
    def n_cam_tokens(self) -> int:
        """Number of camera tokens across all cameras."""
        return len(self.cfg.cameras) * self.cfg.tokens_per_camera

    def _build_mask(self) -> torch.Tensor:
        """Bool mask (S, S), True = blocked.

        The candidate token and the noul question tokens (which read it) are
        invisible to every other token, so the choice heads are exactly
        candidate-invariant however deep the encoder is.
        """
        n = self.n_cam_tokens + 3 + len(QUESTIONS)
        cand = self.n_cam_tokens + 2
        q0 = self.n_cam_tokens + 3
        conditioned = [cand] + [q0 + i for i, q in enumerate(QUESTIONS) if q in NOUL_HEADS]
        mask = torch.zeros(n, n, dtype=torch.bool)
        for row in range(n):
            if row in conditioned:
                continue
            mask[row, conditioned] = True
        return mask

    def forward(
        self,
        cams: torch.Tensor,
        state: torch.Tensor,
        task: torch.Tensor,
        candidate: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """cams (B, n_cams*tokens, feat) · state (B, 6) · task (B,) · candidate (B,) -> logits per head."""
        b = cams.shape[0]
        x_cam = self.cam_proj(cams.float()) + self.cam_pos.reshape(1, -1, self.cfg.d_model)
        if self.cfg.use_proprio:
            x_pro = self.proprio(state / self.state_scale).unsqueeze(1)
        else:
            x_pro = self.proprio_blank.expand(b, 1, -1)
        x_task = self.task_emb(task).unsqueeze(1)
        x_cand = self.cand_emb(candidate).unsqueeze(1)
        x_q = self.question_emb.unsqueeze(0).expand(b, -1, -1)
        x = torch.cat([x_cam, x_pro, x_task, x_cand, x_q], dim=1) + self.type_emb(self.token_types).unsqueeze(0)
        for block in self.blocks:
            x = block(x, self.attn_mask)
        h = self.norm(x)
        q0 = self.n_cam_tokens + 3
        out: dict[str, torch.Tensor] = {}
        for i, q in enumerate(QUESTIONS):
            logits = self.heads[q](h[:, q0 + i])
            out[q] = logits.squeeze(-1) if q in NOUL_HEADS else logits
        return out

    def probabilities(self, logits: dict[str, torch.Tensor], *, calibrated: bool = True) -> dict[str, torch.Tensor]:
        """Softmax / sigmoid with the fitted temperatures (identity when uncalibrated or unfitted)."""
        probs = {}
        for q, z in logits.items():
            t = self.cfg.temperatures.get(q, 1.0) if calibrated else 1.0
            probs[q] = torch.softmax(z / t, dim=-1) if q in CHOICE_HEADS else torch.sigmoid(z / t)
        return probs

    def save_pretrained(self, path: Path) -> None:
        """Write ``config.json`` + ``model.pt`` into ``path``."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.cfg.save(path / "config.json")
        torch.save(self.state_dict(), path / "model.pt")

    @classmethod
    def from_pretrained(cls, path: Path, device: str = "cpu") -> S1VDecider:
        """Load a checkpoint directory written by :meth:`save_pretrained` in eval mode."""
        path = Path(path)
        cfg = S1VConfig.load(path / "config.json")
        model = cls(cfg)
        model.load_state_dict(torch.load(path / "model.pt", map_location=device))
        return model.to(device).eval()


def count_parameters(model: nn.Module) -> int:
    """Trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def describe(model: S1VDecider) -> dict[str, Any]:
    """Parameter count, question list and config as a plain dict."""
    return {"parameters": count_parameters(model), "questions": list(QUESTIONS), "config": asdict(model.cfg)}
